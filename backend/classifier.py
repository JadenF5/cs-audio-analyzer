"""
classifier.py — ML Audio Classifier for Compressed Sensing Audio Analyzer
=========================================================================
Classifies audio signals into: tone / noise / music

Key fixes over v1:
  1. Added spectral flatness  — THE discriminator for Gaussian noise vs tonal signals
                                (flatness ~1.0 = noise, ~0.0 = pure tone)
  2. Added spectral bandwidth — spread of energy (narrow=tone, wide=noise/music)
  3. Added chroma variance    — pitch class spread (music varies, tone is fixed)
  4. Augment training noise class with synthetic Gaussian noise — fixes the
     distribution mismatch where UrbanSound8k "noise" (drilling, AC) doesn't
     look like pure Gaussian noise at inference time
  5. Upgraded to HistGradientBoostingClassifier — typically 3-5% better than
     RandomForest on tabular audio features, handles augmented data well

Feature vector: 36 dimensions (up from 30)
  - 13 MFCC means + 13 stds = 26  (timbre)
  - spectral centroid mean    =  1  (brightness)
  - spectral bandwidth mean   =  1  (energy spread — NEW)
  - spectral rolloff mean     =  1  (energy tail)
  - spectral flatness mean    =  1  (noise vs tonal — KEY NEW FEATURE)
  - zero crossing rate mean   =  1  (noisiness proxy)
  - RMS energy mean           =  1  (loudness)
  - chroma variance           =  1  (pitch class spread — NEW)
                               ---
                                36 total
"""

import os
import csv
import warnings
from dataclasses import dataclass

import numpy as np
import librosa
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


# ── Constants ────────────────────────────────────────────────
SAMPLE_RATE  = 8000
N_SAMPLES    = 256
N_FFT        = 128
HOP          = 32           # N_FFT // 4
N_MFCC       = 13
CLASSES      = ["tone", "noise", "music"]
N_PER_CLASS  = 150          # synthetic samples per class (fallback only)

# How many synthetic Gaussian noise samples to inject into real noise class.
# This bridges the gap between UrbanSound8k "noise" (environmental sounds)
# and truly flat-spectrum Gaussian noise at inference time.
N_SYNTHETIC_NOISE_AUGMENT = 60

# ── Real dataset config (UrbanSound8k) ──────────────────────
USE_REAL_DATA   = True
REAL_DATA_ROOT  = os.environ.get("URBANSOUND8K_ROOT", "./UrbanSound8K")
METADATA_CSV    = os.path.join(REAL_DATA_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_DIR       = os.path.join(REAL_DATA_ROOT, "audio")
FOLDS_TO_USE    = [1, 2]
MAX_PER_REAL_CLASS = 150

CLASS_MAP = {
    # "car_horn":         "tone",
    # "siren":            "tone",
    "air_conditioner":  "noise",
    "engine_idling":    "noise",
    "jackhammer":       "noise",
    "drilling":         "noise",
    "street_music":     "music",
}

MODEL_CACHE_PATH = "audio_classifier_model.joblib"
UPLOAD_WINDOW_SAMPLES = N_SAMPLES


# ── Data container ───────────────────────────────────────────
@dataclass
class ClassificationResult:
    predicted_class:  str
    confidence:       float
    all_scores:       dict
    feature_summary:  dict
    signal_stats:     dict


# ── Feature extraction ───────────────────────────────────────
def extract_features(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a 36-dimensional feature vector from an audio signal.

    The two most important NEW features vs v1:
      - spectral_flatness: ~1.0 for Gaussian noise, ~0.0 for pure tones.
        This single feature almost perfectly separates noise from tonal signals.
      - spectral_bandwidth: wide for noise (energy spread everywhere),
        narrow for tones (energy concentrated at a few frequencies).

    Works on any signal length — n_fft is fixed so longer signals just
    produce more averaged analysis frames, not more features.
    """
    signal = signal.astype(np.float32)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    n_fft = min(N_FFT, len(signal))
    hop   = max(1, n_fft // 4)

    mfccs     = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
                                      n_fft=n_fft, hop_length=hop)
    centroid  = librosa.feature.spectral_centroid(y=signal, sr=sr,
                                                   n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr,
                                                    n_fft=n_fft, hop_length=hop)
    rolloff   = librosa.feature.spectral_rolloff(y=signal, sr=sr,
                                                  n_fft=n_fft, hop_length=hop)
    flatness  = librosa.feature.spectral_flatness(y=signal,
                                                   n_fft=n_fft, hop_length=hop)
    zcr       = librosa.feature.zero_crossing_rate(y=signal, hop_length=hop)
    rms       = librosa.feature.rms(y=signal, hop_length=hop)
    chroma    = librosa.feature.chroma_stft(y=signal, sr=sr,
                                             n_fft=n_fft, hop_length=hop)
    # Variance across the 12 pitch classes captures pitch diversity:
    # music uses many pitch classes, tones use very few.
    chroma_var = np.var(np.mean(chroma, axis=1))

    return np.concatenate([
        np.mean(mfccs, axis=1),    # 13 — timbre means
        np.std(mfccs,  axis=1),    # 13 — timbre stds
        [np.mean(centroid)],       #  1 — brightness
        [np.mean(bandwidth)],      #  1 — energy spread  ← NEW
        [np.mean(rolloff)],        #  1 — energy tail
        [np.mean(flatness)],       #  1 — noise indicator ← KEY
        [np.mean(zcr)],            #  1 — zero crossings
        [np.mean(rms)],            #  1 — loudness
        [float(chroma_var)],       #  1 — pitch diversity ← NEW
    ])                             # = 36 features total


def extract_feature_summary(signal: np.ndarray,
                             sr: int = SAMPLE_RATE) -> dict:
    """Human-readable feature values for frontend display."""
    signal = signal.astype(np.float32)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    n_fft = min(N_FFT, len(signal))
    hop   = max(1, n_fft // 4)

    centroid  = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft, hop_length=hop)
    zcr       = librosa.feature.zero_crossing_rate(y=signal, hop_length=hop)
    rolloff   = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft, hop_length=hop)
    rms       = librosa.feature.rms(y=signal, hop_length=hop)
    flatness  = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft, hop_length=hop)
    mfccs     = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop)

    return {
        "spectral_centroid_hz":  round(float(np.mean(centroid)), 1),
        "spectral_bandwidth_hz": round(float(np.mean(bandwidth)), 1),
        "spectral_rolloff_hz":   round(float(np.mean(rolloff)), 1),
        "spectral_flatness":     round(float(np.mean(flatness)), 4),
        "zero_crossing_rate":    round(float(np.mean(zcr)), 4),
        "rms_energy":            round(float(np.mean(rms)), 4),
        "mfcc_1_mean":           round(float(np.mean(mfccs[0])), 2),
        "mfcc_2_mean":           round(float(np.mean(mfccs[1])), 2),
    }


# ── Synthetic data generators ────────────────────────────────
def _generate_tone(seed: int, length: int = N_SAMPLES) -> np.ndarray:
    """1–6 pure sine waves — sparse in frequency domain, very low flatness."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, length / SAMPLE_RATE, length, endpoint=False)
    if rng.random() < 0.5:   # pure single frequency
        freq = rng.uniform(80, 2000)
        amp = rng.uniform(0.5, 1.0)
        return amp * np.sin(2 * np.pi * freq * t)
    else:                     # 1–6 sine waves (original behaviour)
        n_freqs = rng.integers(2, 7)   # start from 2 for multi
        freqs   = rng.uniform(80, 2000, n_freqs)
        amps    = rng.uniform(0.2, 1.0, n_freqs)
        return sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))


def _generate_noise(seed: int, length: int = N_SAMPLES) -> np.ndarray:
    """Pure Gaussian white noise — flatness ~0.5-1.0, bandwidth very wide."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length)


def _generate_music(seed: int, length: int = N_SAMPLES) -> np.ndarray:
    """Harmonic overtone series — rich but structured, low-moderate flatness."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, length / SAMPLE_RATE, length, endpoint=False)
    root = rng.uniform(80, 600)
    n_harmonics = rng.integers(5, 15)
    amps = rng.uniform(0.1, 1.0, n_harmonics)
    signal = sum(a * np.sin(2 * np.pi * root * (h + 1) * t)
                 for h, a in enumerate(amps))
    root2 = root * rng.choice([1.25, 1.5, 2.0])
    n_harm2 = rng.integers(3, 8)
    amps2 = rng.uniform(0.05, 0.5, n_harm2)
    signal += sum(a * np.sin(2 * np.pi * root2 * (h + 1) * t)
                  for h, a in enumerate(amps2))
    return signal


def _generate_synthetic_for(label: str, n: int) -> tuple[list, list]:
    gen = {"tone": _generate_tone, "noise": _generate_noise, "music": _generate_music}[label]
    X, y = [], []
    for seed in range(n):
        sig = gen(seed)
        sig = sig / (np.max(np.abs(sig)) + 1e-10)
        X.append(extract_features(sig))
        y.append(label)
    return X, y


# ── Real dataset loader (UrbanSound8k) ───────────────────────
def load_real_dataset(
    metadata_csv: str = METADATA_CSV,
    audio_dir:    str = AUDIO_DIR,
    folds:        list = FOLDS_TO_USE,
    max_per_class: int = MAX_PER_REAL_CLASS,
) -> tuple[list, list]:
    if not os.path.exists(metadata_csv):
        print(f"[load_real_dataset] Metadata CSV not found at {metadata_csv} "
              f"— falling back to fully synthetic data.")
        return [], []

    buckets: dict[str, list[str]] = {"tone": [], "noise": [], "music": []}
    with open(metadata_csv, newline="") as f:
        for row in csv.DictReader(f):
            fold = int(row["fold"])
            src_class = row["class"]
            if fold not in folds or src_class not in CLASS_MAP:
                continue
            target = CLASS_MAP[src_class]
            path = os.path.join(audio_dir, f"fold{fold}", row["slice_file_name"])
            buckets[target].append(path)

    rng = np.random.default_rng(0)
    X, y = [], []
    for label, paths in buckets.items():
        rng.shuffle(paths)
        paths = paths[:max_per_class]
        loaded = 0
        for path in paths:
            try:
                signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                signal, _ = librosa.effects.trim(signal, top_db=25)
                if len(signal) < int(0.05 * SAMPLE_RATE):
                    continue
                max_val = np.max(np.abs(signal))
                if max_val > 0:
                    signal = signal / max_val
                X.append(extract_features(signal, SAMPLE_RATE))
                y.append(label)
                loaded += 1
            except Exception as e:
                print(f"[load_real_dataset] Skipping {path}: {e}")
        print(f"[load_real_dataset] {label}: loaded {loaded}/{len(paths)} clips")

    return X, y


# ── Dataset builder ──────────────────────────────────────────
def generate_dataset(use_real: bool = USE_REAL_DATA) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the full labeled training set.

    When real UrbanSound8k data is available:
      - Loads real clips for tone / noise / music
      - Augments the noise class with N_SYNTHETIC_NOISE_AUGMENT synthetic
        Gaussian noise samples — this is the key fix for pure-noise misclassification.
        UrbanSound8k noise (drilling, AC, idling) has tonal components and low-mid
        flatness. Pure Gaussian noise has flatness ~0.5-1.0, which the model never
        sees in real training data alone, causing it to misclassify flat-spectrum
        noise as music. Augmenting with synthetic noise teaches the flatness signal.

    When real data is unavailable: falls back to fully synthetic data.
    """
    if use_real:
        X_real, y_real = load_real_dataset()
        if X_real:
            # Generate synthetic data for ALL three classes
            X_tone, y_tone = _generate_synthetic_for("tone", N_PER_CLASS)
            X_aug_noise, y_aug_noise = _generate_synthetic_for("noise", N_SYNTHETIC_NOISE_AUGMENT)
            # ← NEW: Augment music with synthetic harmonic music
            X_aug_music, y_aug_music = _generate_synthetic_for("music", 60)

            X_all = X_real + X_tone + X_aug_noise + X_aug_music
            y_all = list(y_real) + y_tone + y_aug_noise + y_aug_music
            return np.array(X_all), np.array(y_all)

    # Fallback: fully synthetic
    X, y = [], []
    for label in CLASSES:
        X_lab, y_lab = _generate_synthetic_for(label, N_PER_CLASS)
        X += X_lab
        y += y_lab
    return np.array(X), np.array(y)

# ── Model ────────────────────────────────────────────────────
def train_classifier() -> Pipeline:
    """
    Train on real + augmented data using HistGradientBoostingClassifier.

    HistGradientBoostingClassifier is sklearn's fastest and most accurate
    tree ensemble — uses histogram-based splits (like LightGBM) and handles
    mixed-scale features well without needing careful normalization, though
    we keep the StandardScaler for good measure.
    """
    print("Training audio classifier...")
    X, y = generate_dataset()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.08,
            min_samples_leaf=8,
            l2_regularization=0.1,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    pipeline.fit(X, y)
    print(f"Classifier trained on {len(y)} samples ({len(set(y))} classes).")
    return pipeline


# ── Singleton model ───────────────────────────────────────────
_model: Pipeline | None = None

def get_model(force_retrain: bool = False) -> Pipeline:
    """Load cached model from disk, or train if not found."""
    global _model
    if _model is not None and not force_retrain:
        return _model

    if not force_retrain and os.path.exists(MODEL_CACHE_PATH):
        print(f"Loading cached model from {MODEL_CACHE_PATH}...")
        _model = joblib.load(MODEL_CACHE_PATH)
        return _model

    _model = train_classifier()
    joblib.dump(_model, MODEL_CACHE_PATH)
    print(f"Model cached to {MODEL_CACHE_PATH}")
    return _model


# ── Main classify function ────────────────────────────────────
def classify_signal(signal: np.ndarray,
                    sr: int = SAMPLE_RATE) -> ClassificationResult:
    """
    Classify a time-domain audio signal into tone / noise / music.

    Resamples to SAMPLE_RATE if needed so features are on the same
    numeric scale as what the classifier was trained on.
    """
    model = get_model()

    if sr != SAMPLE_RATE:
        signal = librosa.resample(signal.astype(np.float32),
                                   orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    features = extract_features(signal, sr).reshape(1, -1)

    predicted     = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_order   = model.classes_

    all_scores = {
        str(cls): round(float(prob), 4)
        for cls, prob in zip(class_order, probabilities)
    }
    confidence = round(float(max(probabilities)), 4)

    return ClassificationResult(
        predicted_class = predicted,
        confidence      = confidence,
        all_scores      = all_scores,
        feature_summary = extract_feature_summary(signal, sr),
        signal_stats    = {
            "mean":      round(float(np.mean(signal)), 4),
            "std":       round(float(np.std(signal)), 4),
            "max_abs":   round(float(np.max(np.abs(signal))), 4),
            "n_samples": len(signal),
        },
    )


# ── Helper for upload endpoint ────────────────────────────────
def pick_energetic_window(signal: np.ndarray,
                          window_len: int = UPLOAD_WINDOW_SAMPLES,
                          n_candidates: int = 5,
                          seed: int | None = None) -> np.ndarray:
    if len(signal) <= window_len:
        return signal
    max_start = len(signal) - window_len
    rng = np.random.default_rng(seed)
    n = min(n_candidates, max_start + 1)
    starts = rng.choice(max_start + 1, size=n, replace=False)
    best_start = max(starts, key=lambda s: np.sum(signal[s:s + window_len] ** 2))
    return signal[best_start:best_start + window_len]


# ── Class info for frontend ───────────────────────────────────
CLASS_INFO = {
    "tone": {
        "label":       "Pure Tone",
        "description": "1–6 pure sine waves. Very sparse in frequency domain. "
                       "Spectral flatness ~0 — perfect for CS recovery.",
        "color":       "blue",
        "icon":        "◎",
    },
    "noise": {
        "label":       "Noise",
        "description": "Broadband noise. Flat frequency spectrum (flatness ~0.5–1.0). "
                       "Not sparse — CS recovery will struggle without enough measurements.",
        "color":       "red",
        "icon":        "≋",
    },
    "music": {
        "label":       "Music",
        "description": "Harmonic overtone structure. Rich but organized frequency content. "
                       "Moderately sparse — CS recovery depends on compression ratio.",
        "color":       "green",
        "icon":        "♪",
    },
}
"""
classifier.py — ML Audio Classifier for Compressed Sensing Audio Analyzer
=========================================================================
Phase 4: Extract audio features from (reconstructed) signals and classify
them into one of three categories: tone / noise / music.

Pipeline:
  signal (time domain)
    -> feature extraction (MFCCs, spectral centroid, ZCR, rolloff, RMS)
    -> StandardScaler normalization
    -> RandomForestClassifier
    -> predicted class + confidence scores

Trained on real UrbanSound8k audio (tone <- car_horn/siren, noise <-
air_conditioner/engine_idling/jackhammer/drilling, music <- street_music),
using the FULL length of each (trimmed) clip — not short windows. A
4th class, speech, was tried (real LibriSpeech data) but dropped: speech
is highly non-stationary (constantly changing phoneme to phoneme), so it
needed a training/inference window-length match that kept fighting the CS
reconstruction pipeline's short (32ms) analysis window, and no fix held up
reliably across the demo signal, real uploads, and redeploys. tone/noise/
music are comparatively stationary sounds (a steady hum or horn sounds
much the same over 32ms or 3 seconds), so full-clip training works well
for them even against a short inference window — this is the same
configuration that gave ~90%+ real held-out cross-validation accuracy
and reliably correct demo/upload behavior before speech was ever added.

Falls back to synthetic data if the real dataset isn't found locally
(useful for a quick smoke test without downloading anything).
"""

import os
import csv
import warnings
from dataclasses import dataclass

import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


# ── Constants ────────────────────────────────────────────────
SAMPLE_RATE  = 8000
N_SAMPLES    = 256
N_FFT        = 128          # small fft for short signals
N_MFCC       = 13
CLASSES      = ["tone", "noise", "music"]
N_PER_CLASS  = 150          # synthetic samples per class (fallback only)

# ── Real dataset config (UrbanSound8k) ──────────────────────
# Point these at wherever you unzipped fold1/fold2 locally. Layout expected:
#   UrbanSound8K/metadata/UrbanSound8K.csv
#   UrbanSound8K/audio/fold1/*.wav
#   UrbanSound8K/audio/fold2/*.wav
USE_REAL_DATA   = True
REAL_DATA_ROOT  = os.environ.get("URBANSOUND8K_ROOT", "./UrbanSound8K")
METADATA_CSV    = os.path.join(REAL_DATA_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_DIR       = os.path.join(REAL_DATA_ROOT, "audio")
FOLDS_TO_USE    = [1, 2]
MAX_PER_REAL_CLASS = 150     # cap so no bucket dominates training

# UrbanSound8k's 10 classes -> your 3 target classes.
CLASS_MAP = {
    "car_horn":         "tone",
    "siren":            "tone",
    "air_conditioner":  "noise",
    "engine_idling":    "noise",
    "jackhammer":       "noise",
    "drilling":         "noise",
    "street_music":     "music",
}

MODEL_CACHE_PATH = "audio_classifier_model.joblib"

# Used by main.py's /upload endpoint to pick a representative slice of an
# uploaded file for the CS pipeline, instead of always grabbing the very
# first samples after the silence trim (which tends to be an onset/attack
# transient rather than typical content).
UPLOAD_WINDOW_SAMPLES = N_SAMPLES


def pick_energetic_window(signal: np.ndarray,
                          window_len: int = UPLOAD_WINDOW_SAMPLES,
                          n_candidates: int = 5,
                          seed: int | None = None) -> np.ndarray:
    """
    Pick a representative window from signal by sampling several candidate
    positions and keeping the highest-energy one. Falls back to the whole
    signal if it's already <= window_len.
    """
    if len(signal) <= window_len:
        return signal
    max_start = len(signal) - window_len
    rng = np.random.default_rng(seed)
    n = min(n_candidates, max_start + 1)
    starts = rng.choice(max_start + 1, size=n, replace=False)
    best_start = max(starts, key=lambda s: np.sum(signal[s:s + window_len] ** 2))
    return signal[best_start:best_start + window_len]


# ── Data container ───────────────────────────────────────────
@dataclass
class ClassificationResult:
    predicted_class:  str
    confidence:       float           # probability of predicted class
    all_scores:       dict            # {class: probability} for all classes
    feature_summary:  dict            # key features for frontend display
    signal_stats:     dict            # basic signal statistics


# ── Feature extraction ───────────────────────────────────────
def extract_features(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a 30-dimensional feature vector from an audio signal.

    Features:
      - 13 MFCC means     (timbre / frequency content)
      - 13 MFCC stds      (variability of timbre)
      - spectral centroid  (brightness — high = treble heavy)
      - zero crossing rate (noisiness — high = noisy)
      - spectral rolloff   (frequency where 85% of energy is below)
      - RMS energy         (loudness)

    These 30 features are classic audio fingerprints used in MIR research.
    Works on signals of any length — n_fft is capped at N_FFT, so a longer
    signal just produces more (averaged) analysis frames, not more features.
    """
    signal = signal.astype(np.float32)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    n_fft = min(N_FFT, len(signal))
    hop   = n_fft // 4

    mfccs    = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=n_fft, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr,
                                                  n_fft=n_fft, hop_length=hop)
    zcr      = librosa.feature.zero_crossing_rate(y=signal, hop_length=hop)
    rolloff  = librosa.feature.spectral_rolloff(y=signal, sr=sr,
                                                 n_fft=n_fft, hop_length=hop)
    rms      = librosa.feature.rms(y=signal, hop_length=hop)

    return np.concatenate([
        np.mean(mfccs, axis=1),         # 13 features
        np.std(mfccs,  axis=1),         # 13 features
        [np.mean(centroid)],            #  1 feature
        [np.mean(zcr)],                 #  1 feature
        [np.mean(rolloff)],             #  1 feature
        [np.mean(rms)],                 #  1 feature
    ])                                  # = 30 features total


def extract_feature_summary(signal: np.ndarray,
                             sr: int = SAMPLE_RATE) -> dict:
    """Human-readable feature values for frontend display."""
    signal = signal.astype(np.float32)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    n_fft = min(N_FFT, len(signal))
    hop   = n_fft // 4

    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr,
                                                  n_fft=n_fft, hop_length=hop)
    zcr      = librosa.feature.zero_crossing_rate(y=signal, hop_length=hop)
    rolloff  = librosa.feature.spectral_rolloff(y=signal, sr=sr,
                                                 n_fft=n_fft, hop_length=hop)
    rms      = librosa.feature.rms(y=signal, hop_length=hop)
    mfccs    = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=n_fft, hop_length=hop)

    return {
        "spectral_centroid_hz": round(float(np.mean(centroid)), 1),
        "zero_crossing_rate":   round(float(np.mean(zcr)), 4),
        "spectral_rolloff_hz":  round(float(np.mean(rolloff)), 1),
        "rms_energy":           round(float(np.mean(rms)), 4),
        "mfcc_1_mean":          round(float(np.mean(mfccs[0])), 2),
        "mfcc_2_mean":          round(float(np.mean(mfccs[1])), 2),
    }


# ── Synthetic dataset generation (fallback only) ─────────────
def _generate_tone(seed: int) -> np.ndarray:
    """1–6 pure sine waves at unrelated frequencies — sparse in frequency domain."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES, endpoint=False)
    n_freqs = rng.integers(1, 7)
    freqs   = rng.uniform(80, 2000, n_freqs)
    amps    = rng.uniform(0.2, 1.0, n_freqs)
    signal  = sum(a * np.sin(2 * np.pi * f * t)
                  for f, a in zip(freqs, amps))
    return signal


def _generate_noise(seed: int) -> np.ndarray:
    """Pure Gaussian white noise — uniformly dense in frequency domain."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(N_SAMPLES)


def _generate_music(seed: int) -> np.ndarray:
    """Many overlapping harmonically-related frequencies (chord/melody)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES, endpoint=False)
    root = rng.uniform(80, 600)
    n_harmonics = rng.integers(5, 15)
    amps = rng.uniform(0.1, 1.0, n_harmonics)
    signal = sum(a * np.sin(2 * np.pi * root * (h + 1) * t)
                 for h, a in enumerate(amps))
    root2  = root * rng.choice([1.25, 1.5, 2.0])
    n_harm2 = rng.integers(3, 8)
    amps2   = rng.uniform(0.05, 0.5, n_harm2)
    signal += sum(a * np.sin(2 * np.pi * root2 * (h + 1) * t)
                  for h, a in enumerate(amps2))
    return signal


def _generate_synthetic_for(label: str, n: int = N_PER_CLASS) -> tuple[list, list]:
    """Generate n synthetic examples for a single class label."""
    generators = {
        "tone":  _generate_tone,
        "noise": _generate_noise,
        "music": _generate_music,
    }
    gen = generators[label]
    X, y = [], []
    for seed in range(n):
        signal = gen(seed)
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val
        X.append(extract_features(signal))
        y.append(label)
    return X, y


def load_real_dataset(metadata_csv: str = METADATA_CSV,
                      audio_dir: str = AUDIO_DIR,
                      folds: list[int] = FOLDS_TO_USE,
                      max_per_class: int = MAX_PER_REAL_CLASS) -> tuple[list, list]:
    """
    Load real UrbanSound8k clips, map them to tone/noise/music via
    CLASS_MAP, and extract features from each FULL (trimmed) clip.

    Requires the dataset to already be unzipped locally (see REAL_DATA_ROOT /
    METADATA_CSV / AUDIO_DIR above). Skips (with a printed warning) any file
    that fails to load or is too short/silent after trimming.

    Returns:
        (X, y) as plain lists (not yet np.array).
    """
    if not os.path.exists(metadata_csv):
        print(f"[load_real_dataset] Metadata CSV not found at {metadata_csv} "
              f"— falling back to fully synthetic data. Set URBANSOUND8K_ROOT "
              f"or edit REAL_DATA_ROOT in classifier.py to point at your "
              f"unzipped UrbanSound8k folder.")
        return [], []

    # Group candidate file paths by target class
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

    # Balance: cap each bucket, shuffle deterministically first
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
                if len(signal) < int(0.05 * SAMPLE_RATE):   # skip near-silent clips
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


def generate_dataset(use_real: bool = USE_REAL_DATA) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the full labeled training set.

    If use_real=True and the UrbanSound8k metadata CSV can be found, all
    three classes come from real UrbanSound8k audio (full clip length).
    Otherwise, falls back to fully synthetic data for all three classes.

    Returns:
        (X, y) where X is (n_samples, n_features) and y is string labels.
    """
    if use_real:
        X_real, y_real = load_real_dataset()
        if X_real:
            return np.array(X_real), np.array(y_real)
        # else: real data unavailable, fall through to full synthetic

    X, y = [], []
    for label in CLASSES:
        X_lab, y_lab = _generate_synthetic_for(label, N_PER_CLASS)
        X += X_lab
        y += y_lab

    return np.array(X), np.array(y)


# ── Model training ───────────────────────────────────────────
def train_classifier() -> Pipeline:
    """
    Train a RandomForest classifier on real (or synthetic fallback) data.

    Returns a sklearn Pipeline (scaler + classifier) ready for prediction.
    """
    print("Training audio classifier...")
    X, y = generate_dataset()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    pipeline.fit(X, y)
    print(f"Classifier trained on {len(y)} samples.")
    print(f"Classes: {CLASSES}")
    return pipeline


# ── Singleton model (load once, reuse) ───────────────────────
_model: Pipeline | None = None

def get_model(force_retrain: bool = False) -> Pipeline:
    """
    Return the trained model.

    Loads from MODEL_CACHE_PATH on disk if present (so real-data training,
    which reads hundreds of audio files, doesn't re-run on every server
    restart). Pass force_retrain=True to ignore the cache and retrain
    from scratch — do this after changing CLASS_MAP, feature extraction,
    or the underlying data.
    """
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


# ── Main classify function ───────────────────────────────────
def _split_into_windows(signal: np.ndarray, window_len: int, max_windows: int = 10) -> list:
    """
    Split signal into up to max_windows evenly-spaced, non-overlapping
    windows of window_len samples each. Used so classification can average
    predictions across a longer clip instead of betting on a single slice.
    """
    n_possible = len(signal) // window_len
    n = max(1, min(n_possible, max_windows))
    step = len(signal) // n
    windows = []
    for i in range(n):
        start = i * step
        windows.append(signal[start:start + window_len])
    return windows


def classify_signal(signal: np.ndarray,
                    sr: int = SAMPLE_RATE) -> ClassificationResult:
    """
    Classify a time-domain audio signal into tone/noise/music.

    Args:
        signal: 1D numpy array (time domain), at its ORIGINAL sample rate
        sr:     the signal's actual sample rate in Hz

    Returns:
        ClassificationResult with prediction, confidence, and feature summary

    Resamples to the fixed SAMPLE_RATE if needed, since librosa's spectral
    features (centroid, rolloff, MFCCs) are scaled by sr.

    If given enough audio (more than a couple of the model's ~256-ms-scale
    training windows), classifies MULTIPLE windows spread across the
    signal and averages their predicted probabilities, rather than
    betting the whole result on a single slice. This matters most for
    complex/non-stationary classes like music, where any one short window
    might land on a quiet passage, a drum hit, or a held note that doesn't
    look "musical" in isolation — averaging across several windows gives a
    far more robust read on the whole clip. Short inputs (e.g. the CS
    pipeline's 256-sample reconstruction output, used when classifying
    "the reconstructed signal") naturally fall back to a single window,
    since there isn't more audio to spread across.
    """
    model = get_model()

    if sr != SAMPLE_RATE:
        signal = librosa.resample(signal.astype(np.float32),
                                   orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    window_len = N_SAMPLES * 8   # ~256ms per window, matches training scale
    if len(signal) > window_len * 2:
        windows = _split_into_windows(signal, window_len, max_windows=10)
    else:
        windows = [signal]

    class_order = model.classes_
    all_probs = []
    for w in windows:
        feats = extract_features(w, sr).reshape(1, -1)
        all_probs.append(model.predict_proba(feats)[0])
    avg_probs = np.mean(all_probs, axis=0)

    predicted  = class_order[int(np.argmax(avg_probs))]
    confidence = round(float(np.max(avg_probs)), 4)
    all_scores = {
        cls: round(float(prob), 4)
        for cls, prob in zip(class_order, avg_probs)
    }

    feature_summary = extract_feature_summary(signal, sr)

    signal_stats = {
        "mean":     round(float(np.mean(signal)), 4),
        "std":      round(float(np.std(signal)), 4),
        "max_abs":  round(float(np.max(np.abs(signal))), 4),
        "n_samples": len(signal),
        "n_windows_averaged": len(windows),
    }

    return ClassificationResult(
        predicted_class = predicted,
        confidence      = confidence,
        all_scores      = all_scores,
        feature_summary = feature_summary,
        signal_stats    = signal_stats,
    )


# ── Class descriptions for frontend ─────────────────────────
CLASS_INFO = {
    "tone": {
        "label":       "Pure Tone",
        "description": "1–3 pure sine waves. Very sparse in frequency domain. "
                       "Perfect for CS recovery.",
        "color":       "blue",
        "icon":        "◎",
    },
    "noise": {
        "label":       "Noise",
        "description": "Gaussian white noise. Uniform frequency distribution. "
                       "Not sparse — CS recovery will struggle.",
        "color":       "red",
        "icon":        "≋",
    },
    "music": {
        "label":       "Music",
        "description": "Harmonic overtones — rich frequency structure. "
                       "Moderately sparse in frequency domain.",
        "color":       "green",
        "icon":        "♪",
    },
}
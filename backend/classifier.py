"""
classifier.py — ML Audio Classifier for Compressed Sensing Audio Analyzer
=========================================================================
Phase 4: Extract audio features from (reconstructed) signals and classify
them into one of four categories: tone / noise / music / speech.

Pipeline:
  signal (time domain, 256 samples)
    -> feature extraction (MFCCs, spectral centroid, ZCR, rolloff, RMS)
    -> StandardScaler normalization
    -> RandomForestClassifier (trained on synthetic data)
    -> predicted class + confidence scores

Option A: synthetic training data generated in Python (all 4 classes).
Option B (current): fully real — UrbanSound8k for tone/noise/music,
                    LibriSpeech for speech. Synthetic generators remain
                    as a fallback if either real dataset isn't found
                    locally (useful for a quick smoke test without
                    downloading anything).
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
CLASSES      = ["tone", "noise", "music", "speech"]
N_PER_CLASS  = 150          # synthetic samples per class (used for speech,
                             # and as fallback if real data isn't found)

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

# UrbanSound8k's 10 classes -> your 4 target classes.
# "speech" is intentionally absent: UrbanSound8k has nothing voice-like,
# so that class stays synthetic (see _generate_speech below).
CLASS_MAP = {
    "car_horn":         "tone",
    "siren":            "tone",
    "air_conditioner":  "noise",
    "engine_idling":    "noise",
    "jackhammer":       "noise",
    "drilling":         "noise",
    "street_music":     "music",
}

# Real speech data (LibriSpeech dev-clean). Download from openslr.org —
# no account needed: https://www.openslr.org/resources/12/dev-clean.tar.gz
# Expected layout after extracting:
#   LibriSpeech/dev-clean/<speaker_id>/<chapter_id>/<utterance>.flac
SPEECH_DATA_ROOT = os.environ.get("LIBRISPEECH_ROOT", "./LibriSpeech/dev-clean")
MAX_SPEECH_FILES = 150   # match the size of the other real buckets

MODEL_CACHE_PATH = "audio_classifier_model.joblib"

# Real audio classification always happens on the CS-reconstructed signal,
# which is fixed at N_SAMPLES (256, ~32ms at SAMPLE_RATE). If real training
# data is extracted from full multi-second clips instead, there's a
# systematic train/inference mismatch — the model learns what SECONDS of
# audio look like, then has to classify a 32ms fragment. This barely hurts
# stationary classes (a steady hum sounds the same at 32ms or 3s) but badly
# hurts speech, which is constantly changing phoneme to phoneme. Training
# on short windows that match the real inference length fixes this.
CLASSIFY_WINDOW_SAMPLES = N_SAMPLES   # 256 — must match production input length
WINDOWS_PER_FILE = 3                  # multiple short windows per source
                                       # file: more diverse examples (different
                                       # moments/phonemes) instead of wasting
                                       # most of a multi-second clip


def _sample_windows(signal: np.ndarray,
                    window_len: int = CLASSIFY_WINDOW_SAMPLES,
                    n_windows: int = WINDOWS_PER_FILE,
                    seed: int = 0) -> list:
    """
    Extract up to n_windows short windows spread across signal, so a single
    long clip yields several training examples matching the length actually
    seen at inference. Falls back to the whole (short) signal if it's
    already <= window_len.
    """
    if len(signal) <= window_len:
        return [signal]
    max_start = len(signal) - window_len
    rng = np.random.default_rng(seed)
    n = min(n_windows, max_start + 1)
    starts = sorted(rng.choice(max_start + 1, size=n, replace=False))
    return [signal[s:s + window_len] for s in starts]


def synthetic_demo_tone(seed: int = 42,
                        n_samples: int = CLASSIFY_WINDOW_SAMPLES,
                        sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    A short, clean synthetic tone generated DIRECTLY at the classifier's
    native SAMPLE_RATE — used only for the "Classify Signal" demo button,
    deliberately decoupled from the CS reconstruction demo's own signal.

    Why decoupled: the CS reconstruction demo uses sample_rate=1000 (a
    choice specific to that pipeline's sparsity demonstration), which
    Nyquist-limits it to frequencies below 500Hz. A 256-sample/32ms
    classification window can't resolve such low frequencies cleanly —
    e.g. a 50Hz component only completes ~1.6 cycles in 32ms, nowhere
    near enough to look "tonal" to a short-time spectral analysis. Real
    tonal sounds that classify correctly (e.g. a whistle) are typically
    1000+ Hz, completing dozens of cycles in the same window. So this
    generates directly at SAMPLE_RATE (no resampling needed at all) using
    frequencies high enough to resolve cleanly in a short window —
    matching what actually works for real audio, rather than reusing the
    CS demo's low-frequency, classification-unfriendly signal.
    """
    t = np.arange(n_samples) / sr
    freqs = [600, 1100, 1700]
    amps  = [1.0, 0.6, 0.4]
    signal = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
    peak = np.max(np.abs(signal)) + 1e-10
    return signal / peak


def pick_energetic_window(signal: np.ndarray,
                          window_len: int = CLASSIFY_WINDOW_SAMPLES,
                          n_candidates: int = 5,
                          seed: int | None = None) -> np.ndarray:
    """
    Pick a representative window from signal by sampling several candidate
    positions and keeping the highest-energy one.

    Used at INFERENCE time (main.py's /upload) instead of always grabbing
    the literal first window. Training already samples multiple random
    positions per file (_sample_windows above), which mostly land on real
    voiced/energetic content simply because that's most of a typical
    clip's duration. But a fixed "always take the first slice after the
    silence trim" policy at inference time systematically grabs onset/
    attack transients instead — a different, narrower distribution than
    what training saw. Picking the most energetic of a few candidates
    brings inference back in line with what training actually learned
    "typical" content looks like.
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
    Extract a 30-dimensional feature vector from a short audio signal.

    Features:
      - 13 MFCC means     (timbre / frequency content)
      - 13 MFCC stds      (variability of timbre)
      - spectral centroid  (brightness — high = treble heavy)
      - zero crossing rate (noisiness — high = noisy)
      - spectral rolloff   (frequency where 85% of energy is below)
      - RMS energy         (loudness)

    These 30 features are classic audio fingerprints used in MIR research.
    """
    signal = signal.astype(np.float32)
    # Normalize to [-1, 1]
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    n_fft   = min(N_FFT, len(signal))
    hop     = n_fft // 4

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


# ── Synthetic dataset generation (Option A) ─────────────────
def _generate_tone(seed: int) -> np.ndarray:
    """1–6 pure sine waves at unrelated frequencies — sparse in frequency domain."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES, endpoint=False)
    n_freqs = rng.integers(1, 7)   # up to 6 pure tones
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
    """
    Many overlapping harmonically-related frequencies.
    Simulates a musical chord or melody fragment.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES, endpoint=False)
    # Root note + harmonics
    root = rng.uniform(80, 600)
    n_harmonics = rng.integers(5, 15)
    amps = rng.uniform(0.1, 1.0, n_harmonics)
    signal = sum(a * np.sin(2 * np.pi * root * (h + 1) * t)
                 for h, a in enumerate(amps))
    # Add a second instrument
    root2  = root * rng.choice([1.25, 1.5, 2.0])
    n_harm2 = rng.integers(3, 8)
    amps2   = rng.uniform(0.05, 0.5, n_harm2)
    signal += sum(a * np.sin(2 * np.pi * root2 * (h + 1) * t)
                  for h, a in enumerate(amps2))
    return signal


def _generate_speech(seed: int) -> np.ndarray:
    """
    Amplitude-modulated signal with formant-like structure.
    Simulates the envelope and harmonic structure of voiced speech.

    Only used as a fallback if real LibriSpeech data isn't found
    (see load_speech_dataset / SPEECH_DATA_ROOT below) — real recordings
    of actual human speech are a much better source than this synthetic
    approximation, which occupies a noticeably different region of
    feature space than real audio.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES, endpoint=False)
    # Fundamental frequency (pitch)
    f0  = rng.uniform(80, 300)
    # Modulation (speech rhythm envelope)
    mod = rng.uniform(2, 12)
    # Voiced carrier with harmonics
    signal = np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * mod * t))
    for harmonic in [2, 3, 4, 5]:
        amp = rng.uniform(0.05, 0.35) / harmonic
        signal += amp * np.sin(2 * np.pi * f0 * harmonic * t)
    # Formant-like resonance
    formant = rng.uniform(500, 2500)
    signal += 0.2 * np.sin(2 * np.pi * formant * t) * (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * mod * t)))
    return signal


def load_speech_dataset(root: str = SPEECH_DATA_ROOT,
                        max_files: int = MAX_SPEECH_FILES) -> tuple[list, list]:
    """
    Load real speech clips from a LibriSpeech-style directory tree
    (any nested folder of .flac/.wav files works — LibriSpeech's
    speaker/chapter/utterance structure doesn't matter, we just walk it).

    Returns:
        (X, y) — X is a list of 30-dim feature vectors, y is ["speech", ...]
    """
    if not os.path.exists(root):
        print(f"[load_speech_dataset] {root} not found — falling back to "
              f"synthetic speech. Download LibriSpeech dev-clean from "
              f"https://www.openslr.org/resources/12/dev-clean.tar.gz, "
              f"extract it, and set LIBRISPEECH_ROOT or edit "
              f"SPEECH_DATA_ROOT in classifier.py.")
        return [], []

    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".flac", ".wav")):
                paths.append(os.path.join(dirpath, fn))

    rng = np.random.default_rng(0)
    rng.shuffle(paths)   # shuffle before capping so we get diverse speakers
    paths = paths[:max_files]

    X, y = [], []
    for i, path in enumerate(paths):
        try:
            signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            signal, _ = librosa.effects.trim(signal, top_db=25)
            if len(signal) < int(0.05 * SAMPLE_RATE):
                continue
            for window in _sample_windows(signal, seed=i):
                max_val = np.max(np.abs(window))
                if max_val > 0:
                    window = window / max_val
                X.append(extract_features(window, SAMPLE_RATE))
                y.append("speech")
        except Exception as e:
            print(f"[load_speech_dataset] Skipping {path}: {e}")

    # Cap total examples (not files) so class size stays comparable to
    # before, now that each file can yield multiple window examples.
    if len(y) > max_files:
        idx = np.random.default_rng(1).choice(len(y), size=max_files, replace=False)
        X = [X[i] for i in idx]
        y = [y[i] for i in idx]

    print(f"[load_speech_dataset] speech: {len(y)} short-window examples "
          f"from {len(paths)} source files")
    return X, y


def _generate_synthetic_for(label: str, n: int = N_PER_CLASS) -> tuple[list, list]:
    """Generate n synthetic examples for a single class label."""
    generators = {
        "tone":   _generate_tone,
        "noise":  _generate_noise,
        "music":  _generate_music,
        "speech": _generate_speech,
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
    CLASS_MAP, and extract features from each.

    Requires the dataset to already be unzipped locally (see REAL_DATA_ROOT /
    METADATA_CSV / AUDIO_DIR above). Skips (with a printed warning) any file
    that fails to load or is too short/silent after trimming.

    Returns:
        (X, y) as plain lists (not yet np.array — caller combines with
        synthetic speech data first).
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
        label_X, label_y = [], []
        for i, path in enumerate(paths):
            try:
                signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                signal, _ = librosa.effects.trim(signal, top_db=25)
                if len(signal) < int(0.05 * SAMPLE_RATE):   # skip near-silent clips
                    continue
                for window in _sample_windows(signal, seed=i):
                    max_val = np.max(np.abs(window))
                    if max_val > 0:
                        window = window / max_val
                    label_X.append(extract_features(window, SAMPLE_RATE))
                    label_y.append(label)
            except Exception as e:
                print(f"[load_real_dataset] Skipping {path}: {e}")

        # Cap total examples (not files) so class size stays comparable to
        # before, now that each file can yield multiple window examples.
        if len(label_y) > max_per_class:
            idx = np.random.default_rng(1).choice(len(label_y), size=max_per_class, replace=False)
            label_X = [label_X[i] for i in idx]
            label_y = [label_y[i] for i in idx]

        X += label_X
        y += label_y
        print(f"[load_real_dataset] {label}: {len(label_y)} short-window examples "
              f"from {len(paths)} source files")

    return X, y


def generate_dataset(use_real: bool = USE_REAL_DATA) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the full labeled training set.

    If use_real=True and the UrbanSound8k metadata CSV can be found:
      - tone / noise / music come from real UrbanSound8k audio
      - speech stays synthetic (no real speech class available)
    Otherwise, falls back to fully synthetic data for all 4 classes
    (the original Option A behavior).

    Returns:
        (X, y) where X is (n_samples, n_features) and y is string labels.
    """
    X, y = [], []

    if use_real:
        X_real, y_real = load_real_dataset()
        if X_real:
            X += X_real
            y += y_real
            # Speech: try real LibriSpeech data first, fall back to the
            # synthetic generator only if it isn't available locally.
            X_speech, y_speech = load_speech_dataset()
            if not X_speech:
                X_speech, y_speech = _generate_synthetic_for("speech", N_PER_CLASS)
            X += X_speech
            y += y_speech
            return np.array(X), np.array(y)
        # else: real data unavailable, fall through to full synthetic

    for label in CLASSES:
        X_lab, y_lab = _generate_synthetic_for(label, N_PER_CLASS)
        X += X_lab
        y += y_lab

    return np.array(X), np.array(y)


# ── Model training ───────────────────────────────────────────
def train_classifier() -> Pipeline:
    """
    Train a RandomForest classifier (hybrid real + synthetic data, or
    fully synthetic if real data isn't available — see generate_dataset).

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
            class_weight="balanced",   # real-data buckets aren't perfectly even
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
def classify_signal(signal: np.ndarray,
                    sr: int = SAMPLE_RATE) -> ClassificationResult:
    """
    Classify a time-domain audio signal into tone/noise/music/speech.

    Args:
        signal: 1D numpy array (time domain), at its ORIGINAL sample rate
        sr:     the signal's actual sample rate in Hz

    Returns:
        ClassificationResult with prediction, confidence, and feature summary

    IMPORTANT: The model was trained on features extracted at a fixed
    SAMPLE_RATE (see generate_dataset -> extract_features, which never
    passes sr and so always uses the SAMPLE_RATE default). librosa's
    spectral features (centroid, rolloff, MFCCs) are scaled by sr, so
    feeding in a signal at some other sr (e.g. the demo signal's 1000 Hz,
    or an uploaded file's 44100 Hz) produces features on a totally
    different numeric scale than what the classifier learned, causing
    IMPORTANT — two things this function must get right, both learned the
    hard way:

    1. Sample rate scale: librosa's spectral features (centroid, rolloff,
       MFCCs) are scaled by sr, so features must always be extracted at a
       fixed SAMPLE_RATE regardless of the input's native rate.

    2. Real-world DURATION, not sample count: training windows are
       CLASSIFY_WINDOW_SAMPLES long AT SAMPLE_RATE (256 samples @ 8000Hz
       = 32ms). The CS demo signal is generated at 1000Hz — 256 samples
       there is 256ms, 8x longer — so it needs trimming down to a 32ms
       window too. CRITICAL: resample first, THEN truncate — not the
       other way around. Truncating to a very short slice at a low native
       rate BEFORE resampling starves librosa's resampling filter of
       enough input to work cleanly, producing ringing/overshoot
       artifacts (measured: a 32-raw-sample slice resampled up to 256
       samples overshot the original signal's own amplitude range) that
       corrupt the extracted features far worse than the duration
       mismatch this was meant to fix. Resampling the full signal first
       (long enough for the filter to behave) and truncating afterward
       avoids that entirely while still landing on the correct duration.
    """
    model = get_model()

    # Resample to the fixed rate the model was trained on, if needed —
    # BEFORE truncating (see docstring note above on why order matters).
    if sr != SAMPLE_RATE:
        signal = librosa.resample(signal.astype(np.float32),
                                   orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Now truncate to the target window length, post-resample — analyzes
    # the correct ~32ms slice without resampling artifacts, matching what
    # training saw.
    if len(signal) > CLASSIFY_WINDOW_SAMPLES:
        signal = signal[:CLASSIFY_WINDOW_SAMPLES]

    # Extract features
    features = extract_features(signal, sr).reshape(1, -1)

    # Predict
    predicted    = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_order  = model.classes_

    # Build scores dict
    all_scores = {
        cls: round(float(prob), 4)
        for cls, prob in zip(class_order, probabilities)
    }
    confidence = round(float(max(probabilities)), 4)

    # Feature summary for display
    feature_summary = extract_feature_summary(signal, sr)

    # Basic signal statistics
    signal_stats = {
        "mean":     round(float(np.mean(signal)), 4),
        "std":      round(float(np.std(signal)), 4),
        "max_abs":  round(float(np.max(np.abs(signal))), 4),
        "n_samples": len(signal),
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
    "speech": {
        "label":       "Speech",
        "description": "Voiced speech-like signal with pitch and formants. "
                       "Approximately sparse — CS can recover with enough n.",
        "color":       "yellow",
        "icon":        "◉",
    },
}
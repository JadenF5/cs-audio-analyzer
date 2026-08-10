"""
main.py — FastAPI Backend for Compressed Sensing Audio Analyzer
===============================================================
Endpoints:
  GET  /health          → server health check
  GET  /demo            → run CS on a built-in demo signal (no upload needed)
  POST /upload          → upload audio file, get signal info back
  POST /reconstruct     → run full CS pipeline on uploaded or demo signal

Run locally:
  uvicorn main:app --reload --port 8000

Then visit:
  http://localhost:8000/docs    ← auto-generated interactive API docs
"""

import io
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from cs_engine import (
    run_cs_pipeline,
    get_signal_info,
    generate_demo_signal,
    ReconstructionResult,
    SignalInfo,
)

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="Compressed Sensing Audio Analyzer",
    description="Apply compressed sensing theory to audio signals. "
                "Reconstruct sparse audio from sub-Nyquist measurements.",
    version="1.0.0",
)

# Allow the React frontend to call this API.
# Local dev origins are always allowed. For the deployed frontend, set
# FRONTEND_ORIGIN on Render to your Vercel URL, e.g.
#   FRONTEND_ORIGIN=https://cs-audio-analyzer.vercel.app
# Supports a comma-separated list if you have multiple (e.g. a Vercel
# preview-deployment domain plus your production domain).
_allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
_frontend_origin = os.environ.get("FRONTEND_ORIGIN", "")
if _frontend_origin:
    _allowed_origins += [o.strip() for o in _frontend_origin.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory signal store (simple, no DB needed for Phase 2) ─
# In production this would be a proper cache/storage layer.
_current_signal:     Optional[np.ndarray] = None
_current_samplerate: Optional[int]        = None


# ── Request / Response models ────────────────────────────────
class ReconstructRequest(BaseModel):
    compression_ratio: float = Field(
        default=0.25,
        ge=0.05, le=0.95,
        description="Fraction of signal length to use as measurements (n/N). "
                    "Lower = more compression. Must satisfy n > k*log(N) for good recovery."
    )
    use_demo: bool = Field(
        default=False,
        description="If True, ignore uploaded signal and use built-in demo signal."
    )
    seed: int = Field(
        default=42,
        description="Random seed for measurement matrix reproducibility."
    )


class HealthResponse(BaseModel):
    status: str
    message: str


class SignalInfoResponse(BaseModel):
    duration_sec:   float
    sample_rate:    int
    n_samples:      int
    n_fft_bins:     int
    sparsity_k:     int
    sparsity_ratio: float
    message:        str


class ReconstructResponse(BaseModel):
    # Signals (downsampled for fast JSON transfer)
    original:          list
    reconstructed:     list
    time_axis:         list

    # Frequency domain
    frequencies:       list
    fft_original:      list
    fft_reconstructed: list

    # Quality metrics
    snr_db:            float
    mse:               float
    max_error:         float
    correlation:       float

    # CS parameters
    n_samples:         int
    n_measurements:    int
    sparsity_k:        int
    compression_ratio: float
    theory_min_n:      int
    solver_status:     str

    # Interpretation
    quality_label:     str    # "excellent" | "good" | "poor"
    cs_feasible:       bool   # True if n >= theory_min_n


# ── Helper ───────────────────────────────────────────────────
def _downsample_for_json(arr: list, max_points: int = 512) -> list:
    """Reduce array length for faster JSON transfer while preserving shape."""
    if len(arr) <= max_points:
        return arr
    step = len(arr) // max_points
    return arr[::step][:max_points]


def _quality_label(snr_db: float) -> str:
    if snr_db > 40:
        return "excellent"
    elif snr_db > 20:
        return "good"
    elif snr_db > 10:
        return "fair"
    else:
        return "poor"


def _result_to_response(result: ReconstructionResult) -> ReconstructResponse:
    """Convert internal result dataclass to API response model."""
    return ReconstructResponse(
        original          = _downsample_for_json(result.original),
        reconstructed     = _downsample_for_json(result.reconstructed),
        time_axis         = _downsample_for_json(result.time_axis),
        frequencies       = _downsample_for_json(result.frequencies),
        fft_original      = _downsample_for_json(result.fft_original),
        fft_reconstructed = _downsample_for_json(result.fft_reconstructed),
        snr_db            = result.snr_db,
        mse               = result.mse,
        max_error         = result.max_error,
        correlation       = result.correlation,
        n_samples         = result.n_samples,
        n_measurements    = result.n_measurements,
        sparsity_k        = result.sparsity_k,
        compression_ratio = result.compression_ratio,
        theory_min_n      = result.theory_min_n,
        solver_status     = result.solver_status,
        quality_label     = _quality_label(result.snr_db),
        cs_feasible       = result.n_measurements >= result.theory_min_n,
    )


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check():
    """Check that the API is running."""
    return HealthResponse(
        status="ok",
        message="Compressed Sensing Audio Analyzer API is running."
    )


@app.get("/demo", response_model=ReconstructResponse, tags=["CS Pipeline"])
def run_demo(
    compression_ratio: float = 0.25,
    seed: int = 42
):
    """
    Run the full CS pipeline on a built-in synthetic demo signal.
    No file upload needed — great for testing the frontend.

    The demo signal is a sum of 4 pure sine waves (sparse in frequency domain).
    """
    signal, sr = generate_demo_signal(
        frequencies=[50, 120, 200, 310],
        amplitudes=[1.0, 0.7, 0.5, 0.3],
        sample_rate=1000,
        n_samples=256,
    )

    try:
        result = run_cs_pipeline(
            signal=signal,
            sample_rate=sr,
            compression_ratio=compression_ratio,
            seed=seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CS pipeline failed: {str(e)}")

    return _result_to_response(result)


@app.post("/upload", response_model=SignalInfoResponse, tags=["Audio"])
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (.wav or .mp3).
    Returns signal metadata. Does NOT run CS reconstruction yet.
    Call /reconstruct after this to run the pipeline.

    Stores the signal in memory for the next /reconstruct call.
    """
    global _current_signal, _current_samplerate

    # Validate file type
    if not file.filename.endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .wav, .mp3, .ogg, or .flac"
        )

    try:
        import librosa
        contents = await file.read()
        audio_buffer = io.BytesIO(contents)
        # Resample to a fixed rate (matches classifier.SAMPLE_RATE) rather
        # than keeping each file's native rate. Otherwise, truncating to
        # MAX_SAMPLES below captures a wildly different duration depending
        # on the upload's original sample rate — e.g. 256 raw samples from
        # a 44.1kHz file is ~6ms, way too short to mean anything, while
        # the demo signal's 256 samples represent a much longer window.
        UPLOAD_SR = 8000
        signal, sr = librosa.load(audio_buffer, sr=UPLOAD_SR, mono=True, duration=2.0)

        # Trim leading/trailing silence before truncating below. Otherwise
        # a file with a quiet lead-in (very common in real recordings —
        # room tone before someone starts talking) can have its first
        # MAX_SAMPLES land entirely inside near-total silence: a
        # degenerate, ~zero-variance signal that produces NaN downstream
        # (see compute_metrics in cs_engine.py) and can crash the frontend.
        signal, _ = librosa.effects.trim(signal, top_db=25)
        if len(signal) < int(0.05 * UPLOAD_SR):   # trimmed away to ~nothing
            raise HTTPException(
                status_code=400,
                detail="This file appears to be silent (or nearly silent) "
                       "after trimming. Please upload a file with audible sound."
            )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="librosa not installed. Run: pip install librosa"
        )
    except HTTPException:
        raise   # let our own clean 400s (e.g. "file is silent") pass through untouched
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not load audio file: {str(e)}"
        )

    # Truncate to max 4096 samples for reasonable CS computation time
    # Truncate uploaded audio to a size the CS solve can actually handle
    # quickly. build_sensing_matrix() builds a dense N×N matrix and the
    # l1-minimization solve scales roughly N²-N³ — 256 (matching the demo
    # signal, already proven fast) keeps this responsive even on a
    # constrained free-tier CPU. A much larger N (the old 2048 cap) can
    # take long enough to time out or fail outright under limited compute.
    MAX_SAMPLES = 256
    if len(signal) > MAX_SAMPLES:
        signal = signal[:MAX_SAMPLES]

    # Store in memory
    _current_signal     = signal
    _current_samplerate = int(sr)

    info = get_signal_info(signal, int(sr))

    return SignalInfoResponse(
        duration_sec   = info.duration_sec,
        sample_rate    = info.sample_rate,
        n_samples      = info.n_samples,
        n_fft_bins     = info.n_fft_bins,
        sparsity_k     = info.sparsity_k,
        sparsity_ratio = info.sparsity_ratio,
        message        = (
            f"Loaded '{file.filename}' — "
            f"{info.n_samples} samples at {info.sample_rate} Hz. "
            f"Sparsity: k={info.sparsity_k} of {info.n_fft_bins} FFT bins "
            f"({info.sparsity_ratio:.1%}). "
            f"Call POST /reconstruct to run CS pipeline."
        )
    )


@app.post("/reconstruct", response_model=ReconstructResponse, tags=["CS Pipeline"])
def reconstruct(req: ReconstructRequest):
    """
    Run the full compressed sensing reconstruction pipeline.

    Uses either:
    - The most recently uploaded audio file (via POST /upload), or
    - The built-in demo signal if use_demo=True or no file has been uploaded.

    The pipeline:
      1. Analyze sparsity in frequency domain
      2. Build random measurement matrix Phi (n x N), n = compression_ratio * N
      3. Take compressed measurements y = Phi @ signal
      4. Recover via l1 minimization: min |s|_1 s.t. A @ s = y
      5. Reconstruct via IFFT
      6. Compute SNR, MSE, correlation

    Returns full signal arrays + quality metrics for visualization.
    """
    global _current_signal, _current_samplerate

    # Choose signal source
    if req.use_demo or _current_signal is None:
        signal, sr = generate_demo_signal(seed=req.seed)
    else:
        signal = _current_signal
        sr     = _current_samplerate

    try:
        result = run_cs_pipeline(
            signal=signal,
            sample_rate=sr,
            compression_ratio=req.compression_ratio,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CS pipeline failed: {str(e)}")

    return _result_to_response(result)


# ── Phase 4: Classification endpoint ─────────────────────────

class ClassifyRequest(BaseModel):
    use_reconstructed: bool = Field(
        default=True,
        description="If True classify the reconstructed signal. If False classify original."
    )
    use_demo: bool = Field(default=False)


class ClassifyResponse(BaseModel):
    predicted_class:  str
    confidence:       float
    all_scores:       dict
    feature_summary:  dict
    signal_stats:     dict
    class_label:      str
    class_description: str
    class_color:      str
    class_icon:       str


@app.post("/classify", response_model=ClassifyResponse, tags=["ML Classification"])
def classify(req: ClassifyRequest):
    """
    Classify an audio signal into: tone / noise / music / speech.

    Uses MFCCs, spectral centroid, zero crossing rate, rolloff, and RMS
    as features fed into a RandomForest trained on synthetic data.

    Call this after /reconstruct to classify the reconstructed signal,
    or set use_reconstructed=False to classify the original.
    """
    global _current_signal, _current_samplerate

    if req.use_demo or _current_signal is None:
        signal, sr = generate_demo_signal()
    else:
        signal = _current_signal
        sr     = _current_samplerate

    try:
        from classifier import classify_signal, CLASS_INFO
        result = classify_signal(signal, int(sr))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    info = CLASS_INFO[result.predicted_class]

    # Clean np.str_ keys from sklearn
    clean_scores = {str(k): float(v) for k, v in result.all_scores.items()}

    return ClassifyResponse(
        predicted_class   = result.predicted_class,
        confidence        = result.confidence,
        all_scores        = clean_scores,
        feature_summary   = result.feature_summary,
        signal_stats      = result.signal_stats,
        class_label       = info["label"],
        class_description = info["description"],
        class_color       = info["color"],
        class_icon        = info["icon"],
    )
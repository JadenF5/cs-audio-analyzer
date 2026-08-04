"""
cs_engine.py — Compressed Sensing Audio Engine
===============================================
Clean reusable module wrapping the Phase 1 CS pipeline.
Called by the FastAPI backend to process audio signals.

Pipeline:
  signal (time domain)
    -> FFT -> sparse frequency representation (k << N)
    -> random measurements y = Phi @ signal  (n << N)
    -> l1 minimization: min |s|_1 s.t. A @ s = y
    -> IFFT -> reconstructed signal
    -> quality metrics (SNR, MSE, correlation)
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional


# ── Data containers ─────────────────────────────────────────
@dataclass
class SignalInfo:
    """Metadata about a loaded audio signal."""
    duration_sec:  float
    sample_rate:   int
    n_samples:     int
    n_fft_bins:    int
    sparsity_k:    int          # number of significant FFT bins
    sparsity_ratio: float       # k / n_fft_bins


@dataclass
class ReconstructionResult:
    """Full output of the CS reconstruction pipeline."""
    # Signals (as Python lists for JSON serialization)
    original:       list
    reconstructed:  list
    time_axis:      list

    # Frequency domain
    frequencies:    list        # Hz values
    fft_original:   list        # magnitude of original FFT
    fft_reconstructed: list     # magnitude of reconstructed FFT

    # Quality metrics
    snr_db:         float
    mse:            float
    max_error:      float
    correlation:    float

    # CS parameters used
    n_samples:      int         # N (signal length)
    n_measurements: int         # n (compressed measurements)
    sparsity_k:     int         # k (true sparsity)
    compression_ratio: float    # n / N

    # Solver info
    solver_status:  str
    theory_min_n:   int         # k * log(N)


# ── Core CS functions ────────────────────────────────────────

def analyze_sparsity(signal: np.ndarray,
                     threshold: float = 0.01) -> tuple[int, np.ndarray]:
    """
    Compute FFT and count significant (sparse) frequency components.

    Args:
        signal:    1D time domain signal
        threshold: fraction of max magnitude to count as significant

    Returns:
        (k, fft_magnitudes)  where k = number of significant bins
    """
    W = np.fft.rfft(signal)
    mags = np.abs(W)
    k = int(np.sum(mags > threshold * np.max(mags)))
    return k, mags


def build_sensing_matrix(n_meas: int, N: int,
                         seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build random Gaussian measurement matrix Phi and effective sensing matrix A.

    Phi: n x N random Gaussian (the 'compressed sensor')
    Psi: N x N IFFT matrix     (the sparsity basis)
    A  = Phi @ Psi              (effective sensing matrix in Fourier domain)

    Random Phi satisfies RIP with high probability when n >= O(k log N).

    Returns:
        (Phi, A)
    """
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((n_meas, N)) / np.sqrt(n_meas)
    Psi = np.fft.ifft(np.eye(N)).real          # IFFT matrix
    A   = Phi @ Psi
    return Phi, A


def reconstruct_l1(y: np.ndarray, A: np.ndarray,
                   N: int) -> tuple[np.ndarray, str]:
    """
    Recover sparse Fourier coefficients via l1 minimization:
        min  |s|_1   subject to   A @ s = y

    Under RIP this gives the exact sparse solution.
    Uses SCS solver with high precision — works well for N <= 512.

    Returns:
        (s_recovered, solver_status)
    """
    s_var = cp.Variable(N)
    prob  = cp.Problem(
        cp.Minimize(cp.norm1(s_var)),
        [A @ s_var == y]
    )
    prob.solve(solver=cp.SCS, verbose=False,
               eps=1e-8, max_iters=50000)
    return s_var.value, prob.status


def compute_metrics(original: np.ndarray,
                    reconstructed: np.ndarray) -> dict:
    """Compute reconstruction quality metrics."""
    mse      = float(np.mean((original - reconstructed) ** 2))
    sig_pow  = float(np.mean(original ** 2))
    snr_db   = float(10 * np.log10(sig_pow / max(mse, 1e-12)))
    max_err  = float(np.max(np.abs(original - reconstructed)))
    corr     = float(np.corrcoef(original, reconstructed)[0, 1])
    return dict(snr_db=snr_db, mse=mse, max_error=max_err, correlation=corr)


# ── Main pipeline ────────────────────────────────────────────

def run_cs_pipeline(signal: np.ndarray,
                    sample_rate: int,
                    compression_ratio: float = 0.25,
                    seed: int = 42) -> ReconstructionResult:
    """
    Full compressed sensing pipeline on a 1D audio signal.

    Args:
        signal:            1D numpy array (time domain)
        sample_rate:       samples per second
        compression_ratio: n/N — fraction of samples to use as measurements
        seed:              random seed for reproducibility

    Returns:
        ReconstructionResult with everything the frontend needs
    """
    N = len(signal)
    n = max(10, int(compression_ratio * N))   # number of measurements

    # Sparsity analysis
    k, fft_mags_orig = analyze_sparsity(signal)
    theory_min_n = int(k * np.log(max(N, 2)))

    # Time and frequency axes
    t_axis = (np.arange(N) / sample_rate).tolist()
    freqs  = np.fft.rfftfreq(N, 1 / sample_rate).tolist()

    # Compressed measurements
    Phi, A = build_sensing_matrix(n, N, seed=seed)
    y      = Phi @ signal

    # l1 minimization recovery
    s_recovered, status = reconstruct_l1(y, A, N)

    if s_recovered is None:
        # Fallback: return original if solver failed
        reconstructed = signal.copy()
        status = "failed"
    else:
        reconstructed = np.real(np.fft.ifft(s_recovered))

    # Frequency magnitude of reconstructed
    _, fft_mags_rec = analyze_sparsity(reconstructed)

    # Quality metrics
    metrics = compute_metrics(signal, reconstructed)

    return ReconstructionResult(
        original          = signal.tolist(),
        reconstructed     = reconstructed.tolist(),
        time_axis         = t_axis,
        frequencies       = freqs,
        fft_original      = fft_mags_orig.tolist(),
        fft_reconstructed = fft_mags_rec.tolist(),
        snr_db            = metrics["snr_db"],
        mse               = metrics["mse"],
        max_error         = metrics["max_error"],
        correlation       = metrics["correlation"],
        n_samples         = N,
        n_measurements    = n,
        sparsity_k        = k,
        compression_ratio = n / N,
        solver_status     = status,
        theory_min_n      = theory_min_n,
    )


def get_signal_info(signal: np.ndarray, sample_rate: int) -> SignalInfo:
    """Return metadata about a signal without running full reconstruction."""
    k, fft_mags = analyze_sparsity(signal)
    n_fft = len(fft_mags)
    return SignalInfo(
        duration_sec   = len(signal) / sample_rate,
        sample_rate    = sample_rate,
        n_samples      = len(signal),
        n_fft_bins     = n_fft,
        sparsity_k     = k,
        sparsity_ratio = k / n_fft,
    )


def generate_demo_signal(frequencies: list[float] = None,
                         amplitudes:  list[float] = None,
                         sample_rate: int   = 1000,
                         n_samples:   int   = 256,
                         seed:        int   = 42) -> tuple[np.ndarray, int]:
    """
    Generate a synthetic sparse audio signal for demo purposes.
    Signal is exactly k-sparse in the Fourier basis — ideal for CS demo.

    Uses n_samples=256 by default — this is the sweet spot where
    l1 minimization converges reliably and quickly.

    Returns:
        (signal, sample_rate)
    """
    rng = np.random.default_rng(seed)
    N   = n_samples

    if frequencies is None:
        # Pick k=4 random frequency bins
        freq_indices = sorted(rng.choice(N // 4, 4, replace=False).tolist())
    else:
        # Convert Hz to bin indices
        freq_indices = [int(f * N / sample_rate) for f in frequencies]

    if amplitudes is None:
        amplitudes = [1.0, 0.7, 0.5, 0.3]

    # Build sparse Fourier vector — exactly k nonzero bins
    s_true = np.zeros(N)
    for idx, amp in zip(freq_indices, amplitudes):
        s_true[int(idx)] = amp
        mirror = N - 1 - int(idx)
        if mirror != int(idx):
            s_true[mirror] = amp    # conjugate symmetry → real signal

    # IFFT → real time domain signal
    signal = np.real(np.fft.ifft(s_true))

    # Normalize to [-1, 1]
    signal = signal / np.max(np.abs(signal) + 1e-10)
    return signal, sample_rate
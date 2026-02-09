# analyse/deconvolve.py
"""
Deconvolution-based impulse response extraction using FFT-domain least squares.

Given:
- x[n]: the excitation signal (e.g. log sine sweep) that was played into the system
- y[n]: the recorded system output

We estimate the impulse response h[n] via:
    H(w) = Y(w) * conj(X(w)) / (|X(w)|^2 + eps)

This is a robust, deterministic "Wiener-ish" / Tikhonov-regularized deconvolution.

Notes:
- This approach does NOT perform Farina harmonic separation (nonlinear distortion isolation).
  It yields a single best-fit linear IR estimate.
- Pre/post silence in the sweep WAV is fine (zeros are fine).
- The recorded WAV must include enough post-silence to capture the full tail; padding cannot
  recover missing tail energy.

Outputs:
- Optional: write the extracted IR as a mono/stereo WAV to disk
- The IR can then be used with: ir.py, decay.py, rt60bands.py, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from analyse.io import (
    LoadedAudio,
    convert_wav_samples_to_float32,
    ensure_2d_channel_array,
    load_wav_file,
)

try:
    from scipy.io import wavfile
except ImportError as import_error:  # pragma: no cover
    raise ImportError(
        "scipy is required for WAV writing. Install with: pip install scipy"
    ) from import_error


# --------------------------------------------------------------------------------------
# Public data models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeconvolveSettings:
    # Regularization strength as a fraction of max(|X|^2).
    # Typical range: 1e-12 .. 1e-6. Higher = more stable but more biased.
    regularization_relative: float = 1e-10

    # If True, peak-normalise the output IR before writing/returning.
    normalise_peak: bool = True
    target_peak: float = 0.95

    # If True, remove DC offset from the returned IR (sometimes useful).
    remove_dc: bool = True

    # Length policy for the returned IR:
    # - "recorded": return first len(recorded) samples of the IFFT result (recommended)
    # - "full_fft": return the full FFT length N (rarely needed)
    output_length_mode: str = "recorded"  # "recorded" | "full_fft"

    # If True, trim trailing silence/noise (below -80 dB of peak).
    # Keeps analysis focused on the meaningful decay region.
    trim_trailing_noise: bool = True
    trailing_noise_threshold_db: float = -80.0


@dataclass(frozen=True)
class DeconvolvedImpulseResponse:
    samples: np.ndarray        # shape (N, C), float32
    sample_rate_hz: int
    recorded_file_path: Path
    sweep_file_path: Path


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def _downmix_to_mono_1d(samples_2d: np.ndarray) -> np.ndarray:
    """
    Downmix (N,C) to mono (N,) by averaging channels.
    """
    if samples_2d.ndim != 2:
        raise ValueError("Expected a 2D array (N,C).")
    return np.mean(samples_2d.astype(np.float64, copy=False), axis=1).astype(np.float32)


def _peak_normalise(samples: np.ndarray, target_peak: float) -> np.ndarray:
    samples = samples.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
    if peak <= 0.0:
        return samples
    scale = float(target_peak) / peak
    return (samples * scale).astype(np.float32)


def _write_wav_float32(path: Path, sample_rate_hz: int, samples_2d: np.ndarray) -> None:
    """
    Write float32 WAV using scipy.io.wavfile.write.
    samples_2d must be shaped (N,C).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), int(sample_rate_hz), samples_2d.astype(np.float32, copy=False))


def _trim_trailing_noise(
    samples_2d: np.ndarray,
    threshold_db: float = -80.0,
) -> np.ndarray:
    """
    Trim trailing noise from IR samples.

    Args:
        samples_2d: IR samples, shape (N, C)
        threshold_db: dB threshold relative to peak magnitude

    Returns:
        Trimmed samples where trailing samples below threshold are removed.
        Finds the last sample above threshold across all channels.
    """
    if samples_2d.size == 0:
        return samples_2d

    # Find peak magnitude across all samples and channels
    peak_magnitude = float(np.max(np.abs(samples_2d)))
    if peak_magnitude <= 0:
        return samples_2d

    # Convert threshold from dB to linear
    peak_db = 20 * np.log10(peak_magnitude + 1e-12)
    threshold_linear = 10 ** ((peak_db + threshold_db) / 20)

    # Find magnitude envelope (max across channels for each sample)
    magnitude = np.max(np.abs(samples_2d), axis=1)

    # Find last sample above threshold
    above_threshold = np.where(magnitude > threshold_linear)[0]
    if len(above_threshold) > 0:
        last_index = above_threshold[-1]
        # Keep a small margin (e.g., 1000 samples at 48kHz ≈ 20ms) to preserve natural tail
        margin_samples = max(0, int(0.02 * 48000))  # 20ms margin
        end_index = min(last_index + margin_samples, samples_2d.shape[0])
        return samples_2d[:end_index]

    return samples_2d  # All silence, return as-is


# --------------------------------------------------------------------------------------
# Core deconvolution
# --------------------------------------------------------------------------------------


def deconvolve_impulse_response(
    recorded_samples_2d: np.ndarray,
    sweep_samples_1d: np.ndarray,
    sample_rate_hz: int,
    settings: DeconvolveSettings,
) -> np.ndarray:
    """
    Compute IR for each channel in recorded_samples_2d using the same mono sweep_samples_1d.

    Returns:
        ir_samples_2d: shape (N_out, C), float32
    """
    recorded_samples_2d = ensure_2d_channel_array(convert_wav_samples_to_float32(recorded_samples_2d))
    sweep_samples_1d = np.asarray(sweep_samples_1d, dtype=np.float32)

    if recorded_samples_2d.shape[0] < 8 or sweep_samples_1d.size < 8:
        raise ValueError("Recorded and sweep must both contain at least a few samples.")

    n_recorded = int(recorded_samples_2d.shape[0])
    n_sweep = int(sweep_samples_1d.size)

    # Choose FFT length. For stable deconvolution, N should be >= max(len(recorded), len(sweep)).
    # Using next_pow2 keeps it efficient and avoids accidental truncation.
    n_fft = _next_power_of_two(max(n_recorded, n_sweep))

    # FFT of sweep (mono)
    x = sweep_samples_1d.astype(np.float64, copy=False)
    X = np.fft.rfft(x, n=n_fft)

    # Regularization epsilon: relative to max power of X
    power = np.abs(X) ** 2
    power_max = float(np.max(power)) if power.size > 0 else 0.0
    eps = float(settings.regularization_relative) * max(1e-30, power_max)

    denom = power + eps
    X_conj = np.conj(X)

    channel_count = int(recorded_samples_2d.shape[1])
    ir_channels: list[np.ndarray] = []

    for ch in range(channel_count):
        y = recorded_samples_2d[:, ch].astype(np.float64, copy=False)
        Y = np.fft.rfft(y, n=n_fft)

        # H = Y * conj(X) / (|X|^2 + eps)
        H = (Y * X_conj) / denom

        h = np.fft.irfft(H, n=n_fft).astype(np.float32)

        if settings.output_length_mode == "recorded":
            h = h[:n_recorded]
        elif settings.output_length_mode == "full_fft":
            pass
        else:
            raise ValueError(f"Unknown output_length_mode: {settings.output_length_mode}")

        if settings.remove_dc and h.size > 0:
            h = (h - float(np.mean(h))).astype(np.float32)

        ir_channels.append(h)

    # Stack back to (N,C)
    # Ensure all channels have same length (they should).
    n_out = int(ir_channels[0].size)
    ir_2d = np.stack([c[:n_out] for c in ir_channels], axis=1).astype(np.float32)

    if settings.normalise_peak:
        ir_2d = _peak_normalise(ir_2d, target_peak=float(settings.target_peak))

    if settings.trim_trailing_noise:
        ir_2d = _trim_trailing_noise(
            ir_2d,
            threshold_db=settings.trailing_noise_threshold_db,
        )

    return ir_2d


# --------------------------------------------------------------------------------------
# File-based convenience API
# --------------------------------------------------------------------------------------


def deconvolve_from_wav_files(
    recorded_wav_file_path: str | Path,
    sweep_wav_file_path: str | Path,
    settings: Optional[DeconvolveSettings] = None,
    output_ir_wav_file_path: Optional[str | Path] = None,
) -> DeconvolvedImpulseResponse:
    """
    Load recorded + sweep WAVs and produce a deconvolved IR.

    - recorded: mono or stereo, 48 kHz expected (validated via analyse.io)
    - sweep: mono or stereo, 48 kHz expected (validated via analyse.io)
            sweep will be downmixed to mono internally.

    If output_ir_wav_file_path is provided, writes the IR WAV to disk.

    Returns:
        DeconvolvedImpulseResponse (samples float32 (N,C))
    """
    if settings is None:
        settings = DeconvolveSettings()

    recorded = load_wav_file(
        wav_file_path=recorded_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    sweep = load_wav_file(
        wav_file_path=sweep_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    if recorded.sample_rate_hz != sweep.sample_rate_hz:
        raise ValueError(
            f"Sample rate mismatch: recorded={recorded.sample_rate_hz} Hz, sweep={sweep.sample_rate_hz} Hz"
        )

    sweep_mono = _downmix_to_mono_1d(sweep.samples)

    ir_samples = deconvolve_impulse_response(
        recorded_samples_2d=recorded.samples,
        sweep_samples_1d=sweep_mono,
        sample_rate_hz=recorded.sample_rate_hz,
        settings=settings,
    )

    ir = DeconvolvedImpulseResponse(
        samples=ir_samples,
        sample_rate_hz=int(recorded.sample_rate_hz),
        recorded_file_path=Path(recorded.file_path),
        sweep_file_path=Path(sweep.file_path),
    )

    if output_ir_wav_file_path is not None:
        out_path = Path(output_ir_wav_file_path)
        _write_wav_float32(out_path, ir.sample_rate_hz, ir.samples)

    return ir


def default_output_ir_path(recorded_wav_file_path: str | Path) -> Path:
    """
    Deterministic default output name:
      <recorded_stem>_ir.wav in the same folder.
    """
    p = Path(recorded_wav_file_path)
    return p.with_name(f"{p.stem}_ir.wav")

# analyse/spectrogram.py
"""
Offline spectrogram (time–frequency magnitude) analysis for reverb tails.

Why:
- Bridges fr.py (overall spectral balance) and decay/rt60bands (decay times).
- Makes frequency-dependent decay, modulation, resonances, and noise-floor takeover obvious.

Design conventions:
- load_wav_file(... expected_channel_mode="mono_or_stereo", allow_mono_and_upmix_to_stereo=False)
- analyse L/R unless explicitly downmixed to mono via settings

Output:
- One spectrogram figure per analysed channel (L/R or M).
- If output_basename is provided, writes:
    <basename>_spectrogram_<CH>.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.ticker as mticker

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import create_figure_and_axis, finalize_and_show_or_save


# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrogramAnalysisSettings:
    # Stereo handling:
    # If True and the input is stereo, downmix to mono (single channel "M") before analysis.
    use_mono_downmix_for_stereo: bool = False

    # Time selection:
    trim_to_peak: bool = True
    ignore_leading_seconds: float = 0.0

    # Optional: analyse only this duration from the analysed segment.
    analysis_duration_seconds: Optional[float] = None

    # STFT parameters:
    n_fft: int = 4096
    hop_length: int = 512
    use_hann_window: bool = True

    # Display/frequency bounds:
    floor_db: float = -120.0
    f_min_hz: float = 20.0
    f_max_hz: float = 20000.0

    # Color scaling:
    # If set, clamp display to [max_db - dynamic_range_db, max_db].
    # If None, use percentiles.
    dynamic_range_db: Optional[float] = 90.0


@dataclass(frozen=True)
class SpectrogramPlotSettings:
    # If None, auto-scale via analysis settings (dynamic_range_db / percentiles).
    vmin_db: Optional[float] = None
    vmax_db: Optional[float] = None


@dataclass(frozen=True)
class ChannelSpectrogramResult:
    channel_name: str
    sample_rate_hz: int

    analysis_start_sample_index: int
    analysis_length_samples: int

    time_seconds: np.ndarray          # (T,)
    frequency_hz: np.ndarray          # (F,)
    magnitude_db: np.ndarray          # (F, T)


# --------------------------------------------------------------------------------------
# Core STFT
# --------------------------------------------------------------------------------------


def _format_hz_ticks_for_log_axis(axis) -> None:
    """
    Make log-frequency axis show human-readable ticks like 20, 50, 100, 1k, 2k, 10k...
    """
    major_ticks_hz = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    axis.set_yticks(major_ticks_hz)

    def _hz_formatter(x, pos):
        if x >= 1000.0:
            return f"{int(x / 1000)}k"
        return f"{int(x)}"

    axis.yaxis.set_major_formatter(mticker.FuncFormatter(_hz_formatter))
    axis.yaxis.set_minor_formatter(mticker.NullFormatter())


def _compute_stft_magnitude_db(
    samples: np.ndarray,
    sample_rate_hz: int,
    n_fft: int,
    hop_length: int,
    use_hann_window: bool,
    floor_db: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (time_seconds, frequency_hz, magnitude_db) where magnitude_db shape is (F, T).

    Deterministic, readability over performance.
    """
    if samples.ndim != 1:
        raise ValueError("_compute_stft_magnitude_db expects a 1D mono array.")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive.")
    if samples.size < n_fft:
        raise ValueError("Not enough samples for STFT (need at least n_fft).")

    x = samples.astype(np.float64, copy=False)

    # Number of frames with "valid" framing (no padding).
    num_frames = 1 + (x.size - n_fft) // hop_length
    if num_frames < 1:
        raise ValueError("No STFT frames available with current n_fft/hop_length.")

    if use_hann_window:
        window = np.hanning(n_fft).astype(np.float64)
    else:
        window = np.ones(n_fft, dtype=np.float64)

    # Prepare output arrays
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate_hz)).astype(np.float32)
    num_bins = int(freq_hz.size)

    mag_db = np.empty((num_bins, num_frames), dtype=np.float32)

    # Frame-by-frame STFT (simple and inspectable)
    for frame_index in range(num_frames):
        start = frame_index * hop_length
        frame = x[start : start + n_fft]
        frame_w = frame * window
        spectrum = np.fft.rfft(frame_w)
        mag = np.abs(spectrum).astype(np.float64)

        # Convert magnitude to dB, with floor for stability
        mag = np.maximum(mag, 10.0 ** (float(floor_db) / 20.0))
        mag_db[:, frame_index] = (20.0 * np.log10(mag)).astype(np.float32)

    # Time stamp per frame (at frame start, simple + consistent)
    time_seconds = (np.arange(num_frames, dtype=np.float32) * float(hop_length) / float(sample_rate_hz)).astype(np.float32)

    return time_seconds, freq_hz, mag_db


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def analyse_spectrogram_for_channel(
    samples: np.ndarray,
    sample_rate_hz: int,
    channel_name: str,
    settings: SpectrogramAnalysisSettings,
) -> ChannelSpectrogramResult:
    if samples.ndim != 1:
        raise ValueError("analyse_spectrogram_for_channel expects a 1D mono array.")

    x = samples.astype(np.float64, copy=False)
    start_index = 0

    if settings.trim_to_peak:
        peak_index = int(np.argmax(np.abs(x)))
        start_index = peak_index
        x = x[peak_index:]

    if settings.ignore_leading_seconds > 0.0:
        ignore_count = int(round(float(settings.ignore_leading_seconds) * float(sample_rate_hz)))
        ignore_count = max(0, min(ignore_count, x.size))
        start_index += ignore_count
        x = x[ignore_count:]

    if settings.analysis_duration_seconds is not None:
        n = int(round(float(settings.analysis_duration_seconds) * float(sample_rate_hz)))
        n = max(0, min(n, x.size))
        x = x[:n]

    analysed = x.astype(np.float32)
    if analysed.size < settings.n_fft:
        raise ValueError("Not enough samples after trimming/selection for spectrogram (need at least n_fft).")

    time_s, freq_hz, mag_db = _compute_stft_magnitude_db(
        samples=analysed,
        sample_rate_hz=sample_rate_hz,
        n_fft=int(settings.n_fft),
        hop_length=int(settings.hop_length),
        use_hann_window=bool(settings.use_hann_window),
        floor_db=float(settings.floor_db),
    )

    return ChannelSpectrogramResult(
        channel_name=str(channel_name),
        sample_rate_hz=int(sample_rate_hz),
        analysis_start_sample_index=int(start_index),
        analysis_length_samples=int(analysed.size),
        time_seconds=time_s,
        frequency_hz=freq_hz,
        magnitude_db=mag_db,
    )


def analyse_spectrogram_from_wav_file(
    input_wav_file_path: str | Path,
    settings: Optional[SpectrogramAnalysisSettings] = None,
) -> List[ChannelSpectrogramResult]:
    if settings is None:
        settings = SpectrogramAnalysisSettings()

    loaded_audio = load_wav_file(
        wav_file_path=input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    channel_list = get_analysis_channels(
        loaded_audio=loaded_audio,
        use_mono_downmix_for_stereo=settings.use_mono_downmix_for_stereo,
    )

    results: List[ChannelSpectrogramResult] = []
    for channel_name, channel_samples in channel_list:
        results.append(
            analyse_spectrogram_for_channel(
                samples=channel_samples,
                sample_rate_hz=loaded_audio.sample_rate_hz,
                channel_name=channel_name,
                settings=settings,
            )
        )
    return results


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def plot_spectrogram_figure(
    result: ChannelSpectrogramResult,
    analysis_settings: SpectrogramAnalysisSettings,
    plot_settings: SpectrogramPlotSettings,
    title: Optional[str] = None,
):
    figure, axis = create_figure_and_axis(title=title)

    # Frequency limits
    nyquist = 0.5 * float(result.sample_rate_hz)
    f_min = float(np.clip(analysis_settings.f_min_hz, 1.0, nyquist))
    f_max = float(np.clip(analysis_settings.f_max_hz, f_min, nyquist))

    # Select frequency bins
    fmask = (result.frequency_hz >= f_min) & (result.frequency_hz <= f_max)
    freq = result.frequency_hz[fmask]
    mag = result.magnitude_db[fmask, :]

    if mag.size == 0:
        raise ValueError("Spectrogram frequency selection is empty (check f_min_hz/f_max_hz).")

    # Color scaling
    if plot_settings.vmax_db is not None:
        vmax = float(plot_settings.vmax_db)
    else:
        vmax = float(np.percentile(mag, 99.5))

    if plot_settings.vmin_db is not None:
        vmin = float(plot_settings.vmin_db)
    else:
        if analysis_settings.dynamic_range_db is not None:
            vmin = vmax - float(analysis_settings.dynamic_range_db)
        else:
            vmin = float(np.percentile(mag, 5.0))

    # Plot with log-frequency axis using pcolormesh
    # pcolormesh expects edges; we approximate edges by midpoints.
    t = result.time_seconds.astype(np.float64)
    f = freq.astype(np.float64)

    # Time edges
    if t.size == 1:
        t_edges = np.array([t[0], t[0] + 1e-3], dtype=np.float64)
    else:
        dt = np.diff(t)
        dt0 = float(dt[0])
        t_edges = np.concatenate(([t[0] - 0.5 * dt0], t[:-1] + 0.5 * dt, [t[-1] + 0.5 * float(dt[-1])]))

    # Frequency edges (log-ish spacing assumed; this is fine for rFFT bins too)
    if f.size == 1:
        f_edges = np.array([f[0], f[0] + 1.0], dtype=np.float64)
    else:
        df = np.diff(f)
        df0 = float(df[0])
        f_edges = np.concatenate(([f[0] - 0.5 * df0], f[:-1] + 0.5 * df, [f[-1] + 0.5 * float(df[-1])]))

    # Clamp edges to positive
    f_edges = np.maximum(f_edges, 1e-6)

    mesh = axis.pcolormesh(
        t_edges,
        f_edges,
        mag,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_yscale("log")
    axis.set_ylim(f_min, f_max)
    _format_hz_ticks_for_log_axis(axis)

    axis.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Colorbar
    figure.colorbar(mesh, ax=axis, label="Magnitude (dB)")

    return figure


def plot_spectrogram_from_wav_file(
    input_wav_file_path: str | Path,
    analysis_settings: Optional[SpectrogramAnalysisSettings] = None,
    plot_settings: Optional[SpectrogramPlotSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[ChannelSpectrogramResult]:
    """
    Convenience wrapper: analyse spectrogram then plot per channel.

    If output_basename is provided, writes one PNG per analysed channel:
      <basename>_spectrogram_<CH>.png
    """
    if analysis_settings is None:
        analysis_settings = SpectrogramAnalysisSettings()
    if plot_settings is None:
        plot_settings = SpectrogramPlotSettings()

    results = analyse_spectrogram_from_wav_file(
        input_wav_file_path=input_wav_file_path,
        settings=analysis_settings,
    )

    for result in results:
        title = f"Spectrogram — {input_wav_file_path} — {result.channel_name}"
        fig = plot_spectrogram_figure(
            result=result,
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            title=title,
        )

        if output_basename is None:
            out_path = None
        else:
            base = Path(output_basename)
            out_path = base.with_name(f"{base.stem}_spectrogram_{result.channel_name}.png").with_suffix(".png")

        finalize_and_show_or_save(
            figure=fig,
            output_path=out_path,
            show_interactive=show_interactive,
        )

    return results


# --------------------------------------------------------------------------------------
# CLI-friendly numeric summary
# --------------------------------------------------------------------------------------


def summarise_spectrogram_results_text(results: List[ChannelSpectrogramResult]) -> str:
    lines: List[str] = []
    for r in results:
        duration_s = float(r.analysis_length_samples) / float(r.sample_rate_hz)
        lines.append(
            f"[{r.channel_name}] start_sample={r.analysis_start_sample_index}  "
            f"len_samples={r.analysis_length_samples}  dur={duration_s:.3f}s  "
            f"stft(n_fft={r.magnitude_db.shape[0]*2-2}, frames={r.magnitude_db.shape[1]})"
        )
    return "\n".join(lines)

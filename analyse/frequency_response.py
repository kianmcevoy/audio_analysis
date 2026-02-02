# analyse/fr.py
"""
Offline frequency response / magnitude spectrum analysis.

Purpose:
- Complement decay and RT60 band analysis by showing spectral balance, resonances,
  rolloff, and noise-floor dominance.

Core output:
- Magnitude spectrum (dB) vs frequency (Hz, log-x)

Conventions match other analyse/* modules:
- load_wav_file(... expected_channel_mode="mono_or_stereo", allow_mono_and_upmix_to_stereo=False)
- analyse L/R unless explicitly downmixed to mono via settings

Notes:
- This is not a transfer-function estimate in the strict measurement sense.
  It is a deterministic spectrum view of the chosen segment of the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import matplotlib.ticker as mticker

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import (
    create_figure_and_axis,
    finalize_and_show_or_save,
    label_decibel_axis,
)

# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyResponseAnalysisSettings:
    # Stereo handling:
    # If True and the input is stereo, downmix to mono (single channel "M") before analysis.
    use_mono_downmix_for_stereo: bool = False

    # Time selection:
    # If True, start the analysed segment at the absolute peak (per analysed channel).
    trim_to_peak: bool = True

    # Optional: ignore this many seconds after the trim point (e.g. exclude direct sound).
    ignore_leading_seconds: float = 0.0

    # If provided, analyse only this many seconds from the start of the analysed segment.
    # If None, analyse to the end.
    analysis_duration_seconds: Optional[float] = None

    # Windowing:
    use_hann_window: bool = True

    # Display/robustness:
    magnitude_floor_db: float = -120.0

    # Frequency range for plot/summary
    f_min_hz: float = 20.0
    f_max_hz: float = 20000.0

  # Log-frequency smoothing:
    # 0 disables. Otherwise, this is the smoothing window width in "log bins".
    # Internally we resample magnitude onto a uniform log2(f) grid and apply a moving average.
    smoothing_log_bins: int = 0

    # Log-grid resolution: number of bins per octave for smoothing/interpolation.
    # Higher = smoother/cleaner but slightly more work.
    log_bins_per_octave: int = 96


@dataclass(frozen=True)
class FrequencyResponsePlotSettings:
    secondary_channel_alpha: float = 0.7

    # If None, auto-scale based on plotted data (with margin).
    # Otherwise use explicit (ymin, ymax).
    ylim_db: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ChannelFrequencyResponse:
    channel_name: str
    sample_rate_hz: int

    analysis_start_sample_index: int
    analysis_length_samples: int

    frequency_hz: np.ndarray
    magnitude_db: np.ndarray

    # Simple diagnostics
    peak_frequency_hz: float
    spectral_centroid_hz: float


# --------------------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------------------


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x.astype(np.float64), kernel, mode="same").astype(np.float32)

def _smooth_mag_db_log_frequency(
    frequency_hz: np.ndarray,
    magnitude_db: np.ndarray,
    f_min_hz: float,
    f_max_hz: float,
    smoothing_log_bins: int,
    log_bins_per_octave: int,
) -> np.ndarray:
    """
    Smooth magnitude_db over a uniform log2(f) axis, then interpolate back.

    - frequency_hz and magnitude_db are full rFFT arrays (including DC bin at 0 Hz).
    - We smooth only over [f_min_hz, f_max_hz] and leave outside unchanged.
    """
    if smoothing_log_bins <= 1:
        return magnitude_db

    # Work only on positive frequencies
    freq = frequency_hz.astype(np.float64)
    mag = magnitude_db.astype(np.float64)

    # Avoid log(0); define selection
    f_min = float(max(1.0, f_min_hz))
    f_max = float(max(f_min, f_max_hz))

    mask = (freq >= f_min) & (freq <= f_max)
    if not np.any(mask):
        return magnitude_db

    freq_sel = freq[mask]
    mag_sel = mag[mask]

    # Uniform grid in log2(f)
    log2_min = float(np.log2(freq_sel[0]))
    log2_max = float(np.log2(freq_sel[-1]))

    bins_per_oct = int(max(16, log_bins_per_octave))
    num_bins = int(max(8, np.ceil((log2_max - log2_min) * bins_per_oct))) + 1

    log2_grid = np.linspace(log2_min, log2_max, num_bins, dtype=np.float64)
    freq_grid = (2.0 ** log2_grid).astype(np.float64)

    # Interpolate onto log grid, smooth, then interpolate back
    mag_grid = np.interp(freq_grid, freq_sel, mag_sel)

    kernel = np.ones(int(smoothing_log_bins), dtype=np.float64) / float(smoothing_log_bins)
    mag_grid_smooth = np.convolve(mag_grid, kernel, mode="same")

    mag_sel_smooth = np.interp(freq_sel, freq_grid, mag_grid_smooth)

    out = magnitude_db.copy().astype(np.float32)
    out[mask] = mag_sel_smooth.astype(np.float32)
    return out



def analyse_frequency_response_for_channel(
    samples: np.ndarray,
    sample_rate_hz: int,
    channel_name: str,
    settings: FrequencyResponseAnalysisSettings,
) -> ChannelFrequencyResponse:
    if samples.ndim != 1:
        raise ValueError("analyse_frequency_response_for_channel expects a 1D mono array.")

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

    if x.size < 32:
        raise ValueError("Not enough samples after trimming/selection to analyse spectrum.")

    analysis_length_samples = int(x.size)

    if settings.use_hann_window:
        w = np.hanning(analysis_length_samples).astype(np.float64)
        xw = x * w
    else:
        xw = x

    # rFFT magnitude
    spectrum = np.fft.rfft(xw)
    mag = np.abs(spectrum).astype(np.float64)

    # Convert to dB with floor
    mag = np.maximum(mag, 10.0 ** (float(settings.magnitude_floor_db) / 20.0))
    mag_db = (20.0 * np.log10(mag)).astype(np.float32)

    # Frequency axis
    freq_hz = np.fft.rfftfreq(analysis_length_samples, d=1.0 / float(sample_rate_hz)).astype(np.float32)

    # Optional smoothing in log-frequency domain (more perceptually meaningful)
    if settings.smoothing_log_bins and int(settings.smoothing_log_bins) > 1:
        nyquist = 0.5 * float(sample_rate_hz)
        f_min_s = float(np.clip(settings.f_min_hz, 1.0, nyquist))
        f_max_s = float(np.clip(settings.f_max_hz, f_min_s, nyquist))

        mag_db = _smooth_mag_db_log_frequency(
            frequency_hz=freq_hz,
            magnitude_db=mag_db,
            f_min_hz=f_min_s,
            f_max_hz=f_max_s,
            smoothing_log_bins=int(settings.smoothing_log_bins),
            log_bins_per_octave=int(settings.log_bins_per_octave),
        )

    # Clamp frequency range for diagnostics
    nyquist = 0.5 * float(sample_rate_hz)
    f_min = float(np.clip(settings.f_min_hz, 0.0, nyquist))
    f_max = float(np.clip(settings.f_max_hz, f_min, nyquist))

    mask = (freq_hz >= f_min) & (freq_hz <= f_max)
    if not np.any(mask):
        raise ValueError("Selected frequency range is empty (check f_min_hz/f_max_hz).")

    freq_sel = freq_hz[mask]
    mag_sel_db = mag_db[mask]
    mag_sel_lin = (10.0 ** (mag_sel_db.astype(np.float64) / 20.0)).astype(np.float64)

    # Peak frequency (within selected range)
    peak_idx = int(np.argmax(mag_sel_db))
    peak_frequency_hz = float(freq_sel[peak_idx])

    # Spectral centroid (amplitude-weighted)
    weight_sum = float(np.sum(mag_sel_lin))
    if weight_sum > 0.0:
        spectral_centroid_hz = float(np.sum(freq_sel.astype(np.float64) * mag_sel_lin) / weight_sum)
    else:
        spectral_centroid_hz = float(freq_sel[0])

    return ChannelFrequencyResponse(
        channel_name=channel_name,
        sample_rate_hz=int(sample_rate_hz),
        analysis_start_sample_index=int(start_index),
        analysis_length_samples=int(analysis_length_samples),
        frequency_hz=freq_hz.astype(np.float32),
        magnitude_db=mag_db.astype(np.float32),
        peak_frequency_hz=peak_frequency_hz,
        spectral_centroid_hz=spectral_centroid_hz,
    )


def analyse_frequency_response_from_wav_file(
    input_wav_file_path: str | Path,
    settings: Optional[FrequencyResponseAnalysisSettings] = None,
) -> List[ChannelFrequencyResponse]:
    if settings is None:
        settings = FrequencyResponseAnalysisSettings()

    loaded_audio = load_wav_file(
        wav_file_path=input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    channel_list = get_analysis_channels(
        loaded_audio=loaded_audio,
        use_mono_downmix_for_stereo=settings.use_mono_downmix_for_stereo,
    )

    results: List[ChannelFrequencyResponse] = []
    for channel_name, channel_samples in channel_list:
        results.append(
            analyse_frequency_response_for_channel(
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


def plot_frequency_response_figure(
    channel_results: List[ChannelFrequencyResponse],
    analysis_settings: FrequencyResponseAnalysisSettings,
    plot_settings: FrequencyResponsePlotSettings,
    title: Optional[str] = None,
):
    figure, axis = create_figure_and_axis(title=title)

    # Define frequency range first (needed for auto-scaling)
    nyquist = 0.5 * float(channel_results[0].sample_rate_hz)
    f_min = float(np.clip(analysis_settings.f_min_hz, 1.0, nyquist))
    f_max = float(np.clip(analysis_settings.f_max_hz, f_min, nyquist))

    axis.set_xscale("log")
    
    # Use human-readable Hz ticks instead of 10^x
    major_ticks_hz = [
        20, 50, 100, 200, 500,
        1000, 2000, 5000, 10000, 20000
    ]

    axis.set_xticks(major_ticks_hz)

    def _hz_formatter(x, pos):
        if x >= 1000.0:
            return f"{int(x / 1000)}k"
        return f"{int(x)}"

    axis.xaxis.set_major_formatter(mticker.FuncFormatter(_hz_formatter))

    # Optional: fewer minor ticks for cleanliness
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    
    axis.set_xlabel("Frequency (Hz)")
    label_decibel_axis(axis)
    if plot_settings.ylim_db is not None:
        axis.set_ylim(plot_settings.ylim_db[0], plot_settings.ylim_db[1])
    else:
        # Auto scale: use plotted range with a small headroom margin.
        # Ignore extreme outliers by using percentiles.
        all_vals: List[np.ndarray] = []
        for r in channel_results:
            mask = (r.frequency_hz >= f_min) & (r.frequency_hz <= f_max)
            all_vals.append(r.magnitude_db[mask])

        y = np.concatenate(all_vals) if len(all_vals) else np.array([], dtype=np.float32)
        if y.size > 0:
            y_low = float(np.percentile(y, 1.0))
            y_high = float(np.percentile(y, 99.5))
            axis.set_ylim(y_low - 6.0, y_high + 6.0)

    axis.set_xlim(f_min, f_max)

    for idx, r in enumerate(channel_results):
        alpha = 1.0 if idx == 0 else float(plot_settings.secondary_channel_alpha)

        mask = (r.frequency_hz >= f_min) & (r.frequency_hz <= f_max)
        axis.plot(
            r.frequency_hz[mask],
            r.magnitude_db[mask],
            alpha=alpha,
            label=f"{r.channel_name}  peak={r.peak_frequency_hz:.0f}Hz  centroid={r.spectral_centroid_hz:.0f}Hz",
        )

    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(loc="best")
    return figure


def plot_frequency_response_from_wav_file(
    input_wav_file_path: str | Path,
    analysis_settings: Optional[FrequencyResponseAnalysisSettings] = None,
    plot_settings: Optional[FrequencyResponsePlotSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[ChannelFrequencyResponse]:
    if analysis_settings is None:
        analysis_settings = FrequencyResponseAnalysisSettings()
    if plot_settings is None:
        plot_settings = FrequencyResponsePlotSettings()

    results = analyse_frequency_response_from_wav_file(
        input_wav_file_path=input_wav_file_path,
        settings=analysis_settings,
    )

    title = f"Frequency response (spectrum) — {input_wav_file_path}"
    figure = plot_frequency_response_figure(
        channel_results=results,
        analysis_settings=analysis_settings,
        plot_settings=plot_settings,
        title=title,
    )

    if output_basename is None:
        output_path = None
    else:
        output_basename = Path(output_basename)
        output_path = output_basename.with_name(f"{output_basename.stem}_fr.png").with_suffix(".png")

    finalize_and_show_or_save(
        figure=figure,
        output_path=output_path,
        show_interactive=show_interactive,
    )

    return results


# --------------------------------------------------------------------------------------
# CLI-friendly numeric summary
# --------------------------------------------------------------------------------------


def summarise_frequency_response_results_text(channel_results: List[ChannelFrequencyResponse]) -> str:
    lines: List[str] = []
    for r in channel_results:
        lines.append(
            f"[{r.channel_name}] start_sample={r.analysis_start_sample_index}  "
            f"len_samples={r.analysis_length_samples}  "
            f"peak={r.peak_frequency_hz:.1f}Hz  centroid={r.spectral_centroid_hz:.1f}Hz"
        )
    return "\n".join(lines)

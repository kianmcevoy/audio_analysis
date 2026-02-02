# analyse/filterplot.py
"""
Filter frequency response analysis: magnitude and phase plots.

Purpose:
- Analyse building blocks of reverb algorithms: all-pass filters, comb filters,
  damping filters, etc.
- Display magnitude response (dB) and phase response (degrees or radians)
- Useful for examining filter characteristics and verifying implementations

Core output:
- Magnitude response (dB) vs frequency (Hz, log-x)
- Phase response (degrees) vs frequency (Hz, log-x)

Conventions match other analyse/* modules:
- Load WAV file assuming it's an impulse response or filter output
- Extract frequency response via FFT
- Plot magnitude and phase on separate subplots or combined figure
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
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
class FilterAnalysisSettings:
    # Stereo handling:
    # If True and the input is stereo, downmix to mono before analysis.
    use_mono_downmix_for_stereo: bool = False

    # Time selection:
    # If True, start the analysed segment at the absolute peak sample.
    trim_to_peak: bool = True

    # Optional: ignore this many seconds after the trim point.
    ignore_leading_seconds: float = 0.0

    # If provided, analyse only this many seconds from the start.
    # If None, analyse to the end.
    analysis_duration_seconds: Optional[float] = None

    # Windowing:
    use_hann_window: bool = True

    # Display/robustness:
    magnitude_floor_db: float = -120.0

    # Frequency range for plot
    f_min_hz: float = 20.0
    f_max_hz: float = 20000.0

    # Phase display mode: "degrees" or "radians"
    phase_mode: str = "degrees"

    # Unwrap phase for continuous display
    unwrap_phase: bool = True


@dataclass(frozen=True)
class FilterPlotSettings:
    secondary_channel_alpha: float = 0.7

    # If None, auto-scale based on plotted data.
    # Otherwise use explicit (ymin, ymax) for magnitude plot.
    magnitude_ylim_db: Optional[Tuple[float, float]] = None

    # If None, auto-scale phase plot.
    # Otherwise use explicit (ymin, ymax) for phase plot.
    phase_ylim: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ChannelFilterResponse:
    channel_name: str
    sample_rate_hz: int

    analysis_start_sample_index: int
    analysis_length_samples: int

    frequency_hz: np.ndarray
    magnitude_db: np.ndarray
    phase_response: np.ndarray  # radians or degrees depending on settings

    # Simple diagnostics
    peak_frequency_hz: float
    magnitude_at_1khz_db: float


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def analyse_filter_response_for_channel(
    samples: np.ndarray,
    sample_rate_hz: int,
    channel_name: str,
    settings: FilterAnalysisSettings,
) -> ChannelFilterResponse:
    if samples.ndim != 1:
        raise ValueError("analyse_filter_response_for_channel expects a 1D mono array.")

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
        raise ValueError("Not enough samples after trimming/selection to analyse filter response.")

    analysis_length_samples = int(x.size)

    if settings.use_hann_window:
        w = np.hanning(analysis_length_samples).astype(np.float64)
        xw = x * w
    else:
        xw = x

    # Complex FFT
    spectrum = np.fft.rfft(xw)

    # Magnitude
    mag = np.abs(spectrum).astype(np.float64)
    mag = np.maximum(mag, 10.0 ** (float(settings.magnitude_floor_db) / 20.0))
    mag_db = (20.0 * np.log10(mag)).astype(np.float32)

    # Phase
    phase_rad = np.angle(spectrum).astype(np.float64)
    
    if settings.unwrap_phase:
        phase_rad = np.unwrap(phase_rad)
    
    if settings.phase_mode == "degrees":
        phase_response = np.rad2deg(phase_rad).astype(np.float32)
    else:
        phase_response = phase_rad.astype(np.float32)

    # Frequency axis
    freq_hz = np.fft.rfftfreq(analysis_length_samples, d=1.0 / float(sample_rate_hz)).astype(np.float32)

    # Diagnostics
    nyquist = 0.5 * float(sample_rate_hz)
    f_min = float(np.clip(settings.f_min_hz, 0.0, nyquist))
    f_max = float(np.clip(settings.f_max_hz, f_min, nyquist))

    mask = (freq_hz >= f_min) & (freq_hz <= f_max)
    if not np.any(mask):
        raise ValueError("Selected frequency range is empty.")

    freq_sel = freq_hz[mask]
    mag_sel_db = mag_db[mask]

    # Peak frequency in selected range
    peak_idx_rel = int(np.argmax(mag_sel_db))
    peak_frequency_hz = float(freq_sel[peak_idx_rel])

    # Magnitude at 1 kHz (if in range)
    idx_1k = int(np.argmin(np.abs(freq_hz - 1000.0)))
    magnitude_at_1khz_db = float(mag_db[idx_1k])

    return ChannelFilterResponse(
        channel_name=channel_name,
        sample_rate_hz=sample_rate_hz,
        analysis_start_sample_index=start_index,
        analysis_length_samples=analysis_length_samples,
        frequency_hz=freq_hz,
        magnitude_db=mag_db,
        phase_response=phase_response,
        peak_frequency_hz=peak_frequency_hz,
        magnitude_at_1khz_db=magnitude_at_1khz_db,
    )


def analyse_filter_response_from_wav_file(
    input_wav_file_path: str | Path,
    settings: FilterAnalysisSettings,
) -> List[ChannelFilterResponse]:
    input_wav_file_path = Path(input_wav_file_path)

    loaded = load_wav_file(
        wav_file_path=input_wav_file_path,
        expected_sample_rate_hz=48000,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    channels = get_analysis_channels(
        loaded_audio=loaded,
        use_mono_downmix_for_stereo=bool(settings.use_mono_downmix_for_stereo),
    )

    results: List[ChannelFilterResponse] = []
    for channel_name, channel_samples in channels:
        r = analyse_filter_response_for_channel(
            samples=channel_samples,
            sample_rate_hz=int(loaded.sample_rate_hz),
            channel_name=channel_name,
            settings=settings,
        )
        results.append(r)

    return results


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def plot_filter_response_figure(
    channel_results: List[ChannelFilterResponse],
    analysis_settings: FilterAnalysisSettings,
    plot_settings: FilterPlotSettings,
    title: str,
) -> plt.Figure:
    if not channel_results:
        raise ValueError("No channel results to plot.")

    nyquist = 0.5 * float(channel_results[0].sample_rate_hz)
    f_min = float(np.clip(analysis_settings.f_min_hz, 1.0, nyquist))
    f_max = float(np.clip(analysis_settings.f_max_hz, f_min, nyquist))

    # Create figure with two subplots: magnitude and phase
    figure, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8))
    figure.suptitle(title, fontsize=12, fontweight="bold")

    # Magnitude plot
    ax_mag.set_xscale("log")
    ax_mag.set_xlabel("Frequency (Hz)")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))

    # Auto-scale magnitude or use explicit limits
    if plot_settings.magnitude_ylim_db is None:
        all_vals: List[np.ndarray] = []
        for r in channel_results:
            mask = (r.frequency_hz >= f_min) & (r.frequency_hz <= f_max)
            all_vals.append(r.magnitude_db[mask])

        y = np.concatenate(all_vals) if len(all_vals) else np.array([], dtype=np.float32)
        if y.size > 0:
            y_low = float(np.percentile(y, 1.0))
            y_high = float(np.percentile(y, 99.5))
            ax_mag.set_ylim(y_low - 6.0, y_high + 6.0)
    else:
        ax_mag.set_ylim(plot_settings.magnitude_ylim_db)

    ax_mag.set_xlim(f_min, f_max)

    for idx, r in enumerate(channel_results):
        alpha = 1.0 if idx == 0 else float(plot_settings.secondary_channel_alpha)
        mask = (r.frequency_hz >= f_min) & (r.frequency_hz <= f_max)
        ax_mag.plot(
            r.frequency_hz[mask],
            r.magnitude_db[mask],
            alpha=alpha,
            label=f"{r.channel_name}  peak={r.peak_frequency_hz:.0f}Hz  @1kHz={r.magnitude_at_1khz_db:.1f}dB",
        )

    ax_mag.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax_mag.legend(loc="best", fontsize=9)

    # Phase plot
    ax_phase.set_xscale("log")
    ax_phase.set_xlabel("Frequency (Hz)")
    phase_unit = "degrees" if analysis_settings.phase_mode == "degrees" else "radians"
    ax_phase.set_ylabel(f"Phase ({phase_unit})")
    ax_phase.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))

    # Auto-scale phase or use explicit limits
    if plot_settings.phase_ylim is None:
        all_phase: List[np.ndarray] = []
        for r in channel_results:
            mask = (r.frequency_hz >= f_min) & (r.frequency_hz <= f_max)
            all_phase.append(r.phase_response[mask])

        p = np.concatenate(all_phase) if len(all_phase) else np.array([], dtype=np.float32)
        if p.size > 0:
            p_low = float(np.percentile(p, 1.0))
            p_high = float(np.percentile(p, 99.0))
            margin = (p_high - p_low) * 0.1
            ax_phase.set_ylim(p_low - margin, p_high + margin)
    else:
        ax_phase.set_ylim(plot_settings.phase_ylim)

    ax_phase.set_xlim(f_min, f_max)

    for idx, r in enumerate(channel_results):
        alpha = 1.0 if idx == 0 else float(plot_settings.secondary_channel_alpha)
        mask = (r.frequency_hz >= f_min) & (r.frequency_hz <= f_max)
        ax_phase.plot(
            r.frequency_hz[mask],
            r.phase_response[mask],
            alpha=alpha,
            label=r.channel_name,
        )

    ax_phase.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax_phase.legend(loc="best", fontsize=9)

    plt.tight_layout()
    return figure


def plot_filter_response_from_wav_file(
    input_wav_file_path: str | Path,
    analysis_settings: Optional[FilterAnalysisSettings] = None,
    plot_settings: Optional[FilterPlotSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[ChannelFilterResponse]:
    if analysis_settings is None:
        analysis_settings = FilterAnalysisSettings()
    if plot_settings is None:
        plot_settings = FilterPlotSettings()

    results = analyse_filter_response_from_wav_file(
        input_wav_file_path=input_wav_file_path,
        settings=analysis_settings,
    )

    title = f"Filter frequency response — {input_wav_file_path}"
    figure = plot_filter_response_figure(
        channel_results=results,
        analysis_settings=analysis_settings,
        plot_settings=plot_settings,
        title=title,
    )

    if output_basename is None:
        output_path = None
    else:
        output_basename = Path(output_basename)
        output_path = output_basename.with_name(f"{output_basename.stem}_filter.png").with_suffix(".png")

    finalize_and_show_or_save(
        figure=figure,
        output_path=output_path,
        show_interactive=show_interactive,
    )

    return results


# --------------------------------------------------------------------------------------
# CLI-friendly numeric summary
# --------------------------------------------------------------------------------------


def summarise_filter_response_results_text(channel_results: List[ChannelFilterResponse]) -> str:
    lines: List[str] = []
    for r in channel_results:
        lines.append(
            f"[{r.channel_name}] start_sample={r.analysis_start_sample_index}  "
            f"len_samples={r.analysis_length_samples}  "
            f"peak={r.peak_frequency_hz:.1f}Hz  @1kHz={r.magnitude_at_1khz_db:.1f}dB"
        )
    return "\n".join(lines)

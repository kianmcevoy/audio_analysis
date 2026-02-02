# analyse/waterfall.py
"""
Waterfall (cumulative spectral decay style) plot from STFT magnitude slices.

Why:
- Shows which frequencies "ring" over time (ridges), and whether decay is smooth or modal.
- Complements: fr.py (overall balance), rt60bands.py (broad-band decay), spectrogram.py (continuous TF view).

Design conventions:
- load_wav_file(... expected_channel_mode="mono_or_stereo", allow_mono_and_upmix_to_stereo=False)
- analyse L/R unless explicitly downmixed to mono via settings
- deterministic, readable, CLI-first

Plot styles:
- style="3d" (default): true waterfall in 3D (freq vs time vs dB)
- style="2d": stacked ridges (freq vs dB with time offsets)

Notes on log-frequency in 3D:
- mplot3d doesn't reliably support log axes.
- We plot X = log10(freq_hz), and label ticks in Hz (20, 50, 100, 1k, 2k, 10k, 20k).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import create_figure_and_axis, finalize_and_show_or_save, DEFAULT_FIGURE_SIZE, DEFAULT_DPI


# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterfallAnalysisSettings:
    # Stereo handling
    use_mono_downmix_for_stereo: bool = False

    # Time selection policy (same semantics as decay/spectrogram)
    trim_to_peak: bool = True
    ignore_leading_seconds: float = 0.0
    analysis_duration_seconds: Optional[float] = None

    # STFT parameters (match spectrogram defaults)
    n_fft: int = 4096
    hop_length: int = 512
    use_hann_window: bool = True

    # Frequency display bounds
    f_min_hz: float = 20.0
    f_max_hz: float = 20000.0

    # Slice selection
    slice_mode: str = "auto"  # "auto" | "uniform_time" | "uniform_frames"
    num_slices: int = 18
    slice_spacing_seconds: float = 0.05
    start_time_seconds: float = 0.0
    end_time_seconds: Optional[float] = None

    # dB normalisation
    db_reference: str = "global_max"  # "global_max" | "slice_max"

    # Optional log-frequency smoothing per slice
    smoothing_log_bins: int = 0
    log_bins_per_octave: int = 96

    # Display dynamic range (relative dB)
    dynamic_range_db: float = 80.0

    # Magnitude floor before dB conversion
    floor_db: float = -120.0


@dataclass(frozen=True)
class WaterfallPlotSettings:
    style: str = "3d"  # "3d" | "2d"
    secondary_channel_alpha: float = 0.7

    # 3D view
    elev_deg: float = 30.0
    azim_deg: float = -60.0

    # 2D ridges
    ridge_offset_db: float = 6.0

    # If provided, overrides z/y limits for readability
    zlim_db: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ChannelWaterfallResult:
    channel_name: str
    sample_rate_hz: int

    analysis_start_sample_index: int
    analysis_length_samples: int

    # Slice info
    slice_times_seconds: np.ndarray          # (S,)
    frequency_hz: np.ndarray                # (F,)
    slice_magnitude_rel_db: np.ndarray       # (S, F) in [-dynamic_range_db, 0]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def _hz_major_ticks_for_audio(f_min_hz: float, f_max_hz: float) -> List[float]:
    ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    out = [float(t) for t in ticks if float(t) >= float(f_min_hz) and float(t) <= float(f_max_hz)]
    if not out:
        out = [float(max(1.0, f_min_hz)), float(f_max_hz)]
    return out


def _hz_tick_formatter(x, pos) -> str:
    if x >= 1000.0:
        return f"{int(round(x / 1000.0))}k"
    return f"{int(round(x))}"


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float32, copy=False)
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
    Smooth a single magnitude curve in dB on a uniform log2(f) grid, then interpolate back.
    Deterministic, SciPy-free.
    """
    if smoothing_log_bins <= 1:
        return magnitude_db.astype(np.float32, copy=False)

    freq = frequency_hz.astype(np.float64, copy=False)
    mag = magnitude_db.astype(np.float64, copy=False)

    f_min = float(max(1.0, f_min_hz))
    f_max = float(max(f_min, f_max_hz))

    mask = (freq >= f_min) & (freq <= f_max)
    if not np.any(mask):
        return magnitude_db.astype(np.float32, copy=False)

    freq_sel = freq[mask]
    mag_sel = mag[mask]

    # Uniform grid in log2(f)
    log2_min = float(np.log2(freq_sel[0]))
    log2_max = float(np.log2(freq_sel[-1]))

    bins_per_oct = int(max(16, log_bins_per_octave))
    num_bins = int(max(8, np.ceil((log2_max - log2_min) * bins_per_oct))) + 1

    log2_grid = np.linspace(log2_min, log2_max, num_bins, dtype=np.float64)
    freq_grid = 2.0 ** log2_grid

    mag_grid = np.interp(freq_grid, freq_sel, mag_sel)
    mag_grid_smooth = _moving_average_1d(mag_grid.astype(np.float32), int(smoothing_log_bins)).astype(np.float64)

    mag_sel_smooth = np.interp(freq_sel, freq_grid, mag_grid_smooth)

    out = magnitude_db.astype(np.float32, copy=True)
    out[mask] = mag_sel_smooth.astype(np.float32)
    return out


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
    Frame times are window-start times (simple + consistent).
    """
    if samples.ndim != 1:
        raise ValueError("_compute_stft_magnitude_db expects a 1D mono array.")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive.")
    if samples.size < n_fft:
        raise ValueError("Not enough samples for STFT (need at least n_fft).")

    x = samples.astype(np.float64, copy=False)
    num_frames = 1 + (x.size - n_fft) // hop_length
    if num_frames < 1:
        raise ValueError("No STFT frames available with current n_fft/hop_length.")

    window = np.hanning(n_fft).astype(np.float64) if use_hann_window else np.ones(n_fft, dtype=np.float64)
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate_hz)).astype(np.float32)
    num_bins = int(freq_hz.size)

    mag_db = np.empty((num_bins, num_frames), dtype=np.float32)

    mag_floor_lin = 10.0 ** (float(floor_db) / 20.0)

    for frame_index in range(num_frames):
        start = frame_index * hop_length
        frame = x[start : start + n_fft]
        frame_w = frame * window
        spectrum = np.fft.rfft(frame_w)
        mag = np.abs(spectrum).astype(np.float64)
        mag = np.maximum(mag, mag_floor_lin)
        mag_db[:, frame_index] = (20.0 * np.log10(mag)).astype(np.float32)

    time_seconds = (np.arange(num_frames, dtype=np.float32) * float(hop_length) / float(sample_rate_hz)).astype(np.float32)
    return time_seconds, freq_hz, mag_db


def _select_slice_frame_indices(
    frame_times_seconds: np.ndarray,
    settings: WaterfallAnalysisSettings,
) -> np.ndarray:
    """
    Choose STFT frame indices for slices based on slice_mode.
    Returns ordered unique indices.
    """
    if frame_times_seconds.size == 0:
        return np.zeros((0,), dtype=np.int32)

    start_t = float(max(0.0, settings.start_time_seconds))
    end_t = float(settings.end_time_seconds) if settings.end_time_seconds is not None else float(frame_times_seconds[-1])

    if end_t <= start_t:
        end_t = float(frame_times_seconds[-1])

    in_range = (frame_times_seconds >= start_t) & (frame_times_seconds <= end_t)
    if not np.any(in_range):
        return np.zeros((0,), dtype=np.int32)

    idx_min = int(np.argmax(in_range))
    idx_max = int(np.max(np.nonzero(in_range)))

    mode = str(settings.slice_mode).lower()

    if mode == "uniform_frames":
        # Evenly spaced indices in frame domain
        count = int(max(1, settings.num_slices))
        indices = np.linspace(idx_min, idx_max, count, dtype=np.int32)
        return np.unique(indices)

    if mode == "uniform_time":
        spacing = float(max(1e-4, settings.slice_spacing_seconds))
        times = np.arange(start_t, end_t + 1e-9, spacing, dtype=np.float64)
        indices = []
        for t in times:
            # nearest frame
            j = int(np.argmin(np.abs(frame_times_seconds - float(t))))
            if j >= idx_min and j <= idx_max:
                indices.append(j)
        if len(indices) == 0:
            indices = [idx_min, idx_max]
        return np.unique(np.array(indices, dtype=np.int32))

    # auto: evenly spaced in time range using num_slices
    count = int(max(2, settings.num_slices))
    target_times = np.linspace(start_t, end_t, count, dtype=np.float64)
    indices = []
    for t in target_times:
        j = int(np.argmin(np.abs(frame_times_seconds - float(t))))
        if j >= idx_min and j <= idx_max:
            indices.append(j)
    return np.unique(np.array(indices, dtype=np.int32))


def _build_rel_db_slices(
    freq_hz: np.ndarray,
    mag_db: np.ndarray,                 # (F, T)
    frame_indices: np.ndarray,          # (S,)
    f_min_hz: float,
    f_max_hz: float,
    settings: WaterfallAnalysisSettings,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (freq_selected, slices_rel_db) where slices_rel_db is (S, Fsel) in [-dyn, 0].
    """
    nyquist = float(freq_hz[-1]) if freq_hz.size else 0.0
    f_min = float(np.clip(f_min_hz, 1.0, nyquist))
    f_max = float(np.clip(f_max_hz, f_min, nyquist))

    fmask = (freq_hz >= f_min) & (freq_hz <= f_max)
    if not np.any(fmask):
        raise ValueError("Waterfall frequency selection is empty (check f_min_hz/f_max_hz).")

    f_sel = freq_hz[fmask].astype(np.float32)
    slices_db = mag_db[fmask][:, frame_indices].T.astype(np.float32)  # (S, F)

    # Optional per-slice log-frequency smoothing
    if settings.smoothing_log_bins and int(settings.smoothing_log_bins) > 1:
        smoothed = []
        for s in range(slices_db.shape[0]):
            smoothed.append(
                _smooth_mag_db_log_frequency(
                    frequency_hz=f_sel,
                    magnitude_db=slices_db[s],
                    f_min_hz=f_min,
                    f_max_hz=f_max,
                    smoothing_log_bins=int(settings.smoothing_log_bins),
                    log_bins_per_octave=int(settings.log_bins_per_octave),
                )
            )
        slices_db = np.stack(smoothed, axis=0).astype(np.float32)

    # Normalise to relative dB
    ref_mode = str(settings.db_reference).lower()

    if ref_mode == "slice_max":
        ref = np.max(slices_db, axis=1, keepdims=True)  # (S,1)
        rel = slices_db - ref
    else:
        # global_max (default)
        ref = float(np.max(slices_db))
        rel = slices_db - ref

    dyn = float(max(10.0, settings.dynamic_range_db))
    rel = np.clip(rel, -dyn, 0.0).astype(np.float32)

    return f_sel, rel


# --------------------------------------------------------------------------------------
# Analysis entrypoints
# --------------------------------------------------------------------------------------


def analyse_waterfall_for_channel(
    samples: np.ndarray,
    sample_rate_hz: int,
    channel_name: str,
    settings: WaterfallAnalysisSettings,
) -> ChannelWaterfallResult:
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
        raise ValueError("Not enough samples after trimming/selection for waterfall (need at least n_fft).")

    frame_times_s, freq_hz, mag_db = _compute_stft_magnitude_db(
        samples=analysed,
        sample_rate_hz=sample_rate_hz,
        n_fft=int(settings.n_fft),
        hop_length=int(settings.hop_length),
        use_hann_window=bool(settings.use_hann_window),
        floor_db=float(settings.floor_db),
    )

    frame_indices = _select_slice_frame_indices(frame_times_s, settings)
    if frame_indices.size < 2:
        raise ValueError("Not enough slices selected for waterfall (increase duration or num_slices).")

    f_sel, slices_rel_db = _build_rel_db_slices(
        freq_hz=freq_hz,
        mag_db=mag_db,
        frame_indices=frame_indices,
        f_min_hz=float(settings.f_min_hz),
        f_max_hz=float(settings.f_max_hz),
        settings=settings,
    )

    slice_times = frame_times_s[frame_indices].astype(np.float32)

    return ChannelWaterfallResult(
        channel_name=str(channel_name),
        sample_rate_hz=int(sample_rate_hz),
        analysis_start_sample_index=int(start_index),
        analysis_length_samples=int(analysed.size),
        slice_times_seconds=slice_times,
        frequency_hz=f_sel,
        slice_magnitude_rel_db=slices_rel_db,
    )


def analyse_waterfall_from_wav_file(
    input_wav_file_path: str | Path,
    settings: Optional[WaterfallAnalysisSettings] = None,
) -> List[ChannelWaterfallResult]:
    if settings is None:
        settings = WaterfallAnalysisSettings()

    loaded_audio = load_wav_file(
        wav_file_path=input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    channel_list = get_analysis_channels(
        loaded_audio=loaded_audio,
        use_mono_downmix_for_stereo=settings.use_mono_downmix_for_stereo,
    )

    results: List[ChannelWaterfallResult] = []
    for channel_name, channel_samples in channel_list:
        results.append(
            analyse_waterfall_for_channel(
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


def _apply_log_hz_ticks_to_axis(axis, f_min_hz: float, f_max_hz: float) -> None:
    axis.set_xscale("log")
    axis.set_xlim(float(f_min_hz), float(f_max_hz))
    ticks = _hz_major_ticks_for_audio(f_min_hz, f_max_hz)
    axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(mticker.FuncFormatter(_hz_tick_formatter))
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())


def plot_waterfall_figure(
    result: ChannelWaterfallResult,
    analysis_settings: WaterfallAnalysisSettings,
    plot_settings: WaterfallPlotSettings,
    title: Optional[str] = None,
):
    style = str(plot_settings.style).lower()
    nyquist = 0.5 * float(result.sample_rate_hz)

    f_min = float(np.clip(analysis_settings.f_min_hz, 1.0, nyquist))
    f_max = float(np.clip(analysis_settings.f_max_hz, f_min, nyquist))

    dyn = float(max(10.0, analysis_settings.dynamic_range_db))

    if style == "2d":
        figure, axis = create_figure_and_axis(title=title)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Magnitude (dB, offset by time slice)")
        _apply_log_hz_ticks_to_axis(axis, f_min, f_max)

        ridge_offset = float(max(0.0, plot_settings.ridge_offset_db))
        S = int(result.slice_times_seconds.size)

        # Baseline: newest slice at top (offset 0), older slices shifted downward
        # This reads naturally as "decay over time".
        for i in range(S):
            t = float(result.slice_times_seconds[i])
            offset = -float(i) * ridge_offset
            y = result.slice_magnitude_rel_db[i] + offset
            axis.plot(result.frequency_hz, y, alpha=0.9, label=None)

        # Annotate a few slice times for readability (first, middle, last)
        for idx in [0, S // 2, S - 1]:
            t = float(result.slice_times_seconds[idx])
            offset = -float(idx) * ridge_offset
            axis.text(
                float(result.frequency_hz[0]),
                float(offset),
                f"{t:.2f}s",
                fontsize=9,
                verticalalignment="bottom",
            )

        axis.grid(True, which="both", linestyle=":", linewidth=0.5)

        # y-limits
        if plot_settings.zlim_db is not None:
            axis.set_ylim(plot_settings.zlim_db[0], plot_settings.zlim_db[1])
        else:
            # show full range of offsets + dyn
            axis.set_ylim(-float(S - 1) * ridge_offset - dyn, 2.0)

        return figure

    # --- 3D default ---
    figure = plt.figure(figsize=DEFAULT_FIGURE_SIZE, dpi=DEFAULT_DPI)
    axis = figure.add_subplot(111, projection='3d')
    if title:
        axis.set_title(title)

    # Plot X as log10(freq) because 3D log axes are unreliable
    x_log = np.log10(result.frequency_hz.astype(np.float64))
    y_time = result.slice_times_seconds.astype(np.float64)
    z_db = result.slice_magnitude_rel_db.astype(np.float64)  # (S, F)

    # Create mesh grid for surface plot
    X, Y = np.meshgrid(x_log, y_time)
    
    # Draw as 3D surface mesh
    axis.plot_surface(
        X,
        Y,
        z_db,
        cmap='viridis',
        alpha=0.8,
        antialiased=True,
        edgecolor='none',
        linewidth=0,
    )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Time (s)")
    axis.set_zlabel("Magnitude (dB rel)")
    
    # Invert Y axis so earliest time is furthest away (back to front decay)
    axis.invert_yaxis()

    # X ticks: label in Hz
    ticks_hz = _hz_major_ticks_for_audio(f_min, f_max)
    ticks_x = [np.log10(float(t)) for t in ticks_hz]
    axis.set_xlim(np.log10(f_min), np.log10(f_max))
    axis.set_xticks(ticks_x)
    axis.set_xticklabels([_hz_tick_formatter(t, None) for t in ticks_hz])

    # Z limits
    if plot_settings.zlim_db is not None:
        axis.set_zlim(plot_settings.zlim_db[0], plot_settings.zlim_db[1])
    else:
        axis.set_zlim(-dyn, 2.0)

    axis.view_init(elev=float(plot_settings.elev_deg), azim=float(plot_settings.azim_deg))

    return figure


def plot_waterfall_from_wav_file(
    input_wav_file_path: str | Path,
    analysis_settings: Optional[WaterfallAnalysisSettings] = None,
    plot_settings: Optional[WaterfallPlotSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[ChannelWaterfallResult]:
    """
    Convenience wrapper: analyse + plot per channel.

    If output_basename is provided, writes one PNG per analysed channel:
      <basename>_waterfall_<CH>.png
    """
    if analysis_settings is None:
        analysis_settings = WaterfallAnalysisSettings()
    if plot_settings is None:
        plot_settings = WaterfallPlotSettings()

    results = analyse_waterfall_from_wav_file(
        input_wav_file_path=input_wav_file_path,
        settings=analysis_settings,
    )

    for r in results:
        title = f"Waterfall — {input_wav_file_path} — {r.channel_name}"
        fig = plot_waterfall_figure(
            result=r,
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            title=title,
        )

        if output_basename is None:
            out_path = None
        else:
            base = Path(output_basename)
            out_path = base.with_name(f"{base.stem}_waterfall_{r.channel_name}.png").with_suffix(".png")

        finalize_and_show_or_save(
            figure=fig,
            output_path=out_path,
            show_interactive=show_interactive,
        )

    return results


# --------------------------------------------------------------------------------------
# CLI-friendly numeric summary
# --------------------------------------------------------------------------------------


def summarise_waterfall_results_text(results: List[ChannelWaterfallResult]) -> str:
    lines: List[str] = []
    for r in results:
        dur = float(r.analysis_length_samples) / float(r.sample_rate_hz)
        lines.append(
            f"[{r.channel_name}] start_sample={r.analysis_start_sample_index}  dur={dur:.3f}s  "
            f"slices={int(r.slice_times_seconds.size)}  f_bins={int(r.frequency_hz.size)}"
        )
    return "\n".join(lines)

# analyse/modalcloud.py
"""
Modal cloud plot: per-frequency-bin decay time estimates (RT60) from STFT magnitude vs time.

What it shows:
- Scatter of (frequency, RT60) points.
- Highlights resonant ridges as clusters of long-decay points at specific frequencies.
- Complements waterfall: waterfall is "lines over time"; cloud is "summary points".

Approach (simple, deterministic, SciPy-free):
1) Compute STFT magnitude in dB (like spectrogram).
2) Aggregate to log-frequency bins (stable + perceptual).
3) For each bin, form a decay curve in dB relative to its own maximum (0 dB at peak).
4) Fit a line over a standard range (T30 default: -5 .. -35 dB), extrapolate RT60.

Notes:
- This is not a room-acoustics “true modal analysis”; it’s a practical diagnostic for reverb tails.
- Uses rFFT bins => frequency resolution depends on n_fft and hop_length.

Outputs:
- One plot per analysed channel (L/R or M).
- If output_basename provided:
    <basename>_modalcloud_<CH>.png
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.ticker as mticker

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import create_figure_and_axis, finalize_and_show_or_save


# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalCloudAnalysisSettings:
    # Stereo handling
    use_mono_downmix_for_stereo: bool = False

    # Time selection (same semantics as other modules)
    trim_to_peak: bool = True
    ignore_leading_seconds: float = 0.0
    analysis_duration_seconds: Optional[float] = None

    # STFT parameters
    n_fft: int = 8192
    hop_length: int = 512
    use_hann_window: bool = True

    # Frequency range
    f_min_hz: float = 20.0
    f_max_hz: float = 20000.0

    # Log binning
    log_bins_per_octave: int = 24  # perceptual-ish resolution
    min_bins: int = 24

    # Normalisation per bin: dB relative to max over time in that bin (peak becomes 0 dB)
    floor_db: float = -120.0

    # Decay fit windows (like decay.py, but per-bin)
    fit_lower_limit_db: float = -80.0
    # Default metric is T30-like:
    t30_range_db: Tuple[float, float] = (-5.0, -35.0)
    t20_range_db: Tuple[float, float] = (-5.0, -25.0)
    edt_range_db: Tuple[float, float] = (0.0, -10.0)

    metric: str = "t30"  # "t30" | "t20" | "edt"

    # Reliability / pruning
    min_fit_points: int = 10
    min_peak_db_above_floor: float = 20.0  # ignore bins whose peak is too close to floor


@dataclass(frozen=True)
class ModalCloudPlotSettings:
    secondary_channel_alpha: float = 0.7

    # If True, plot a median-smoothed curve over the cloud (helps readability)
    show_median_curve: bool = True
    median_octave_window: float = 0.25  # window for median curve in octaves

    # Axis limits
    ylim_seconds: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ModalPoint:
    centre_hz: float
    rt60_seconds: float
    r_squared: float


@dataclass(frozen=True)
class ChannelModalCloudResult:
    channel_name: str
    sample_rate_hz: int

    analysis_start_sample_index: int
    analysis_length_samples: int

    metric: str  # "t30"/"t20"/"edt"
    points: List[ModalPoint]


# --------------------------------------------------------------------------------------
# Core STFT
# --------------------------------------------------------------------------------------


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
    """
    if samples.ndim != 1:
        raise ValueError("_compute_stft_magnitude_db expects a 1D mono array.")
    if samples.size < n_fft:
        raise ValueError("Not enough samples for STFT (need at least n_fft).")

    x = samples.astype(np.float64, copy=False)
    num_frames = 1 + (x.size - n_fft) // hop_length
    if num_frames < 1:
        raise ValueError("No STFT frames available.")

    window = np.hanning(n_fft).astype(np.float64) if use_hann_window else np.ones(n_fft, dtype=np.float64)
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate_hz)).astype(np.float32)
    num_bins = int(freq_hz.size)

    mag_db = np.empty((num_bins, num_frames), dtype=np.float32)
    mag_floor_lin = 10.0 ** (float(floor_db) / 20.0)

    for frame_index in range(num_frames):
        start = frame_index * hop_length
        frame = x[start : start + n_fft]
        spectrum = np.fft.rfft(frame * window)
        mag = np.abs(spectrum).astype(np.float64)
        mag = np.maximum(mag, mag_floor_lin)
        mag_db[:, frame_index] = (20.0 * np.log10(mag)).astype(np.float32)

    time_seconds = (np.arange(num_frames, dtype=np.float32) * float(hop_length) / float(sample_rate_hz)).astype(np.float32)
    return time_seconds, freq_hz, mag_db


# --------------------------------------------------------------------------------------
# Log-frequency binning
# --------------------------------------------------------------------------------------


def _build_log_bins(f_min_hz: float, f_max_hz: float, bins_per_octave: int, min_bins: int) -> np.ndarray:
    f_min = float(max(1.0, f_min_hz))
    f_max = float(max(f_min * 1.001, f_max_hz))
    octaves = float(log2(f_max / f_min))
    n = int(max(min_bins, ceil(octaves * float(max(4, bins_per_octave)))))
    # edges in log space
    edges = f_min * (2.0 ** (np.linspace(0.0, octaves, n + 1, dtype=np.float64)))
    return edges.astype(np.float32)


def _aggregate_to_log_bins(
    freq_hz: np.ndarray,         # (F,)
    mag_db: np.ndarray,          # (F,T)
    bin_edges_hz: np.ndarray,    # (B+1,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate STFT magnitude into log-frequency bins.

    Returns:
      centres_hz: (B,)
      mag_db_bins: (B,T)  (computed by averaging in linear magnitude, then converting to dB)
    """
    F, T = mag_db.shape
    edges = bin_edges_hz.astype(np.float64)
    centres = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)

    # Convert to linear magnitude for averaging
    mag_lin = 10.0 ** (mag_db.astype(np.float64) / 20.0)

    out = np.full((centres.size, T), np.nan, dtype=np.float32)

    for b in range(centres.size):
        lo = float(edges[b])
        hi = float(edges[b + 1])
        mask = (freq_hz >= lo) & (freq_hz < hi)
        if not np.any(mask):
            continue
        m = np.mean(mag_lin[mask, :], axis=0)
        m = np.maximum(m, 1e-30)
        out[b, :] = (20.0 * np.log10(m)).astype(np.float32)

    return centres, out


# --------------------------------------------------------------------------------------
# Per-bin decay fitting
# --------------------------------------------------------------------------------------


def _interpolated_crossing_time_seconds(
    time_seconds: np.ndarray,
    curve_db: np.ndarray,
    target_db: float,
) -> Optional[float]:
    below = curve_db <= float(target_db)
    if not np.any(below):
        return None
    idx = int(np.argmax(below))
    if idx == 0:
        return float(time_seconds[0])

    t0 = float(time_seconds[idx - 1])
    t1 = float(time_seconds[idx])
    y0 = float(curve_db[idx - 1])
    y1 = float(curve_db[idx])
    if y1 == y0:
        return t1
    frac = (float(target_db) - y0) / (y1 - y0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return t0 + frac * (t1 - t0)


def _fit_decay_slope(
    time_seconds: np.ndarray,
    curve_db: np.ndarray,
    range_db: Tuple[float, float],
    fit_lower_limit_db: float,
    min_points: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit y = m t + b over curve section between range_db[0] and range_db[1].
    Returns (slope, intercept, r2, rt60) or None.
    """
    high_db, low_db = float(range_db[0]), float(range_db[1])
    if low_db > high_db:
        raise ValueError("range_db should be (higher_db, lower_db), e.g. (-5, -35).")

    effective_low = max(low_db, float(fit_lower_limit_db))

    t_start = _interpolated_crossing_time_seconds(time_seconds, curve_db, high_db)
    t_end = _interpolated_crossing_time_seconds(time_seconds, curve_db, effective_low)
    if t_start is None or t_end is None or t_end <= t_start:
        return None

    mask = (time_seconds >= t_start) & (time_seconds <= t_end)
    if int(np.sum(mask)) < int(min_points):
        return None

    t = time_seconds[mask].astype(np.float64)
    y = curve_db[mask].astype(np.float64)

    A = np.column_stack([t, np.ones_like(t)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    if slope >= 0.0:
        return None

    y_pred = slope * t + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    rt60 = float(-60.0 / slope)
    return slope, intercept, r2, rt60


# --------------------------------------------------------------------------------------
# Analysis entrypoints
# --------------------------------------------------------------------------------------


def analyse_modal_cloud_for_channel(
    samples: np.ndarray,
    sample_rate_hz: int,
    channel_name: str,
    settings: ModalCloudAnalysisSettings,
) -> ChannelModalCloudResult:
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
        raise ValueError("Not enough samples after trimming/selection for modal cloud (need at least n_fft).")

    time_s, freq_hz, mag_db = _compute_stft_magnitude_db(
        samples=analysed,
        sample_rate_hz=sample_rate_hz,
        n_fft=int(settings.n_fft),
        hop_length=int(settings.hop_length),
        use_hann_window=bool(settings.use_hann_window),
        floor_db=float(settings.floor_db),
    )

    nyquist = 0.5 * float(sample_rate_hz)
    f_min = float(np.clip(settings.f_min_hz, 1.0, nyquist))
    f_max = float(np.clip(settings.f_max_hz, f_min, nyquist))

    # Restrict frequency rows first (for speed + consistency)
    fmask = (freq_hz >= f_min) & (freq_hz <= f_max)
    freq_sel = freq_hz[fmask]
    mag_sel = mag_db[fmask, :]

    # Log bins
    edges = _build_log_bins(f_min, f_max, int(settings.log_bins_per_octave), int(settings.min_bins))
    centres_hz, mag_bins_db = _aggregate_to_log_bins(freq_sel, mag_sel, edges)

    metric = str(settings.metric).lower()
    if metric == "t20":
        range_db = settings.t20_range_db
    elif metric == "edt":
        range_db = settings.edt_range_db
    else:
        metric = "t30"
        range_db = settings.t30_range_db

    points: List[ModalPoint] = []

    for b in range(centres_hz.size):
        curve = mag_bins_db[b, :]
        if not np.all(np.isfinite(curve)):
            continue

        # Normalize each bin to its own max across time: peak becomes 0 dB
        peak = float(np.max(curve))
        if (peak - float(settings.floor_db)) < float(settings.min_peak_db_above_floor):
            continue

        rel = (curve - peak).astype(np.float32)

        fit = _fit_decay_slope(
            time_seconds=time_s,
            curve_db=rel,
            range_db=range_db,
            fit_lower_limit_db=float(settings.fit_lower_limit_db),
            min_points=int(settings.min_fit_points),
        )
        if fit is None:
            continue

        _, _, r2, rt60 = fit
        points.append(
            ModalPoint(
                centre_hz=float(centres_hz[b]),
                rt60_seconds=float(rt60),
                r_squared=float(r2),
            )
        )

    points.sort(key=lambda p: p.centre_hz)

    return ChannelModalCloudResult(
        channel_name=str(channel_name),
        sample_rate_hz=int(sample_rate_hz),
        analysis_start_sample_index=int(start_index),
        analysis_length_samples=int(analysed.size),
        metric=metric,
        points=points,
    )


def analyse_modal_cloud_from_wav_file(
    input_wav_file_path: str | Path,
    settings: Optional[ModalCloudAnalysisSettings] = None,
) -> List[ChannelModalCloudResult]:
    if settings is None:
        settings = ModalCloudAnalysisSettings()

    loaded_audio = load_wav_file(
        wav_file_path=input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    channel_list = get_analysis_channels(
        loaded_audio=loaded_audio,
        use_mono_downmix_for_stereo=settings.use_mono_downmix_for_stereo,
    )

    results: List[ChannelModalCloudResult] = []
    for channel_name, channel_samples in channel_list:
        results.append(
            analyse_modal_cloud_for_channel(
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


def _apply_hz_ticks(axis, f_min_hz: float, f_max_hz: float) -> None:
    axis.set_xscale("log")
    axis.set_xlim(float(f_min_hz), float(f_max_hz))
    major_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    major_ticks = [t for t in major_ticks if float(t) >= float(f_min_hz) and float(t) <= float(f_max_hz)]
    if major_ticks:
        axis.set_xticks(major_ticks)
    axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())


def _median_curve(points: List[ModalPoint], window_octaves: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(points) < 8:
        return None
    window_oct = float(max(0.01, window_octaves))
    freqs = np.array([p.centre_hz for p in points], dtype=np.float64)
    rt60 = np.array([p.rt60_seconds for p in points], dtype=np.float64)

    logf = np.log2(freqs)
    out_f = []
    out_y = []

    for i in range(freqs.size):
        lo = logf[i] - 0.5 * window_oct
        hi = logf[i] + 0.5 * window_oct
        m = (logf >= lo) & (logf <= hi)
        if int(np.sum(m)) < 3:
            continue
        out_f.append(freqs[i])
        out_y.append(float(np.median(rt60[m])))

    if len(out_f) < 4:
        return None
    return np.array(out_f, dtype=np.float32), np.array(out_y, dtype=np.float32)


def plot_modal_cloud_figure(
    result: ChannelModalCloudResult,
    analysis_settings: ModalCloudAnalysisSettings,
    plot_settings: ModalCloudPlotSettings,
    title: Optional[str] = None,
):
    figure, axis = create_figure_and_axis(title=title)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel(f"RT60 estimate (s) [{result.metric.upper()}]")

    nyquist = 0.5 * float(result.sample_rate_hz)
    f_min = float(np.clip(analysis_settings.f_min_hz, 1.0, nyquist))
    f_max = float(np.clip(analysis_settings.f_max_hz, f_min, nyquist))
    _apply_hz_ticks(axis, f_min, f_max)

    if len(result.points) == 0:
        axis.text(0.5, 0.5, "No valid points (insufficient decay range).", transform=axis.transAxes, ha="center")
        axis.grid(True, which="both", linestyle=":", linewidth=0.5)
        return figure

    freqs = np.array([p.centre_hz for p in result.points], dtype=np.float32)
    rt60 = np.array([p.rt60_seconds for p in result.points], dtype=np.float32)

    axis.scatter(freqs, rt60, s=12, alpha=0.85, label=f"{result.channel_name} ({len(result.points)} pts)")

    if plot_settings.show_median_curve:
        med = _median_curve(result.points, plot_settings.median_octave_window)
        if med is not None:
            f_med, y_med = med
            axis.plot(f_med, y_med, alpha=0.9, linestyle="-", label=f"{result.channel_name} median")

    if plot_settings.ylim_seconds is not None:
        axis.set_ylim(plot_settings.ylim_seconds[0], plot_settings.ylim_seconds[1])

    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(loc="best")
    return figure


def plot_modal_cloud_from_wav_file(
    input_wav_file_path: str | Path,
    analysis_settings: Optional[ModalCloudAnalysisSettings] = None,
    plot_settings: Optional[ModalCloudPlotSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[ChannelModalCloudResult]:
    """
    Convenience wrapper: analyse + plot per channel.

    If output_basename is provided, writes one PNG per analysed channel:
      <basename>_modalcloud_<CH>.png
    """
    if analysis_settings is None:
        analysis_settings = ModalCloudAnalysisSettings()
    if plot_settings is None:
        plot_settings = ModalCloudPlotSettings()

    results = analyse_modal_cloud_from_wav_file(
        input_wav_file_path=input_wav_file_path,
        settings=analysis_settings,
    )

    for r in results:
        title = f"Modal cloud — {input_wav_file_path} — {r.channel_name}"
        fig = plot_modal_cloud_figure(
            result=r,
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            title=title,
        )

        if output_basename is None:
            out_path = None
        else:
            base = Path(output_basename)
            out_path = base.with_name(f"{base.stem}_modalcloud_{r.channel_name}.png").with_suffix(".png")

        finalize_and_show_or_save(
            figure=fig,
            output_path=out_path,
            show_interactive=show_interactive,
        )

    return results


# --------------------------------------------------------------------------------------
# CLI-friendly numeric summary
# --------------------------------------------------------------------------------------


def summarise_modal_cloud_results_text(results: List[ChannelModalCloudResult]) -> str:
    lines: List[str] = []
    for r in results:
        dur = float(r.analysis_length_samples) / float(r.sample_rate_hz)
        lines.append(
            f"[{r.channel_name}] metric={r.metric} start_sample={r.analysis_start_sample_index} dur={dur:.3f}s points={len(r.points)}"
        )
        if len(r.points) > 0:
            rt = np.array([p.rt60_seconds for p in r.points], dtype=np.float64)
            lines.append(f"  rt60: median={np.median(rt):.3f}s  p90={np.percentile(rt,90):.3f}s  max={np.max(rt):.3f}s")
    return "\n".join(lines)

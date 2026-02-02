# analyse/diffusion.py
"""
Diffusion / decorrelation analysis over time.

Outputs (per time window):
- Autocorrelation "peakedness" vs time:
    max_abs_autocorr = max_{lag=1..L} |r_norm(lag)|
  Lower values generally indicate a more noise-like / diffuse tail.
  High values indicate periodicity / ringing / strong modes.

- Echo density vs time (simple, robust heuristic):
    density_fraction = fraction(|x| > threshold_rms * rms)
  Optionally normalized so that a Gaussian signal yields ~1.0:
    density_normalized = density_fraction / gaussian_expected_fraction

- Stereo decorrelation metrics vs time:
    corr0 = Pearson correlation coefficient (zero-lag) in each window
    iacc_max = max cross-correlation over +-lag (normalized)

This is intended as a diagnostic view for "diffuse field onset", modulation,
and stereo diffusion, not a standards-grade room-acoustics tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import create_figure_and_axis, finalize_and_show_or_save, label_time_axis_seconds


# --------------------------------------------------------------------------------------
# Settings / results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffusionAnalysisSettings:
    # Stereo policy (matches other modules)
    use_mono_downmix_for_stereo: bool = False

    # Time-zero policy (applied per analysed channel unless downmixed)
    trim_to_peak: bool = True
    ignore_leading_seconds: float = 0.0

    # Windowing for time-series metrics
    window_seconds: float = 0.050
    hop_seconds: float = 0.010

    # Autocorrelation / IACC lag range
    max_lag_milliseconds: float = 10.0

    # Echo density threshold definition
    echo_density_threshold_rms: float = 1.0
    echo_density_normalise_to_gaussian: bool = True


@dataclass(frozen=True)
class DiffusionTimeSeries:
    time_seconds: np.ndarray  # (num_frames,)

    # 0..1-ish
    max_abs_autocorr: np.ndarray  # (num_frames,)
    echo_density: np.ndarray      # (num_frames,)  fraction or gaussian-normalized

    # Stereo-only (None for mono)
    corr0: Optional[np.ndarray] = None       # (num_frames,)
    iacc_max: Optional[np.ndarray] = None    # (num_frames,)


@dataclass(frozen=True)
class DiffusionChannelResult:
    channel_name: str
    sample_rate_hz: int
    series: DiffusionTimeSeries


# --------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------


def _trim_and_ignore(
    samples: np.ndarray,
    sample_rate_hz: int,
    trim_to_peak: bool,
    ignore_leading_seconds: float,
) -> Tuple[np.ndarray, int]:
    """
    Return (trimmed_samples, start_index_in_original).
    """
    x = samples.astype(np.float64, copy=False)
    start_index = 0

    if trim_to_peak:
        peak_index = int(np.argmax(np.abs(x)))
        start_index += peak_index
        x = x[peak_index:]

    if ignore_leading_seconds > 0.0:
        ignore_count = int(round(ignore_leading_seconds * float(sample_rate_hz)))
        ignore_count = max(0, min(ignore_count, x.size))
        start_index += ignore_count
        x = x[ignore_count:]

    return x.astype(np.float32), start_index


def _frame_count(num_samples: int, win: int, hop: int) -> int:
    if num_samples < win:
        return 0
    return 1 + (num_samples - win) // hop


def _expected_gaussian_abs_exceedance(threshold_rms: float) -> float:
    """
    For x ~ N(0, sigma^2), with threshold = threshold_rms * sigma:
      P(|x| > k*sigma) = 2 * (1 - Phi(k))
    Implemented via erf for determinism without SciPy stats.
    """
    k = float(threshold_rms)
    # Phi(k) = 0.5 * (1 + erf(k/sqrt(2)))
    phi = 0.5 * (1.0 + math.erf(k / np.sqrt(2.0)))
    return 2.0 * (1.0 - phi)


def _windowed_max_abs_autocorr(x: np.ndarray, max_lag: int) -> float:
    """
    Normalized autocorrelation within a window. Returns max |r(lag)| for lag=1..max_lag.
    Uses unbiased-ish normalization by energy at lag 0.
    """
    if x.size < 4:
        return float("nan")

    x0 = x - float(np.mean(x))
    denom = float(np.dot(x0, x0))
    if denom <= 1e-20:
        return float("nan")

    # r(l) = sum_{n} x[n] x[n+l] / denom
    best = 0.0
    L = min(max_lag, x0.size - 2)
    for lag in range(1, L + 1):
        r = float(np.dot(x0[:-lag], x0[lag:]) / denom)
        best = max(best, abs(r))
    return best


def _windowed_corr0(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation coefficient (zero lag) for a window.
    """
    if x.size != y.size or x.size < 4:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    xx = float(np.dot(x0, x0))
    yy = float(np.dot(y0, y0))
    if xx <= 1e-20 or yy <= 1e-20:
        return float("nan")
    return float(np.dot(x0, y0) / np.sqrt(xx * yy))


def _windowed_iacc_max(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    """
    IACC-like metric: max normalized cross-correlation over lags in [-max_lag, +max_lag].
    Normalization: sqrt(Ex * Ey) in the window.
    """
    if x.size != y.size or x.size < 4:
        return float("nan")

    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))

    ex = float(np.dot(x0, x0))
    ey = float(np.dot(y0, y0))
    denom = np.sqrt(ex * ey)
    if denom <= 1e-20:
        return float("nan")

    L = min(max_lag, x0.size - 2)
    best = 0.0

    # Positive lag: correlate x[n] with y[n+lag]
    for lag in range(0, L + 1):
        if lag == 0:
            r = float(np.dot(x0, y0) / denom)
        else:
            r = float(np.dot(x0[:-lag], y0[lag:]) / denom)
        best = max(best, abs(r))

    # Negative lag: correlate x[n+lag] with y[n]
    for lag in range(1, L + 1):
        r = float(np.dot(x0[lag:], y0[:-lag]) / denom)
        best = max(best, abs(r))

    return best


def _windowed_echo_density(x: np.ndarray, threshold_rms: float, normalise_to_gaussian: bool) -> float:
    """
    Echo density proxy:
      fraction of samples whose abs amplitude exceeds threshold_rms * rms(window)
    """
    if x.size < 4:
        return float("nan")
    x0 = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x0 * x0)))
    if rms <= 1e-20:
        return float("nan")

    thr = float(threshold_rms) * rms
    frac = float(np.mean(np.abs(x0) > thr))

    if not normalise_to_gaussian:
        return frac

    expected = _expected_gaussian_abs_exceedance(threshold_rms)
    if expected <= 1e-12:
        return float("nan")
    return frac / expected


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def analyse_diffusion_for_channel(
    samples: np.ndarray,
    sample_rate_hz: int,
    channel_name: str,
    settings: DiffusionAnalysisSettings,
) -> DiffusionChannelResult:
    x, _ = _trim_and_ignore(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        trim_to_peak=settings.trim_to_peak,
        ignore_leading_seconds=settings.ignore_leading_seconds,
    )

    win = int(round(settings.window_seconds * float(sample_rate_hz)))
    hop = int(round(settings.hop_seconds * float(sample_rate_hz)))
    win = max(16, win)
    hop = max(1, hop)

    num_frames = _frame_count(x.size, win, hop)
    if num_frames <= 0:
        raise ValueError("Not enough samples for diffusion analysis windows.")

    max_lag = int(round((settings.max_lag_milliseconds / 1000.0) * float(sample_rate_hz)))
    max_lag = max(1, max_lag)

    times = np.zeros((num_frames,), dtype=np.float32)
    max_abs_ac = np.zeros((num_frames,), dtype=np.float32)
    echo_den = np.zeros((num_frames,), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop
        end = start + win
        w = x[start:end]

        times[i] = (start + win * 0.5) / float(sample_rate_hz)
        max_abs_ac[i] = float(_windowed_max_abs_autocorr(w, max_lag=max_lag))
        echo_den[i] = float(
            _windowed_echo_density(
                w,
                threshold_rms=settings.echo_density_threshold_rms,
                normalise_to_gaussian=settings.echo_density_normalise_to_gaussian,
            )
        )

    series = DiffusionTimeSeries(
        time_seconds=times,
        max_abs_autocorr=max_abs_ac,
        echo_density=echo_den,
        corr0=None,
        iacc_max=None,
    )

    return DiffusionChannelResult(
        channel_name=channel_name,
        sample_rate_hz=sample_rate_hz,
        series=series,
    )


def analyse_diffusion_from_wav_file(
    input_wav_file_path: str | Path,
    settings: Optional[DiffusionAnalysisSettings] = None,
) -> List[DiffusionChannelResult]:
    if settings is None:
        settings = DiffusionAnalysisSettings()

    loaded = load_wav_file(
        wav_file_path=input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    channels = get_analysis_channels(
        loaded_audio=loaded,
        use_mono_downmix_for_stereo=settings.use_mono_downmix_for_stereo,
    )

    results: List[DiffusionChannelResult] = []
    for ch_name, ch_samples in channels:
        results.append(
            analyse_diffusion_for_channel(
                samples=ch_samples,
                sample_rate_hz=loaded.sample_rate_hz,
                channel_name=ch_name,
                settings=settings,
            )
        )

    # If we have true stereo (L and R), compute stereo metrics once and attach to both.
    if (not settings.use_mono_downmix_for_stereo) and len(channels) == 2:
        # Re-trim using same policy, but ensure both are aligned consistently:
        # Use peak index from the *sum* to avoid L/R slight peak differences.
        left_raw = channels[0][1]
        right_raw = channels[1][1]
        combined = (left_raw.astype(np.float64) + right_raw.astype(np.float64)) * 0.5

        combined_trimmed, start_idx = _trim_and_ignore(
            samples=combined.astype(np.float32),
            sample_rate_hz=loaded.sample_rate_hz,
            trim_to_peak=settings.trim_to_peak,
            ignore_leading_seconds=settings.ignore_leading_seconds,
        )

        # Apply same start index to L/R
        l = left_raw.astype(np.float32)[start_idx:start_idx + combined_trimmed.size]
        r = right_raw.astype(np.float32)[start_idx:start_idx + combined_trimmed.size]

        win = int(round(settings.window_seconds * float(loaded.sample_rate_hz)))
        hop = int(round(settings.hop_seconds * float(loaded.sample_rate_hz)))
        win = max(16, win)
        hop = max(1, hop)

        num_frames = _frame_count(combined_trimmed.size, win, hop)
        max_lag = int(round((settings.max_lag_milliseconds / 1000.0) * float(loaded.sample_rate_hz)))
        max_lag = max(1, max_lag)

        corr0 = np.zeros((num_frames,), dtype=np.float32)
        iacc = np.zeros((num_frames,), dtype=np.float32)

        for i in range(num_frames):
            start = i * hop
            end = start + win
            wl = l[start:end]
            wr = r[start:end]
            corr0[i] = float(_windowed_corr0(wl, wr))
            iacc[i] = float(_windowed_iacc_max(wl, wr, max_lag=max_lag))

        # Attach to results (same timeline as existing series by construction)
        for idx in range(len(results)):
            s = results[idx].series
            results[idx] = DiffusionChannelResult(
                channel_name=results[idx].channel_name,
                sample_rate_hz=results[idx].sample_rate_hz,
                series=DiffusionTimeSeries(
                    time_seconds=s.time_seconds,
                    max_abs_autocorr=s.max_abs_autocorr,
                    echo_density=s.echo_density,
                    corr0=corr0,
                    iacc_max=iacc,
                ),
            )

    return results


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def plot_diffusion_from_wav_file(
    input_wav_file_path: str | Path,
    analysis_settings: Optional[DiffusionAnalysisSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[DiffusionChannelResult]:
    if analysis_settings is None:
        analysis_settings = DiffusionAnalysisSettings()

    results = analyse_diffusion_from_wav_file(
        input_wav_file_path=input_wav_file_path,
        settings=analysis_settings,
    )

    title = f"Diffusion — {input_wav_file_path}"
    figure, axis = create_figure_and_axis(title=title)
    label_time_axis_seconds(axis)
    axis.set_ylabel("Metric (unitless)")
    axis.set_ylim(-0.05, 1.25)

    # Plot per-channel autocorr and echo density.
    for ch_i, r in enumerate(results):
        alpha = 1.0 if ch_i == 0 else 0.7
        axis.plot(
            r.series.time_seconds,
            r.series.max_abs_autocorr,
            alpha=alpha,
            label=f"max|autocorr| {r.channel_name}",
        )
        axis.plot(
            r.series.time_seconds,
            r.series.echo_density,
            alpha=alpha,
            linestyle="--",
            label=f"echo_density {r.channel_name}",
        )

    # Stereo-only overlays (plotted once; same arrays attached to both results)
    if results and results[0].series.corr0 is not None and results[0].series.iacc_max is not None:
        axis.plot(
            results[0].series.time_seconds,
            results[0].series.corr0,
            linestyle=":",
            label="corr0 (L,R)",
        )
        axis.plot(
            results[0].series.time_seconds,
            results[0].series.iacc_max,
            linestyle="-.",
            label="IACC max (±lag)",
        )

    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(loc="best")

    if output_basename is None:
        output_path = None
    else:
        output_basename = Path(output_basename)
        output_path = output_basename.with_name(f"{output_basename.stem}_diffusion.png").with_suffix(".png")

    finalize_and_show_or_save(
        figure=figure,
        output_path=output_path,
        show_interactive=show_interactive,
    )

    return results


# --------------------------------------------------------------------------------------
# CLI-friendly summary
# --------------------------------------------------------------------------------------


def summarise_diffusion_results_text(results: List[DiffusionChannelResult]) -> str:
    """
    Deterministic summary: report medians over time (robust).
    """
    lines: List[str] = []
    for r in results:
        ac = r.series.max_abs_autocorr
        ed = r.series.echo_density

        lines.append(f"[{r.channel_name}]")
        lines.append(f"  median_max_abs_autocorr={float(np.nanmedian(ac)):.3f}")
        lines.append(f"  median_echo_density={float(np.nanmedian(ed)):.3f}")

        if r.series.corr0 is not None and r.series.iacc_max is not None:
            lines.append(f"  median_corr0={float(np.nanmedian(r.series.corr0)):.3f}")
            lines.append(f"  median_iacc_max={float(np.nanmedian(r.series.iacc_max)):.3f}")

    return "\n".join(lines)

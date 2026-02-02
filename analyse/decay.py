# analyse/decay.py
"""
Offline decay analysis (Schroeder EDC + T20/T30-derived RT60).

This module is designed to be:
- deterministic
- inspectable/readable (not performance-first)
- reusable later for bandpassed RT60 analysis (rt60bands.py)

Core outputs:
- Schroeder Energy Decay Curve (EDC) in dB
- Linear regression fits over standard decay ranges:
    T20:  -5 .. -25 dB
    T30:  -5 .. -35 dB
  with RT60 derived from slope: RT60 = -60 / slope

Stereo handling matches analyse/ir.py:
- load_wav_file(... expected_channel_mode="mono_or_stereo", allow_mono_and_upmix_to_stereo=False)
- analyse L/R unless explicitly downmixed to mono via settings
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import (
    create_figure_and_axis,
    finalize_and_show_or_save,
    label_decibel_axis,
    label_time_axis_seconds,
)


# --------------------------------------------------------------------------------------
# Public data models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DecayAnalysisSettings:
    # Stereo handling:
    # If True and the input is stereo, downmix to mono (single channel "M") before analysis.
    use_mono_downmix_for_stereo: bool = False

    # Time-zero policy:
    # If True, analysis begins at the absolute peak sample in each analysed channel.
    trim_to_peak: bool = True

    # Optional: ignore this many seconds after the trim point (e.g., to exclude direct sound).
    ignore_leading_seconds: float = 0.0

    # EDC representation:
    edc_floor_db: float = -120.0
    edc_epsilon: float = 1e-20

    # Fit policy:
    fit_lower_limit_db: float = -80.0
    t20_range_db: Tuple[float, float] = (-5.0, -25.0)
    t30_range_db: Tuple[float, float] = (-5.0, -35.0)

    # EDT (Early Decay Time):
    # If enabled, fit 0..-10 dB and extrapolate to RT60 via slope.
    compute_edt: bool = False
    edt_range_db: Tuple[float, float] = (0.0, -10.0)

    # Optional smoothing:
    edc_smoothing_window_samples: int = 0
    
    


@dataclass(frozen=True)
class LinearDecayFit:
    name: str
    range_db: Tuple[float, float]
    start_time_seconds: float
    end_time_seconds: float
    slope_db_per_second: float
    intercept_db: float
    r_squared: float
    rt60_seconds: float


@dataclass(frozen=True)
class ChannelDecayAnalysis:
    channel_name: str
    sample_rate_hz: int

    analysis_start_sample_index: int
    time_seconds: np.ndarray
    edc_db: np.ndarray

    early_decay_10db_time_seconds: Optional[float]

    fits: Dict[str, LinearDecayFit]


@dataclass(frozen=True)
class DecayPlotSettings:
    show_fit_lines: bool = True
    secondary_channel_alpha: float = 0.7
    ylim_db: Tuple[float, float] = (-120.0, 5.0)


# --------------------------------------------------------------------------------------
# Core primitives (reusable later for rt60bands.py)
# --------------------------------------------------------------------------------------


def compute_schroeder_edc_db(
    samples: np.ndarray,
    sample_rate_hz: int,
    settings: DecayAnalysisSettings,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute Schroeder Energy Decay Curve (EDC).

    Returns:
      time_seconds: (N,)
      edc_db:       (N,) with 0 dB at the first sample of the analysed segment
      analysis_start_sample_index: offset into the original samples where analysis begins
    """
    if samples.ndim != 1:
        raise ValueError("compute_schroeder_edc_db expects a 1D mono array.")

    analysis_start_sample_index = 0

    analysed_samples = samples.astype(np.float64, copy=False)

    if settings.trim_to_peak:
        peak_index = int(np.argmax(np.abs(analysed_samples)))
        analysis_start_sample_index = peak_index
        analysed_samples = analysed_samples[peak_index:]

    if settings.ignore_leading_seconds > 0.0:
        ignore_count = int(round(settings.ignore_leading_seconds * float(sample_rate_hz)))
        ignore_count = max(0, min(ignore_count, len(analysed_samples)))
        analysis_start_sample_index += ignore_count
        analysed_samples = analysed_samples[ignore_count:]

    if analysed_samples.size < 4:
        raise ValueError("Not enough samples after trimming/ignoring to compute EDC.")

    # Energy and reverse cumulative sum (Schroeder integration).
    energy = analysed_samples * analysed_samples
    edc_linear = np.cumsum(energy[::-1])[::-1]

    # Avoid zeros for log10.
    edc_linear = np.maximum(edc_linear, float(settings.edc_epsilon))

    # Normalise to 0 dB at start.
    edc_linear /= edc_linear[0]
    edc_db = 10.0 * np.log10(edc_linear)

    # Optional smoothing in dB domain.
    if settings.edc_smoothing_window_samples and settings.edc_smoothing_window_samples > 1:
        window = int(settings.edc_smoothing_window_samples)
        kernel = np.ones(window, dtype=np.float64) / float(window)
        edc_db = np.convolve(edc_db, kernel, mode="same")

    # Apply floor (primarily for display/robust range handling).
    edc_db = np.maximum(edc_db, float(settings.edc_floor_db)).astype(np.float32)

    time_seconds = (np.arange(edc_db.size, dtype=np.float32) / float(sample_rate_hz)).astype(np.float32)
    return time_seconds, edc_db, analysis_start_sample_index


def _interpolated_crossing_time_seconds(
    time_seconds: np.ndarray,
    edc_db: np.ndarray,
    target_db: float,
) -> Optional[float]:
    """
    Return the first time where edc_db crosses <= target_db, linearly interpolated.
    """
    below = edc_db <= target_db
    if not np.any(below):
        return None

    idx = int(np.argmax(below))  # first True
    if idx == 0:
        return float(time_seconds[0])

    t0 = float(time_seconds[idx - 1])
    t1 = float(time_seconds[idx])
    y0 = float(edc_db[idx - 1])
    y1 = float(edc_db[idx])

    if y1 == y0:
        return t1

    frac = (target_db - y0) / (y1 - y0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return t0 + frac * (t1 - t0)


def fit_decay_slope_over_db_range(
    time_seconds: np.ndarray,
    edc_db: np.ndarray,
    range_db: Tuple[float, float],
    fit_lower_limit_db: float,
    fit_name: str,
) -> Optional[LinearDecayFit]:
    """
    Fit a straight line in dB over the portion of EDC between range_db[0] and range_db[1],
    clamping the lower bound to fit_lower_limit_db (to avoid fitting deep into noise).

    Returns LinearDecayFit or None if the range is not available / invalid.
    """
    high_db, low_db = float(range_db[0]), float(range_db[1])
    if low_db > high_db:
        raise ValueError("range_db should be (higher_db, lower_db), e.g. (-5, -25).")

    effective_low_db = max(low_db, float(fit_lower_limit_db))

    start_t = _interpolated_crossing_time_seconds(time_seconds, edc_db, high_db)
    end_t = _interpolated_crossing_time_seconds(time_seconds, edc_db, effective_low_db)

    if start_t is None or end_t is None or end_t <= start_t:
        return None

    mask = (time_seconds >= start_t) & (time_seconds <= end_t)
    if int(np.sum(mask)) < 8:
        return None

    t = time_seconds[mask].astype(np.float64)
    y = edc_db[mask].astype(np.float64)

    # Least squares fit: y = m t + b
    A = np.column_stack([t, np.ones_like(t)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # A decay should have negative slope.
    if slope >= 0.0:
        return None

    y_pred = slope * t + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    rt60_seconds = float(-60.0 / slope)

    return LinearDecayFit(
        name=fit_name,
        range_db=(high_db, low_db),
        start_time_seconds=float(start_t),
        end_time_seconds=float(end_t),
        slope_db_per_second=float(slope),
        intercept_db=float(intercept),
        r_squared=float(r_squared),
        rt60_seconds=rt60_seconds,
    )


# --------------------------------------------------------------------------------------
# Analysis entrypoints
# --------------------------------------------------------------------------------------


def analyse_decay_for_channel(
    samples: np.ndarray,
    sample_rate_hz: int,
    channel_name: str,
    settings: DecayAnalysisSettings,
) -> ChannelDecayAnalysis:
    time_seconds, edc_db, start_sample_index = compute_schroeder_edc_db(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        settings=settings,
    )

    t_0 = _interpolated_crossing_time_seconds(time_seconds, edc_db, 0.0)
    t_minus_10 = _interpolated_crossing_time_seconds(time_seconds, edc_db, -10.0)

    if t_0 is not None and t_minus_10 is not None and t_minus_10 >= t_0:
        early_decay_10db_time_seconds: Optional[float] = float(t_minus_10 - t_0)
    else:
        early_decay_10db_time_seconds = None

    fits: Dict[str, LinearDecayFit] = {}

    if settings.compute_edt:
        edt_fit = fit_decay_slope_over_db_range(
            time_seconds=time_seconds,
            edc_db=edc_db,
            range_db=settings.edt_range_db,
            fit_lower_limit_db=settings.fit_lower_limit_db,
            fit_name="EDT",
        )
        if edt_fit is not None:
            fits["EDT"] = edt_fit

    t20_fit = fit_decay_slope_over_db_range(
        time_seconds=time_seconds,
        edc_db=edc_db,
        range_db=settings.t20_range_db,
        fit_lower_limit_db=settings.fit_lower_limit_db,
        fit_name="T20",
    )
    if t20_fit is not None:
        fits["T20"] = t20_fit

    t30_fit = fit_decay_slope_over_db_range(
        time_seconds=time_seconds,
        edc_db=edc_db,
        range_db=settings.t30_range_db,
        fit_lower_limit_db=settings.fit_lower_limit_db,
        fit_name="T30",
    )
    if t30_fit is not None:
        fits["T30"] = t30_fit

    return ChannelDecayAnalysis(
        channel_name=channel_name,
        sample_rate_hz=sample_rate_hz,
        analysis_start_sample_index=start_sample_index,
        time_seconds=time_seconds,
        edc_db=edc_db,
        early_decay_10db_time_seconds=early_decay_10db_time_seconds,
        fits=fits,
    )



def analyse_decay_from_wav_file(
    input_wav_file_path: str | Path,
    settings: Optional[DecayAnalysisSettings] = None,
) -> List[ChannelDecayAnalysis]:
    """
    Analysis-only: load WAV and compute EDC + fits per channel.
    No plotting side-effects; suitable for reuse by future analyses (e.g. rt60bands).
    """
    if settings is None:
        settings = DecayAnalysisSettings()

    loaded_audio = load_wav_file(
        wav_file_path=input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    channel_list = get_analysis_channels(
        loaded_audio=loaded_audio,
        use_mono_downmix_for_stereo=settings.use_mono_downmix_for_stereo,
    )

    results: List[ChannelDecayAnalysis] = []
    for channel_name, channel_samples in channel_list:
        results.append(
            analyse_decay_for_channel(
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


def plot_decay_figure(
    channel_analyses: List[ChannelDecayAnalysis],
    analysis_settings: DecayAnalysisSettings,
    plot_settings: DecayPlotSettings,
    title: Optional[str] = None,
):
    figure, axis = create_figure_and_axis(title=title)

    label_time_axis_seconds(axis)
    label_decibel_axis(axis)
    axis.set_ylim(plot_settings.ylim_db[0], plot_settings.ylim_db[1])

    for channel_index, result in enumerate(channel_analyses):
        alpha = 1.0 if channel_index == 0 else float(plot_settings.secondary_channel_alpha)

        # EDC curve (no legend entry)
        axis.plot(
            result.time_seconds,
            result.edc_db,
            alpha=alpha,
            label=None,
        )

        if plot_settings.show_fit_lines:
            for fit_name in ("EDT", "T20", "T30"):
                if fit_name not in result.fits:
                    continue

                fit = result.fits[fit_name]

                t_line = np.array([fit.start_time_seconds, fit.end_time_seconds], dtype=np.float32)
                y_line = (fit.slope_db_per_second * t_line + fit.intercept_db).astype(np.float32)

                if fit.name == "EDT":
                    if result.early_decay_10db_time_seconds is not None:
                        fit_label = (
                            f"EDT {result.channel_name}  "
                            f"{fit.rt60_seconds:.2f}s  "
                            f"Δ10dB={result.early_decay_10db_time_seconds:.3f}s"
                        )
                    else:
                        fit_label = f"EDT {result.channel_name}  {fit.rt60_seconds:.2f}s  Δ10dB=NA"
                else:
                    fit_label = f"{fit.name} {result.channel_name}  {fit.rt60_seconds:.2f}s"

                axis.plot(
                    t_line,
                    y_line,
                    alpha=alpha,
                    linestyle="--",
                    label=fit_label,
                )

    # Visual guides
    axis.axhline(float(analysis_settings.t20_range_db[0]), linestyle=":", linewidth=1.0)
    axis.axhline(float(analysis_settings.t20_range_db[1]), linestyle=":", linewidth=1.0)
    axis.axhline(float(analysis_settings.t30_range_db[1]), linestyle=":", linewidth=1.0)
    axis.axhline(float(analysis_settings.fit_lower_limit_db), linestyle=":", linewidth=1.0)

    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(loc="best")

    return figure



    # Visual guides for common ranges (helps interpret fits).
    axis.axhline(float(analysis_settings.t20_range_db[0]), linestyle=":", linewidth=1.0)
    axis.axhline(float(analysis_settings.t20_range_db[1]), linestyle=":", linewidth=1.0)
    axis.axhline(float(analysis_settings.t30_range_db[1]), linestyle=":", linewidth=1.0)
    axis.axhline(float(analysis_settings.fit_lower_limit_db), linestyle=":", linewidth=1.0)

    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(loc="best")

    return figure


def plot_decay_from_wav_file(
    input_wav_file_path: str | Path,
    analysis_settings: Optional[DecayAnalysisSettings] = None,
    plot_settings: Optional[DecayPlotSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[ChannelDecayAnalysis]:
    """
    Convenience wrapper: analyse decay then plot EDC with fit overlays.

    If output_basename is provided, one PNG will be written:
      - <basename>_decay.png
    """
    if analysis_settings is None:
        analysis_settings = DecayAnalysisSettings()
    if plot_settings is None:
        plot_settings = DecayPlotSettings()

    results = analyse_decay_from_wav_file(
        input_wav_file_path=input_wav_file_path,
        settings=analysis_settings,
    )

    title = f"Decay (EDC) — {input_wav_file_path}"
    figure = plot_decay_figure(
        channel_analyses=results,
        analysis_settings=analysis_settings,
        plot_settings=plot_settings,
        title=title,
    )

    if output_basename is None:
        output_path = None
    else:
        output_basename = Path(output_basename)
        output_path = output_basename.with_name(f"{output_basename.stem}_decay.png").with_suffix(".png")

    finalize_and_show_or_save(
        figure=figure,
        output_path=output_path,
        show_interactive=show_interactive,
    )

    return results


# --------------------------------------------------------------------------------------
# CLI-friendly numeric summaries
# --------------------------------------------------------------------------------------


def summarise_decay_results_text(channel_analyses: List[ChannelDecayAnalysis]) -> str:
    """
    Format a deterministic multiline summary of decay metrics.

    This is intentionally plain text and stable across runs so it can be diffed.
    """
    lines: List[str] = []

    for result in channel_analyses:
        lines.append(f"[{result.channel_name}] analysis_start_sample_index={result.analysis_start_sample_index}")

        if result.early_decay_10db_time_seconds is None:
            lines.append("  early_0_to_-10_time=NA")
        else:
            lines.append(f"  early_0_to_-10_time={result.early_decay_10db_time_seconds:.4f}s")

        if not result.fits:
            lines.append("  fits=NA")
            lines.append("")
            continue

        # Print fits in a fixed order.
        for fit_name in ("EDT", "T20", "T30"):
            fit = result.fits.get(fit_name)
            if fit is None:
                lines.append(f"  {fit_name}: NA")
                continue

            lines.append(
                "  "
                f"{fit.name}: "
                f"range=[{fit.range_db[0]:.1f},{fit.range_db[1]:.1f}]dB "
                f"time=[{fit.start_time_seconds:.4f},{fit.end_time_seconds:.4f}]s "
                f"slope={fit.slope_db_per_second:.6f}dB/s "
                f"r2={fit.r_squared:.6f} "
                f"rt60={fit.rt60_seconds:.4f}s"
            )

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"

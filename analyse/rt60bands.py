# analyse/rt60bands.py
"""
Band-limited RT60 analysis using FFT-domain filtering.

Design:
- A single plot API that works for:
    * 3-band summary (Low/Mid/High)  [default]
    * octave bands
    * 1/3-octave bands
- Filtering is done using rFFT magnitude masks with raised-cosine transitions
  (deterministic, no SciPy).

Default behaviour:
- band_mode="three"
- compute RT60 from T30 in each band
- optional include_t20 / include_edt

Plot behaviour:
- If number of bands <= 6 -> grouped bar chart (categorical)
- Else -> line plot vs band centre frequency (log-x)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import create_figure_and_axis, finalize_and_show_or_save
from analyse.decay import (
    DecayAnalysisSettings,
    compute_schroeder_edc_db,
    fit_decay_slope_over_db_range,
)

# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rt60BandsAnalysisSettings:
    # "three" -> Low/Mid/High summary
    # "octave" -> 1 band per octave
    # "third" -> 3 bands per octave
    band_mode: str = "three"  # "three" | "octave" | "third"

    # Used when band_mode == "three"
    low_upper_hz: float = 250.0
    mid_center_hz: float = 1000.0
    mid_width_octaves: float = 2.0
    high_lower_hz: float = 4000.0

    # Used when band_mode in {"octave","third"}
    f_min_hz: float = 31.5
    f_max_hz: float = 16000.0

    # Smooth transition width for FFT masks (in octaves).
    transition_width_octaves: float = (1.0 / 6.0)

    # Optional extra metrics
    include_t20: bool = False
    include_edt: bool = False

    # Reuse decay analysis policy from decay.py
    decay_settings: DecayAnalysisSettings = DecayAnalysisSettings()


@dataclass(frozen=True)
class Rt60BandsPlotSettings:
    ylim_seconds: Optional[Tuple[float, float]] = None
    secondary_channel_alpha: float = 0.7

    # If True, legend entries include per-band numeric values.
    # Recommended: True for 3-band mode, False for dense octave/third.
    legend_values: bool = True



@dataclass(frozen=True)
class BandDefinition:
    name: str                 # e.g. "Low", "Mid", "High", or "1000Hz"
    centre_hz: float          # representative / centre frequency for plotting
    kind: str                 # "lowpass" | "bandpass" | "highpass"
    low_edge_hz: Optional[float] = None
    high_edge_hz: Optional[float] = None


@dataclass(frozen=True)
class Rt60BandMetrics:
    rt60_t30_seconds: Optional[float]
    rt60_t20_seconds: Optional[float]
    edt_seconds: Optional[float]


@dataclass(frozen=True)
class Rt60BandsChannelResult:
    channel_name: str
    sample_rate_hz: int
    band_definitions: List[BandDefinition]              # ordered
    band_metrics_by_name: Dict[str, Rt60BandMetrics]    # keyed by band.name


# --------------------------------------------------------------------------------------
# FFT mask helpers
# --------------------------------------------------------------------------------------


def _octave_factor(octaves: float) -> float:
    return float(2.0 ** float(octaves))


def _raised_cosine_ramp(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """
    Smooth ramp from 0 at x<=x0 to 1 at x>=x1 using half-cosine.
    """
    if x1 <= x0:
        return (x >= x1).astype(np.float32)

    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def _make_lowpass_mask(freqs_hz: np.ndarray, pass_hz: float, transition_oct: float, nyquist_hz: float) -> np.ndarray:
    pass_hz = float(np.clip(pass_hz, 1.0, nyquist_hz))
    stop_hz = float(min(nyquist_hz, pass_hz * _octave_factor(transition_oct)))
    if stop_hz <= pass_hz:
        stop_hz = min(nyquist_hz, pass_hz + 1.0)

    ramp = _raised_cosine_ramp(freqs_hz, pass_hz, stop_hz)  # 0->1
    mask = 1.0 - ramp
    mask[freqs_hz <= pass_hz] = 1.0
    mask[freqs_hz >= stop_hz] = 0.0
    return mask.astype(np.float32)


def _make_highpass_mask(freqs_hz: np.ndarray, pass_hz: float, transition_oct: float, nyquist_hz: float) -> np.ndarray:
    pass_hz = float(np.clip(pass_hz, 1.0, nyquist_hz))
    stop_hz = float(max(1.0, pass_hz / _octave_factor(transition_oct)))
    if pass_hz <= stop_hz:
        stop_hz = max(1.0, pass_hz - 1.0)

    ramp = _raised_cosine_ramp(freqs_hz, stop_hz, pass_hz)
    mask = ramp
    mask[freqs_hz <= stop_hz] = 0.0
    mask[freqs_hz >= pass_hz] = 1.0
    return mask.astype(np.float32)


def _make_bandpass_mask(
    freqs_hz: np.ndarray,
    low_edge_hz: float,
    high_edge_hz: float,
    transition_oct: float,
    nyquist_hz: float,
) -> np.ndarray:
    low_edge_hz = float(np.clip(low_edge_hz, 1.0, nyquist_hz))
    high_edge_hz = float(np.clip(high_edge_hz, 1.0, nyquist_hz))
    if high_edge_hz <= low_edge_hz:
        return np.zeros_like(freqs_hz, dtype=np.float32)

    hp = _make_highpass_mask(freqs_hz, pass_hz=low_edge_hz, transition_oct=transition_oct, nyquist_hz=nyquist_hz)
    lp = _make_lowpass_mask(freqs_hz, pass_hz=high_edge_hz, transition_oct=transition_oct, nyquist_hz=nyquist_hz)
    return (hp * lp).astype(np.float32)


def _apply_fft_mask(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = int(samples.size)
    spectrum = np.fft.rfft(samples.astype(np.float64, copy=False))
    masked = spectrum * mask.astype(np.float64, copy=False)
    out = np.fft.irfft(masked, n=n)
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Band definition generation
# --------------------------------------------------------------------------------------


def _build_three_band_definitions(settings: Rt60BandsAnalysisSettings, sample_rate_hz: int) -> List[BandDefinition]:
    nyquist = 0.5 * float(sample_rate_hz)

    low_upper = float(np.clip(settings.low_upper_hz, 20.0, nyquist))
    mid_center = float(np.clip(settings.mid_center_hz, 20.0, nyquist))
    mid_width = float(max(0.1, settings.mid_width_octaves))
    high_lower = float(np.clip(settings.high_lower_hz, 20.0, nyquist))

    half = 0.5 * mid_width
    mid_low = mid_center / _octave_factor(half)
    mid_high = mid_center * _octave_factor(half)

    mid_low = float(np.clip(mid_low, 20.0, nyquist))
    mid_high = float(np.clip(mid_high, 20.0, nyquist))

    low_centre = float(np.sqrt(20.0 * low_upper))
    high_centre = float(np.sqrt(max(20.0, high_lower) * nyquist))

    return [
        BandDefinition(name="Low", centre_hz=low_centre, kind="lowpass", high_edge_hz=low_upper),
        BandDefinition(name="Mid", centre_hz=mid_center, kind="bandpass", low_edge_hz=mid_low, high_edge_hz=mid_high),
        BandDefinition(name="High", centre_hz=high_centre, kind="highpass", low_edge_hz=high_lower),
    ]


def _build_fractional_octave_band_definitions(
    settings: Rt60BandsAnalysisSettings,
    sample_rate_hz: int,
    bands_per_octave: int,
) -> List[BandDefinition]:
    nyquist = 0.5 * float(sample_rate_hz)
    f_min = float(max(20.0, min(settings.f_min_hz, nyquist)))
    f_max = float(max(f_min, min(settings.f_max_hz, nyquist)))

    n = float(bands_per_octave)
    step = 2.0 ** (1.0 / n)           # centre spacing
    half_band = 2.0 ** (1.0 / (2.0*n))  # edge ratio

    anchor = 1000.0  # deterministic, familiar reference

    # fc = anchor * step^k. Solve for k range that covers [f_min, f_max].
    k_min = int(np.floor(np.log(f_min / anchor) / np.log(step)))
    k_max = int(np.ceil(np.log(f_max / anchor) / np.log(step)))

    bands: List[BandDefinition] = []
    for k in range(k_min, k_max + 1):
        fc = anchor * (step ** float(k))
        if fc < f_min or fc > f_max:
            continue

        low = fc / half_band
        high = fc * half_band

        low = float(np.clip(low, 20.0, nyquist))
        high = float(np.clip(high, 20.0, nyquist))
        if high <= low:
            continue

        name = f"{int(round(fc))}Hz"
        bands.append(
            BandDefinition(
                name=name,
                centre_hz=float(fc),
                kind="bandpass",
                low_edge_hz=low,
                high_edge_hz=high,
            )
        )

    bands.sort(key=lambda b: b.centre_hz)
    return bands


def _build_band_definitions(settings: Rt60BandsAnalysisSettings, sample_rate_hz: int) -> List[BandDefinition]:
    mode = str(settings.band_mode).lower()
    if mode == "three":
        return _build_three_band_definitions(settings, sample_rate_hz)
    if mode == "octave":
        return _build_fractional_octave_band_definitions(settings, sample_rate_hz, bands_per_octave=1)
    if mode == "third":
        return _build_fractional_octave_band_definitions(settings, sample_rate_hz, bands_per_octave=3)
    raise ValueError(f"Unknown band_mode: {settings.band_mode}")


# --------------------------------------------------------------------------------------
# Metrics per band
# --------------------------------------------------------------------------------------


def _compute_band_metrics_from_samples(
    bandpassed_samples: np.ndarray,
    sample_rate_hz: int,
    decay_settings: DecayAnalysisSettings,
    include_t20: bool,
    include_edt: bool,
) -> Rt60BandMetrics:
    time_s, edc_db, _ = compute_schroeder_edc_db(
        samples=bandpassed_samples,
        sample_rate_hz=sample_rate_hz,
        settings=decay_settings,
    )

    # Primary metric: T30-derived RT60
    t30_fit = fit_decay_slope_over_db_range(
        time_seconds=time_s,
        edc_db=edc_db,
        range_db=decay_settings.t30_range_db,
        fit_lower_limit_db=decay_settings.fit_lower_limit_db,
        fit_name="T30",
    )
    rt60_t30 = t30_fit.rt60_seconds if t30_fit is not None else None

    rt60_t20: Optional[float] = None
    if include_t20:
        t20_fit = fit_decay_slope_over_db_range(
            time_seconds=time_s,
            edc_db=edc_db,
            range_db=decay_settings.t20_range_db,
            fit_lower_limit_db=decay_settings.fit_lower_limit_db,
            fit_name="T20",
        )
        rt60_t20 = t20_fit.rt60_seconds if t20_fit is not None else None

    edt_seconds: Optional[float] = None
    if include_edt:
        edt_fit = fit_decay_slope_over_db_range(
            time_seconds=time_s,
            edc_db=edc_db,
            range_db=decay_settings.edt_range_db,
            fit_lower_limit_db=decay_settings.fit_lower_limit_db,
            fit_name="EDT",
        )
        edt_seconds = edt_fit.rt60_seconds if edt_fit is not None else None

    return Rt60BandMetrics(
        rt60_t30_seconds=rt60_t30,
        rt60_t20_seconds=rt60_t20,
        edt_seconds=edt_seconds,
    )


def analyse_rt60_bands_for_channel(
    samples: np.ndarray,
    sample_rate_hz: int,
    channel_name: str,
    settings: Rt60BandsAnalysisSettings,
) -> Rt60BandsChannelResult:
    full_samples = samples.astype(np.float32, copy=False)

    # Determine a consistent time-zero from the *full-band* signal
    peak_index = 0
    if settings.decay_settings.trim_to_peak:
        peak_index = int(np.argmax(np.abs(full_samples)))

    ignore_count = 0
    if settings.decay_settings.ignore_leading_seconds > 0.0:
        ignore_count = int(round(settings.decay_settings.ignore_leading_seconds * float(sample_rate_hz)))
        ignore_count = max(0, min(ignore_count, full_samples.size))

    start_index = min(full_samples.size, peak_index + ignore_count)

    # FFT will be computed on the full signal length (avoid pre-trim artifacts)
    n = int(full_samples.size)
    if n < 8:
        raise ValueError("Not enough samples for rt60bands analysis.")

    nyquist = 0.5 * float(sample_rate_hz)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz)).astype(np.float32)

    band_defs = _build_band_definitions(settings=settings, sample_rate_hz=sample_rate_hz)
    band_metrics: Dict[str, Rt60BandMetrics] = {}

    # Create decay settings with trimming disabled because we apply a fixed trim ourselves
    band_decay_settings = replace(
        settings.decay_settings,
        trim_to_peak=False,
        ignore_leading_seconds=0.0,
    )

    for band in band_defs:
        if band.kind == "lowpass":
            assert band.high_edge_hz is not None
            mask = _make_lowpass_mask(
                freqs_hz=freqs,
                pass_hz=band.high_edge_hz,
                transition_oct=settings.transition_width_octaves,
                nyquist_hz=nyquist,
            )
        elif band.kind == "highpass":
            assert band.low_edge_hz is not None
            mask = _make_highpass_mask(
                freqs_hz=freqs,
                pass_hz=band.low_edge_hz,
                transition_oct=settings.transition_width_octaves,
                nyquist_hz=nyquist,
            )
        elif band.kind == "bandpass":
            assert band.low_edge_hz is not None and band.high_edge_hz is not None
            mask = _make_bandpass_mask(
                freqs_hz=freqs,
                low_edge_hz=band.low_edge_hz,
                high_edge_hz=band.high_edge_hz,
                transition_oct=settings.transition_width_octaves,
                nyquist_hz=nyquist,
            )
        else:
            raise ValueError(f"Unknown band kind: {band.kind}")

        # Filter full signal, then apply consistent trim
        band_full = _apply_fft_mask(samples=full_samples, mask=mask)
        band_trimmed = band_full[start_index:]

        if band_trimmed.size < 8:
            band_metrics[band.name] = Rt60BandMetrics(None, None, None)
            continue

        metrics = _compute_band_metrics_from_samples(
            bandpassed_samples=band_trimmed,
            sample_rate_hz=sample_rate_hz,
            decay_settings=band_decay_settings,
            include_t20=settings.include_t20,
            include_edt=settings.include_edt,
        )
        band_metrics[band.name] = metrics

    return Rt60BandsChannelResult(
        channel_name=channel_name,
        sample_rate_hz=sample_rate_hz,
        band_definitions=band_defs,
        band_metrics_by_name=band_metrics,
    )



def analyse_rt60_bands_from_wav_file(
    input_wav_file_path: str | Path,
    settings: Optional[Rt60BandsAnalysisSettings] = None,
) -> List[Rt60BandsChannelResult]:
    if settings is None:
        settings = Rt60BandsAnalysisSettings()

    loaded_audio = load_wav_file(
        wav_file_path=input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )

    channel_list = get_analysis_channels(
        loaded_audio=loaded_audio,
        use_mono_downmix_for_stereo=settings.decay_settings.use_mono_downmix_for_stereo,
    )

    results: List[Rt60BandsChannelResult] = []
    for channel_name, channel_samples in channel_list:
        results.append(
            analyse_rt60_bands_for_channel(
                samples=channel_samples,
                sample_rate_hz=loaded_audio.sample_rate_hz,
                channel_name=channel_name,
                settings=settings,
            )
        )
    return results


# --------------------------------------------------------------------------------------
# Plotting + summary
# --------------------------------------------------------------------------------------


def plot_rt60_bands_figure(
    channel_results: List[Rt60BandsChannelResult],
    settings: Rt60BandsAnalysisSettings,
    plot_settings: Rt60BandsPlotSettings,
    title: Optional[str] = None,
):
    """
    One plot API that adapts based on band count:
    - <= 6 bands: grouped bar chart (categorical)
    - > 6 bands: line plot vs centre frequency (log-x)
    Legend includes the numeric results for readability and screenshot-value.
    """
    if len(channel_results) == 0:
        raise ValueError("No channel results to plot.")

    # Assume all channels share the same band_definitions (constructed from settings)
    bands = channel_results[0].band_definitions
    band_names = [b.name for b in bands]
    centres_hz = np.array([b.centre_hz for b in bands], dtype=np.float32)

    metrics = ["T30"]
    if settings.include_t20:
        metrics.append("T20")
    if settings.include_edt:
        metrics.append("EDT")

    def metric_value(m: Rt60BandMetrics, metric: str) -> Optional[float]:
        if metric == "T30":
            return m.rt60_t30_seconds
        if metric == "T20":
            return m.rt60_t20_seconds
        if metric == "EDT":
            return m.edt_seconds
        raise ValueError(metric)

    figure, axis = create_figure_and_axis(title=title)

    # Decide plot mode
    use_bar = len(bands) <= 6

    if use_bar:
        axis.set_xlabel("Band")
        axis.set_ylabel("RT60 (seconds)")

        x = np.arange(len(bands), dtype=np.float32)
        axis.set_xticks(x)
        axis.set_xticklabels(band_names)

        n_metrics = len(metrics)
        n_channels = len(channel_results)
        total_groups = n_metrics * n_channels
        bar_width = 0.8 / max(1, total_groups)

        offset_index = 0

        for channel_index, channel in enumerate(channel_results):
            alpha = 1.0 if channel_index == 0 else float(plot_settings.secondary_channel_alpha)

            for metric in metrics:
                values: List[float] = []
                label_parts: List[str] = []

                for band in band_names:
                    bm = channel.band_metrics_by_name.get(band)
                    v = None if bm is None else metric_value(bm, metric)
                    values.append(np.nan if v is None else float(v))
                    label_parts.append(f"{band}={'NA' if v is None else f'{v:.2f}s'}")

                if plot_settings.legend_values:
                    label = f"{metric} {channel.channel_name}  " + "  ".join(label_parts)
                else:
                    label = f"{metric} {channel.channel_name}"

                axis.bar(
                    x + (offset_index - total_groups / 2) * bar_width + bar_width / 2,
                    values,
                    width=bar_width,
                    alpha=alpha,
                    label=label,
                )
                offset_index += 1

        axis.grid(True, axis="y", linestyle=":", linewidth=0.5)

    else:
        axis.set_xlabel("Band centre frequency (Hz)")
        axis.set_ylabel("RT60 (seconds)")
        axis.set_xscale("log")
        axis.grid(True, which="both", linestyle=":", linewidth=0.5)

        metric_linestyle = {"T30": "-", "T20": "--", "EDT": ":"}

        for channel_index, channel in enumerate(channel_results):
            alpha = 1.0 if channel_index == 0 else float(plot_settings.secondary_channel_alpha)

            for metric in metrics:
                y = []
                label_parts: List[str] = []

                for band in band_names:
                    bm = channel.band_metrics_by_name.get(band)
                    v = None if bm is None else metric_value(bm, metric)
                    y.append(np.nan if v is None else float(v))
                    label_parts.append(f"{band}={'NA' if v is None else f'{v:.2f}s'}")

                y_arr = np.array(y, dtype=np.float32)
                if plot_settings.legend_values:
                    label = f"{metric} {channel.channel_name}  " + "  ".join(label_parts)
                else:
                    label = f"{metric} {channel.channel_name}"

                axis.plot(
                    centres_hz,
                    y_arr,
                    linestyle=metric_linestyle[metric],
                    marker="o",
                    alpha=alpha,
                    label=label,
                )

    if plot_settings.ylim_seconds is not None:
        axis.set_ylim(plot_settings.ylim_seconds[0], plot_settings.ylim_seconds[1])

    axis.legend(loc="best")
    return figure


def plot_rt60_bands_from_wav_file(
    input_wav_file_path: str | Path,
    settings: Optional[Rt60BandsAnalysisSettings] = None,
    plot_settings: Optional[Rt60BandsPlotSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[Rt60BandsChannelResult]:
    if settings is None:
        settings = Rt60BandsAnalysisSettings()
    if plot_settings is None:
        plot_settings = Rt60BandsPlotSettings()
    # Default: show numeric legend values for 3-band, suppress for dense modes.
    if plot_settings.legend_values and str(settings.band_mode).lower() in ("octave", "third"):
        plot_settings = Rt60BandsPlotSettings(
            ylim_seconds=plot_settings.ylim_seconds,
            secondary_channel_alpha=plot_settings.secondary_channel_alpha,
            legend_values=False,
        )

    results = analyse_rt60_bands_from_wav_file(
        input_wav_file_path=input_wav_file_path,
        settings=settings,
    )

    title = f"RT60 bands — {input_wav_file_path}"
    figure = plot_rt60_bands_figure(
        channel_results=results,
        settings=settings,
        plot_settings=plot_settings,
        title=title,
    )

    if output_basename is None:
        output_path = None
    else:
        output_basename = Path(output_basename)
        output_path = output_basename.with_name(f"{output_basename.stem}_rt60bands.png").with_suffix(".png")

    finalize_and_show_or_save(
        figure=figure,
        output_path=output_path,
        show_interactive=show_interactive,
    )

    return results


def summarise_rt60_bands_results_text(
    channel_results: List[Rt60BandsChannelResult],
    include_t20: bool,
    include_edt: bool,
) -> str:
    lines: List[str] = []
    metrics = ["T30"]
    if include_t20:
        metrics.append("T20")
    if include_edt:
        metrics.append("EDT")

    for channel in channel_results:
        lines.append(f"[{channel.channel_name}]")
        header = ["Band"] + [f"{m}_RT60(s)" for m in metrics]
        lines.append("  ".join(header))

        for band in channel.band_definitions:
            bm = channel.band_metrics_by_name.get(band.name)
            row = [band.name]
            for m in metrics:
                if bm is None:
                    row.append("NA")
                    continue
                if m == "T30":
                    v = bm.rt60_t30_seconds
                elif m == "T20":
                    v = bm.rt60_t20_seconds
                elif m == "EDT":
                    v = bm.edt_seconds
                else:
                    raise ValueError(m)

                row.append("NA" if v is None else f"{float(v):.3f}")

            lines.append("  ".join(row))

        lines.append("")

    return "\n".join(lines)

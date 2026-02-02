# analyse/group_delay.py
"""
Group delay analysis (from an impulse response / filter output).

What it shows:
- Group delay vs frequency derived from the unwrapped phase of the FFT.

Notes / interpretation:
- For minimum-phase filters, group delay tends to be "small and smooth".
- For allpass / dispersive structures, group delay reveals where energy is delayed.
- For comb-like structures, group delay can oscillate rapidly; smoothing can help.

Computation:
- Compute H(e^jw) from an analysed segment (typically starting at the IR peak).
- Phase unwrap (optional)
- Group delay in samples:  gd(w) = - d(phi) / d(w)
  where w is in radians/sample.

Outputs:
- One plot per analysed channel (L/R or M).
- If output_basename provided:
    <basename>_groupdelay_<CH>.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.ticker as mticker

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import create_figure_and_axis, finalize_and_show_or_save


@dataclass(frozen=True)
class GroupDelayAnalysisSettings:
    # Stereo handling
    use_mono_downmix_for_stereo: bool = False

    # Time selection
    trim_to_peak: bool = True
    ignore_leading_seconds: float = 0.0
    analysis_duration_seconds: Optional[float] = None

    # FFT/windowing
    use_hann_window: bool = True
    fft_size: Optional[int] = None  # if None, next pow2 >= segment length (capped)

    # Robustness/display
    f_min_hz: float = 20.0
    f_max_hz: float = 20000.0
    unwrap_phase: bool = True

    # Optional smoothing in frequency domain (moving average over bins)
    smoothing_bins: int = 0  # 0 disables


@dataclass(frozen=True)
class GroupDelayPlotSettings:
    secondary_channel_alpha: float = 0.7
    ylim_samples: Optional[Tuple[float, float]] = None
    show_zero_line: bool = True


@dataclass(frozen=True)
class ChannelGroupDelayResult:
    channel_name: str
    sample_rate_hz: int
    frequency_hz: np.ndarray        # shape (F,)
    group_delay_samples: np.ndarray # shape (F,)


def _next_pow2(n: int) -> int:
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def _compute_group_delay_from_ir(
    samples: np.ndarray,
    sample_rate_hz: int,
    settings: GroupDelayAnalysisSettings,
) -> ChannelGroupDelayResult:
    assert samples.ndim == 1

    # Window
    segment = samples.astype(np.float64, copy=False)
    if settings.use_hann_window:
        segment = segment * np.hanning(len(segment))

    # FFT size
    if settings.fft_size is None:
        n_fft = _next_pow2(len(segment))
        # cap to keep runtime sane for very long tails
        n_fft = min(n_fft, 1 << 20)  # 1,048,576
    else:
        n_fft = int(settings.fft_size)

    H = np.fft.rfft(segment, n=n_fft)
    # Frequency axis
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate_hz))
    # Phase
    phase = np.angle(H)
    if settings.unwrap_phase:
        phase = np.unwrap(phase)

    # w axis in rad/sample
    w = 2.0 * np.pi * (freq_hz / float(sample_rate_hz))

    # group delay = -dphi/dw
    dphi_dw = np.gradient(phase, w)
    gd = -dphi_dw

    if settings.smoothing_bins and settings.smoothing_bins > 1:
        gd = _moving_average(gd, int(settings.smoothing_bins))

    # Apply frequency range mask
    mask = (freq_hz >= float(settings.f_min_hz)) & (freq_hz <= float(settings.f_max_hz))
    freq_hz = freq_hz[mask]
    gd = gd[mask]

    return ChannelGroupDelayResult(
        channel_name="",
        sample_rate_hz=sample_rate_hz,
        frequency_hz=freq_hz.astype(np.float64, copy=False),
        group_delay_samples=gd.astype(np.float64, copy=False),
    )


def plot_group_delay_from_wav_file(
    input_wav_file_path: str,
    settings: GroupDelayAnalysisSettings,
    plot_settings: GroupDelayPlotSettings,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[ChannelGroupDelayResult]:
    loaded = load_wav_file(
        input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )
    channels = get_analysis_channels(loaded, use_mono_downmix_for_stereo=settings.use_mono_downmix_for_stereo)

    results: List[ChannelGroupDelayResult] = []

    for channel_name, channel_samples in channels:
        samples = channel_samples

        # Time selection consistent with other modules:
        start_index = 0
        if settings.trim_to_peak:
            start_index = int(np.argmax(np.abs(samples)))

        start_index += int(round(float(settings.ignore_leading_seconds) * loaded.sample_rate_hz))
        start_index = max(0, min(start_index, len(samples)))

        if settings.analysis_duration_seconds is None:
            segment = samples[start_index:]
        else:
            length = int(round(float(settings.analysis_duration_seconds) * loaded.sample_rate_hz))
            segment = samples[start_index : start_index + max(1, length)]

        result = _compute_group_delay_from_ir(segment, loaded.sample_rate_hz, settings)
        result = ChannelGroupDelayResult(
            channel_name=channel_name,
            sample_rate_hz=result.sample_rate_hz,
            frequency_hz=result.frequency_hz,
            group_delay_samples=result.group_delay_samples,
        )
        results.append(result)

        title = f"Group delay ({channel_name})"
        fig, ax = create_figure_and_axis(title=title)

        ax.plot(result.frequency_hz, result.group_delay_samples, alpha=plot_settings.secondary_channel_alpha if channel_name != "L" else 1.0)

        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Group delay (samples)")

        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        if plot_settings.show_zero_line:
            ax.axhline(0.0, linestyle="--", linewidth=1.0)

        if plot_settings.ylim_samples is not None:
            ax.set_ylim(plot_settings.ylim_samples[0], plot_settings.ylim_samples[1])

        if output_basename is not None:
            output_path = str(Path(output_basename).with_suffix("")) + f"_groupdelay_{channel_name}.png"
        else:
            output_path = None

        finalize_and_show_or_save(fig, output_path=output_path, show_interactive=show_interactive)

    return results


def summarise_group_delay_results_text(results: List[ChannelGroupDelayResult]) -> str:
    lines: List[str] = []
    for r in results:
        gd = r.group_delay_samples
        if gd.size == 0:
            continue
        lines.append(f"- {r.channel_name}: gd median={float(np.median(gd)):.3f} samples, "
                     f"p10={float(np.percentile(gd, 10)):.3f}, p90={float(np.percentile(gd, 90)):.3f}")
    if not lines:
        return "No group delay results."
    return "Group delay summary:\n" + "\n".join(lines)

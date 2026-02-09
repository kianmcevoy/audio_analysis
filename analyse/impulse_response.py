# analyse/ir.py
"""
Impulse-response style visualisation for reverb analysis.

This module assumes the input WAV is already an impulse response, OR is a response
to an impulse/click/noise burst where IR-like inspection is still useful.

Plots:
1) Waveform (linear time) with an early zoom (default first 80 ms)
2) Log-magnitude over time (dB), useful for tail inspection

Stereo handling:
- plots left and right separately (two lines)
- also supports plotting a mono downmix if requested
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from analyse.io import (
    LoadedAudio,
    load_wav_file,
    get_analysis_channels,
)

from analyse.plotting import (
    create_figure_and_axis,
    finalize_and_show_or_save,
    label_amplitude_axis,
    label_decibel_axis,
    label_time_axis_seconds,
    plot_log_magnitude_over_time,
    plot_time_series,
    time_axis_from_sample_count,
)


@dataclass(frozen=True)
class ImpulseResponseViewSettings:
    """
    Settings controlling the IR plots.
    """
    early_window_seconds: float = 0.08
    log_magnitude_floor_db: float = -120.0
    use_mono_downmix: bool = False
    trim_leading_silence_db: float = -60.0  # Trim silence below this level (dB below peak)


def compute_log_magnitude(samples: np.ndarray) -> np.ndarray:
    """
    Compute a magnitude-like envelope for log plotting.

    For readability and robustness, we use absolute value here.
    Later we can replace this with an RMS envelope or Hilbert envelope.
    """
    return np.abs(samples).astype(np.float32)


def trim_leading_silence(
    samples: np.ndarray,
    threshold_db: float = -60.0,
) -> np.ndarray:
    """
    Remove leading silence from audio samples.

    Args:
        samples: Audio samples (1D or 2D). If 2D (N, C), uses max magnitude across channels.
        threshold_db: Silence threshold relative to peak magnitude in dB.
                      Samples above this level are kept.

    Returns:
        Trimmed samples with leading silence removed.
    """
    if samples.size == 0:
        return samples

    # Compute magnitude for detection
    if samples.ndim == 1:
        magnitude = compute_log_magnitude(samples)
    else:
        # For 2D array (N, C), find peak across all channels
        magnitude = np.max(compute_log_magnitude(samples), axis=1)

    peak_magnitude = np.max(magnitude)
    if peak_magnitude <= 0:
        return samples  # All silence, return as-is

    # Convert to dB and find threshold
    magnitude_db = 20 * np.log10(peak_magnitude + 1e-12)
    threshold_absolute = 10 ** ((magnitude_db + threshold_db) / 20)

    # Find first sample above threshold
    above_threshold = np.where(magnitude > threshold_absolute)[0]
    if len(above_threshold) > 0:
        trim_index = above_threshold[0]
        return samples[trim_index:]

    return samples  # All silence, return as-is


def plot_impulse_response_waveform(
    loaded_audio: LoadedAudio,
    settings: ImpulseResponseViewSettings,
    output_path: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> None:
    """
    Plot waveform and early waveform zoom for the input audio.
    """

    # Trim leading silence
    trimmed_samples = trim_leading_silence(
        loaded_audio.samples,
        threshold_db=settings.trim_leading_silence_db,
    )

    total_samples = trimmed_samples.shape[0]
    sample_rate_hz = loaded_audio.sample_rate_hz
    full_time_axis_seconds = time_axis_from_sample_count(total_samples, sample_rate_hz)

    # ---- Full waveform
    full_figure, full_axis = create_figure_and_axis(
        title=f"Waveform (full) - {loaded_audio.file_path.name}"
    )

    # Extract channels from trimmed audio
    if trimmed_samples.ndim == 1:
        # Mono
        plot_channels = [("Mono", trimmed_samples, 1.0)]
    else:
        # Stereo or multichannel
        channel_names = ["Left", "Right"] if trimmed_samples.shape[1] >= 2 else [f"Channel {i}" for i in range(trimmed_samples.shape[1])]
        plot_channels = []
        for idx in range(trimmed_samples.shape[1]):
            alpha = 1.0 if idx == 0 else 0.5
            plot_channels.append((channel_names[idx], trimmed_samples[:, idx], alpha))

    for channel_name, channel_samples, channel_alpha in plot_channels:
        plot_time_series(
            axis=full_axis,
            time_seconds=full_time_axis_seconds,
            samples=channel_samples,
            label=channel_name,
            alpha=channel_alpha,
        )

    label_time_axis_seconds(full_axis)
    label_amplitude_axis(full_axis, unit="Amplitude")
    finalize_and_show_or_save(
        figure=full_figure, output_path=output_path, show_interactive=show_interactive
    )

    # ---- Early zoom waveform
    early_window_samples = int(round(settings.early_window_seconds * sample_rate_hz))
    early_window_samples = max(1, min(early_window_samples, total_samples))

    early_time_axis_seconds = full_time_axis_seconds[:early_window_samples]

    early_figure, early_axis = create_figure_and_axis(
        title=f"Waveform (early {settings.early_window_seconds*1000:.0f} ms) - {loaded_audio.file_path.name}"
    )
    for channel_name, channel_samples, channel_alpha in plot_channels:
        plot_time_series(
            axis=early_axis,
            time_seconds=early_time_axis_seconds,
            samples=channel_samples[:early_window_samples],
            label=channel_name,
            alpha=channel_alpha,
        )

    label_time_axis_seconds(early_axis)
    label_amplitude_axis(early_axis, unit="Amplitude")
    finalize_and_show_or_save(
        figure=early_figure,
        output_path=None if output_path is None else _suffix_output_path(output_path, "_early"),
        show_interactive=show_interactive,
    )


def plot_impulse_response_log_magnitude(
    loaded_audio: LoadedAudio,
    settings: ImpulseResponseViewSettings,
    output_path: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> None:
    """
    Plot log-magnitude (dB) over time for tail inspection.
    """

    # Trim leading silence
    trimmed_samples = trim_leading_silence(
        loaded_audio.samples,
        threshold_db=settings.trim_leading_silence_db,
    )

    # Extract channels from trimmed audio
    if trimmed_samples.ndim == 1:
        # Mono
        plot_channels = [("Mono", trimmed_samples, 1.0)]
    else:
        # Stereo or multichannel
        channel_names = ["Left", "Right"] if trimmed_samples.shape[1] >= 2 else [f"Channel {i}" for i in range(trimmed_samples.shape[1])]
        plot_channels = []
        for idx in range(trimmed_samples.shape[1]):
            alpha = 1.0 if idx == 0 else 0.5
            plot_channels.append((channel_names[idx], trimmed_samples[:, idx], alpha))

    total_samples = trimmed_samples.shape[0]
    sample_rate_hz = loaded_audio.sample_rate_hz
    time_axis_seconds = time_axis_from_sample_count(total_samples, sample_rate_hz)

    figure, axis = create_figure_and_axis(
        title=f"Log magnitude (tail) - {loaded_audio.file_path.name}"
    )

    for channel_name, channel_samples, channel_alpha in plot_channels:
        magnitude = compute_log_magnitude(channel_samples)
        # plot helper expects linear magnitude; it converts to dB internally
        plot_log_magnitude_over_time(
            axis=axis,
            time_seconds=time_axis_seconds,
            magnitude=magnitude,
            floor_db=settings.log_magnitude_floor_db,
            alpha=channel_alpha,
            label=channel_name,
        )

    label_time_axis_seconds(axis)
    label_decibel_axis(axis)

    # Add legend for channel identification
    if not settings.use_mono_downmix:
        axis.legend()

    finalize_and_show_or_save(
        figure=figure, output_path=output_path, show_interactive=show_interactive
    )


def _suffix_output_path(output_path: str | Path, suffix: str) -> Path:
    """
    Add a suffix before the file extension:
      "plot.png" + "_early" -> "plot_early.png"
    """
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def plot_ir_from_wav_file(
    wav_file_path: str | Path,
    settings: Optional[ImpulseResponseViewSettings] = None,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> None:
    """
    Convenience wrapper: load WAV then plot waveform + log-magnitude.

    If output_basename is provided, two PNGs will be written:
      - <basename>.png           (full waveform)
      - <basename>_early.png     (early waveform)
      - <basename>_tail.png      (log magnitude tail)
    """
    if settings is None:
        settings = ImpulseResponseViewSettings()

    loaded_audio = load_wav_file(
        wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )
    if output_basename is None:
        waveform_output_path = None
        tail_output_path = None
    else:
        output_basename = Path(output_basename)
        waveform_output_path = output_basename.with_suffix(".png")
        tail_output_path = output_basename.with_name(f"{output_basename.stem}_tail.png").with_suffix(".png")

    plot_impulse_response_waveform(
        loaded_audio=loaded_audio,
        settings=settings,
        output_path=waveform_output_path,
        show_interactive=show_interactive,
    )

    plot_impulse_response_log_magnitude(
        loaded_audio=loaded_audio,
        settings=settings,
        output_path=tail_output_path,
        show_interactive=show_interactive,
    )

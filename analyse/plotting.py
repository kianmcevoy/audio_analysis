# analyse/plotting.py
"""
Shared plotting helpers for reverb analysis.

Design goals:
- consistent plot appearance across all analyses
- explicit labels, units, and titles
- no hidden global matplotlib state
- readable, boring plotting code
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Global plotting defaults (local, not rcParams-global)
# -------------------------------------------------------------------

DEFAULT_FIGURE_SIZE = (10.0, 6.0)
DEFAULT_DPI = 100
DEFAULT_GRID = True


# -------------------------------------------------------------------
# Figure / axis helpers
# -------------------------------------------------------------------

def create_figure_and_axis(
    title: Optional[str] = None,
    figure_size: Tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a matplotlib figure and axis with consistent defaults.
    """
    figure, axis = plt.subplots(figsize=figure_size, dpi=DEFAULT_DPI)

    if title is not None:
        axis.set_title(title)

    axis.grid(DEFAULT_GRID)
    return figure, axis


def finalize_and_show_or_save(
    figure: plt.Figure,
    output_path: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> None:
    """
    Finalise a plot: either show it interactively or save to disk.

    If output_path is provided:
        - the figure is saved as PNG
        - the figure is closed
    Otherwise:
        - the figure is shown interactively (unless show_interactive=False)
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        return

    if show_interactive:
        plt.show()

    plt.close(figure)


# -------------------------------------------------------------------
# Axis labelling helpers
# -------------------------------------------------------------------

def label_time_axis_seconds(axis: plt.Axes) -> None:
    axis.set_xlabel("Time (seconds)")


def label_frequency_axis_hz(axis: plt.Axes, log_scale: bool = False) -> None:
    axis.set_xlabel("Frequency (Hz)")
    if log_scale:
        axis.set_xscale("log")


def label_amplitude_axis(
    axis: plt.Axes,
    unit: str = "Amplitude",
) -> None:
    axis.set_ylabel(unit)


def label_decibel_axis(axis: plt.Axes) -> None:
    axis.set_ylabel("Level (dB)")


# -------------------------------------------------------------------
# Common plot patterns
# -------------------------------------------------------------------

def plot_time_series(
    axis: plt.Axes,
    time_seconds: np.ndarray,
    samples: np.ndarray,
    label: Optional[str] = None,
    color: Optional[str] = None,
    alpha: float = 1.0,
) -> None:
    axis.plot(
        time_seconds,
        samples,
        label=label,
        color=color,
        alpha=alpha,
    )
    if label is not None:
        axis.legend(loc="best")



def plot_log_magnitude_over_time(
    axis: plt.Axes,
    time_seconds: np.ndarray,
    magnitude: np.ndarray,
    floor_db: float = -120.0,
    alpha: float = 1.0,
    label: str | None = None,
) -> None:
    """
    Plot magnitude in decibels over time.
    """
    magnitude = np.maximum(magnitude, 10 ** (floor_db / 20.0))
    magnitude_db = 20.0 * np.log10(magnitude)

    axis.plot(time_seconds, magnitude_db, alpha=alpha, label=label)
    axis.set_ylim(bottom=floor_db)


def plot_spectrogram(
    axis: plt.Axes,
    spectrogram_magnitude: np.ndarray,
    time_seconds: np.ndarray,
    frequency_hz: np.ndarray,
    magnitude_floor_db: float = -120.0,
) -> None:
    """
    Plot a log-magnitude spectrogram using pcolormesh.
    """
    magnitude_db = 20.0 * np.log10(
        np.maximum(spectrogram_magnitude, 10 ** (magnitude_floor_db / 20.0))
    )

    mesh = axis.pcolormesh(
        time_seconds,
        frequency_hz,
        magnitude_db,
        shading="nearest",
        cmap="magma",
    )

    axis.set_ylabel("Frequency (Hz)")
    axis.set_ylim(bottom=frequency_hz[1])
    axis.set_yscale("log")

    plt.colorbar(mesh, ax=axis, label="Magnitude (dB)")


def plot_waterfall_lines(
    axis: plt.Axes,
    frequency_hz: np.ndarray,
    magnitude_slices: np.ndarray,
    time_offsets: np.ndarray,
    offset_scale: float = 1.0,
) -> None:
    """
    Plot a waterfall / cumulative spectral decay as stacked line plots.

    magnitude_slices shape: (num_slices, num_frequency_bins)
    time_offsets shape: (num_slices,)
    """
    for slice_index in range(magnitude_slices.shape[0]):
        axis.plot(
            frequency_hz,
            magnitude_slices[slice_index] + time_offsets[slice_index] * offset_scale,
            linewidth=1.0,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Magnitude + time offset")


def plot_scatter(
    axis: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    size_values: Optional[np.ndarray] = None,
    alpha: float = 0.7,
) -> None:
    """
    Generic scatter plot helper (used for mode clouds).
    """
    if size_values is not None:
        axis.scatter(x_values, y_values, s=size_values, alpha=alpha)
    else:
        axis.scatter(x_values, y_values, alpha=alpha)

    axis.grid(True)


# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

def time_axis_from_sample_count(
    number_of_samples: int,
    sample_rate_hz: int,
) -> np.ndarray:
    """
    Generate a time axis in seconds for a given sample count.
    """
    return np.arange(number_of_samples, dtype=np.float32) / float(sample_rate_hz)

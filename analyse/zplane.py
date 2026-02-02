# analyse/zplane.py
"""
Z-plane pole/zero "cloud" plot from an impulse response.

Goal:
- Quickly visualise pole locations (and optional zeros) so you can iterate on DSP
  and re-run analysis without manually deriving polynomials.

Two modes:
1) AR (all-pole) fit (default): fast + robust for resonators and feedback combs.
   - Fits an all-pole IIR model to the selected segment using least squares.
   - Poles = roots(a), where a[0]=1.
   - Good for: Schroeder feedback comb, resonators, lightly coloured loops.

2) AR + FIR numerator (optional zeros): after AR fit, derive a short FIR numerator
   that matches the first samples of the segment:
       b[n] = sum_{k=0..p} a[k] * h[n-k]
   - This is not a full Prony ARMA fit; it's a pragmatic "good enough" numerator
     for plotting zeros that affect early response.

Limitations:
- Pole/zero estimation from IR is model-based: results depend on chosen model order
  and the segment used.
- For heavily time-varying systems (modulated delay, noise modulation), poles are
  not strictly defined; treat the plot as a diagnostic, not ground truth.

Outputs:
- One plot per analysed channel (L/R or M).
- If output_basename provided:
    <basename>_zplane_<CH>.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from analyse.io import get_analysis_channels, load_wav_file
from analyse.plotting import create_figure_and_axis, finalize_and_show_or_save


@dataclass(frozen=True)
class ZPlaneAnalysisSettings:
    # Stereo handling
    use_mono_downmix_for_stereo: bool = False

    # Time selection
    trim_to_peak: bool = True
    ignore_leading_seconds: float = 0.0
    analysis_duration_seconds: Optional[float] = None

    # Model fit
    model: str = "ar"            # "ar"
    ar_order: int = 256          # poles count
    derive_zeros: bool = False   # derive FIR numerator to show zeros (approx)
    zero_order: int = 64         # FIR length-1 when derive_zeros=True

    # Numerical robustness
    normalise_segment: bool = True
    ridge_lambda: float = 0.0    # >0 adds tiny ridge for ill-conditioned fits


@dataclass(frozen=True)
class ZPlanePlotSettings:
    secondary_channel_alpha: float = 0.7
    show_unit_circle: bool = True
    show_axes: bool = True
    limit_radius: float = 1.2    # plot limits in x/y
    annotate_stats: bool = True


@dataclass(frozen=True)
class ChannelZPlaneResult:
    channel_name: str
    sample_rate_hz: int
    poles: np.ndarray   # complex
    zeros: Optional[np.ndarray]  # complex or None


def _fit_ar_least_squares(x: np.ndarray, order: int, ridge_lambda: float = 0.0) -> np.ndarray:
    """
    Fit an AR model:
        x[n] + sum_{k=1..p} a[k] x[n-k] = e[n]
    Return a with a[0]=1.

    Uses least squares (and optional ridge regularisation).
    """
    x = np.asarray(x, dtype=np.float64)
    p = int(order)
    if p < 1:
        return np.array([1.0], dtype=np.float64)

    # Need at least p+1 samples.
    if x.size <= p:
        p = max(1, x.size - 1)

    # Build regression: A * a = y
    # For n=p..N-1:
    #   y = -x[n]
    #   A row = [x[n-1], x[n-2], ..., x[n-p]]
    N = x.size
    y = -x[p:N]
    A = np.empty((N - p, p), dtype=np.float64)
    for k in range(1, p + 1):
        A[:, k - 1] = x[p - k : N - k]

    if ridge_lambda and ridge_lambda > 0.0:
        # Solve (A^T A + λI) a = A^T y
        ATA = A.T @ A
        ATy = A.T @ y
        ATA.flat[:: p + 1] += float(ridge_lambda)
        a_rest = np.linalg.solve(ATA, ATy)
    else:
        a_rest, *_ = np.linalg.lstsq(A, y, rcond=None)

    a = np.concatenate(([1.0], a_rest))
    return a


def _derive_fir_numerator_from_ar(a: np.ndarray, h: np.ndarray, zero_order: int) -> np.ndarray:
    """
    Given AR denominator a (a[0]=1) and an impulse response segment h,
    derive a short FIR numerator b so that (a * h)[0:Q+1] = b[0:Q+1].

    This is a pragmatic way to get "some zeros" for plotting without a full ARMA fit.
    """
    p = len(a) - 1
    Q = int(max(0, zero_order))
    b = np.zeros(Q + 1, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    for n in range(Q + 1):
        acc = 0.0
        for k in range(0, p + 1):
            if n - k < 0 or n - k >= h.size:
                continue
            acc += a[k] * h[n - k]
        b[n] = acc
    return b


def _roots_from_poly_descending(poly: np.ndarray) -> np.ndarray:
    """
    np.roots expects descending powers of z.
    Our AR poly is in z^-1 form: A(z) = 1 + a1 z^-1 + ... + ap z^-p.
    Multiply by z^p: z^p + a1 z^{p-1} + ... + ap.
    So coefficients in descending powers are: [1, a1, a2, ..., ap].
    """
    poly = np.asarray(poly, dtype=np.float64)
    # Trim tiny trailing coefficients for stability
    while poly.size > 1 and abs(poly[-1]) < 1e-14:
        poly = poly[:-1]
    if poly.size <= 1:
        return np.array([], dtype=np.complex128)
    return np.roots(poly)


def _rt60_from_pole_radius(r: float, sample_rate_hz: int) -> float:
    """
    Approximate RT60 for a single pole radius r (0<r<1) assuming exponential decay:
        r^n = exp(n ln r) = exp(-n / tau_samples)
    tau_samples = -1 / ln(r)
    RT60 ≈ ln(1000) * tau_seconds
    """
    r = float(r)
    if r <= 0.0 or r >= 1.0:
        return float("inf")
    tau_samples = -1.0 / np.log(r)
    tau_seconds = tau_samples / float(sample_rate_hz)
    return np.log(1000.0) * tau_seconds


def plot_zplane_from_wav_file(
    input_wav_file_path: str,
    settings: ZPlaneAnalysisSettings,
    plot_settings: ZPlanePlotSettings,
    output_basename: Optional[str | Path] = None,
    show_interactive: bool = True,
) -> List[ChannelZPlaneResult]:
    loaded = load_wav_file(
        input_wav_file_path,
        expected_channel_mode="mono_or_stereo",
        allow_mono_and_upmix_to_stereo=False,
    )
    channels = get_analysis_channels(loaded, use_mono_downmix_for_stereo=settings.use_mono_downmix_for_stereo)

    results: List[ChannelZPlaneResult] = []

    for channel_name, channel_samples in channels:
        samples = channel_samples

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

        seg = segment.astype(np.float64, copy=False)

        if settings.normalise_segment:
            peak = float(np.max(np.abs(seg))) if seg.size else 1.0
            if peak > 0.0:
                seg = seg / peak

        # Fit AR denominator
        a = _fit_ar_least_squares(seg, order=int(settings.ar_order), ridge_lambda=float(settings.ridge_lambda))
        poles = _roots_from_poly_descending(a)

        zeros: Optional[np.ndarray] = None
        if settings.derive_zeros:
            b = _derive_fir_numerator_from_ar(a, seg, zero_order=int(settings.zero_order))
            # FIR numerator in z^-1 => B(z) = b0 + b1 z^-1 + ...
            # Multiply by z^Q: b0 z^Q + b1 z^{Q-1} + ... + bQ
            # descending coeffs: [b0, b1, ..., bQ]
            zeros = _roots_from_poly_descending(b)

        results.append(ChannelZPlaneResult(
            channel_name=channel_name,
            sample_rate_hz=loaded.sample_rate_hz,
            poles=poles,
            zeros=zeros,
        ))

        title = f"Z-plane pole cloud ({channel_name})"
        fig, ax = create_figure_and_axis(title=title, figure_size=(7.5, 7.5))

        if plot_settings.show_axes:
            ax.axhline(0.0, linewidth=1.0)
            ax.axvline(0.0, linewidth=1.0)

        if plot_settings.show_unit_circle:
            t = np.linspace(0.0, 2.0 * np.pi, 512)
            ax.plot(np.cos(t), np.sin(t), linestyle="--", linewidth=1.0)

        # Poles and zeros
        if poles.size:
            ax.scatter(np.real(poles), np.imag(poles), marker="x", s=30, label="Poles")

        if zeros is not None and zeros.size:
            ax.scatter(np.real(zeros), np.imag(zeros), marker="o", s=18, facecolors="none", label="Zeros")

        ax.set_aspect("equal", adjustable="box")
        lim = float(plot_settings.limit_radius)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Re{z}")
        ax.set_ylabel("Im{z}")
        ax.legend(loc="upper right")

        if plot_settings.annotate_stats and poles.size:
            radii = np.abs(poles)
            max_r = float(np.max(radii))
            med_r = float(np.median(radii))
            unstable = int(np.sum(radii >= 1.0))
            rt60_med = _rt60_from_pole_radius(min(med_r, 0.999999), loaded.sample_rate_hz)
            rt60_max = _rt60_from_pole_radius(min(max_r, 0.999999), loaded.sample_rate_hz)
            txt = (
                f"AR order: {int(settings.ar_order)}\n"
                f"poles: {poles.size}\n"
                f"unstable (|p|>=1): {unstable}\n"
                f"radius median: {med_r:.6f}\n"
                f"radius max: {max_r:.6f}\n"
                f"RT60~ (median r): {rt60_med:.3f} s\n"
                f"RT60~ (max r): {rt60_max:.3f} s"
            )
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

        if output_basename is not None:
            output_path = str(Path(output_basename).with_suffix("")) + f"_zplane_{channel_name}.png"
        else:
            output_path = None

        finalize_and_show_or_save(fig, output_path=output_path, show_interactive=show_interactive)

    return results


def summarise_zplane_results_text(results: List[ChannelZPlaneResult]) -> str:
    lines: List[str] = []
    for r in results:
        if r.poles.size == 0:
            lines.append(f"- {r.channel_name}: no poles (fit failed or order=0)")
            continue
        radii = np.abs(r.poles)
        lines.append(
            f"- {r.channel_name}: poles={r.poles.size}, "
            f"max|p|={float(np.max(radii)):.6f}, median|p|={float(np.median(radii)):.6f}, "
            f"unstable(|p|>=1)={int(np.sum(radii>=1.0))}"
        )
    if not lines:
        return "No z-plane results."
    return "Z-plane summary:\n" + "\n".join(lines)

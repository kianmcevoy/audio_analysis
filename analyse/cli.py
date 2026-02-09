# analyse/cli.py
"""
Command Line Interface (CLI) for offline reverb analysis.

Usage examples:
  python -m analyse.cli --help
  python -m analyse.cli ir --help
  python -m analyse.cli ir --input output.wav
  python -m analyse.cli ir --input output.wav --output plots/output_ir

Notes:
- This CLI is intentionally simple: one command produces one plot set.
- Default expectations are stereo WAV at 48 kHz.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from analyse.impulse_response import ImpulseResponseViewSettings, plot_ir_from_wav_file

from analyse.decay import (
    DecayAnalysisSettings,
    DecayPlotSettings,
    plot_decay_from_wav_file,
    summarise_decay_results_text,
)

from analyse.rt60bands import (
    Rt60BandsAnalysisSettings,
    Rt60BandsPlotSettings,
    plot_rt60_bands_from_wav_file,
    summarise_rt60_bands_results_text,
)

from analyse.frequency_response import (
    FrequencyResponseAnalysisSettings,
    FrequencyResponsePlotSettings,
    plot_frequency_response_from_wav_file,
    summarise_frequency_response_results_text,
)

from analyse.filterplot import (
    FilterAnalysisSettings,
    FilterPlotSettings,
    plot_filter_response_from_wav_file,
    summarise_filter_response_results_text,
)

from analyse.diffusion import (
    DiffusionAnalysisSettings,
    plot_diffusion_from_wav_file,
    summarise_diffusion_results_text,
)

from analyse.spectrogram import (
    SpectrogramAnalysisSettings,
    SpectrogramPlotSettings,
    plot_spectrogram_from_wav_file,
    summarise_spectrogram_results_text,
)

from analyse.waterfall import (
    WaterfallAnalysisSettings,
    WaterfallPlotSettings,
    plot_waterfall_from_wav_file,
    summarise_waterfall_results_text,
)

from analyse.modalcloud import (
    ModalCloudAnalysisSettings,
    ModalCloudPlotSettings,
    plot_modal_cloud_from_wav_file,
    summarise_modal_cloud_results_text,
)

from analyse.zplane import (
    ZPlaneAnalysisSettings,
    ZPlanePlotSettings,
    plot_zplane_from_wav_file,
    summarise_zplane_results_text,
)

from analyse.group_delay import (
    GroupDelayAnalysisSettings,
    GroupDelayPlotSettings,
    plot_group_delay_from_wav_file,
    summarise_group_delay_results_text,
)

from analyse.deconvolve import (
    DeconvolveSettings,
    deconvolve_from_wav_files,
    default_output_ir_path,
)

from analyse.report import (
    ReportSettings,
    run_report_from_wav_file,
)

from analyse.bundle import (
    BundleRunSettings,
    run_bundle_report,
)


def parse_arguments() -> argparse.Namespace:
    top_level_parser = argparse.ArgumentParser(
        prog="analyse",
        description="Offline analysis tools for reverb outputs (plots, metrics).",
    )

    subparsers = top_level_parser.add_subparsers(
        dest="command_name",
        required=True,
        help="Analysis to run. Use: analyse <command> --help",
    )

    # ------------------------------------------------------------------
    # IR (waveform + early zoom + tail log-magnitude)
    # ------------------------------------------------------------------
    impulse_response_parser = subparsers.add_parser(
        "ir",
        help="Plot waveform (full + early zoom) and log-magnitude tail view.",
    )

    impulse_response_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo, 48 kHz expected).",
    )

    impulse_response_parser.add_argument(
        "--early-window",
        dest="early_window_seconds",
        type=float,
        default=0.08,
        help="Early zoom window length in seconds (default: 0.08).",
    )

    impulse_response_parser.add_argument(
        "--floor-db",
        dest="log_magnitude_floor_db",
        type=float,
        default=-120.0,
        help="Minimum dB floor for log-magnitude tail plot (default: -120).",
    )

    impulse_response_parser.add_argument(
        "--mono",
        dest="use_mono_downmix",
        action="store_true",
        help="If set, downmix stereo to mono for plotting.",
    )

    impulse_response_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help=(
            "If provided, saves PNGs instead of showing plots. "
            "Example: --output plots/my_preset "
            "will write plots/my_preset.png, plots/my_preset_early.png, plots/my_preset_tail.png"
        ),
    )

    impulse_response_parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not display plots interactively (useful when saving files).",
    )
    

    # ------------------------------------------------------------------
    # Z-plane (pole/zero cloud)
    # ------------------------------------------------------------------
    zplane_parser = subparsers.add_parser(
        "zplane",
        help="Estimate poles (and optional zeros) from an IR and plot them on the z-plane.",
    )

    zplane_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo).",
    )

    zplane_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="Output basename (no extension). e.g. plots/my_ir  -> plots/my_ir_zplane_L.png",
    )
    zplane_parser.add_argument(
        "--no-show",
        dest="no_show",
        action="store_true",
        help="Do not show plots interactively (save only if --output is provided).",
    )


    zplane_parser.add_argument(
        "--mono",
        dest="use_mono_downmix_for_stereo",
        action="store_true",
        help="If input is stereo, downmix to mono before analysis.",
    )

    zplane_parser.add_argument(
        "--no-trim",
        dest="trim_to_peak",
        action="store_false",
        help="Do not trim analysis start to peak sample (use start of file).",
    )

    zplane_parser.add_argument(
        "--ignore-leading",
        dest="ignore_leading_seconds",
        type=float,
        default=0.0,
        help="Seconds to skip after trim point (default: 0).",
    )

    zplane_parser.add_argument(
        "--duration",
        dest="analysis_duration_seconds",
        type=float,
        default=None,
        help="Analyse only this many seconds (default: to end).",
    )

    zplane_parser.add_argument(
        "--ar-order",
        dest="ar_order",
        type=int,
        default=256,
        help="AR model order (pole count). Default: 256.",
    )

    zplane_parser.add_argument(
        "--zeros",
        dest="derive_zeros",
        action="store_true",
        help="Also derive a short FIR numerator and plot zeros (approx).",
    )

    zplane_parser.add_argument(
        "--zero-order",
        dest="zero_order",
        type=int,
        default=64,
        help="FIR numerator order (when --zeros). Default: 64.",
    )

    zplane_parser.add_argument(
        "--radius",
        dest="limit_radius",
        type=float,
        default=1.2,
        help="Plot radius limit (default: 1.2).",
    )

    zplane_parser.add_argument(
        "--ridge",
        dest="ridge_lambda",
        type=float,
        default=0.0,
        help="Optional ridge regularisation for AR fit (default: 0). Try 1e-6 .. 1e-3 if fit is unstable.",
    )

    # ------------------------------------------------------------------
    # Group delay
    # ------------------------------------------------------------------
    groupdelay_parser = subparsers.add_parser(
        "groupdelay",
        help="Plot group delay vs frequency from an IR/filter output.",
    )

    groupdelay_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo).",
    )

    groupdelay_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="Output basename (no extension). e.g. plots/my_ir  -> plots/my_ir_groupdelay_L.png",
    )
    groupdelay_parser.add_argument(
        "--no-show",
        dest="no_show",
        action="store_true",
        help="Do not show plots interactively (save only if --output is provided).",
    )


    groupdelay_parser.add_argument(
        "--mono",
        dest="use_mono_downmix_for_stereo",
        action="store_true",
        help="If input is stereo, downmix to mono before analysis.",
    )

    groupdelay_parser.add_argument(
        "--no-trim",
        dest="trim_to_peak",
        action="store_false",
        help="Do not trim analysis start to peak sample (use start of file).",
    )

    groupdelay_parser.add_argument(
        "--ignore-leading",
        dest="ignore_leading_seconds",
        type=float,
        default=0.0,
        help="Seconds to skip after trim point (default: 0).",
    )

    groupdelay_parser.add_argument(
        "--duration",
        dest="analysis_duration_seconds",
        type=float,
        default=None,
        help="Analyse only this many seconds (default: to end).",
    )

    groupdelay_parser.add_argument(
        "--fft",
        dest="fft_size",
        type=int,
        default=None,
        help="FFT size (default: next pow2 >= segment length, capped).",
    )

    groupdelay_parser.add_argument(
        "--smooth",
        dest="smoothing_bins",
        type=int,
        default=0,
        help="Moving-average smoothing window in FFT bins (default: 0 disables).",
    )

    groupdelay_parser.add_argument(
        "--fmin",
        dest="f_min_hz",
        type=float,
        default=20.0,
        help="Min frequency for plot (Hz). Default: 20.",
    )

    groupdelay_parser.add_argument(
        "--fmax",
        dest="f_max_hz",
        type=float,
        default=20000.0,
        help="Max frequency for plot (Hz). Default: 20000.",
    )
    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    # Bundle (analyse a folder of taps + meta.json)
    # ------------------------------------------------------------------
    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Analyse an IR bundle folder (meta.json + taps/*.wav) and write per-tap reports.",
    )
    bundle_parser.add_argument("--input", dest="bundle_root", type=str, required=True, help="Bundle root folder")
    bundle_parser.add_argument(
        "--reports-subdir",
        dest="reports_subdir",
        type=str,
        default="reports",
        help="Subfolder under bundle root to write outputs (default: reports)",
    )

# Deconvolve (sweep -> impulse response)
    # ------------------------------------------------------------------
    deconvolve_parser = subparsers.add_parser(
        "deconvolve",
        help="Deconvolve recorded sweep output into an impulse response WAV.",
    )

    deconvolve_parser.add_argument(
        "--recorded_wav_file_path",
        type=str,
        required=True,
        help="Recorded output WAV (mono or stereo, 48 kHz expected).",
    )

    deconvolve_parser.add_argument(
        "--sweep_wav_file_path",
        type=str,
        required=True,
        help="Original sweep WAV that was played (mono or stereo, 48 kHz expected).",
    )

    deconvolve_parser.add_argument(
        "--output_ir_wav_file_path",
        type=str,
        default=None,
        help=(
            "Output IR WAV path. If omitted, writes next to recorded file as: "
            "<recorded_stem>_ir.wav"
        ),
    )

    deconvolve_parser.add_argument(
        "--regularization_relative",
        type=float,
        default=1e-10,
        help="Regularization as fraction of max(|X|^2). Default: 1e-10.",
    )

    deconvolve_parser.add_argument(
        "--normalise_peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, peak-normalise the output IR. Default: true.",
    )

    deconvolve_parser.add_argument(
        "--target_peak",
        type=float,
        default=0.95,
        help="Target peak value if normalise_peak is enabled. Default: 0.95.",
    )

    deconvolve_parser.add_argument(
        "--remove_dc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, remove DC offset from the output IR. Default: true.",
    )

    deconvolve_parser.add_argument(
        "--output_length_mode",
        type=str,
        choices=["recorded", "full_fft"],
        default="recorded",
        help="Length of output IR. Default: recorded.",
    )


    # ------------------------------------------------------------------
    # RT60 Decay (T20/T30/EDT) 
    # ------------------------------------------------------------------
    decay_parser = subparsers.add_parser(
        "decay",
        help="Schroeder EDC + T20/T30/RT60 decay estimation"
        )

    decay_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo, 48 kHz expected).",
    )
    decay_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="If provided, saves a PNG: <basename>_decay.png",
    )
    decay_parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not display plots interactively (useful when saving files).",
    )

    decay_parser.add_argument(
        "--trim_to_peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, start analysis at the absolute peak sample (default: true).",
        )
    
    decay_parser.add_argument(
        "--ignore-leading",
        dest="ignore_leading_seconds",
        type=float,
        default=0.0
        )

    decay_parser.add_argument(
        "--edc_floor_db",
        type=float,
        default=-120.0
        )
    
    decay_parser.add_argument(
        "--fit_lower_limit_db",
        type=float,
        default=-80.0
        )

    decay_parser.add_argument(
        "--smoothing",
        dest="edc_smoothing_window_samples",
        type=int,
        default=0
        )

    decay_parser.add_argument(
        "--mono",
        dest="use_mono_downmix",
        action="store_true",
        default=False
        )
    
    decay_parser.add_argument(
        "--compute_edt",
        action=argparse.BooleanOptionalAction,
        help="If set, also compute EDT (fit 0..-10 dB and extrapolate).",
        default=True
        )
    
    # ------------------------------------------------------------------
    # RT60 bands (Low/Mid/High by default)
    # ------------------------------------------------------------------
    rt60bands_parser = subparsers.add_parser(
        "rt60bands",
        help="Band-limited RT60: default Low/Mid/High T30 on one plot (optional T20/EDT).",
    )

    rt60bands_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo, 48 kHz expected).",
    )

    rt60bands_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="If provided, saves one PNG: <basename>_rt60bands.png",
    )

    rt60bands_parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not display plots interactively (useful when saving files).",
    )
    
    rt60bands_parser.add_argument(
    "--band_mode",
    type=str,
    default="three",
    choices=["three", "octave", "third"],
    help="Band mode: three (Low/Mid/High), octave, or third (1/3-octave). Default: three.",
    )

    rt60bands_parser.add_argument(
        "--f_min_hz",
        type=float,
        default=31.5,
        help="Minimum band centre frequency for octave/third modes (Hz).",
    )
    rt60bands_parser.add_argument(
        "--f_max_hz",
        type=float,
        default=16000.0,
        help="Maximum band centre frequency for octave/third modes (Hz).",
    )
    
    rt60bands_parser.add_argument(
        "--legend_values",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include per-band numeric values in the legend. "
            "Default: enabled for band_mode=three, disabled otherwise."
        ),
    )

    # Defaults are Low/Mid/High broad bands; these args override them
    rt60bands_parser.add_argument("--low_upper_hz", type=float, default=250.0, help="Low band upper cutoff (Hz).")
    rt60bands_parser.add_argument("--mid_center_hz", type=float, default=1000.0, help="Mid band centre (Hz).")
    rt60bands_parser.add_argument(
        "--mid_width_octaves",
        type=float,
        default=2.0,
        help="Mid band width in octaves (default 2.0 => roughly 500..2000 Hz around 1 kHz).",
    )
    rt60bands_parser.add_argument("--high_lower_hz", type=float, default=4000.0, help="High band lower cutoff (Hz).")

    rt60bands_parser.add_argument(
        "--transition_width_octaves",
        type=float,
        default=(1.0 / 6.0),
        help="FFT mask transition width (octaves). Default: 1/6 octave.",
    )

    rt60bands_parser.add_argument(
        "--include_t20",
        action="store_true",
        help="If set, also compute T20-derived RT60 for each band.",
    )
    rt60bands_parser.add_argument(
        "--include_edt",
        action="store_true",
        help="If set, also compute EDT (0..-10 dB fit extrapolated) for each band.",
    )

    # Reuse the same stereo policy as other analyses
    rt60bands_parser.add_argument(
        "--mono",
        dest="use_mono_downmix",
        action="store_true",
        help="If set, downmix stereo to mono before analysis.",
    )

    # Reuse core decay policy knobs (kept minimal)
    rt60bands_parser.add_argument("--trim_to_peak", action="store_true", default=True)
    rt60bands_parser.add_argument("--ignore-leading", dest="ignore_leading_seconds", type=float, default=0.0)
    rt60bands_parser.add_argument("--edc_floor_db", type=float, default=-120.0)
    rt60bands_parser.add_argument("--fit_lower_limit_db", type=float, default=-80.0)
    rt60bands_parser.add_argument("--smoothing", dest="edc_smoothing_window_samples", type=int, default=0)
    
    # ------------------------------------------------------------------
    # Frequency response (spectrum)
    # ------------------------------------------------------------------
    fr_parser = subparsers.add_parser(
        "fr",
        help="Plot magnitude spectrum (dB) vs frequency (log-x) for a selected segment.",
    )

    fr_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo, 48 kHz expected).",
    )

    fr_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="If provided, saves a PNG: <basename>_fr.png",
    )

    fr_parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not display plots interactively (useful when saving files).",
    )

    fr_parser.add_argument(
        "--mono",
        dest="use_mono_downmix",
        action="store_true",
        help="If set, downmix stereo to mono before analysis.",
    )

    fr_parser.add_argument(
        "--trim_to_peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, start analysis at the absolute peak sample (default: true).",
    )

    fr_parser.add_argument(
        "--ignore-leading",
        dest="ignore_leading_seconds",
        type=float,
        default=0.0,
        help="Ignore this many seconds after trim point (default: 0.0).",
    )

    fr_parser.add_argument(
        "--duration",
        dest="analysis_duration_seconds",
        type=float,
        default=None,
        help="If set, analyse only this many seconds from the start of the analysed segment.",
    )

    fr_parser.add_argument(
        "--magnitude_floor_db",
        type=float,
        default=-120.0,
        help="Floor for magnitude dB (default: -120).",
    )

    fr_parser.add_argument(
        "--f_min_hz",
        type=float,
        default=20.0,
        help="Minimum plotted frequency (default: 20).",
    )
    fr_parser.add_argument(
        "--f_max_hz",
        type=float,
        default=20000.0,
        help="Maximum plotted frequency (default: 20000).",
    )

    fr_parser.add_argument(
        "--smoothing_log_bins",
        type=int,
        default=0,
        help="Optional log-frequency smoothing window (bins). 0 disables.",
    )

    fr_parser.add_argument(
        "--log_bins_per_octave",
        type=int,
        default=96,
        help="Resolution of the internal log-frequency grid (default: 96 bins/octave).",
    )

    fr_parser.add_argument(
        "--no_hann_window",
        action="store_true",
        help="If set, do not apply a Hann window before FFT.",
    )
    
    # ------------------------------------------------------------------
    # Filter response (magnitude and phase)
    # ------------------------------------------------------------------
    filter_parser = subparsers.add_parser(
        "filter",
        help="Plot filter frequency response: magnitude (dB) and phase.",
    )

    filter_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo, 48 kHz expected).",
    )

    filter_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="If provided, saves a PNG: <basename>_filter.png",
    )

    filter_parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not display plots interactively (useful when saving files).",
    )

    filter_parser.add_argument(
        "--mono",
        dest="use_mono_downmix",
        action="store_true",
        help="If set, downmix stereo to mono before analysis.",
    )

    filter_parser.add_argument(
        "--trim_to_peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, start analysis at the absolute peak sample (default: true).",
    )

    filter_parser.add_argument(
        "--ignore-leading",
        dest="ignore_leading_seconds",
        type=float,
        default=0.0,
        help="Ignore this many seconds after trim point (default: 0.0).",
    )

    filter_parser.add_argument(
        "--duration",
        dest="analysis_duration_seconds",
        type=float,
        default=None,
        help="If set, analyse only this many seconds from the start of the analysed segment.",
    )

    filter_parser.add_argument(
        "--magnitude_floor_db",
        type=float,
        default=-120.0,
        help="Floor for magnitude dB (default: -120).",
    )

    filter_parser.add_argument(
        "--f_min_hz",
        type=float,
        default=20.0,
        help="Minimum plotted frequency (default: 20).",
    )
    
    filter_parser.add_argument(
        "--f_max_hz",
        type=float,
        default=20000.0,
        help="Maximum plotted frequency (default: 20000).",
    )

    filter_parser.add_argument(
        "--phase_mode",
        type=str,
        choices=["degrees", "radians"],
        default="degrees",
        help="Phase display mode: degrees or radians (default: degrees).",
    )

    filter_parser.add_argument(
        "--no_unwrap_phase",
        action="store_true",
        help="If set, do not unwrap phase for continuous display.",
    )

    filter_parser.add_argument(
        "--no_hann_window",
        action="store_true",
        help="If set, do not apply a Hann window before FFT.",
    )
    
    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------
    spectrogram_parser = subparsers.add_parser(
        "spectrogram",
        help="Plot time–frequency magnitude spectrogram (log-frequency).",
    )

    spectrogram_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo, 48 kHz expected).",
    )

    spectrogram_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="If provided, saves PNG(s): <basename>_spectrogram_<CH>.png",
    )

    spectrogram_parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not display plots interactively (useful when saving files).",
    )

    spectrogram_parser.add_argument(
        "--mono",
        dest="use_mono_downmix",
        action="store_true",
        help="If set, downmix stereo to mono before analysis.",
    )

    spectrogram_parser.add_argument(
        "--trim_to_peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, start analysis at the absolute peak sample (default: true).",
    )

    spectrogram_parser.add_argument(
        "--ignore-leading",
        dest="ignore_leading_seconds",
        type=float,
        default=0.0,
        help="Ignore this many seconds after trim point (default: 0.0).",
    )

    spectrogram_parser.add_argument(
        "--duration",
        dest="analysis_duration_seconds",
        type=float,
        default=None,
        help="If set, analyse only this many seconds from the analysed segment.",
    )

    spectrogram_parser.add_argument("--n_fft", type=int, default=4096, help="STFT FFT size (default: 4096).")
    spectrogram_parser.add_argument("--hop_length", type=int, default=512, help="STFT hop length (default: 512).")

    spectrogram_parser.add_argument(
        "--no_hann_window",
        action="store_true",
        help="If set, do not apply a Hann window to each frame.",
    )

    spectrogram_parser.add_argument("--floor_db", type=float, default=-120.0, help="Magnitude floor in dB (default: -120).")
    spectrogram_parser.add_argument("--f_min_hz", type=float, default=20.0, help="Min frequency to display (default: 20).")
    spectrogram_parser.add_argument("--f_max_hz", type=float, default=20000.0, help="Max frequency to display (default: 20000).")

    spectrogram_parser.add_argument(
        "--dynamic_range_db",
        type=float,
        default=90.0,
        help="Color scale range below max (default: 90). Use 0 to disable and use percentiles.",
    )

        # ------------------------------------------------------------------
    # Diffusion (autocorr / echo density / decorrelation)
    # ------------------------------------------------------------------
    diffusion_parser = subparsers.add_parser(
        "diffusion",
        help="Diffusion metrics over time: autocorr, echo density, stereo decorrelation.",
    )

    diffusion_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo, 48 kHz expected).",
    )

    diffusion_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="If provided, saves one PNG: <basename>_diffusion.png",
    )

    diffusion_parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not display plots interactively (useful when saving files).",
    )

    diffusion_parser.add_argument(
        "--mono",
        dest="use_mono_downmix",
        action="store_true",
        help="If set, downmix stereo to mono before analysis.",
    )

    diffusion_parser.add_argument(
        "--trim_to_peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, start analysis at the absolute peak sample (default: true).",
    )

    diffusion_parser.add_argument(
        "--ignore-leading",
        dest="ignore_leading_seconds",
        type=float,
        default=0.0,
        help="Ignore this many seconds after the trim point (default: 0.0).",
    )

    diffusion_parser.add_argument(
        "--window_seconds",
        type=float,
        default=0.050,
        help="Analysis window length in seconds (default: 0.050).",
    )

    diffusion_parser.add_argument(
        "--hop_seconds",
        type=float,
        default=0.010,
        help="Hop size in seconds (default: 0.010).",
    )

    diffusion_parser.add_argument(
        "--max_lag_milliseconds",
        type=float,
        default=10.0,
        help="Max lag for autocorr/IACC (ms). Default: 10 ms.",
    )

    diffusion_parser.add_argument(
        "--echo_density_threshold_rms",
        type=float,
        default=1.0,
        help="Echo density threshold in RMS multiples. Default: 1.0.",
    )

    diffusion_parser.add_argument(
        "--echo_density_normalise_to_gaussian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, normalize echo density so Gaussian noise ~1.0 (default: true).",
    )

    
    # ------------------------------------------------------------------
    # Waterfall
    # ------------------------------------------------------------------
    waterfall_parser = subparsers.add_parser(
        "waterfall",
        help="Waterfall (CSD-style) plot: spectral slices over time (3D default, 2D ridges optional).",
    )

    waterfall_parser.add_argument(
        "--input",
        dest="input_wav_file_path",
        type=str,
        required=True,
        help="Path to input WAV file (mono or stereo, 48 kHz expected).",
    )

    waterfall_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        default=None,
        help="If provided, saves PNG(s): <basename>_waterfall_<CH>.png",
    )

    waterfall_parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not display plots interactively (useful when saving files).",
    )

    waterfall_parser.add_argument(
        "--mono",
        dest="use_mono_downmix",
        action="store_true",
        help="If set, downmix stereo to mono before analysis.",
    )

    waterfall_parser.add_argument(
        "--trim_to_peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, start analysis at the absolute peak sample (default: true).",
    )

    waterfall_parser.add_argument("--ignore-leading", dest="ignore_leading_seconds", type=float, default=0.0)
    waterfall_parser.add_argument("--duration", dest="analysis_duration_seconds", type=float, default=None)

    waterfall_parser.add_argument("--n_fft", type=int, default=4096)
    waterfall_parser.add_argument("--hop_length", type=int, default=512)
    waterfall_parser.add_argument("--no_hann_window", action="store_true")

    waterfall_parser.add_argument("--f_min_hz", type=float, default=20.0)
    waterfall_parser.add_argument("--f_max_hz", type=float, default=20000.0)

    waterfall_parser.add_argument(
        "--style",
        type=str,
        choices=["3d", "2d"],
        default="3d",
        help="Plot style: 3d waterfall (default) or 2d stacked ridges.",
    )

    waterfall_parser.add_argument(
        "--slice_mode",
        type=str,
        choices=["auto", "uniform_time", "uniform_frames"],
        default="auto",
        help="How to choose slice times/frames.",
    )
    waterfall_parser.add_argument("--num_slices", type=int, default=18)
    waterfall_parser.add_argument("--slice_spacing_seconds", type=float, default=0.05)
    waterfall_parser.add_argument("--start_time_seconds", type=float, default=0.0)
    waterfall_parser.add_argument("--end_time_seconds", type=float, default=None)

    waterfall_parser.add_argument(
        "--db_reference",
        type=str,
        choices=["global_max", "slice_max"],
        default="global_max",
        help="Normalisation: global_max keeps absolute decay; slice_max shows shape only.",
    )

    waterfall_parser.add_argument("--dynamic_range_db", type=float, default=80.0)
    waterfall_parser.add_argument("--floor_db", type=float, default=-120.0)

    waterfall_parser.add_argument("--smoothing_log_bins", type=int, default=0)
    waterfall_parser.add_argument("--log_bins_per_octave", type=int, default=96)

    # 3D view
    waterfall_parser.add_argument("--elev_deg", type=float, default=30.0)
    waterfall_parser.add_argument("--azim_deg", type=float, default=-60.0)

    # 2D ridges
    waterfall_parser.add_argument("--ridge_offset_db", type=float, default=6.0)

    # ------------------------------------------------------------------
    # Modal cloud
    # ------------------------------------------------------------------
    modalcloud_parser = subparsers.add_parser(
        "modalcloud",
        help="Modal cloud: frequency vs RT60 points from per-bin STFT decay fits.",
    )

    modalcloud_parser.add_argument("--input", dest="input_wav_file_path", type=str, required=True)
    modalcloud_parser.add_argument("--output", dest="output_basename", type=str, default=None)
    modalcloud_parser.add_argument("--no_show", action="store_true")

    modalcloud_parser.add_argument("--mono", dest="use_mono_downmix", action="store_true")
    modalcloud_parser.add_argument("--trim_to_peak", action=argparse.BooleanOptionalAction, default=True)
    modalcloud_parser.add_argument("--ignore-leading", dest="ignore_leading_seconds", type=float, default=0.0)
    modalcloud_parser.add_argument("--duration", dest="analysis_duration_seconds", type=float, default=None)

    modalcloud_parser.add_argument("--n_fft", type=int, default=8192)
    modalcloud_parser.add_argument("--hop_length", type=int, default=512)
    modalcloud_parser.add_argument("--no_hann_window", action="store_true")

    modalcloud_parser.add_argument("--f_min_hz", type=float, default=20.0)
    modalcloud_parser.add_argument("--f_max_hz", type=float, default=20000.0)

    modalcloud_parser.add_argument(
        "--metric",
        type=str,
        choices=["t30", "t20", "edt"],
        default="t30",
        help="Which decay fit window to use per bin (default: t30).",
    )

    modalcloud_parser.add_argument("--log_bins_per_octave", type=int, default=24)
    modalcloud_parser.add_argument("--min_bins", type=int, default=24)

    modalcloud_parser.add_argument("--fit_lower_limit_db", type=float, default=-80.0)
    modalcloud_parser.add_argument("--min_fit_points", type=int, default=10)
    modalcloud_parser.add_argument("--min_peak_db_above_floor", type=float, default=20.0)
    modalcloud_parser.add_argument("--floor_db", type=float, default=-120.0)

    modalcloud_parser.add_argument("--show_median_curve", action=argparse.BooleanOptionalAction, default=True)
    modalcloud_parser.add_argument("--median_octave_window", type=float, default=0.25)
    modalcloud_parser.add_argument("--ylim_seconds_min", type=float, default=None)
    modalcloud_parser.add_argument("--ylim_seconds_max", type=float, default=None)


    # ------------------------------------------------------------------
    # Report (run a standard suite and write plots + text summary)
    # ------------------------------------------------------------------
    report_parser = subparsers.add_parser(
        "report",
        help="Run a standard analysis suite and write plots + a text summary.",
    )
    report_parser.add_argument("--input", dest="input_wav_file_path", type=str, required=True)
    report_parser.add_argument(
        "--output",
        dest="output_basename",
        type=str,
        required=True,
        help=(
            "Output basename/prefix (folder + base name). "
            "Example: --output plots/my_ir_report "
            "will write plots/my_ir_report_<analysis>.png and plots/my_ir_report_report.txt"
        ),
    )

    # Common report controls (propagated into sub-analyses):
    report_parser.add_argument("--mono", dest="use_mono_downmix", action="store_true", help="Downmix stereo to mono for analysis.")
    report_parser.add_argument("--trim_to_peak", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--ignore_leading_seconds", type=float, default=0.0)

    # Enable/disable blocks:
    report_parser.add_argument("--ir", dest="run_ir", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--decay", dest="run_decay", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--rt60bands", dest="run_rt60bands", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--fr", dest="run_fr", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--gd", dest="run_gd", action=argparse.BooleanOptionalAction, default=True, help="Group delay vs frequency")
    report_parser.add_argument("--spectrogram", dest="run_spectrogram", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--waterfall", dest="run_waterfall", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--diffusion", dest="run_diffusion", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--modalcloud", dest="run_modalcloud", action=argparse.BooleanOptionalAction, default=True)
    report_parser.add_argument("--echodensity", dest="run_echodensity", action=argparse.BooleanOptionalAction, default=True)


    return top_level_parser.parse_args()


def run_impulse_response_command(parsed_arguments: argparse.Namespace) -> None:
    settings = ImpulseResponseViewSettings(
        early_window_seconds=float(parsed_arguments.early_window_seconds),
        log_magnitude_floor_db=float(parsed_arguments.log_magnitude_floor_db),
        use_mono_downmix=bool(parsed_arguments.use_mono_downmix),
    )

    show_interactive = not bool(parsed_arguments.no_show)

    output_basename: Optional[str] = parsed_arguments.output_basename
    if output_basename is not None:
        output_basename = str(Path(output_basename))

    plot_ir_from_wav_file(
        wav_file_path=str(parsed_arguments.input_wav_file_path),
        settings=settings,
        output_basename=output_basename,
        show_interactive=show_interactive,
    )


def main() -> None:
    parsed_arguments = parse_arguments()

    command_name = str(parsed_arguments.command_name)

    if command_name == "ir":
        run_impulse_response_command(parsed_arguments)
        return
    
    elif command_name == "deconvolve":
        output_path: Optional[str] = parsed_arguments.output_ir_wav_file_path
        if output_path is None:
            output_path = str(default_output_ir_path(parsed_arguments.recorded_wav_file_path))
        else:
            output_path = str(Path(output_path))

        settings = DeconvolveSettings(
            regularization_relative=float(parsed_arguments.regularization_relative),
            normalise_peak=bool(parsed_arguments.normalise_peak),
            target_peak=float(parsed_arguments.target_peak),
            remove_dc=bool(parsed_arguments.remove_dc),
            output_length_mode=str(parsed_arguments.output_length_mode),
        )

        result = deconvolve_from_wav_files(
            recorded_wav_file_path=str(parsed_arguments.recorded_wav_file_path),
            sweep_wav_file_path=str(parsed_arguments.sweep_wav_file_path),
            settings=settings,
            output_ir_wav_file_path=output_path,
        )

        # Deterministic, CLI-friendly confirmation
        print(f"Wrote IR WAV: {output_path}")
        print(f"  sample_rate_hz={result.sample_rate_hz}")
        print(f"  channels={result.samples.shape[1]}")
        print(f"  length_seconds={result.samples.shape[0] / float(result.sample_rate_hz):.3f}")
        return


    elif command_name == "decay":
        analysis_settings = DecayAnalysisSettings(
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            edc_floor_db=float(parsed_arguments.edc_floor_db),
            fit_lower_limit_db=float(parsed_arguments.fit_lower_limit_db),
            edc_smoothing_window_samples=int(parsed_arguments.edc_smoothing_window_samples),
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            compute_edt=bool(parsed_arguments.compute_edt),
        )

        plot_settings = DecayPlotSettings()

        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        results = plot_decay_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_decay_results_text(results))
        return
    
    elif command_name == "rt60bands":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        decay_settings = DecayAnalysisSettings(
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            edc_floor_db=float(parsed_arguments.edc_floor_db),
            fit_lower_limit_db=float(parsed_arguments.fit_lower_limit_db),
            edc_smoothing_window_samples=int(parsed_arguments.edc_smoothing_window_samples),
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            # EDT fit window exists in decay settings; include_edt flag controls whether it is computed per band
            compute_edt=bool(parsed_arguments.include_edt),
        )

        rt_settings = Rt60BandsAnalysisSettings(
            band_mode=str(parsed_arguments.band_mode),
            low_upper_hz=float(parsed_arguments.low_upper_hz),
            mid_center_hz=float(parsed_arguments.mid_center_hz),
            mid_width_octaves=float(parsed_arguments.mid_width_octaves),
            high_lower_hz=float(parsed_arguments.high_lower_hz),
            f_min_hz=float(parsed_arguments.f_min_hz),
            f_max_hz=float(parsed_arguments.f_max_hz),
            transition_width_octaves=float(parsed_arguments.transition_width_octaves),
            include_t20=bool(parsed_arguments.include_t20),
            include_edt=bool(parsed_arguments.include_edt),
            decay_settings=decay_settings,
        )

        if parsed_arguments.legend_values is None:
            legend_values = (str(parsed_arguments.band_mode) == "three")
        else:
            legend_values = bool(parsed_arguments.legend_values)

        plot_settings = Rt60BandsPlotSettings(legend_values=legend_values)

        results = plot_rt60_bands_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            settings=rt_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_rt60_bands_results_text(results, include_t20=rt_settings.include_t20, include_edt=rt_settings.include_edt))
        return
    
    elif command_name == "fr":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        analysis_settings = FrequencyResponseAnalysisSettings(
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            analysis_duration_seconds=parsed_arguments.analysis_duration_seconds,
            use_hann_window=not bool(parsed_arguments.no_hann_window),
            magnitude_floor_db=float(parsed_arguments.magnitude_floor_db),
            f_min_hz=float(parsed_arguments.f_min_hz),
            f_max_hz=float(parsed_arguments.f_max_hz),
            smoothing_log_bins=int(parsed_arguments.smoothing_log_bins),
            log_bins_per_octave=int(parsed_arguments.log_bins_per_octave),
        )

        plot_settings = FrequencyResponsePlotSettings()

        results = plot_frequency_response_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_frequency_response_results_text(results))
        return

    elif command_name == "filter":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        analysis_settings = FilterAnalysisSettings(
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            analysis_duration_seconds=parsed_arguments.analysis_duration_seconds,
            use_hann_window=not bool(parsed_arguments.no_hann_window),
            magnitude_floor_db=float(parsed_arguments.magnitude_floor_db),
            f_min_hz=float(parsed_arguments.f_min_hz),
            f_max_hz=float(parsed_arguments.f_max_hz),
            phase_mode=str(parsed_arguments.phase_mode),
            unwrap_phase=not bool(parsed_arguments.no_unwrap_phase),
        )

        plot_settings = FilterPlotSettings()

        results = plot_filter_response_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_filter_response_results_text(results))
        return

    elif command_name == "spectrogram":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        dynamic_range_db = float(parsed_arguments.dynamic_range_db)
        if dynamic_range_db <= 0.0:
            dynamic_range_db_value = None
        else:
            dynamic_range_db_value = dynamic_range_db

        analysis_settings = SpectrogramAnalysisSettings(
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            analysis_duration_seconds=parsed_arguments.analysis_duration_seconds,
            n_fft=int(parsed_arguments.n_fft),
            hop_length=int(parsed_arguments.hop_length),
            use_hann_window=not bool(parsed_arguments.no_hann_window),
            floor_db=float(parsed_arguments.floor_db),
            f_min_hz=float(parsed_arguments.f_min_hz),
            f_max_hz=float(parsed_arguments.f_max_hz),
            dynamic_range_db=dynamic_range_db_value,
        )

        plot_settings = SpectrogramPlotSettings()

        results = plot_spectrogram_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_spectrogram_results_text(results))
        return

    elif command_name == "diffusion":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        settings = DiffusionAnalysisSettings(
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            window_seconds=float(parsed_arguments.window_seconds),
            hop_seconds=float(parsed_arguments.hop_seconds),
            max_lag_milliseconds=float(parsed_arguments.max_lag_milliseconds),
            echo_density_threshold_rms=float(parsed_arguments.echo_density_threshold_rms),
            echo_density_normalise_to_gaussian=bool(parsed_arguments.echo_density_normalise_to_gaussian),
        )

        results = plot_diffusion_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            analysis_settings=settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_diffusion_results_text(results))
        return

    
    elif command_name == "waterfall":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        analysis_settings = WaterfallAnalysisSettings(
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            analysis_duration_seconds=parsed_arguments.analysis_duration_seconds,
            n_fft=int(parsed_arguments.n_fft),
            hop_length=int(parsed_arguments.hop_length),
            use_hann_window=not bool(parsed_arguments.no_hann_window),
            f_min_hz=float(parsed_arguments.f_min_hz),
            f_max_hz=float(parsed_arguments.f_max_hz),
            slice_mode=str(parsed_arguments.slice_mode),
            num_slices=int(parsed_arguments.num_slices),
            slice_spacing_seconds=float(parsed_arguments.slice_spacing_seconds),
            start_time_seconds=float(parsed_arguments.start_time_seconds),
            end_time_seconds=parsed_arguments.end_time_seconds,
            db_reference=str(parsed_arguments.db_reference),
            smoothing_log_bins=int(parsed_arguments.smoothing_log_bins),
            log_bins_per_octave=int(parsed_arguments.log_bins_per_octave),
            dynamic_range_db=float(parsed_arguments.dynamic_range_db),
            floor_db=float(parsed_arguments.floor_db),
        )

        plot_settings = WaterfallPlotSettings(
            style=str(parsed_arguments.style),
            elev_deg=float(parsed_arguments.elev_deg),
            azim_deg=float(parsed_arguments.azim_deg),
            ridge_offset_db=float(parsed_arguments.ridge_offset_db),
        )

        results = plot_waterfall_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_waterfall_results_text(results))
        return

    elif command_name == "modalcloud":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        analysis_settings = ModalCloudAnalysisSettings(
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            analysis_duration_seconds=parsed_arguments.analysis_duration_seconds,
            n_fft=int(parsed_arguments.n_fft),
            hop_length=int(parsed_arguments.hop_length),
            use_hann_window=not bool(parsed_arguments.no_hann_window),
            f_min_hz=float(parsed_arguments.f_min_hz),
            f_max_hz=float(parsed_arguments.f_max_hz),
            log_bins_per_octave=int(parsed_arguments.log_bins_per_octave),
            min_bins=int(parsed_arguments.min_bins),
            metric=str(parsed_arguments.metric),
            fit_lower_limit_db=float(parsed_arguments.fit_lower_limit_db),
            min_fit_points=int(parsed_arguments.min_fit_points),
            min_peak_db_above_floor=float(parsed_arguments.min_peak_db_above_floor),
            floor_db=float(parsed_arguments.floor_db),
        )

        ylim = None
        if parsed_arguments.ylim_seconds_min is not None and parsed_arguments.ylim_seconds_max is not None:
            ylim = (float(parsed_arguments.ylim_seconds_min), float(parsed_arguments.ylim_seconds_max))

        plot_settings = ModalCloudPlotSettings(
            show_median_curve=bool(parsed_arguments.show_median_curve),
            median_octave_window=float(parsed_arguments.median_octave_window),
            ylim_seconds=ylim,
        )

        results = plot_modal_cloud_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            analysis_settings=analysis_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_modal_cloud_results_text(results))
        return


    


    elif command_name == "zplane":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        analysis_settings = ZPlaneAnalysisSettings(
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix_for_stereo),
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            analysis_duration_seconds=parsed_arguments.analysis_duration_seconds,
            ar_order=int(parsed_arguments.ar_order),
            derive_zeros=bool(parsed_arguments.derive_zeros),
            zero_order=int(parsed_arguments.zero_order),
            ridge_lambda=float(parsed_arguments.ridge_lambda),
        )

        plot_settings = ZPlanePlotSettings(
            limit_radius=float(parsed_arguments.limit_radius),
        )

        results = plot_zplane_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            settings=analysis_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_zplane_results_text(results))
        return

    elif command_name == "groupdelay":
        show_interactive = not bool(parsed_arguments.no_show)

        output_basename: Optional[str] = parsed_arguments.output_basename
        if output_basename is not None:
            output_basename = str(Path(output_basename))

        analysis_settings = GroupDelayAnalysisSettings(
            use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix_for_stereo),
            trim_to_peak=bool(parsed_arguments.trim_to_peak),
            ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            analysis_duration_seconds=parsed_arguments.analysis_duration_seconds,
            fft_size=parsed_arguments.fft_size,
            smoothing_bins=int(parsed_arguments.smoothing_bins),
            f_min_hz=float(parsed_arguments.f_min_hz),
            f_max_hz=float(parsed_arguments.f_max_hz),
        )

        plot_settings = GroupDelayPlotSettings()

        results = plot_group_delay_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            settings=analysis_settings,
            plot_settings=plot_settings,
            output_basename=output_basename,
            show_interactive=show_interactive,
        )

        print(summarise_group_delay_results_text(results))
        return

    if command_name == "report":
        output_basename = str(Path(parsed_arguments.output_basename))

        report_settings = ReportSettings(
            common_use_mono_downmix_for_stereo=bool(parsed_arguments.use_mono_downmix),
            common_trim_to_peak=bool(parsed_arguments.trim_to_peak),
            common_ignore_leading_seconds=float(parsed_arguments.ignore_leading_seconds),
            run_impulse_response_plots=bool(parsed_arguments.run_ir),
            run_decay=bool(parsed_arguments.run_decay),
            run_rt60_bands=bool(parsed_arguments.run_rt60bands),
            run_frequency_response=bool(parsed_arguments.run_fr),
            run_group_delay=bool(parsed_arguments.run_gd),
            run_spectrogram=bool(parsed_arguments.run_spectrogram),
            run_waterfall=bool(parsed_arguments.run_waterfall),
            run_diffusion=bool(parsed_arguments.run_diffusion),
            run_modal_cloud=bool(parsed_arguments.run_modalcloud),
            run_echo_density=bool(parsed_arguments.run_echodensity),
        )

        results = run_report_from_wav_file(
            input_wav_file_path=str(parsed_arguments.input_wav_file_path),
            output_basename=output_basename,
            settings=report_settings,
        )

        print(results.summary_markdown)
        print(f"Wrote: {results.summary_markdown_path}")
        return

    
    if command_name == "bundle":
        settings = BundleRunSettings(reports_subdir=str(parsed_arguments.reports_subdir))
        index = run_bundle_report(str(parsed_arguments.bundle_root), settings=settings)
        print(f"Wrote bundle report index: {index}")
        return

    raise ValueError(f"Unknown command: {command_name}")


if __name__ == "__main__":
    main()

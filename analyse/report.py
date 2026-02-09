"""
Report mode: run a standard suite of offline analyses on an input WAV (usually an IR).

Design goals (consistent with the rest of this project):
- deterministic, CLI-friendly output
- readability over performance
- one entry point that produces a folder full of plots + a Markdown summary
- reuses existing per-analysis modules (no duplicate maths)

Conventions:
- output_basename points to the desired output prefix, e.g. "plots/my_ir"
- each analysis module appends its own suffix when saving PNGs
- report summary is written to "<output_basename>_report.md"
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence, Any, Dict, List

from analyse.io import load_wav_file, DEFAULT_EXPECTED_SAMPLE_RATE_HZ

from analyse.impulse_response import ImpulseResponseViewSettings, plot_ir_from_wav_file

from analyse.decay import (
    DecayAnalysisSettings,
    DecayPlotSettings,
    plot_decay_from_wav_file,
    summarise_decay_results_text,
    ChannelDecayAnalysis,
)

from analyse.rt60bands import (
    Rt60BandsAnalysisSettings,
    Rt60BandsPlotSettings,
    plot_rt60_bands_from_wav_file,
    summarise_rt60_bands_results_text,
    Rt60BandsChannelResult,
)

from analyse.frequency_response import (
    FrequencyResponseAnalysisSettings,
    FrequencyResponsePlotSettings,
    plot_frequency_response_from_wav_file,
    summarise_frequency_response_results_text,
    ChannelFrequencyResponse,
)


from analyse.group_delay import (
    GroupDelayAnalysisSettings,
    GroupDelayPlotSettings,
    plot_group_delay_from_wav_file,
    summarise_group_delay_results_text,
)

from analyse.spectrogram import (
    SpectrogramAnalysisSettings,
    SpectrogramPlotSettings,
    plot_spectrogram_from_wav_file,
    summarise_spectrogram_results_text,
    ChannelSpectrogramResult,
)

from analyse.waterfall import (
    WaterfallAnalysisSettings,
    WaterfallPlotSettings,
    plot_waterfall_from_wav_file,
    summarise_waterfall_results_text,
    ChannelWaterfallResult,
)

from analyse.diffusion import (
    DiffusionAnalysisSettings,
    plot_diffusion_from_wav_file,
    summarise_diffusion_results_text,
    DiffusionChannelResult,
)

from analyse.modalcloud import (
    ModalCloudAnalysisSettings,
    ModalCloudPlotSettings,
    plot_modal_cloud_from_wav_file,
    summarise_modal_cloud_results_text,
    ChannelModalCloudResult,
)


# --------------------------------------------------------------------------------------
# Settings + results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportSettings:
    # Common analysis controls:
    common_use_mono_downmix_for_stereo: bool = False
    common_trim_to_peak: bool = True
    common_ignore_leading_seconds: float = 0.0

    # Enable/disable blocks:
    run_impulse_response_plots: bool = True
    run_decay: bool = True
    run_rt60_bands: bool = True
    run_frequency_response: bool = True
    run_group_delay: bool = True
    run_spectrogram: bool = True
    run_waterfall: bool = True
    run_diffusion: bool = True
    run_modal_cloud: bool = True
    run_echo_density: bool = True

    expected_sample_rate_hz: int = DEFAULT_EXPECTED_SAMPLE_RATE_HZ

    ir_view_settings: Optional[ImpulseResponseViewSettings] = None
    decay_analysis_settings: Optional[DecayAnalysisSettings] = None
    decay_plot_settings: Optional[DecayPlotSettings] = None
    rt60_bands_settings: Optional[Rt60BandsAnalysisSettings] = None
    rt60_bands_plot_settings: Optional[Rt60BandsPlotSettings] = None
    frequency_response_analysis_settings: Optional[FrequencyResponseAnalysisSettings] = None
    frequency_response_plot_settings: Optional[FrequencyResponsePlotSettings] = None
    group_delay_analysis_settings: Optional[GroupDelayAnalysisSettings] = None
    group_delay_plot_settings: Optional[GroupDelayPlotSettings] = None
    spectrogram_analysis_settings: Optional[SpectrogramAnalysisSettings] = None
    spectrogram_plot_settings: Optional[SpectrogramPlotSettings] = None
    waterfall_analysis_settings: Optional[WaterfallAnalysisSettings] = None
    waterfall_plot_settings: Optional[WaterfallPlotSettings] = None
    diffusion_analysis_settings: Optional[DiffusionAnalysisSettings] = None
    modal_cloud_analysis_settings: Optional[ModalCloudAnalysisSettings] = None
    modal_cloud_plot_settings: Optional[ModalCloudPlotSettings] = None


@dataclass(frozen=True)
class ReportResults:
    input_wav_file_path: Path
    output_basename: Path
    summary_markdown_path: Path
    summary_markdown: str


# --------------------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------------------


def _md_section(title: str) -> str:
    return f"\n## {title}\n\n"


def _md_codeblock(text: str) -> str:
    text = text.strip()
    if not text:
        return "_(no output)_\n"
    return f"```text\n{text}\n```\n"


def _md_image(basename: Path, suffix: str, alt_text: str = "") -> str:
    """Create markdown image link for PNG file."""
    filename = f"{basename.name}{suffix}.png"
    if not alt_text:
        alt_text = filename
    return f"![{alt_text}]({filename})\n\n"


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _apply_common_overrides(settings_obj: Any, report_settings: ReportSettings) -> Any:
    if settings_obj is None:
        return None

    field_names = {f.name for f in dataclasses.fields(settings_obj)}
    kwargs: Dict[str, Any] = {}

    if "use_mono_downmix_for_stereo" in field_names:
        kwargs["use_mono_downmix_for_stereo"] = report_settings.common_use_mono_downmix_for_stereo
    if "trim_to_peak" in field_names:
        kwargs["trim_to_peak"] = report_settings.common_trim_to_peak
    if "ignore_leading_seconds" in field_names:
        kwargs["ignore_leading_seconds"] = report_settings.common_ignore_leading_seconds

    return replace(settings_obj, **kwargs) if kwargs else settings_obj


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_header_block(input_wav_file_path: Path, expected_sample_rate_hz: int) -> str:
    loaded = load_wav_file(
        input_wav_file_path,
        expected_sample_rate_hz=expected_sample_rate_hz,
        expected_channel_mode="stereo",
        allow_mono_and_upmix_to_stereo=True,
    )

    n_samples = int(loaded.samples.shape[0])
    sr = int(loaded.sample_rate_hz)
    ch = int(loaded.samples.shape[1])
    duration_seconds = n_samples / sr if sr > 0 else 0.0

    return (
        "# Offline Reverb Analysis Report\n\n"
        f"**Input WAV:** `{input_wav_file_path}`  \n"
        f"**Sample rate:** {sr} Hz (expected {expected_sample_rate_hz} Hz)  \n"
        f"**Channels:** {ch}  \n"
        f"**Samples:** {n_samples}  \n"
        f"**Duration:** {duration_seconds:.6f} s\n\n"
        "---\n"
    )


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def run_report_from_wav_file(
    input_wav_file_path: str | Path,
    output_basename: str | Path,
    settings: Optional[ReportSettings] = None,
) -> ReportResults:
    if settings is None:
        settings = ReportSettings()

    input_wav_file_path = Path(input_wav_file_path)
    output_basename = Path(output_basename)
    _ensure_parent_dir(output_basename)

    md_parts: List[str] = []
    md_parts.append(_format_header_block(input_wav_file_path, settings.expected_sample_rate_hz))

    if settings.run_impulse_response_plots:
        ir_settings = _apply_common_overrides(
            settings.ir_view_settings or ImpulseResponseViewSettings(), settings
        )
        plot_ir_from_wav_file(
            wav_file_path=input_wav_file_path,
            settings=ir_settings,
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("Impulse response"))
        md_parts.append(_md_image(output_basename, "", "Impulse response overview"))
        md_parts.append(_md_image(output_basename, "_early", "Early reflections"))
        md_parts.append(_md_image(output_basename, "_tail", "Tail (log magnitude)"))

    if settings.run_decay:
        decay_results = plot_decay_from_wav_file(
            input_wav_file_path=input_wav_file_path,
            analysis_settings=_apply_common_overrides(
                settings.decay_analysis_settings or DecayAnalysisSettings(), settings
            ),
            plot_settings=settings.decay_plot_settings or DecayPlotSettings(),
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("Decay / EDC"))
        md_parts.append(_md_image(output_basename, "_decay", "Decay analysis (T20/T30/RT60/EDT)"))
        md_parts.append(_md_codeblock(summarise_decay_results_text(decay_results)))

    if settings.run_rt60_bands:
        rt60_settings = _apply_common_overrides(
            settings.rt60_bands_settings or Rt60BandsAnalysisSettings(), settings
        )
        rt60_results = plot_rt60_bands_from_wav_file(
            input_wav_file_path=input_wav_file_path,
            settings=rt60_settings,
            plot_settings=settings.rt60_bands_plot_settings or Rt60BandsPlotSettings(),
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("RT60 by band"))
        md_parts.append(_md_image(output_basename, "_rt60bands", "RT60 by frequency band"))
        md_parts.append(
            _md_codeblock(
                summarise_rt60_bands_results_text(
                    rt60_results,
                    include_t20=bool(rt60_settings.include_t20),
                    include_edt=bool(rt60_settings.include_edt),
                )
            )
        )

    if settings.run_frequency_response:
        fr_results = plot_frequency_response_from_wav_file(
            input_wav_file_path=input_wav_file_path,
            analysis_settings=_apply_common_overrides(
                settings.frequency_response_analysis_settings
                or FrequencyResponseAnalysisSettings(),
                settings,
            ),
            plot_settings=settings.frequency_response_plot_settings
            or FrequencyResponsePlotSettings(),
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("Frequency response"))
        md_parts.append(_md_image(output_basename, "_fr", "Frequency response spectrum"))
        md_parts.append(_md_codeblock(summarise_frequency_response_results_text(fr_results)))

    
    if settings.run_group_delay:
        gd_results = plot_group_delay_from_wav_file(
            input_wav_file_path=input_wav_file_path,
            settings=_apply_common_overrides(
                settings.group_delay_analysis_settings or GroupDelayAnalysisSettings(),
                settings,
            ),
            plot_settings=settings.group_delay_plot_settings or GroupDelayPlotSettings(),
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("Group delay"))
        md_parts.append(_md_image(output_basename, "_groupdelay", "Group delay vs frequency"))
        md_parts.append(_md_codeblock(summarise_group_delay_results_text(gd_results)))

    if settings.run_spectrogram:
        spec_results = plot_spectrogram_from_wav_file(
            input_wav_file_path=input_wav_file_path,
            analysis_settings=_apply_common_overrides(
                settings.spectrogram_analysis_settings or SpectrogramAnalysisSettings(),
                settings,
            ),
            plot_settings=settings.spectrogram_plot_settings or SpectrogramPlotSettings(),
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("Spectrogram"))
        md_parts.append(_md_image(output_basename, "_spectrogram_left", "Spectrogram (left)"))
        if not settings.common_use_mono_downmix_for_stereo:
            md_parts.append(_md_image(output_basename, "_spectrogram_right", "Spectrogram (right)"))
        md_parts.append(_md_codeblock(summarise_spectrogram_results_text(spec_results)))

    if settings.run_waterfall:
        wf_results = plot_waterfall_from_wav_file(
            input_wav_file_path=input_wav_file_path,
            analysis_settings=_apply_common_overrides(
                settings.waterfall_analysis_settings or WaterfallAnalysisSettings(),
                settings,
            ),
            plot_settings=settings.waterfall_plot_settings or WaterfallPlotSettings(),
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("Waterfall"))
        md_parts.append(_md_image(output_basename, "_waterfall_left", "Waterfall plot (left)"))
        if not settings.common_use_mono_downmix_for_stereo:
            md_parts.append(_md_image(output_basename, "_waterfall_right", "Waterfall plot (right)"))
        md_parts.append(_md_codeblock(summarise_waterfall_results_text(wf_results)))

    if settings.run_diffusion:
        diff_results = plot_diffusion_from_wav_file(
            input_wav_file_path=input_wav_file_path,
            analysis_settings=_apply_common_overrides(
                settings.diffusion_analysis_settings
                or DiffusionAnalysisSettings(hop_seconds=0.05, max_lag_milliseconds=5.0),
                settings,
            ),
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("Diffusion / echo density proxy"))
        md_parts.append(_md_image(output_basename, "_diffusion", "Diffusion metrics over time"))
        md_parts.append(_md_codeblock(summarise_diffusion_results_text(diff_results)))

    if settings.run_modal_cloud:
        modal_results = plot_modal_cloud_from_wav_file(
            input_wav_file_path=input_wav_file_path,
            analysis_settings=_apply_common_overrides(
                settings.modal_cloud_analysis_settings or ModalCloudAnalysisSettings(),
                settings,
            ),
            plot_settings=settings.modal_cloud_plot_settings or ModalCloudPlotSettings(),
            output_basename=output_basename,
            show_interactive=False,
        )
        md_parts.append(_md_section("Modal cloud"))
        md_parts.append(_md_image(output_basename, "_modalcloud_left", "Modal cloud (left)"))
        if not settings.common_use_mono_downmix_for_stereo:
            md_parts.append(_md_image(output_basename, "_modalcloud_right", "Modal cloud (right)"))
        md_parts.append(_md_codeblock(summarise_modal_cloud_results_text(modal_results)))

    summary_markdown = "".join(md_parts).rstrip() + "\n"
    summary_path = Path(f"{output_basename}_report.md")
    _ensure_parent_dir(summary_path)
    summary_path.write_text(summary_markdown, encoding="utf-8")

    return ReportResults(
        input_wav_file_path=input_wav_file_path,
        output_basename=output_basename,
        summary_markdown_path=summary_path,
        summary_markdown=summary_markdown,
    )

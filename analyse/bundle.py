# analyse/bundle.py
"""
Bundle runner: analyse a folder containing:
  meta.json
  taps/*.wav

It runs the existing report pipeline on each tap WAV and writes an index Markdown file.

Bundle layout (from C++ harness):
  <bundle_root>/
    meta.json
    taps/<tapname>.wav

Outputs:
  <bundle_root>/reports/<tapname>/...plots...
  <bundle_root>/reports/bundle_report.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from analyse.report import ReportSettings, run_report_from_wav_file


@dataclass(frozen=True)
class BundleRunSettings:
    reports_subdir: str = "reports"
    report_settings: Optional[ReportSettings] = None


def run_bundle_report(bundle_root: str | Path, settings: Optional[BundleRunSettings] = None) -> Path:
    if settings is None:
        settings = BundleRunSettings()

    bundle_root = Path(bundle_root)
    meta_path = bundle_root / "meta.json"
    taps_dir = bundle_root / "taps"

    meta = json.loads(meta_path.read_text())
    tap_names: List[str] = list(meta.get("taps", []))

    reports_root = bundle_root / settings.reports_subdir
    reports_root.mkdir(parents=True, exist_ok=True)

    index_lines: List[str] = []
    index_lines.append("# IR Bundle Report\n")
    index_lines.append(f"**Bundle:** `{bundle_root}`\n")
    index_lines.append(f"**Sample rate:** {meta.get('sample_rate_hz')}\n")
    index_lines.append(f"**Length (samples):** {meta.get('length_samples')}\n")
    index_lines.append("\n## Taps\n")

    for tap in tap_names:
        wav_path = taps_dir / f"{tap}.wav"
        out_dir = reports_root / tap
        out_dir.mkdir(parents=True, exist_ok=True)

        output_basename = out_dir / tap

        run_report_from_wav_file(
            input_wav_file_path=wav_path,
            output_basename=output_basename,
            settings=settings.report_settings,
        )

        report_md = out_dir / f"{tap}_report.md"
        index_lines.append(f"- [{tap}]({settings.reports_subdir}/{tap}/{report_md.name})")

    index_path = reports_root / "bundle_report.md"
    index_path.write_text("\n".join(index_lines) + "\n")
    return index_path

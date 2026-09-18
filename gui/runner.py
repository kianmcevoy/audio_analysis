# gui/runner.py
"""
Runs analyse.cli subcommands as subprocesses and collects the PNGs they write.

Each run writes into its own directory with the fixed basename "out", so an
output file "out_tail.png" has the suffix "_tail". The GUI uses suffixes to
line up rows across files.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_STEM = "out"


@dataclass(frozen=True)
class AnalysisSpec:
    command: str
    label: str
    no_show_flag: str


# The CLI spells its no-show flag inconsistently across subcommands.
ANALYSES: List[AnalysisSpec] = [
    AnalysisSpec("ir", "Impulse response", "--no_show"),
    AnalysisSpec("decay", "Decay (EDC / RT60)", "--no_show"),
    AnalysisSpec("rt60bands", "RT60 bands", "--no_show"),
    AnalysisSpec("fr", "Frequency response", "--no_show"),
    AnalysisSpec("filter", "Filter (mag + phase)", "--no_show"),
    AnalysisSpec("groupdelay", "Group delay", "--no-show"),
    AnalysisSpec("spectrogram", "Spectrogram", "--no_show"),
    AnalysisSpec("diffusion", "Diffusion", "--no_show"),
    AnalysisSpec("waterfall", "Waterfall (CSD)", "--no_show"),
    AnalysisSpec("modalcloud", "Modal cloud", "--no_show"),
    AnalysisSpec("zplane", "Z-plane", "--no-show"),
]


@dataclass
class AnalysisRun:
    wav_path: Path
    ok: bool
    # suffix (e.g. "", "_tail", "_spectrogram_left") -> PNG path
    images: Dict[str, Path] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""


def run_analysis(spec: AnalysisSpec, wav_path: Path, out_dir: Path, mono: bool) -> AnalysisRun:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    completed = _run_cli(spec.command, wav_path, out_dir / OUTPUT_STEM, mono, [spec.no_show_flag])

    images = {
        png.stem[len(OUTPUT_STEM):]: png
        for png in sorted(out_dir.glob(f"{OUTPUT_STEM}*.png"))
    }
    stderr = completed.stderr
    if completed.returncode == 0 and not images:
        stderr = stderr or "No plots were written."

    return AnalysisRun(
        wav_path=wav_path,
        ok=completed.returncode == 0 and bool(images),
        images=images,
        stdout=completed.stdout,
        stderr=stderr,
    )


@dataclass
class ReportRun:
    wav_path: Path
    ok: bool
    report_path: Path
    stdout: str = ""
    stderr: str = ""


def run_report(wav_path: Path, out_dir: Path, mono: bool) -> ReportRun:
    """Full `analyse.cli report` suite: writes <out_dir>/<stem>_*.png and <stem>_report.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_basename = out_dir / wav_path.stem
    completed = _run_cli("report", wav_path, output_basename, mono, [])
    report_path = Path(f"{output_basename}_report.md")
    ok = completed.returncode == 0 and report_path.exists()
    if not ok and not any(out_dir.iterdir()):
        out_dir.rmdir()
    return ReportRun(
        wav_path=wav_path,
        ok=ok,
        report_path=report_path,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _run_cli(command: str, wav_path: Path, output_basename: Path, mono: bool, extra: List[str]) -> subprocess.CompletedProcess:
    args = [
        sys.executable, "-m", "analyse.cli", command,
        "--input", str(wav_path),
        "--output", str(output_basename),
        *extra,
    ]
    if mono:
        args.append("--mono")
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, env=_subprocess_env())


def _subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"  # never pop up windows from the child
    return env

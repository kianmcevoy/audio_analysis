# gui/app.py
"""
Tk window: file list + one button per analysis on the left, composite plot on
the right (one column per file, one row per output image), CLI summaries below.

Plots are produced by the unmodified analyse.cli commands (see gui/runner.py);
this module only arranges the resulting PNGs for side-by-side comparison.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from gui.runner import ANALYSES, AnalysisRun, AnalysisSpec, ReportRun, run_analysis, run_report

# Tk sets the X11 WM_CLASS to this (capitalised); the .desktop StartupWMClass
# written by analyse_gui.sh must match so the dock uses the launcher's icon.
WINDOW_CLASS_NAME = "audio-analysis"
ICON_PATH = Path(__file__).resolve().parent / "icon.png"

EXPORT_DPI_MIN = 100
EXPORT_DPI_MAX = 600


class AnalysisApp:
    def __init__(self, root: tk.Tk, initial_files: List[Path]) -> None:
        self.root = root
        self.root.title("Audio Analysis")
        self.root.geometry("1400x900")

        self.session_dir = tempfile.TemporaryDirectory(prefix="analyse_gui_")
        self.files: List[Path] = []
        self.last_dir = str(Path.cwd())
        self.last_spec: Optional[AnalysisSpec] = None
        self.last_runs: List[AnalysisRun] = []
        self.busy_widgets: List[ttk.Widget] = []

        self.mono = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Add one or more WAV files, then pick an analysis.")

        self._build_layout()
        self._add_files(initial_files)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(panes, padding=8)
        right = ttk.Frame(panes)
        panes.add(left, weight=0)
        panes.add(right, weight=1)

        self._build_file_panel(left)
        self._build_analysis_panel(left)
        self._build_export_panel(left)
        self._build_plot_panel(right)

        ttk.Label(self.root, textvariable=self.status, anchor=tk.W, padding=(8, 2)).pack(fill=tk.X, side=tk.BOTTOM)

    def _build_file_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Files", padding=6)
        frame.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.file_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, width=36, height=8)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scroll.set)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(6, 0))
        for text, command in (("Add files…", self._on_add_files), ("Remove", self._on_remove), ("Clear", self._on_clear)):
            button = ttk.Button(buttons, text=text, command=command)
            button.pack(side=tk.LEFT, expand=True, fill=tk.X)
            self.busy_widgets.append(button)

        mono_check = ttk.Checkbutton(frame, text="Mono downmix", variable=self.mono)
        mono_check.pack(anchor=tk.W, pady=(6, 0))
        self.busy_widgets.append(mono_check)

    def _build_analysis_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Analyse", padding=6)
        frame.pack(fill=tk.X, pady=(8, 0))
        for spec in ANALYSES:
            button = ttk.Button(frame, text=spec.label, command=lambda s=spec: self._on_run(s))
            button.pack(fill=tk.X, pady=1)
            self.busy_widgets.append(button)

    def _build_export_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Export", padding=6)
        frame.pack(fill=tk.X, pady=(8, 0))
        for text, command in (
            ("Export plot…", self._on_export_plot),
            ("Save individual PNGs…", self._on_save_pngs),
            ("Generate report…", self._on_generate_report),
        ):
            button = ttk.Button(frame, text=text, command=command)
            button.pack(fill=tk.X, pady=1)
            self.busy_widgets.append(button)

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        panes = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(panes)
        summary_frame = ttk.Frame(panes)
        panes.add(plot_frame, weight=4)
        panes.add(summary_frame, weight=1)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.summary = tk.Text(summary_frame, height=8, wrap=tk.NONE, font="TkFixedFont", state=tk.DISABLED)
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary.yview)
        self.summary.configure(yscrollcommand=summary_scroll.set)
        self.summary.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------
    # File list
    # ------------------------------------------------------------------

    def _add_files(self, paths: List[Path]) -> None:
        for path in paths:
            path = path.resolve()
            if path not in self.files:
                self.files.append(path)
                self.file_list.insert(tk.END, path.name)
        if paths:
            self.last_dir = str(paths[-1].resolve().parent)

    def _on_add_files(self) -> None:
        chosen = filedialog.askopenfilenames(
            parent=self.root,
            title="Select audio files",
            initialdir=self.last_dir,
            filetypes=[("WAV files", "*.wav *.WAV"), ("All files", "*")],
        )
        self._add_files([Path(p) for p in chosen])

    def _on_remove(self) -> None:
        for index in reversed(self.file_list.curselection()):
            self.file_list.delete(index)
            del self.files[index]

    def _on_clear(self) -> None:
        self.file_list.delete(0, tk.END)
        self.files.clear()

    # ------------------------------------------------------------------
    # Running analyses
    # ------------------------------------------------------------------

    def _on_run(self, spec: AnalysisSpec) -> None:
        if not self.files:
            messagebox.showinfo("No files", "Add at least one WAV file first.", parent=self.root)
            return

        files = list(self.files)
        mono = self.mono.get()
        self._set_busy(True)

        def worker() -> None:
            runs: List[AnalysisRun] = []
            for index, wav_path in enumerate(files):
                self.root.after(0, self.status.set, f"{spec.label}: {wav_path.name} ({index + 1}/{len(files)})…")
                out_dir = Path(self.session_dir.name) / spec.command / f"{index}_{wav_path.stem}"
                try:
                    runs.append(run_analysis(spec, wav_path, out_dir, mono))
                except Exception as error:  # keep the GUI alive whatever the child does
                    runs.append(AnalysisRun(wav_path=wav_path, ok=False, stderr=str(error)))
            self.root.after(0, self._show_results, spec, runs)

        threading.Thread(target=worker, daemon=True).start()

    def _show_results(self, spec: AnalysisSpec, runs: List[AnalysisRun]) -> None:
        self._set_busy(False)
        self.last_spec = spec
        self.last_runs = [run for run in runs if run.ok]

        self._draw_composite(spec, self.last_runs)
        self._write_summary(runs)

        failures = [run for run in runs if not run.ok]
        self.status.set(f"{spec.label}: {len(self.last_runs)} of {len(runs)} file(s) analysed.")
        if failures:
            detail = "\n\n".join(f"{run.wav_path.name}:\n{_tail(run.stderr)}" for run in failures)
            messagebox.showerror(f"{spec.label} failed", detail, parent=self.root)

    def _draw_composite(self, spec: AnalysisSpec, runs: List[AnalysisRun]) -> None:
        self.figure.clear()
        if not runs:
            self.canvas.draw_idle()
            return

        suffixes: List[str] = []
        for run in runs:
            for suffix in run.images:
                if suffix not in suffixes:
                    suffixes.append(suffix)

        column_titles = _unique_labels([run.wav_path for run in runs])
        axes = self.figure.subplots(len(suffixes), len(runs), squeeze=False)
        for column, run in enumerate(runs):
            for row, suffix in enumerate(suffixes):
                axis = axes[row][column]
                axis.set_axis_off()
                image_path = run.images.get(suffix)
                if image_path is None:
                    axis.text(0.5, 0.5, "(no output)", ha="center", va="center", transform=axis.transAxes)
                else:
                    axis.imshow(mpimg.imread(image_path))
                if row == 0:
                    axis.set_title(column_titles[column], fontsize=10, fontweight="bold")

        self.figure.suptitle(spec.label)
        self.figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.02, hspace=0.04)
        self.canvas.draw_idle()

    def _write_summary(self, runs: List[AnalysisRun]) -> None:
        sections = []
        for run in runs:
            body = run.stdout.strip() if run.ok else f"FAILED\n{_tail(run.stderr)}"
            sections.append(f"=== {run.wav_path.name} ===\n{body}")
        self._set_summary_text("\n\n".join(sections))

    def _set_summary_text(self, text: str) -> None:
        self.summary.configure(state=tk.NORMAL)
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", text)
        self.summary.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for widget in self.busy_widgets:
            widget.configure(state=state)
        self.root.configure(cursor="watch" if busy else "")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_plot(self) -> None:
        if not self.last_runs or self.last_spec is None:
            messagebox.showinfo("Nothing to export", "Run an analysis first.", parent=self.root)
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export plot",
            initialdir=self.last_dir,
            initialfile=f"{self.last_spec.command}_comparison.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not path:
            return

        self.figure.savefig(path, dpi=self._native_export_dpi(), facecolor="white")
        self.status.set(f"Exported {path}")

    def _native_export_dpi(self) -> float:
        """DPI at which each composite cell is at least as wide as its source PNG."""
        widest_image_px = max(
            mpimg.imread(image).shape[1] for run in self.last_runs for image in run.images.values()
        )
        cell_width_in = self.figure.get_figwidth() * 0.98 / len(self.last_runs)
        return min(EXPORT_DPI_MAX, max(EXPORT_DPI_MIN, widest_image_px / cell_width_in))

    def _on_save_pngs(self) -> None:
        if not self.last_runs or self.last_spec is None:
            messagebox.showinfo("Nothing to save", "Run an analysis first.", parent=self.root)
            return

        directory = filedialog.askdirectory(parent=self.root, title="Save PNGs to folder", initialdir=self.last_dir)
        if not directory:
            return

        count = 0
        for run in self.last_runs:
            for suffix, image in run.images.items():
                shutil.copy2(image, Path(directory) / f"{run.wav_path.stem}_{self.last_spec.command}{suffix}.png")
                count += 1
        self.status.set(f"Saved {count} PNG(s) to {directory}")

    def _on_generate_report(self) -> None:
        if not self.files:
            messagebox.showinfo("No files", "Add at least one WAV file first.", parent=self.root)
            return

        directory = filedialog.askdirectory(parent=self.root, title="Write reports to folder", initialdir=self.last_dir)
        if not directory:
            return

        files = list(self.files)
        mono = self.mono.get()
        self._set_busy(True)

        def worker() -> None:
            runs: List[ReportRun] = []
            for index, wav_path in enumerate(files):
                self.root.after(0, self.status.set, f"Report: {wav_path.name} ({index + 1}/{len(files)}), this can take a while…")
                out_dir = Path(directory) / f"{wav_path.stem}_report"
                try:
                    runs.append(run_report(wav_path, out_dir, mono))
                except Exception as error:
                    runs.append(ReportRun(wav_path=wav_path, ok=False, report_path=out_dir, stderr=str(error)))
            self.root.after(0, self._show_report_results, Path(directory), runs)

        threading.Thread(target=worker, daemon=True).start()

    def _show_report_results(self, directory: Path, runs: List[ReportRun]) -> None:
        self._set_busy(False)

        sections = []
        for run in runs:
            body = run.stdout.strip() if run.ok else f"FAILED\n{_tail(run.stderr)}"
            sections.append(f"=== {run.wav_path.name} report ===\n{body}")
        self._set_summary_text("\n\n".join(sections))

        succeeded = [run for run in runs if run.ok]
        failures = [run for run in runs if not run.ok]
        self.status.set(f"Report: {len(succeeded)} of {len(runs)} written to {directory}")
        if failures:
            detail = "\n\n".join(f"{run.wav_path.name}:\n{_tail(run.stderr)}" for run in failures)
            messagebox.showerror("Report failed", detail, parent=self.root)
        if succeeded and messagebox.askyesno(
            "Report written",
            "\n".join(str(run.report_path) for run in succeeded) + "\n\nOpen the folder?",
            parent=self.root,
        ):
            _open_in_file_manager(directory)

    def _on_close(self) -> None:
        self.session_dir.cleanup()
        self.root.destroy()


def _unique_labels(paths: List[Path]) -> List[str]:
    """File names, prefixed with the parent folder where names collide."""
    names = [path.name for path in paths]
    return [
        f"{path.parent.name}/{path.name}" if names.count(path.name) > 1 else path.name
        for path in paths
    ]


def _open_in_file_manager(directory: Path) -> None:
    try:
        subprocess.Popen(["xdg-open", str(directory)])
    except OSError:
        pass


def _tail(text: str, max_lines: int = 15) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:]) if lines else "(no error output)"


def main() -> None:
    root = tk.Tk(className=WINDOW_CLASS_NAME)
    root.icon_image = tk.PhotoImage(file=str(ICON_PATH))  # keep a reference or Tk drops the icon
    root.iconphoto(True, root.icon_image)
    AnalysisApp(root, [Path(arg) for arg in sys.argv[1:]])
    root.mainloop()

# Audio Analysis Framework

A comprehensive Python framework for audio impulse response analysis and test signal generation, designed for **reverb**, **delay**, **comb-filter**, and **physical-modelling** DSP workflows.

The framework is intentionally **offline, deterministic, and IR-centric**, making it suitable for iterative DSP development:

> change DSP → render audio → analyse → inspect → repeat

---

## Features

### Analysis Commands (`analyse`)
- **ir**: Impulse Response visualization (waveform, early reflections, frequency tail)
- **decay**: Schroeder EDC with T20 / T30 / RT60 / EDT decay metrics
- **rt60bands**: Band-limited RT60 analysis (Low/Mid/High, octave, third-octave)
- **fr**: Frequency-response magnitude spectrum
- **filter**: Filter frequency-response analysis (magnitude + phase)
- **groupdelay**: Group-delay analysis derived from phase response
- **spectrogram**: Time-frequency spectrogram visualization
- **diffusion**: Diffusion / decorrelation metrics (autocorr, echo density, stereo decorrelation)
- **waterfall**: 3D cumulative spectral decay (CSD) waterfall plots
- **modalcloud**: Modal decay analysis (frequency vs RT60 scatter)
- **zplane**: Estimated z-plane pole (and optional zero) cloud plots
- **report**: Generate comprehensive analysis report with plots + summary markdown

### Signal Generation (`gen`)
- **impulse**: Single-sample Dirac impulse
- **click**: Windowed short pulse
- **impulse_train**: Periodic click train
- **noise_long**: Extended white / pink noise
- **noise_burst**: Short windowed noise burst
- **sine_sustain**: Sustained sine wave
- **sine_burst**: Windowed sine burst
- **sweep**: Logarithmic sine sweep (with silence padding for deconvolution)
- **pluck**: Synthetic muted-pluck proxy
- **karplus_pluck**: Karplus–Strong physical model
- **all**: Generate all test tones with sensible defaults

---

## Quick Start

### Installation
```bash
bash setup.sh
```
### Activate
```bash
source .venv/bin/activate
```

### Generate Test Signals
```bash
python -m gen.cli all
python -m gen.cli sweep --start-freq 20 --end-freq 20000 --duration_seconds 10 --output my_sweep.wav
python -m gen.cli click --duration 0.001 --output click.wav
```

### Analyse Impulse Responses
```bash
python -m analyse.cli ir --input my_ir.wav --output plots/my_ir
python -m analyse.cli decay --input my_ir.wav --output plots/decay --mono
python -m analyse.cli rt60bands --input my_ir.wav --output plots/bands --mono
python -m analyse.cli filter --input my_ir.wav --output plots/filter --mono
python -m analyse.cli groupdelay --input my_ir.wav --output plots/groupdelay --mono
python -m analyse.cli zplane --input my_ir.wav --output plots/zplane --mono
python -m analyse.cli report --input my_ir.wav --output plots/my_ir_report --mono
```

---

## Z-Plane Pole Cloud Analysis

The **zplane** command estimates pole locations from an impulse response using an
all-pole (AR) least-squares fit and plots them on the complex plane with a unit-circle overlay.

```bash
python -m analyse.cli zplane --input my_ir.wav --output plots/zplane
```

Options:
- `--ar-order`: Number of poles to fit (controls modal density)
- `--duration`: Analysis window after IR peak
- `--ignore-leading`: Ignore early transient samples
- `--zeros`: Also estimate and plot zeros (approximate)
- `--radius`: Plot radius (default slightly > 1.0)

Notes:
- Pole plots assume **LTI behaviour**
- For time-varying systems (e.g. modulated delays), results are approximate
- Best used on static IRs when tuning combs, resonators, and Schroeder structures

---

## Group Delay Analysis

The **groupdelay** command derives group delay from the unwrapped phase response:

\[
GD(\omega) = -\frac{d\phi(\omega)}{d\omega}
\]

This is useful for:
- comparing allpass vs comb structures
- diagnosing dispersion
- visualising time-smearing behaviour

---

## Typical Analysis Workflow

1. Generate or render an impulse response
2. Inspect time-domain structure with `ir`
3. Analyse decay with `decay` and `rt60bands`
4. Examine spectral behaviour with `filter` and `spectrogram`
5. Inspect temporal behaviour with `groupdelay`
6. Visualise modal structure with `zplane` and `modalcloud`
7. Iterate DSP parameters and repeat

---

## Project Structure

```
analyse/
  cli.py
  impulse_response.py
  decay.py
  rt60bands.py
  frequency_response.py
  filterplot.py
  group_delay.py
  zplane.py
  spectrogram.py
  diffusion.py
  waterfall.py
  modalcloud.py
  report.py
  deconvolve.py
  io.py
  plotting.py

gen/
  cli.py
  signals.py
```

---

## Technical Notes

- Sample rate assumed: **48 kHz**
- Designed for offline inspection, not real-time use
- Pole/zero estimation is diagnostic, not symbolic
- Emphasis is on **understanding DSP behaviour**, not black-box metrics


## Bundle analysis (meta.json + taps/*.wav)

If you have a captured IR bundle folder (e.g. from a C++ probe harness):

```bash
python -m analyse.cli bundle --input analysis_runs/20260205_123456
```

Or use the helper script:

```bash
./scripts/analyse_bundle.sh analysis_runs/20260205_123456
```

This runs `report` on each tap WAV and writes an index Markdown at `reports/bundle_report.md`.

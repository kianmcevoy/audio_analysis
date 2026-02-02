"""
Make the two plots we discussed:

1) "Actual attenuation at the specified cutoff frequency" for the ORIGINAL mapping
2) Overlay ORIGINAL vs STANDARD one-pole on the same plot

Assumptions:
- Fs is fixed (needed to label x-axis in Hz)
- We evaluate the filter magnitude at the SAME frequency that you pass as the cutoff.
- fc_norm = fc_hz / Fs  (cycles/sample), with 0 < fc_norm < 0.5

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Config ----------
Fs = 48_000.0

# Build a frequency grid with good resolution at low frequencies + full coverage.
# Use log spacing for low-to-mid, linear to near Nyquist.
fc_norm = np.concatenate([
    np.logspace(-6, np.log10(0.05), 800, base=10),
    np.linspace(0.05, 0.49, 800),
])
fc_norm = np.unique(fc_norm)
fc_hz = fc_norm * Fs
w = 2.0 * np.pi * fc_norm

# ---------- One-pole frequency response ----------
def onepole_mag_at_fc_from_pole(a, w):
    """
    For H(z) = (1-a)/(1 - a z^-1), evaluate |H(e^jw)| at given w.
    a can be vector, w vector (same shape).
    """
    ejw = np.exp(-1j * w)
    H = (1.0 - a) / (1.0 - a * ejw)
    return np.abs(H)

def original_mapping_pole(fc_norm):
    """
    Your original mapping:
      coef = 1/(pi*fc_norm)
      pole a = (coef - 1)/(coef + 1)
    """
    coef = 1.0 / (np.pi * fc_norm)
    return (coef - 1.0) / (coef + 1.0)

def standard_mapping_pole(fc_norm):
    """
    Standard one-pole mapping:
      a = exp(-2*pi*fc_norm)
    """
    return np.exp(-2.0 * np.pi * fc_norm)

# ---------- Compute curves ----------
a_orig = original_mapping_pole(fc_norm)
a_std  = standard_mapping_pole(fc_norm)

mag_orig = onepole_mag_at_fc_from_pole(a_orig, w)
mag_std  = onepole_mag_at_fc_from_pole(a_std,  w)

db_orig = 20.0 * np.log10(np.maximum(mag_orig, 1e-300))
db_std  = 20.0 * np.log10(np.maximum(mag_std,  1e-300))

target_db = 20.0 * np.log10(1.0 / np.sqrt(2.0))  # -3.0103 dB

# Filter out frequencies below 100 Hz
mask = fc_hz >= 100.0
fc_hz = fc_hz[mask]
db_orig = db_orig[mask]
db_std = db_std[mask]

# ---------- Plot 1: error vs frequency (original mapping) ----------
# err_db = db_orig - target_db

# plt.figure(figsize=(8, 4.8))
# plt.plot(fc_hz, err_db)
# plt.axhline(0.0, linestyle="--")
# plt.axvline(Fs / np.pi, linestyle=":", label="Fs/pi")
# plt.xscale("log")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Error at fc (dB)  [actual - (-3.01 dB)]")
# plt.title("Original mapping: dB error at the specified cutoff frequency")
# plt.grid(True, which="both")
# plt.legend()
# plt.tight_layout()
# plt.show()

# ---------- Plot 2: actual dB at fc, overlay original vs standard ----------
plt.figure(figsize=(8, 4.8))
plt.plot(fc_hz, db_orig, label="Original mapping (crude damping)")
plt.plot(fc_hz, db_std,  label="Standard one-pole (exp mapping)")
plt.axhline(target_db, linestyle="--", color="gray", label="-3.01 dB reference")
plt.axvline(Fs / np.pi, linestyle=":",  color="gray", label="Fs / π ≈ 15.3 kHz")

plt.xscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude at specified cutoff (dB)")
plt.title("Attenuation at the specified cutoff frequency (Fs = 48 kHz)")
plt.ylim(-5, 0)
plt.grid(True, which="both")
plt.legend()

# More readable Hz/kHz ticks (optional; log axis)
ticks_hz = np.array([100, 200, 500,
                     1e3, 2e3, 5e3, 10e3, 20e3])
tick_labels = [f"{int(t)} Hz" if t < 1000 else f"{int(t/1000)} kHz" for t in ticks_hz]
plt.xticks(ticks_hz, tick_labels)

plt.tight_layout()
plt.show()

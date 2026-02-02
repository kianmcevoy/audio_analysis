# gen/__init__.py
"""
gen package

Offline test-signal generation for reverb analysis.

This package contains:
- signal generator functions (gen.signals)
- command line interface entrypoint (gen.cli)

Typical usage:
    from gen.signals import generate_noise_burst
"""

from .signals import (
    GeneratedSignal,
    duplicate_mono_to_stereo,
    generate_click,
    generate_impulse,
    generate_impulse_train,
    generate_karplus_strong_pluck,
    generate_log_sine_sweep,
    generate_noise,
    generate_noise_burst,
    generate_pluck_like,
    generate_sine,
    generate_sine_burst,
)

__all__ = [
    "GeneratedSignal",
    "duplicate_mono_to_stereo",
    "generate_click",
    "generate_impulse",
    "generate_impulse_train",
    "generate_karplus_strong_pluck",
    "generate_log_sine_sweep",
    "generate_noise",
    "generate_noise_burst",
    "generate_pluck_like",
    "generate_sine",
    "generate_sine_burst",
]

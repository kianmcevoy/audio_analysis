# analyse/__init__.py
"""
analyse package

Offline analysis tools for reverb outputs.

This package contains:
- WAV loading and normalisation (analyse.io)
- analysis routines (IR, decay, spectrogram, waterfall, modes, diffusion)
- command line interface entrypoint (analyse.cli) [to be added]

Typical usage:
    from analyse.io import load_wav_file
"""

from .io import (
    LoadedAudio,
    DEFAULT_EXPECTED_SAMPLE_RATE_HZ,
    convert_wav_samples_to_float32,
    downmix_to_mono,
    duplicate_mono_to_stereo,
    get_channel,
    get_left_right,
    load_wav_file,
    validate_audio_format,
)

__all__ = [
    "LoadedAudio",
    "DEFAULT_EXPECTED_SAMPLE_RATE_HZ",
    "convert_wav_samples_to_float32",
    "downmix_to_mono",
    "duplicate_mono_to_stereo",
    "get_channel",
    "get_left_right",
    "load_wav_file",
    "validate_audio_format",
]

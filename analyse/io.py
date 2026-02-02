# analyse/io.py
"""
WAV I/O utilities for offline reverb analysis.

Design goals:
- readable and explicit
- robust handling of common WAV encodings (int16, int32, float32)
- consistent internal format: float32 in range [-1, 1]
- clear errors when input doesn't match expectations

Assumptions for this project:
- target sample rate is 48 kHz
- most inputs are stereo, but mono can be accepted when requested
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

try:
    from scipy.io import wavfile
except ImportError as import_error:  # pragma: no cover
    raise ImportError(
        "scipy is required for WAV reading. Install with: pip install scipy"
    ) from import_error


ChannelMode = Literal["mono", "stereo", "mono_or_stereo"]
DEFAULT_EXPECTED_SAMPLE_RATE_HZ = 48_000


@dataclass(frozen=True)
class LoadedAudio:
    """
    Container for loaded audio with a consistent internal representation.
    """
    samples: np.ndarray          # shape (num_samples, num_channels), float32 in [-1, 1]
    sample_rate_hz: int          # e.g. 48000
    file_path: Path              # original file


def _convert_integer_pcm_to_float32(samples: np.ndarray) -> np.ndarray:
    """
    Convert integer PCM to float32 in [-1, 1].

    Supports:
    - int16: scale by 32768
    - int32: scale by 2147483648

    Note:
    Some WAV files store 24-bit PCM in an int32 container.
    This function still produces a valid float range mapping.
    """
    if samples.dtype == np.int16:
        return (samples.astype(np.float32) / 32768.0)

    if samples.dtype == np.int32:
        return (samples.astype(np.float32) / 2147483648.0)

    raise ValueError(f"Unsupported integer PCM dtype: {samples.dtype}")

def get_analysis_channels(
    loaded_audio: LoadedAudio,
    use_mono_downmix_for_stereo: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """
    Return a list of channels to analyse as (name, 1D samples).

    Rules:
    - mono input: returns [("mono", samples)]
    - stereo input:
        - if use_mono_downmix_for_stereo: returns [("mono", 0.5*(L+R))]
        - else returns [("left", L), ("right", R)]
    """
    channel_count = loaded_audio.samples.shape[1]

    if channel_count == 1:
        mono_samples = loaded_audio.samples[:, 0].astype(np.float32, copy=False)
        return [("mono", mono_samples)]

    if channel_count == 2:
        left_samples = loaded_audio.samples[:, 0].astype(np.float32, copy=False)
        right_samples = loaded_audio.samples[:, 1].astype(np.float32, copy=False)

        if use_mono_downmix_for_stereo:
            mono_samples = 0.5 * (left_samples + right_samples)
            return [("mono", mono_samples)]

        return [("left", left_samples), ("right", right_samples)]

    raise ValueError(f"Unsupported channel count: {channel_count}")


def convert_wav_samples_to_float32(samples_from_wav: np.ndarray) -> np.ndarray:
    """
    Convert WAV samples to float32 in [-1, 1] regardless of source dtype.

    - float32/float64: passed through (clipped to [-1,1])
    - int16/int32: scaled appropriately
    """
    if np.issubdtype(samples_from_wav.dtype, np.floating):
        float_samples = samples_from_wav.astype(np.float32, copy=False)
        return np.clip(float_samples, -1.0, 1.0).astype(np.float32)

    if np.issubdtype(samples_from_wav.dtype, np.integer):
        float_samples = _convert_integer_pcm_to_float32(samples_from_wav)
        return np.clip(float_samples, -1.0, 1.0).astype(np.float32)

    raise ValueError(f"Unsupported WAV dtype: {samples_from_wav.dtype}")


def ensure_2d_channel_array(float_samples: np.ndarray) -> np.ndarray:
    """
    Ensure samples are shaped (num_samples, num_channels).
    """
    if float_samples.ndim == 1:
        return float_samples.reshape((-1, 1))

    if float_samples.ndim == 2:
        return float_samples

    raise ValueError(f"Expected 1D or 2D audio array, got shape {float_samples.shape}")


def duplicate_mono_to_stereo(float_samples: np.ndarray) -> np.ndarray:
    """
    Convert mono (N,1) or (N,) to stereo (N,2) by duplicating channels.
    """
    float_samples = ensure_2d_channel_array(float_samples)

    if float_samples.shape[1] == 1:
        mono_channel = float_samples[:, 0]
        return np.stack([mono_channel, mono_channel], axis=1).astype(np.float32)

    if float_samples.shape[1] == 2:
        return float_samples.astype(np.float32)

    raise ValueError(f"Expected mono or stereo for upmix, got {float_samples.shape[1]} channels")


def downmix_to_mono(float_samples: np.ndarray) -> np.ndarray:
    """
    Downmix stereo (or multichannel) to mono by averaging channels.
    Returns shape (num_samples, 1).
    """
    float_samples = ensure_2d_channel_array(float_samples)

    mono_samples = np.mean(float_samples, axis=1, dtype=np.float32)
    return mono_samples.reshape((-1, 1)).astype(np.float32)


def validate_audio_format(
    loaded_audio: LoadedAudio,
    expected_sample_rate_hz: int = DEFAULT_EXPECTED_SAMPLE_RATE_HZ,
    expected_channel_mode: ChannelMode = "stereo",
) -> None:
    if loaded_audio.sample_rate_hz != expected_sample_rate_hz:
        raise ValueError(
            f"Expected sample rate {expected_sample_rate_hz} Hz, "
            f"but got {loaded_audio.sample_rate_hz} Hz for file {loaded_audio.file_path}"
        )

    channel_count = loaded_audio.samples.shape[1]

    if expected_channel_mode == "mono" and channel_count != 1:
        raise ValueError(f"Expected mono (1 channel) but got {channel_count} channels for file {loaded_audio.file_path}")

    if expected_channel_mode == "stereo" and channel_count != 2:
        raise ValueError(f"Expected stereo (2 channels) but got {channel_count} channels for file {loaded_audio.file_path}")

    if expected_channel_mode == "mono_or_stereo" and channel_count not in (1, 2):
        raise ValueError(
            f"Expected mono or stereo (1 or 2 channels) but got {channel_count} channels for file {loaded_audio.file_path}"
        )


def load_wav_file(
    wav_file_path: str | Path,
    expected_sample_rate_hz: int = DEFAULT_EXPECTED_SAMPLE_RATE_HZ,
    expected_channel_mode: ChannelMode = "stereo",
    allow_mono_and_upmix_to_stereo: bool = True,
) -> LoadedAudio:
    """
    Load a WAV file, convert to float32, ensure shape (N, C),
    optionally upmix mono to stereo, and validate expected format.

    Typical usage:
        loaded = load_wav_file("reverb_output.wav")
        # loaded.samples is float32 stereo at 48k by default

    If allow_mono_and_upmix_to_stereo is True:
        - mono input is accepted and duplicated to stereo when stereo is expected
    """
    wav_file_path = Path(wav_file_path)

    sample_rate_hz, samples_from_wav = wavfile.read(str(wav_file_path))

    float_samples = convert_wav_samples_to_float32(samples_from_wav)
    float_samples = ensure_2d_channel_array(float_samples)

    if expected_channel_mode == "stereo" and allow_mono_and_upmix_to_stereo:
        if float_samples.shape[1] == 1:
            float_samples = duplicate_mono_to_stereo(float_samples)

    loaded_audio = LoadedAudio(
        samples=float_samples.astype(np.float32, copy=False),
        sample_rate_hz=int(sample_rate_hz),
        file_path=wav_file_path,
    )

    validate_audio_format(
        loaded_audio=loaded_audio,
        expected_sample_rate_hz=expected_sample_rate_hz,
        expected_channel_mode=expected_channel_mode,
    )

    return loaded_audio


def get_channel(
    loaded_audio: LoadedAudio,
    channel_index: int,
) -> np.ndarray:
    """
    Return a single channel as a 1D float32 array of length N.
    """
    channel_count = loaded_audio.samples.shape[1]
    if not (0 <= channel_index < channel_count):
        raise ValueError(f"channel_index out of range: {channel_index} for {channel_count} channels")

    return loaded_audio.samples[:, channel_index].astype(np.float32, copy=False)


def get_left_right(
    loaded_audio: LoadedAudio,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience: return left and right as 1D arrays.
    """
    validate_audio_format(loaded_audio, expected_channel_mode="stereo")
    left = get_channel(loaded_audio, 0)
    right = get_channel(loaded_audio, 1)
    return left, right

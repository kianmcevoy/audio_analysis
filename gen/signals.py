# gen/signals.py
"""
Offline test-signal generators for reverb analysis.

All generators return mono NumPy arrays (float32, range [-1, 1]).
The CLI layer is responsible for duplicating mono -> stereo (L = R)
and writing 48 kHz WAV files.

Design goals:
- clarity over cleverness
- deterministic and repeatable signals
- readable DSP, suitable for study and modification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# -------------------------------------------------------------------
# Type aliases
# -------------------------------------------------------------------

WindowType = Literal["rect", "hann", "hamming", "blackman"]
NoiseType = Literal["white", "pink"]


# -------------------------------------------------------------------
# Data container
# -------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedSignal:
    """
    Container for a generated mono signal.
    """
    samples: np.ndarray      # shape (num_samples,), dtype float32
    sample_rate_hz: int


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def convert_to_float32_and_limit_peak(samples: np.ndarray) -> np.ndarray:
    """
    Convert array to float32 and ensure peak magnitude <= 1.0.
    """
    samples = np.asarray(samples, dtype=np.float32)

    if samples.size == 0:
        return samples

    peak_magnitude = float(np.max(np.abs(samples)))
    if peak_magnitude > 1.0:
        samples = samples / peak_magnitude

    return samples


def seconds_to_samples(duration_seconds: float, sample_rate_hz: int) -> int:
    """
    Convert duration in seconds to integer sample count.
    """
    if duration_seconds < 0.0:
        raise ValueError("Duration must be non-negative")

    return int(round(duration_seconds * sample_rate_hz))


def generate_window(
    number_of_samples: int,
    window_type: WindowType = "hann",
) -> np.ndarray:
    """
    Generate a window of the requested type.
    """
    if number_of_samples <= 0:
        return np.zeros((0,), dtype=np.float32)

    if window_type == "rect":
        window_values = np.ones(number_of_samples, dtype=np.float32)
    elif window_type == "hann":
        window_values = np.hanning(number_of_samples).astype(np.float32)
    elif window_type == "hamming":
        window_values = np.hamming(number_of_samples).astype(np.float32)
    elif window_type == "blackman":
        window_values = np.blackman(number_of_samples).astype(np.float32)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    return window_values


def normalise_peak_amplitude(
    samples: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    Peak-normalise a signal to a target absolute amplitude.
    """
    samples = np.asarray(samples, dtype=np.float32)

    if samples.size == 0:
        return samples

    current_peak = float(np.max(np.abs(samples)))
    if current_peak <= 0.0:
        return samples

    return samples * (target_peak / current_peak)


# -------------------------------------------------------------------
# Signal generators
# -------------------------------------------------------------------

def generate_impulse(
    sample_rate_hz: int = 48_000,
    impulse_sample_index: int = 0,
    total_duration_seconds: float = 1.0,
) -> GeneratedSignal:
    """
    Generate a Dirac impulse inside a buffer of fixed length.
    """
    total_number_of_samples = seconds_to_samples(
        total_duration_seconds, sample_rate_hz
    )

    impulse_samples = np.zeros(
        (total_number_of_samples,), dtype=np.float32
    )

    if 0 <= impulse_sample_index < total_number_of_samples:
        impulse_samples[impulse_sample_index] = 1.0

    return GeneratedSignal(
        samples=impulse_samples,
        sample_rate_hz=sample_rate_hz,
    )


def generate_click(
    sample_rate_hz: int = 48_000,
    click_duration_seconds: float = 0.001,
    window_type: WindowType = "hann",
) -> GeneratedSignal:
    """
    Generate a short, windowed pulse.

    This behaves better than a single-sample impulse through
    systems that include modulation, dynamics, or smoothing.
    """
    number_of_samples = max(
        1, seconds_to_samples(click_duration_seconds, sample_rate_hz)
    )

    raw_pulse = np.ones(
        (number_of_samples,), dtype=np.float32
    )

    window = generate_window(number_of_samples, window_type)
    windowed_pulse = raw_pulse * window

    windowed_pulse = normalise_peak_amplitude(windowed_pulse, 0.95)

    return GeneratedSignal(
        samples=windowed_pulse,
        sample_rate_hz=sample_rate_hz,
    )


def generate_impulse_train(
    sample_rate_hz: int = 48_000,
    total_duration_seconds: float = 2.0,
    impulse_period_seconds: float = 0.25,
    click_duration_seconds: float = 0.001,
    window_type: WindowType = "hann",
) -> GeneratedSignal:
    """
    Generate a periodic train of short clicks.

    Useful for testing time-variance, modulation, and periodicity.
    """
    total_number_of_samples = seconds_to_samples(
        total_duration_seconds, sample_rate_hz
    )

    output_samples = np.zeros(
        (total_number_of_samples,), dtype=np.float32
    )

    impulse_period_samples = max(
        1, seconds_to_samples(impulse_period_seconds, sample_rate_hz)
    )

    click_signal = generate_click(
        sample_rate_hz=sample_rate_hz,
        click_duration_seconds=click_duration_seconds,
        window_type=window_type,
    ).samples

    click_length_samples = click_signal.size

    for start_sample in range(0, total_number_of_samples, impulse_period_samples):
        end_sample = min(
            total_number_of_samples,
            start_sample + click_length_samples,
        )
        output_samples[start_sample:end_sample] += click_signal[
            : end_sample - start_sample
        ]

    output_samples = normalise_peak_amplitude(output_samples, 0.95)

    return GeneratedSignal(
        samples=output_samples,
        sample_rate_hz=sample_rate_hz,
    )


def generate_noise(
    sample_rate_hz: int = 48_000,
    duration_seconds: float = 1.0,
    noise_type: NoiseType = "white",
    random_seed: int = 0,
) -> GeneratedSignal:
    """
    Generate white or pink noise.
    """
    number_of_samples = seconds_to_samples(
        duration_seconds, sample_rate_hz
    )

    random_generator = np.random.default_rng(random_seed)

    if noise_type == "white":
        noise_samples = random_generator.standard_normal(
            number_of_samples
        ).astype(np.float32)

        noise_samples = normalise_peak_amplitude(noise_samples, 0.95)

        return GeneratedSignal(
            samples=noise_samples,
            sample_rate_hz=sample_rate_hz,
        )

    if noise_type == "pink":
        white_noise = random_generator.standard_normal(
            number_of_samples
        ).astype(np.float32)

        frequency_domain = np.fft.rfft(white_noise)
        frequency_axis_hz = np.fft.rfftfreq(
            number_of_samples, d=1.0 / sample_rate_hz
        )

        pink_scaling = np.ones_like(
            frequency_axis_hz, dtype=np.float32
        )

        nonzero_mask = frequency_axis_hz > 0.0
        pink_scaling[nonzero_mask] = (
            1.0 / np.sqrt(frequency_axis_hz[nonzero_mask])
        )

        frequency_domain *= pink_scaling

        pink_noise = np.fft.irfft(
            frequency_domain, n=number_of_samples
        ).astype(np.float32)

        pink_noise -= float(np.mean(pink_noise))
        pink_noise = normalise_peak_amplitude(pink_noise, 0.95)

        return GeneratedSignal(
            samples=pink_noise,
            sample_rate_hz=sample_rate_hz,
        )

    raise ValueError(f"Unknown noise type: {noise_type}")


def generate_noise_burst(
    sample_rate_hz: int = 48_000,
    burst_duration_seconds: float = 0.02,
    noise_type: NoiseType = "white",
    random_seed: int = 0,
    window_type: WindowType = "hann",
) -> GeneratedSignal:
    """
    Generate a short windowed noise burst.
    """
    base_noise = generate_noise(
        sample_rate_hz=sample_rate_hz,
        duration_seconds=burst_duration_seconds,
        noise_type=noise_type,
        random_seed=random_seed,
    ).samples

    window = generate_window(base_noise.size, window_type)
    windowed_noise = base_noise * window

    windowed_noise = normalise_peak_amplitude(windowed_noise, 0.95)

    return GeneratedSignal(
        samples=windowed_noise,
        sample_rate_hz=sample_rate_hz,
    )


def generate_sine(
    sample_rate_hz: int = 48_000,
    frequency_hz: float = 440.0,
    duration_seconds: float = 2.0,
    amplitude: float = 0.5,
    initial_phase_radians: float = 0.0,
) -> GeneratedSignal:
    """
    Generate a sustained sine wave.
    """
    number_of_samples = seconds_to_samples(
        duration_seconds, sample_rate_hz
    )

    time_axis_seconds = (
        np.arange(number_of_samples, dtype=np.float32)
        / float(sample_rate_hz)
    )

    sine_samples = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * time_axis_seconds
        + initial_phase_radians
    )

    sine_samples = convert_to_float32_and_limit_peak(sine_samples)

    return GeneratedSignal(
        samples=sine_samples,
        sample_rate_hz=sample_rate_hz,
    )


def generate_sine_burst(
    sample_rate_hz: int = 48_000,
    frequency_hz: float = 220.0,
    burst_duration_seconds: float = 0.1,
    amplitude: float = 0.7,
    window_type: WindowType = "hann",
) -> GeneratedSignal:
    """
    Generate a windowed sine burst.
    """
    sine_signal = generate_sine(
        sample_rate_hz=sample_rate_hz,
        frequency_hz=frequency_hz,
        duration_seconds=burst_duration_seconds,
        amplitude=amplitude,
    ).samples

    window = generate_window(sine_signal.size, window_type)
    windowed_sine = sine_signal * window

    windowed_sine = normalise_peak_amplitude(windowed_sine, 0.95)

    return GeneratedSignal(
        samples=windowed_sine,
        sample_rate_hz=sample_rate_hz,
    )


def generate_log_sine_sweep(
    sample_rate_hz: int = 48_000,
    duration_seconds: float = 10.0,
    start_frequency_hz: float = 20.0,
    end_frequency_hz: float = 20_000.0,
    amplitude: float = 0.5,
    fade_duration_seconds: float = 0.01,
    pre_silence_seconds: float = 0.0,
    post_silence_seconds: float = 0.0,
) -> GeneratedSignal:
    """
    Generate a logarithmic sine sweep.

    Intended for deconvolution-based impulse response extraction.
    
    Parameters:
        pre_silence_seconds: Silent padding before the sweep (for deconvolution).
        post_silence_seconds: Silent padding after the sweep (for deconvolution tail).
    """
    number_of_samples = seconds_to_samples(
        duration_seconds, sample_rate_hz
    )

    if number_of_samples <= 1:
        return GeneratedSignal(
            samples=np.zeros((number_of_samples,), dtype=np.float32),
            sample_rate_hz=sample_rate_hz,
        )

    if start_frequency_hz <= 0.0 or end_frequency_hz <= start_frequency_hz:
        raise ValueError("Require 0 < start_frequency_hz < end_frequency_hz")

    time_axis_seconds = (
        np.arange(number_of_samples, dtype=np.float64)
        / float(sample_rate_hz)
    )

    sweep_duration = float(duration_seconds)
    sweep_constant = sweep_duration / np.log(
        end_frequency_hz / start_frequency_hz
    )

    instantaneous_phase = (
        2.0
        * np.pi
        * start_frequency_hz
        * sweep_constant
        * (np.exp(time_axis_seconds / sweep_constant) - 1.0)
    )

    sweep_samples = amplitude * np.sin(instantaneous_phase)
    sweep_samples = sweep_samples.astype(np.float32)

    fade_samples = seconds_to_samples(
        fade_duration_seconds, sample_rate_hz
    )
    fade_samples = min(fade_samples, number_of_samples // 2)

    if fade_samples > 0:
        # Half-cosine fade: smoother than linear
        t = np.linspace(0.0, np.pi, fade_samples, dtype=np.float32)
        fade_window = 0.5 - 0.5 * np.cos(t)  # 0..1
        sweep_samples[:fade_samples] *= fade_window
        sweep_samples[-fade_samples:] *= fade_window[::-1]

    # Optional DC removal (usually near-zero already)
    sweep_samples -= float(np.mean(sweep_samples))

    # Add silence padding if requested
    pre_silence_samples = seconds_to_samples(pre_silence_seconds, sample_rate_hz)
    post_silence_samples = seconds_to_samples(post_silence_seconds, sample_rate_hz)
    
    if pre_silence_samples > 0 or post_silence_samples > 0:
        pre_padding = np.zeros(pre_silence_samples, dtype=np.float32)
        post_padding = np.zeros(post_silence_samples, dtype=np.float32)
        sweep_samples = np.concatenate([pre_padding, sweep_samples, post_padding])

    return GeneratedSignal(
        samples=sweep_samples,
        sample_rate_hz=sample_rate_hz,
    )


def generate_pluck_like(
    sample_rate_hz: int = 48_000,
    duration_seconds: float = 0.15,
    bandlimit_frequency_hz: float = 8000.0,
    decay_time_constant_seconds: float = 0.03,
    random_seed: int = 0,
) -> GeneratedSignal:
    """
    Generate a synthetic muted-pluck excitation.

    - band-limited noise
    - exponential decay envelope
    """
    number_of_samples = seconds_to_samples(
        duration_seconds, sample_rate_hz
    )

    if number_of_samples <= 0:
        return GeneratedSignal(
            samples=np.zeros((0,), dtype=np.float32),
            sample_rate_hz=sample_rate_hz,
        )

    random_generator = np.random.default_rng(random_seed)
    excitation_noise = random_generator.standard_normal(
        number_of_samples
    ).astype(np.float32)

    frequency_domain = np.fft.rfft(excitation_noise)
    frequency_axis_hz = np.fft.rfftfreq(
        number_of_samples, d=1.0 / sample_rate_hz
    )

    frequency_domain[
        frequency_axis_hz > bandlimit_frequency_hz
    ] = 0.0

    bandlimited_noise = np.fft.irfft(
        frequency_domain, n=number_of_samples
    ).astype(np.float32)

    time_axis_seconds = (
        np.arange(number_of_samples, dtype=np.float32)
        / float(sample_rate_hz)
    )

    decay_envelope = np.exp(
        -time_axis_seconds / decay_time_constant_seconds
    ).astype(np.float32)

    pluck_samples = bandlimited_noise * decay_envelope
    pluck_samples = normalise_peak_amplitude(pluck_samples, 0.95)

    return GeneratedSignal(
        samples=pluck_samples,
        sample_rate_hz=sample_rate_hz,
    )

def generate_karplus_strong_pluck(
    sample_rate_hz: int = 48_000,
    fundamental_frequency_hz: float = 110.0,
    duration_seconds: float = 2.0,
    excitation_noise_bandlimit_hz: float = 8000.0,
    feedback_decay_factor: float = 0.996,
    lowpass_blend: float = 0.5,
    random_seed: int = 0,
) -> GeneratedSignal:
    """
    Generate a Karplus–Strong pluck (simple string-like physical model).

    Model:
    - Initialise a delay-line buffer with band-limited noise.
    - Re-circulate through a simple 2-point averaging filter (a lowpass) and a decay factor.

    Parameters:
    - fundamental_frequency_hz: sets delay length approximately (pitch).
    - feedback_decay_factor: closer to 1.0 = longer sustain. Must be < 1.0 for stability.
    - lowpass_blend: 0..1, how much averaging (damping) is applied per loop.
      0.0 = no averaging (brighter, noisier), 1.0 = full 2-point average (darker).
    - excitation_noise_bandlimit_hz: lowpass applied to initial noise burst (offline FFT).
    """
    if fundamental_frequency_hz <= 0.0:
        raise ValueError("fundamental_frequency_hz must be > 0")

    if not (0.0 < feedback_decay_factor < 1.0):
        raise ValueError("feedback_decay_factor must be between 0 and 1 (exclusive)")

    if not (0.0 <= lowpass_blend <= 1.0):
        raise ValueError("lowpass_blend must be between 0 and 1 (inclusive)")

    total_number_of_samples = seconds_to_samples(duration_seconds, sample_rate_hz)
    if total_number_of_samples <= 0:
        return GeneratedSignal(samples=np.zeros((0,), dtype=np.float32), sample_rate_hz=sample_rate_hz)

    # Delay line length in samples (nearest integer).
    delay_line_length_samples = int(round(sample_rate_hz / fundamental_frequency_hz))
    delay_line_length_samples = max(2, delay_line_length_samples)

    # Initialise delay line with band-limited noise (deterministic).
    random_generator = np.random.default_rng(random_seed)
    initial_noise = random_generator.standard_normal(delay_line_length_samples).astype(np.float32)

    frequency_domain = np.fft.rfft(initial_noise)
    frequency_axis_hz = np.fft.rfftfreq(delay_line_length_samples, d=1.0 / sample_rate_hz)
    frequency_domain[frequency_axis_hz > float(excitation_noise_bandlimit_hz)] = 0.0
    bandlimited_initial_noise = np.fft.irfft(frequency_domain, n=delay_line_length_samples).astype(np.float32)

    delay_line_buffer = bandlimited_initial_noise.copy()

    output_samples = np.zeros((total_number_of_samples,), dtype=np.float32)

    delay_line_read_index = 0

    # Previous sample for the simple 2-point averaging filter.
    previous_delay_line_sample = delay_line_buffer[-1]

    for output_sample_index in range(total_number_of_samples):
        current_delay_line_sample = float(delay_line_buffer[delay_line_read_index])

        # 2-point average (a simple lowpass / loss model)
        two_point_average = 0.5 * (previous_delay_line_sample + current_delay_line_sample)

        filtered_sample = (
            (1.0 - lowpass_blend) * current_delay_line_sample
            + lowpass_blend * two_point_average
        )

        next_sample_value = feedback_decay_factor * filtered_sample

        output_samples[output_sample_index] = current_delay_line_sample

        delay_line_buffer[delay_line_read_index] = next_sample_value

        previous_delay_line_sample = current_delay_line_sample

        delay_line_read_index += 1
        if delay_line_read_index >= delay_line_length_samples:
            delay_line_read_index = 0

    output_samples = normalise_peak_amplitude(output_samples, 0.95)

    return GeneratedSignal(samples=output_samples, sample_rate_hz=sample_rate_hz)



def duplicate_mono_to_stereo(
    mono_samples: np.ndarray,
) -> np.ndarray:
    """
    Duplicate a mono signal to stereo (L = R).
    """
    mono_samples = np.asarray(mono_samples, dtype=np.float32)
    return np.stack(
        [mono_samples, mono_samples],
        axis=1,
    )

# gen/cli.py
"""
Command Line Interface (CLI) for generating offline reverb test signals.

Usage examples:
  python -m gen.cli --help
  python -m gen.cli impulse --output impulse.wav
  python -m gen.cli noise_burst --burst_duration_seconds 0.02 --window_type hann
  python -m gen.cli sweep --duration_seconds 10 --start_frequency_hz 20 --end_frequency_hz 20000

All outputs are stereo WAV at 48 kHz by default (L = R).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from scipy.io import wavfile
except ImportError as import_error:  # pragma: no cover
    raise ImportError(
        "scipy is required for WAV writing. Install with: pip install scipy"
    ) from import_error

from gen.signals import (
    GeneratedSignal,
    duplicate_mono_to_stereo,
    generate_click,
    generate_impulse,
    generate_impulse_train,
    generate_log_sine_sweep,
    generate_noise,
    generate_noise_burst,
    generate_pluck_like,
    generate_karplus_strong_pluck,
    generate_sine,
    generate_sine_burst,
)


DEFAULT_SAMPLE_RATE_HZ = 48_000


def write_wav_file_pcm16(
    output_file_path: Path,
    samples_float32: np.ndarray,
    sample_rate_hz: int,
) -> None:
    """
    Write mono or stereo float32 samples to 16-bit PCM WAV.

    Accepted shapes:
    - mono:  (num_samples,) or (num_samples, 1)
    - stereo:(num_samples, 2)
    """
    samples_float32 = np.asarray(samples_float32, dtype=np.float32)

    if samples_float32.ndim == 2 and samples_float32.shape[1] == 1:
        samples_float32 = samples_float32[:, 0]  # flatten to (N,)

    if samples_float32.ndim == 1:
        pass  # mono is fine
    elif samples_float32.ndim == 2 and samples_float32.shape[1] == 2:
        pass  # stereo is fine
    else:
        raise ValueError(
            f"Expected mono (N) or stereo (N,2). Got shape {samples_float32.shape}"
        )

    clipped_samples = np.clip(samples_float32, -1.0, 1.0)
    int16_samples = (clipped_samples * 32767.0).astype(np.int16)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_file_path), sample_rate_hz, int16_samples)



def ensure_wav_suffix(output_file_path: Path) -> Path:
    if output_file_path.suffix.lower() != ".wav":
        return output_file_path.with_suffix(".wav")
    return output_file_path


def default_output_filename(signal_name: str) -> str:
    return f"{signal_name}.wav"


def parse_arguments() -> argparse.Namespace:
    top_level_parser = argparse.ArgumentParser(
        prog="gen",
        description="Generate offline stereo WAV test signals for reverb analysis (48 kHz by default).",
    )

    top_level_parser.add_argument(
        "--output-dir",
        dest="output_directory",
        type=str,
        default="test_tones",
        help="Directory to write generated WAV files (default: ./test_tones).",
    )

    top_level_parser.add_argument(
        "--channel_mode",
        type=str,
        default="mono",
        choices=["mono", "stereo"],
        help="Output channel mode (default: mono).",
    )

    top_level_parser.add_argument(
        "--sample_rate_hz",
        type=int,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="Sample rate in Hz (default: 48000).",
    )

    subparsers = top_level_parser.add_subparsers(
        dest="command_name",
        required=True,
        help="Signal type to generate. Use: gen <command> --help",
    )

    # -------------------------
    # impulse
    # -------------------------
    impulse_parser = subparsers.add_parser(
        "impulse",
        help="Single-sample Dirac impulse inside a fixed-length buffer.",
    )
    impulse_parser.add_argument(
        "--duration",
        dest="total_duration_seconds",
        type=float,
        default=1.0,
        help="Total buffer duration in seconds (default: 1.0).",
    )
    impulse_parser.add_argument(
        "--impulse_sample_index",
        type=int,
        default=0,
        help="Sample index where the impulse occurs (default: 0).",
    )
    impulse_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("impulse"),
        help="Output WAV filename (default: impulse.wav).",
    )

    # -------------------------
    # click
    # -------------------------
    click_parser = subparsers.add_parser(
        "click",
        help="Short windowed pulse (often more practical than a single-sample impulse).",
    )
    click_parser.add_argument(
        "--duration",
        dest="click_duration_seconds",
        type=float,
        default=0.001,
        help="Click duration in seconds (default: 0.001).",
    )
    click_parser.add_argument(
        "--window_type",
        type=str,
        default="hann",
        choices=["rect", "hann", "hamming", "blackman"],
        help="Window shape applied to the click (default: hann).",
    )
    click_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("click"),
        help="Output WAV filename (default: click.wav).",
    )

    # -------------------------
    # impulse_train
    # -------------------------
    impulse_train_parser = subparsers.add_parser(
        "impulse_train",
        help="Periodic train of clicks for time-variance and periodicity checks.",
    )
    impulse_train_parser.add_argument(
        "--duration",
        dest="total_duration_seconds",
        type=float,
        default=2.0,
        help="Total duration in seconds (default: 2.0).",
    )
    impulse_train_parser.add_argument(
        "--period",
        dest="impulse_period_seconds",
        type=float,
        default=0.25,
        help="Time between impulses in seconds (default: 0.25).",
    )
    impulse_train_parser.add_argument(
        "--click-duration",
        dest="click_duration_seconds",
        type=float,
        default=0.001,
        help="Click duration in seconds (default: 0.001).",
    )
    impulse_train_parser.add_argument(
        "--window_type",
        type=str,
        default="hann",
        choices=["rect", "hann", "hamming", "blackman"],
        help="Window shape applied to each click (default: hann).",
    )
    impulse_train_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("impulse_train"),
        help="Output WAV filename (default: impulse_train.wav).",
    )

    # -------------------------
    # noise_long
    # -------------------------
    noise_long_parser = subparsers.add_parser(
        "noise_long",
        help="Long noise signal for steady-state behaviour (diffusion / modulation stats).",
    )
    noise_long_parser.add_argument(
        "--duration_seconds",
        type=float,
        default=3.0,
        help="Noise duration in seconds (default: 3.0).",
    )
    noise_long_parser.add_argument(
        "--noise_type",
        type=str,
        default="white",
        choices=["white", "pink"],
        help="Noise type (default: white).",
    )
    noise_long_parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for repeatability (default: 0).",
    )
    noise_long_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("noise_long"),
        help="Output WAV filename (default: noise_long.wav).",
    )

    # -------------------------
    # noise_burst
    # -------------------------
    noise_burst_parser = subparsers.add_parser(
        "noise_burst",
        help="Short windowed noise burst (10–50 ms typical) for density/diffusion tests.",
    )
    noise_burst_parser.add_argument(
        "--duration",
        dest="burst_duration_seconds",
        type=float,
        default=0.02,
        help="Burst duration in seconds (default: 0.02).",
    )
    noise_burst_parser.add_argument(
        "--noise_type",
        type=str,
        default="white",
        choices=["white", "pink"],
        help="Noise type (default: white).",
    )
    noise_burst_parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for repeatability (default: 0).",
    )
    noise_burst_parser.add_argument(
        "--window_type",
        type=str,
        default="hann",
        choices=["rect", "hann", "hamming", "blackman"],
        help="Window applied to the burst (default: hann).",
    )
    noise_burst_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("noise_burst"),
        help="Output WAV filename (default: noise_burst.wav).",
    )

    # -------------------------
    # sine_sustain
    # -------------------------
    sine_sustain_parser = subparsers.add_parser(
        "sine_sustain",
        help="Sustained sine wave for modulation/pitch-stability tests.",
    )
    sine_sustain_parser.add_argument(
        "--freq",
        dest="frequency_hz",
        type=float,
        default=440.0,
        help="Sine frequency in Hz (default: 440).",
    )
    sine_sustain_parser.add_argument(
        "--duration_seconds",
        type=float,
        default=5.0,
        help="Duration in seconds (default: 5.0).",
    )
    sine_sustain_parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Amplitude (0..1) (default: 0.5).",
    )
    sine_sustain_parser.add_argument(
        "--initial_phase_radians",
        type=float,
        default=0.0,
        help="Initial phase in radians (default: 0.0).",
    )
    sine_sustain_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("sine_sustain"),
        help="Output WAV filename (default: sine_sustain.wav).",
    )

    # -------------------------
    # sine_burst
    # -------------------------
    sine_burst_parser = subparsers.add_parser(
        "sine_burst",
        help="Windowed sine burst for modal decay / ringing tests.",
    )
    sine_burst_parser.add_argument(
        "--freq",
        dest="frequency_hz",
        type=float,
        default=220.0,
        help="Sine frequency in Hz (default: 220).",
    )
    sine_burst_parser.add_argument(
        "--duration",
        dest="burst_duration_seconds",
        type=float,
        default=0.1,
        help="Burst duration in seconds (default: 0.1).",
    )
    sine_burst_parser.add_argument(
        "--amplitude",
        type=float,
        default=0.7,
        help="Amplitude (0..1) (default: 0.7).",
    )
    sine_burst_parser.add_argument(
        "--window_type",
        type=str,
        default="hann",
        choices=["rect", "hann", "hamming", "blackman"],
        help="Window applied to the burst (default: hann).",
    )
    sine_burst_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("sine_burst"),
        help="Output WAV filename (default: sine_burst.wav).",
    )

    # -------------------------
    # sweep
    # -------------------------
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Logarithmic sine sweep for robust IR extraction via deconvolution.",
    )
    sweep_parser.add_argument(
        "--duration_seconds",
        type=float,
        default=10.0,
        help="Sweep duration in seconds (default: 10.0).",
    )
    sweep_parser.add_argument(
        "--start-freq",
        dest="start_frequency_hz",
        type=float,
        default=20.0,
        help="Start frequency in Hz (default: 20).",
    )
    sweep_parser.add_argument(
        "--end-freq",
        dest="end_frequency_hz",
        type=float,
        default=20_000.0,
        help="End frequency in Hz (default: 20000).",
    )
    sweep_parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Amplitude (0..1) (default: 0.5).",
    )
    sweep_parser.add_argument(
        "--fade_duration_seconds",
        type=float,
        default=0.01,
        help="Fade-in/out duration in seconds (default: 0.01).",
    )
    sweep_parser.add_argument(
        "--pre_silence_seconds",
        type=float,
        default=1.0,
        help="Silent padding before sweep for deconvolution (default: 1.0).",
    )
    sweep_parser.add_argument(
        "--post_silence_seconds",
        type=float,
        default=2.0,
        help="Silent padding after sweep for IR tail capture (default: 2.0).",
    )
    sweep_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("sweep"),
        help="Output WAV filename (default: sweep.wav).",
    )

    # -------------------------
    # pluck
    # -------------------------
    pluck_parser = subparsers.add_parser(
        "pluck",
        help="Synthetic muted-pluck proxy (band-limited noise with exponential decay).",
    )
    pluck_parser.add_argument(
        "--duration_seconds",
        type=float,
        default=0.15,
        help="Pluck duration in seconds (default: 0.15).",
    )
    pluck_parser.add_argument(
        "--bandlimit",
        dest="bandlimit_frequency_hz",
        type=float,
        default=8000.0,
        help="Lowpass cutoff for the excitation in Hz (default: 8000).",
    )
    pluck_parser.add_argument(
        "--decay",
        dest="decay_time_constant_seconds",
        type=float,
        default=0.03,
        help="Exponential decay time constant in seconds (default: 0.03).",
    )
    pluck_parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for repeatability (default: 0).",
    )
    pluck_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("pluck"),
        help="Output WAV filename (default: pluck.wav).",
    )

    # -------------------------
    # karplus_pluck
    # -------------------------
    karplus_pluck_parser = subparsers.add_parser(
        "karplus_pluck",
        help="Karplus–Strong pluck (string-like physical model).",
    )
    karplus_pluck_parser.add_argument(
        "--freq",
        dest="fundamental_frequency_hz",
        type=float,
        default=110.0,
        help="Fundamental frequency in Hz (default: 110).",
    )
    karplus_pluck_parser.add_argument(
        "--duration_seconds",
        type=float,
        default=2.0,
        help="Output duration in seconds (default: 2.0).",
    )
    karplus_pluck_parser.add_argument(
        "--bandlimit",
        dest="excitation_noise_bandlimit_hz",
        type=float,
        default=8000.0,
        help="Lowpass cutoff for initial excitation noise in Hz (default: 8000).",
    )
    karplus_pluck_parser.add_argument(
        "--feedback_decay_factor",
        type=float,
        default=0.996,
        help="Feedback decay factor (0..1, exclusive). Closer to 1 = longer sustain (default: 0.996).",
    )
    karplus_pluck_parser.add_argument(
        "--lowpass_blend",
        type=float,
        default=0.5,
        help="0..1 blend amount for 2-point averaging loss filter (default: 0.5).",
    )
    karplus_pluck_parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for repeatability (default: 0).",
    )
    karplus_pluck_parser.add_argument(
        "--output",
        type=str,
        default=default_output_filename("karplus_pluck"),
        help="Output WAV filename (default: karplus_pluck.wav).",
    )

    # -------------------------
    # all
    # -------------------------
    all_parser = subparsers.add_parser(
        "all",
        help="Generate all test tones with default settings.",
    )

    return top_level_parser.parse_args()


def generate_signal_from_arguments(parsed_arguments: argparse.Namespace) -> Tuple[str, GeneratedSignal, Path]:
    sample_rate_hz = int(parsed_arguments.sample_rate_hz)
    command_name = str(parsed_arguments.command_name)

    if command_name == "impulse":
        generated_signal = generate_impulse(
            sample_rate_hz=sample_rate_hz,
            impulse_sample_index=int(parsed_arguments.impulse_sample_index),
            total_duration_seconds=float(parsed_arguments.total_duration_seconds),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "click":
        generated_signal = generate_click(
            sample_rate_hz=sample_rate_hz,
            click_duration_seconds=float(parsed_arguments.click_duration_seconds),
            window_type=str(parsed_arguments.window_type),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "impulse_train":
        generated_signal = generate_impulse_train(
            sample_rate_hz=sample_rate_hz,
            total_duration_seconds=float(parsed_arguments.total_duration_seconds),
            impulse_period_seconds=float(parsed_arguments.impulse_period_seconds),
            click_duration_seconds=float(parsed_arguments.click_duration_seconds),
            window_type=str(parsed_arguments.window_type),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "noise_long":
        generated_signal = generate_noise(
            sample_rate_hz=sample_rate_hz,
            duration_seconds=float(parsed_arguments.duration_seconds),
            noise_type=str(parsed_arguments.noise_type),
            random_seed=int(parsed_arguments.random_seed),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "noise_burst":
        generated_signal = generate_noise_burst(
            sample_rate_hz=sample_rate_hz,
            burst_duration_seconds=float(parsed_arguments.burst_duration_seconds),
            noise_type=str(parsed_arguments.noise_type),
            random_seed=int(parsed_arguments.random_seed),
            window_type=str(parsed_arguments.window_type),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "sine_sustain":
        generated_signal = generate_sine(
            sample_rate_hz=sample_rate_hz,
            frequency_hz=float(parsed_arguments.frequency_hz),
            duration_seconds=float(parsed_arguments.duration_seconds),
            amplitude=float(parsed_arguments.amplitude),
            initial_phase_radians=float(parsed_arguments.initial_phase_radians),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "sine_burst":
        generated_signal = generate_sine_burst(
            sample_rate_hz=sample_rate_hz,
            frequency_hz=float(parsed_arguments.frequency_hz),
            burst_duration_seconds=float(parsed_arguments.burst_duration_seconds),
            amplitude=float(parsed_arguments.amplitude),
            window_type=str(parsed_arguments.window_type),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "sweep":
        generated_signal = generate_log_sine_sweep(
            sample_rate_hz=sample_rate_hz,
            duration_seconds=float(parsed_arguments.duration_seconds),
            start_frequency_hz=float(parsed_arguments.start_frequency_hz),
            end_frequency_hz=float(parsed_arguments.end_frequency_hz),
            amplitude=float(parsed_arguments.amplitude),
            fade_duration_seconds=float(parsed_arguments.fade_duration_seconds),
            pre_silence_seconds=float(parsed_arguments.pre_silence_seconds),
            post_silence_seconds=float(parsed_arguments.post_silence_seconds),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "pluck":
        generated_signal = generate_pluck_like(
            sample_rate_hz=sample_rate_hz,
            duration_seconds=float(parsed_arguments.duration_seconds),
            bandlimit_frequency_hz=float(parsed_arguments.bandlimit_frequency_hz),
            decay_time_constant_seconds=float(parsed_arguments.decay_time_constant_seconds),
            random_seed=int(parsed_arguments.random_seed),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    if command_name == "karplus_pluck":
        generated_signal = generate_karplus_strong_pluck(
            sample_rate_hz=sample_rate_hz,
            fundamental_frequency_hz=float(parsed_arguments.fundamental_frequency_hz),
            duration_seconds=float(parsed_arguments.duration_seconds),
            excitation_noise_bandlimit_hz=float(parsed_arguments.excitation_noise_bandlimit_hz),
            feedback_decay_factor=float(parsed_arguments.feedback_decay_factor),
            lowpass_blend=float(parsed_arguments.lowpass_blend),
            random_seed=int(parsed_arguments.random_seed),
        )
        output_path = Path(parsed_arguments.output)
        return command_name, generated_signal, output_path

    raise ValueError(f"Unknown command: {command_name}")


def main() -> None:
    parsed_arguments = parse_arguments()

    command_name = str(parsed_arguments.command_name)
    
    if command_name == "all":
        # Generate all test tones with default settings
        sample_rate_hz = int(parsed_arguments.sample_rate_hz)
        channel_mode = str(parsed_arguments.channel_mode)
        output_dir = Path(parsed_arguments.output_directory)
        
        all_signals = [
            ("impulse", generate_impulse(sample_rate_hz=sample_rate_hz)),
            ("click", generate_click(sample_rate_hz=sample_rate_hz)),
            ("impulse_train", generate_impulse_train(sample_rate_hz=sample_rate_hz)),
            ("noise_long", generate_noise(sample_rate_hz=sample_rate_hz, duration_seconds=10.0)),
            ("noise_burst", generate_noise_burst(sample_rate_hz=sample_rate_hz)),
            ("sine_sustain", generate_sine(sample_rate_hz=sample_rate_hz, frequency_hz=1000.0, duration_seconds=1.0)),
            ("sine_burst", generate_sine_burst(sample_rate_hz=sample_rate_hz, frequency_hz=1000.0)),
            ("sweep", generate_log_sine_sweep(sample_rate_hz=sample_rate_hz)),
            ("pluck", generate_pluck_like(sample_rate_hz=sample_rate_hz)),
            ("karplus_pluck", generate_karplus_strong_pluck(sample_rate_hz=sample_rate_hz, fundamental_frequency_hz=110.0)),
        ]
        
        for name, signal in all_signals:
            output_path = ensure_wav_suffix(output_dir / default_output_filename(name))
            
            if channel_mode == "mono":
                output_samples = signal.samples
            elif channel_mode == "stereo":
                output_samples = duplicate_mono_to_stereo(signal.samples)
            else:
                raise ValueError(f"Unknown channel_mode: {channel_mode}")
            
            write_wav_file_pcm16(
                output_file_path=output_path,
                samples_float32=output_samples,
                sample_rate_hz=signal.sample_rate_hz,
            )
            
            if output_samples.ndim == 1:
                channel_count = 1
            else:
                channel_count = int(output_samples.shape[1])
            
            print(f"Wrote {output_path} ({output_samples.shape[0]} samples, {signal.sample_rate_hz} Hz, {channel_count} channel(s))")
        
        return

    command_name, generated_signal, output_file_path = generate_signal_from_arguments(parsed_arguments)

    # Combine with output_directory
    output_dir = Path(parsed_arguments.output_directory)
    output_file_path = output_dir / output_file_path
    output_file_path = ensure_wav_suffix(output_file_path)

    channel_mode = str(parsed_arguments.channel_mode)

    if channel_mode == "mono":
        output_samples = generated_signal.samples  # shape (N,)
    elif channel_mode == "stereo":
        output_samples = duplicate_mono_to_stereo(generated_signal.samples)  # shape (N,2)
    else:
        raise ValueError(f"Unknown channel_mode: {channel_mode}")

    write_wav_file_pcm16(
        output_file_path=output_file_path,
        samples_float32=output_samples,
        sample_rate_hz=generated_signal.sample_rate_hz,
    )

    if output_samples.ndim == 1:
        channel_count = 1
    else:
        channel_count = int(output_samples.shape[1])

    print(f"Wrote {output_file_path} ({output_samples.shape[0]} samples, {generated_signal.sample_rate_hz} Hz, {channel_count} channel(s))")


if __name__ == "__main__":
    main()

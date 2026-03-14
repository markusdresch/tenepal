"""Audio preprocessing: normalization, resampling, format conversion."""

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
from scipy import signal

from .loader import AudioData


def preprocess_audio(audio: AudioData, target_sr: int = 16000) -> AudioData:
    """Preprocess audio: convert to mono, resample, normalize.

    Args:
        audio: Input AudioData
        target_sr: Target sample rate in Hz (default: 16000 for Allosaurus)

    Returns:
        Preprocessed AudioData with mono samples at target_sr
    """
    samples = audio.samples

    # Convert stereo to mono (average channels)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    # Resample if needed
    if audio.sample_rate != target_sr:
        # Calculate number of samples after resampling
        num_samples = int(len(samples) * target_sr / audio.sample_rate)
        samples = signal.resample(samples, num_samples)

    # Normalize to peak amplitude of 0.9 (prevent clipping)
    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples * (0.9 / peak)

    # Ensure float32 dtype
    samples = samples.astype(np.float32)

    # Calculate new duration
    duration = len(samples) / target_sr

    return AudioData(
        samples=samples,
        sample_rate=target_sr,
        duration=duration,
        source_format=audio.source_format
    )


def save_wav(audio: AudioData, path: Union[str, Path]) -> Path:
    """Save AudioData to WAV file.

    Args:
        audio: AudioData to save
        path: Output file path

    Returns:
        Path object of the saved file
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write WAV file
    sf.write(str(path), audio.samples, audio.sample_rate)

    return path

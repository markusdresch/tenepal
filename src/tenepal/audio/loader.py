"""Audio file loading for multiple formats."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
from pydub import AudioSegment


@dataclass
class AudioData:
    """Container for audio data with metadata."""
    samples: np.ndarray  # Audio samples as float32 array
    sample_rate: int  # Sample rate in Hz
    duration: float  # Duration in seconds
    source_format: str  # Original file format (wav, mp3, flac)


def load_audio(path: Union[str, Path]) -> AudioData:
    """Load audio file from WAV, MP3, or FLAC format.

    Args:
        path: Path to audio file

    Returns:
        AudioData object containing samples and metadata

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported
    """
    path = Path(path)

    # Validate file exists
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Validate file extension
    extension = path.suffix.lower()
    supported_formats = {".wav", ".mp3", ".flac"}
    if extension not in supported_formats:
        raise ValueError(
            f"Unsupported audio format: {extension}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    # Load based on format
    if extension == ".wav":
        # Use soundfile for WAV (efficient, preserves metadata)
        samples, sample_rate = sf.read(str(path), dtype="float32")
    else:
        # Use pydub for MP3 and FLAC (requires ffmpeg)
        audio_segment = AudioSegment.from_file(str(path), format=extension[1:])

        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

        # Normalize from int16 range to float32 [-1.0, 1.0]
        samples = samples / 32768.0

        # Handle stereo: reshape if needed
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))

        sample_rate = audio_segment.frame_rate

    # Calculate duration
    if samples.ndim == 1:
        duration = len(samples) / sample_rate
    else:
        duration = samples.shape[0] / sample_rate

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        source_format=extension[1:]
    )

"""Silero-VAD based speech segmentation with torch.no_grad wrapper."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from .extractor import SpeechSegment

logger = logging.getLogger(__name__)

# Module-level cache for VAD model (lazy-load)
_vad_model = None
_vad_utils = None
_vad_loaded = False


def _load_silero_vad() -> tuple[Optional[torch.nn.Module], Optional[tuple]]:
    """Lazy-load Silero-VAD model and utilities.

    Returns:
        Tuple of (model, utils) if successful, (None, None) otherwise
    """
    global _vad_model, _vad_utils, _vad_loaded

    if _vad_loaded:
        return (_vad_model, _vad_utils)

    try:
        # Set single thread for Silero recommendation
        torch.set_num_threads(1)

        logger.info("Loading Silero-VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )

        _vad_model = model
        _vad_utils = utils
        _vad_loaded = True
        logger.info("Silero-VAD model loaded successfully")
        return (_vad_model, _vad_utils)

    except Exception as exc:
        logger.warning("Failed to load Silero-VAD model: %s", exc)
        _vad_model = None
        _vad_utils = None
        _vad_loaded = True
        return (None, None)


def segment_speech(
    audio_path: Path,
    output_dir: Path,
    min_silence_ms: int = 500,
    padding_ms: int = 100,
) -> list[SpeechSegment]:
    """Segment audio into speech chunks using Silero-VAD.

    Args:
        audio_path: Path to input audio file (WAV)
        output_dir: Directory for output (used for debugging, not saved here)
        min_silence_ms: Minimum silence duration between segments (default: 500)
        padding_ms: Padding before/after each segment boundary (default: 100)

    Returns:
        List of SpeechSegment objects with timestamped audio data

    Raises:
        RuntimeError: If Silero-VAD failed to load
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAD model
    model, utils = _load_silero_vad()
    if model is None or utils is None:
        raise RuntimeError("Silero-VAD failed to load. Check torch installation.")

    # Unpack utils
    get_speech_timestamps, _, read_audio, *_ = utils

    # Read audio at 16000 Hz (Silero requirement)
    logger.info("Segmenting speech in %s", audio_path.name)
    wav_16k = read_audio(str(audio_path), sampling_rate=16000)

    # Run VAD inference with torch.no_grad to prevent memory leak
    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(
            wav_16k,
            model,
            sampling_rate=16000,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=padding_ms,
            return_seconds=False,  # Return sample indices
        )

    # Load original audio at its native sample rate for slicing
    original_audio, original_sr = sf.read(str(audio_path), dtype="float32")

    # Ensure mono
    if original_audio.ndim > 1:
        original_audio = np.mean(original_audio, axis=1)

    # Convert VAD timestamps (16kHz sample indices) to milliseconds
    # Then convert milliseconds to sample indices at original sample rate
    segments: list[SpeechSegment] = []

    for ts in speech_timestamps:
        # Convert 16kHz sample indices to milliseconds
        start_ms = (ts['start'] * 1000) // 16000
        end_ms = (ts['end'] * 1000) // 16000

        # Convert milliseconds to sample indices at original sample rate
        start_sample = int((start_ms * original_sr) / 1000)
        end_sample = int((end_ms * original_sr) / 1000)

        # Clamp to valid range
        start_sample = max(0, min(start_sample, len(original_audio)))
        end_sample = max(0, min(end_sample, len(original_audio)))

        # Skip zero-length segments
        if end_sample <= start_sample:
            continue

        # Slice audio data
        audio_slice = original_audio[start_sample:end_sample]

        # Create SpeechSegment
        segment = SpeechSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            audio_data=audio_slice,
            sample_rate=original_sr,
        )
        segments.append(segment)

    # Calculate total duration for logging
    duration_sec = len(original_audio) / original_sr
    logger.info(
        "Found %d speech segments in %.1fs audio",
        len(segments),
        duration_sec,
    )

    return segments

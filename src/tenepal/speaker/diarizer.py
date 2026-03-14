"""Speaker diarization using pyannote.audio with graceful fallback."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

from ..audio.loader import AudioData

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A labeled speaker segment with timing information."""
    speaker: str       # "Speaker A", "Speaker B", etc.
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds


# Module-level cache for pyannote pipeline (lazy-load)
_pipeline = None
_pipeline_loaded = False


def _letter_label(index: int) -> str:
    """Convert 0-based index to letter label.

    0 -> 'Speaker A', 25 -> 'Speaker Z', 26 -> 'Speaker AA', 27 -> 'Speaker AB'.
    """
    if index < 26:
        return f"Speaker {chr(65 + index)}"
    # For >26 speakers, use double letters
    first = chr(65 + (index // 26) - 1)
    second = chr(65 + (index % 26))
    return f"Speaker {first}{second}"


def _load_pipeline():
    """Lazy-load pyannote speaker diarization pipeline.

    Returns the pipeline if available, or None for fallback mode.
    Caches result to avoid repeated load attempts.
    """
    global _pipeline, _pipeline_loaded

    if _pipeline_loaded:
        return _pipeline

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.info("HUGGINGFACE_TOKEN not set, using fallback diarization")
        _pipeline_loaded = True
        return None

    try:
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
        _pipeline_loaded = True
        return _pipeline
    except ImportError:
        logger.warning("pyannote.audio not installed, using fallback diarization")
        _pipeline_loaded = True
        return None
    except Exception as exc:
        logger.warning("Failed to load pyannote pipeline: %s", exc)
        _pipeline_loaded = True
        return None


def diarize(audio_path: Union[str, Path], use_docker: bool = False) -> list[SpeakerSegment]:
    """Diarize audio file to detect and label speakers.

    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC)
        use_docker: Whether to run diarization inside Docker GPU container

    Returns:
        List of SpeakerSegment objects sorted by start_time, then speaker label.
        Falls back to a single 'Speaker ?' segment when pyannote is unavailable.
    """
    audio_path = Path(audio_path)

    if use_docker:
        from .docker_diarizer import DockerDiarizer

        if DockerDiarizer.is_available():
            logger.info("Using Docker GPU diarization")
            return DockerDiarizer().diarize(audio_path)
        logger.warning("Docker diarization unavailable, falling back to host CPU")
    pipeline = _load_pipeline()

    if pipeline is None:
        # Fallback: return single segment spanning entire audio
        info = sf.info(str(audio_path))
        duration = info.duration
        return [SpeakerSegment(speaker="Speaker ?", start_time=0.0, end_time=duration)]

    # Run pyannote diarization
    result = pipeline(str(audio_path))

    # pyannote 4.x returns DiarizeOutput dataclass; extract Annotation
    annotation = getattr(result, "speaker_diarization", result)

    # Build label mapping: assign lettered labels by order of first appearance
    label_map: dict[str, str] = {}
    label_counter = 0
    segments: list[SpeakerSegment] = []

    for segment, _track, speaker_label in annotation.itertracks(yield_label=True):
        if speaker_label not in label_map:
            label_map[speaker_label] = _letter_label(label_counter)
            label_counter += 1

        segments.append(SpeakerSegment(
            speaker=label_map[speaker_label],
            start_time=segment.start,
            end_time=segment.end,
        ))

    # Sort by start_time, then by speaker label for deterministic output
    segments.sort(key=lambda s: (s.start_time, s.speaker))

    return segments


def slice_audio_by_speaker(
    audio: AudioData,
    segments: list[SpeakerSegment],
) -> list[tuple[SpeakerSegment, AudioData]]:
    """Slice audio data by speaker segments.

    Args:
        audio: AudioData object with samples and metadata
        segments: List of SpeakerSegment objects defining time ranges

    Returns:
        List of (SpeakerSegment, AudioData) tuples with extracted audio.
        Segments with zero-length audio after clamping are skipped.
    """
    total_samples = len(audio.samples)
    result: list[tuple[SpeakerSegment, AudioData]] = []

    for seg in segments:
        # Convert seconds to sample indices
        start_idx = int(seg.start_time * audio.sample_rate)
        end_idx = int(seg.end_time * audio.sample_rate)

        # Clamp to valid range
        start_idx = max(0, min(start_idx, total_samples))
        end_idx = max(0, min(end_idx, total_samples))

        # Skip zero-length segments
        if end_idx <= start_idx:
            continue

        sliced_samples = audio.samples[start_idx:end_idx]
        sliced_duration = len(sliced_samples) / audio.sample_rate

        sliced_audio = AudioData(
            samples=sliced_samples,
            sample_rate=audio.sample_rate,
            duration=sliced_duration,
            source_format=audio.source_format,
        )

        result.append((seg, sliced_audio))

    return result

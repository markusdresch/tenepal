"""Pipeline orchestrator for video preprocessing: extract -> isolate -> segment."""

import json
import logging
from pathlib import Path

from .extractor import extract_audio, SpeechSegment
from .isolator import isolate_vocals
from .segmenter import segment_speech

logger = logging.getLogger(__name__)


def preprocess_video(
    video_path: Path,
    output_dir: Path,
    skip_isolation: bool = False,
    demucs_segment: int = 300,
) -> list[SpeechSegment]:
    """Preprocess video through three-stage pipeline: extract -> isolate -> segment.

    Args:
        video_path: Path to input video file (MKV or MP4)
        output_dir: Directory for intermediate and output files
        skip_isolation: If True, skip Demucs vocal isolation (for clean audio)
        demucs_segment: Segment size in seconds for Demucs processing

    Returns:
        List of SpeechSegment objects with timestamped audio data

    Raises:
        FileNotFoundError: If video file does not exist
        ValueError: If video format is not supported
        RuntimeError: If required tools are not installed
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Validate video path exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Extract audio from video
    logger.info("Stage 1/3: Extracting audio from %s...", video_path.name)
    extracted_wav = extract_audio(video_path, output_dir)

    # Stage 2: Isolate vocals (optional)
    if skip_isolation:
        logger.info("Stage 2/3: Skipping vocal isolation (--skip-isolation)")
        audio_to_segment = extracted_wav
    else:
        logger.info("Stage 2/3: Isolating vocals...")
        audio_to_segment = isolate_vocals(
            extracted_wav,
            output_dir,
            segment_size=demucs_segment,
        )

    # Stage 3: Segment speech
    logger.info("Stage 3/3: Segmenting speech...")
    segments = segment_speech(
        audio_to_segment,
        output_dir,
        min_silence_ms=500,  # User decision from CONTEXT
        padding_ms=100,      # User decision from CONTEXT
    )

    logger.info("Preprocessing complete: %d speech segments found", len(segments))

    # Export segments manifest
    manifest_path = output_dir / f"{video_path.stem}_segments.json"
    export_segments_json(segments, manifest_path)

    return segments


def export_segments_json(segments: list[SpeechSegment], output_path: Path) -> None:
    """Export segments manifest as JSON for debugging/inspection.

    Args:
        segments: List of SpeechSegment objects
        output_path: Path to output JSON file
    """
    # Calculate total speech duration
    total_speech_ms = sum(seg.end_ms - seg.start_ms for seg in segments)

    # Build manifest
    manifest = {
        "version": "1.0",
        "segment_count": len(segments),
        "total_speech_ms": total_speech_ms,
        "segments": [
            {
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "duration_ms": seg.end_ms - seg.start_ms,
            }
            for seg in segments
        ],
    }

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Exported segments manifest: %s", output_path)

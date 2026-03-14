"""FFmpeg-based audio extraction from video files."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A timestamped speech segment with audio data."""
    start_ms: int          # Start timestamp in milliseconds
    end_ms: int            # End timestamp in milliseconds
    audio_data: np.ndarray # Mono float32 audio samples
    sample_rate: int = 22050  # Default sample rate


def extract_audio(video_path: Path, output_dir: Path, sample_rate: int = 22050) -> Path:
    """Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file (MKV or MP4)
        output_dir: Directory for output WAV file
        sample_rate: Target sample rate in Hz (default: 22050)

    Returns:
        Path to extracted WAV file

    Raises:
        FileNotFoundError: If video file does not exist or has no audio track
        ValueError: If video format is not MKV or MP4
        subprocess.CalledProcessError: If ffmpeg extraction fails
    """
    video_path = Path(video_path)

    # Validate video file exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Validate video format
    extension = video_path.suffix.lower()
    if extension not in {".mkv", ".mp4"}:
        raise ValueError(
            f"Unsupported video format: {extension}. "
            f"Supported formats: .mkv, .mp4"
        )

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for audio streams using ffprobe
    logger.info("Checking audio streams in %s", video_path.name)
    try:
        probe_result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not probe_result.stdout.strip():
            raise FileNotFoundError(f"No audio track found in video: {video_path}")
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(
            f"Failed to probe video for audio streams: {video_path}\n{exc.stderr}"
        ) from exc

    # Construct output path
    output_path = output_dir / f"{video_path.stem}_extracted.wav"

    # Build ffmpeg command for audio extraction
    # -vn: no video
    # -map 0:a:0: select first audio stream
    # -ac 1: mono
    # -ar {sample_rate}: resample to target rate
    # -acodec pcm_s16le: 16-bit PCM WAV
    # -y: overwrite output file
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-map", "0:a:0",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        "-y",
        str(output_path),
    ]

    logger.info("Extracting audio from %s to %s", video_path.name, output_path.name)

    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise subprocess.CalledProcessError(
            exc.returncode,
            exc.cmd,
            exc.output,
            f"FFmpeg extraction failed:\n{exc.stderr}",
        ) from exc

    # Log extraction complete with file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Audio extraction complete: %s (%.2f MB)",
        output_path.name,
        file_size_mb,
    )

    return output_path

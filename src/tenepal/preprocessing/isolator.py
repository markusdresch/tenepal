"""Demucs-based vocal isolation with lazy model loading."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Module-level cache for Demucs separator (lazy-load)
_separator = None
_separator_loaded = False
_separator_segment_size = None


def _load_separator(segment_size: int = 300) -> Optional["Separator"]:
    """Lazy-load Demucs separator model.

    Args:
        segment_size: Segment size in seconds for processing long audio

    Returns:
        Separator instance if available, or None if demucs not installed
    """
    global _separator, _separator_loaded, _separator_segment_size

    # Return cached separator if already loaded with same segment size
    if _separator_loaded and _separator_segment_size == segment_size:
        return _separator

    # Try importing demucs
    try:
        from demucs.api import Separator
    except ImportError:
        logger.error("Demucs not installed. Install with: pip install demucs")
        _separator_loaded = True
        _separator = None
        _separator_segment_size = segment_size
        return None

    # Try instantiating separator
    try:
        logger.info("Loading Demucs model (htdemucs)...")
        # Note: segment parameter omitted to use model default; custom values
        # cause tensor shape mismatches in htdemucs. split=True enables chunked
        # processing for long audio.
        _separator = Separator(
            model="htdemucs",
            split=True,
        )
        _separator_loaded = True
        _separator_segment_size = segment_size
        logger.info("Demucs model loaded successfully")
        return _separator
    except Exception as exc:
        logger.warning("Failed to load Demucs model: %s", exc)
        _separator_loaded = True
        _separator = None
        _separator_segment_size = segment_size
        return None


def isolate_vocals(
    audio_path: Path,
    output_dir: Path,
    segment_size: int = 300,
) -> Path:
    """Isolate vocals from audio using Demucs source separation.

    Args:
        audio_path: Path to input audio file (WAV)
        output_dir: Directory for output isolated vocals WAV
        segment_size: Segment size in seconds for processing (default: 300)

    Returns:
        Path to isolated vocals WAV file

    Raises:
        RuntimeError: If Demucs is not installed
        MemoryError: If processing runs out of memory (with advice)
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load separator
    separator = _load_separator(segment_size)
    if separator is None:
        raise RuntimeError(
            "Demucs not installed. Install: pip install demucs\n"
            "Or bypass with --skip-isolation"
        )

    logger.info("Isolating vocals from %s", audio_path.name)

    # Separate audio using Demucs
    try:
        # Demucs 4.1+ returns (origin, dict); 4.0 returns dict only
        result = separator.separate_audio_file(str(audio_path))
        if isinstance(result, tuple):
            _origin, separated = result
        else:
            separated = result
    except (MemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower() or isinstance(exc, MemoryError):
            suggested_segment = segment_size // 2
            raise RuntimeError(
                f"Out of memory during vocal isolation. "
                f"Try --demucs-segment {suggested_segment} or reduce audio length"
            ) from exc
        raise

    # Extract vocals tensor
    vocals = separated["vocals"]  # Shape: (channels, samples)
    other = separated["other"]    # For confidence proxy

    # Compute isolation confidence proxy: RMS ratio
    # Lower ratio = better isolation (other stem is quieter relative to vocals)
    # Convert to numpy first since Demucs may return tensors
    vocals_np_rms = vocals.cpu().numpy() if hasattr(vocals, "cpu") else vocals
    other_np_rms = other.cpu().numpy() if hasattr(other, "cpu") else other
    vocals_rms = np.sqrt(np.mean(vocals_np_rms ** 2))
    other_rms = np.sqrt(np.mean(other_np_rms ** 2))

    if vocals_rms > 0:
        confidence_ratio = other_rms / vocals_rms
        if confidence_ratio > 0.8:
            logger.warning(
                "Low isolation confidence (%.2f). "
                "Consider --skip-isolation for this audio.",
                confidence_ratio,
            )
    else:
        logger.warning("Vocals stem is silent - isolation may have failed")

    # Save vocals to WAV
    output_path = output_dir / f"{audio_path.stem}_isolated.wav"

    # Demucs outputs shape (channels, samples), need (samples, channels) for soundfile
    # Also convert from tensor to numpy if needed
    if hasattr(vocals, "cpu"):  # PyTorch tensor
        vocals_np = vocals.cpu().numpy()
    else:
        vocals_np = vocals

    # Transpose to (samples, channels)
    vocals_np = vocals_np.T

    # Get sample rate from original file
    info = sf.info(str(audio_path))
    sample_rate = info.samplerate

    # Save isolated vocals
    sf.write(str(output_path), vocals_np, sample_rate, subtype="PCM_16")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Vocal isolation complete: %s (%.2f MB)",
        output_path.name,
        file_size_mb,
    )

    return output_path


def is_demucs_available() -> bool:
    """Check if Demucs is available for import.

    Returns:
        True if demucs.api can be imported, False otherwise
    """
    try:
        import demucs.api  # noqa: F401
        return True
    except ImportError:
        return False

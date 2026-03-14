"""Voice onset/offset detection for segment trimming.

Diarization segments (pyannote) include silence padding at start/end (50-200ms).
IPA backends (Allosaurus, wav2vec2) interpret silence frames as phonemes → garbage IPA.

This module provides signal-level trimming to find the exact voice onset/offset
within a diarization segment, using intensity analysis via Parselmouth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import parselmouth

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_sound_from_array(audio: NDArray, sr: int) -> parselmouth.Sound:
    """Create a parselmouth Sound from a numpy array.

    Args:
        audio: Audio samples as numpy array (mono, float or int)
        sr: Sample rate

    Returns:
        parselmouth.Sound object
    """
    # Ensure float64 in range [-1, 1]
    if audio.dtype != np.float64:
        audio = audio.astype(np.float64)
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0  # Assume int16 range
    return parselmouth.Sound(audio, sampling_frequency=float(sr))


def trim_to_voice(
    audio_or_sound: str | parselmouth.Sound | NDArray,
    sr: int | None,
    start_s: float,
    end_s: float,
    margin: float = 0.3,
    threshold_db: float = 15.0,
) -> tuple[float, float, dict]:
    """Find exact voice onset/offset within a diarization segment.

    Uses intensity contour analysis to detect when speech actually starts/ends,
    trimming silence padding that causes garbage phoneme output.

    Args:
        audio_or_sound: Path to wav/mp4, parselmouth.Sound, or numpy array
        sr: Sample rate (required if audio_or_sound is numpy array)
        start_s: Diarization start time (seconds)
        end_s: Diarization end time (seconds)
        margin: Extra context to load around segment (seconds)
        threshold_db: dB below peak to consider as voice activity

    Returns:
        (onset_s, offset_s, stats) — trimmed absolute timestamps and stats dict
    """
    if isinstance(audio_or_sound, (str, bytes)):
        snd = parselmouth.Sound(str(audio_or_sound))
    elif isinstance(audio_or_sound, np.ndarray):
        if sr is None:
            raise ValueError("sr (sample rate) required when passing numpy array")
        snd = create_sound_from_array(audio_or_sound, sr)
    else:
        snd = audio_or_sound

    # Extract chunk with margin for analysis
    chunk_start = max(0, start_s - margin)
    chunk_end = min(snd.xmax, end_s + margin)

    try:
        chunk = snd.extract_part(from_time=chunk_start, to_time=chunk_end)
        intensity = chunk.to_intensity(time_step=0.01)
    except Exception:
        # Too short for intensity analysis
        return start_s, end_s, {"trimmed": False, "reason": "too_short"}

    times = intensity.xs()
    values = np.array([intensity.get_value(t) for t in times])

    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return start_s, end_s, {"trimmed": False, "reason": "no_valid_intensity"}

    # Peak and threshold
    peak_db = np.nanmax(values)
    thresh = peak_db - threshold_db

    above = np.where(values > thresh)[0]
    if len(above) == 0:
        return start_s, end_s, {"trimmed": False, "reason": "below_threshold"}

    # Convert to absolute time
    onset_abs = times[above[0]] + chunk_start
    offset_abs = times[above[-1]] + chunk_start

    # Safety: don't expand beyond original + 100ms
    onset_abs = max(onset_abs, start_s - 0.1)
    offset_abs = min(offset_abs, end_s + 0.1)

    # Minimum segment duration: 80ms
    if offset_abs - onset_abs < 0.08:
        return start_s, end_s, {"trimmed": False, "reason": "too_short_after_trim"}

    stats = {
        "trimmed": True,
        "trim_start_ms": round((start_s - onset_abs) * 1000),
        "trim_end_ms": round((end_s - offset_abs) * 1000),
        "original_ms": round((end_s - start_s) * 1000),
        "trimmed_ms": round((offset_abs - onset_abs) * 1000),
    }
    return onset_abs, offset_abs, stats


def compute_trim_stats(
    original_start: float,
    original_end: float,
    trimmed_start: float,
    trimmed_end: float,
) -> dict:
    """Compute trimming statistics for a segment.

    Returns:
        dict with trim_start_ms, trim_end_ms, original_duration_ms, trimmed_duration_ms
    """
    return {
        "trim_start_ms": round((original_start - trimmed_start) * 1000),
        "trim_end_ms": round((original_end - trimmed_end) * 1000),
        "original_duration_ms": round((original_end - original_start) * 1000),
        "trimmed_duration_ms": round((trimmed_end - trimmed_start) * 1000),
    }

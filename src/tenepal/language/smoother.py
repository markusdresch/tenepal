"""Speaker-level language smoothing.

This module provides speaker-level language smoothing to correct short,
low-confidence outlier segments that arise from foreign names, brand words,
or ASR artifacts. Each speaker's primary language is determined by phoneme
count, and outlier segments are reclassified to match the speaker's dominant
language while preserving genuine code-switching.
"""

from dataclasses import dataclass

from tenepal.language.identifier import LanguageSegment

PROTECTED_SMOOTHING_LANGUAGES = frozenset({"may", "lat"})


@dataclass
class SpeakerLanguageStats:
    """Accumulated language statistics for one speaker.

    Attributes:
        primary_language: Most-spoken language (by phoneme count, excluding "other")
        total_phonemes: Total phonemes across all segments
        language_phoneme_counts: Phoneme count per language (dict[str, int])
        language_confidence: Sum of confidence per language (dict[str, float])
    """
    primary_language: str
    total_phonemes: int
    language_phoneme_counts: dict[str, int]
    language_confidence: dict[str, float]


def smooth_by_speaker(
    segments: list[LanguageSegment],
    min_phonemes: int = 5,
    min_confidence_ratio: float = 0.3
) -> list[LanguageSegment]:
    """Smooth language segments by speaker's primary language.

    Accumulates each speaker's primary language (by phoneme count) and
    reclassifies short or low-confidence outlier segments to the speaker's
    dominant language. Preserves genuine code-switching (long, high-confidence
    segments in a different language).

    Segments without speaker labels (speaker=None) are left untouched.

    Algorithm:
    1. Group segments by speaker (skip None speakers)
    2. For each speaker, compute SpeakerLanguageStats:
       - Count phonemes per language (excluding "other")
       - Primary language = language with most phonemes
       - Need at least 2 segments to establish primary language
    3. For each non-primary segment of each speaker:
       - If segment has fewer than min_phonemes → reclassify to primary
       - Else if segment confidence < threshold → reclassify to primary
         Threshold = min_confidence_ratio * (primary avg conf) * segment_phoneme_count
       - Otherwise → keep (genuine code-switching)
    4. After reclassification, merge adjacent segments with same language+speaker
    5. Return full list (smoothed speakers + untouched None-speaker segments),
       sorted by start_time

    Args:
        segments: List of LanguageSegment objects (may include speaker labels)
        min_phonemes: Minimum phonemes for a segment to avoid smoothing (default 5)
        min_confidence_ratio: Confidence ratio threshold for smoothing (default 0.3)

    Returns:
        List of LanguageSegment objects after speaker-level smoothing and merging
    """
    if not segments:
        return []

    # Separate speaker-labeled segments from None-speaker segments
    speaker_segments: dict[str, list[LanguageSegment]] = {}
    no_speaker_segments: list[LanguageSegment] = []

    for seg in segments:
        if seg.speaker is None:
            no_speaker_segments.append(seg)
        else:
            if seg.speaker not in speaker_segments:
                speaker_segments[seg.speaker] = []
            speaker_segments[seg.speaker].append(seg)

    # Process each speaker's segments independently
    smoothed_segments: list[LanguageSegment] = []

    for speaker, segs in speaker_segments.items():
        # Need at least 2 segments to establish a primary language
        if len(segs) < 2:
            smoothed_segments.extend(segs)
            continue

        # Compute speaker language statistics
        stats = _compute_speaker_stats(segs)

        # If no primary language (all "other"), keep as-is
        if not stats.primary_language:
            smoothed_segments.extend(segs)
            continue

        # Smooth each segment
        primary_avg_conf_per_phoneme = (
            stats.language_confidence[stats.primary_language] /
            stats.language_phoneme_counts[stats.primary_language]
        )

        for seg in segs:
            if seg.language == stats.primary_language:
                # Primary language → keep as-is
                smoothed_segments.append(seg)
            elif seg.language in PROTECTED_SMOOTHING_LANGUAGES:
                # Preserve explicit Maya/Latin detections through smoothing.
                smoothed_segments.append(seg)
            elif seg.language == "other":
                # Short OTH → reclassify to primary; long OTH → keep
                seg_phoneme_count = len(seg.phonemes)
                if seg_phoneme_count < min_phonemes:
                    smoothed_segments.append(_reclassify_segment(seg, stats.primary_language))
                else:
                    smoothed_segments.append(seg)
            else:
                # Non-primary segment: check if it should be smoothed
                seg_phoneme_count = len(seg.phonemes)

                # Condition 1: Too short
                if seg_phoneme_count < min_phonemes:
                    # Smooth to primary
                    smoothed_segments.append(_reclassify_segment(seg, stats.primary_language))
                else:
                    # Condition 2: Low confidence relative to primary
                    threshold = min_confidence_ratio * primary_avg_conf_per_phoneme * seg_phoneme_count
                    if seg.confidence < threshold:
                        # Smooth to primary
                        smoothed_segments.append(_reclassify_segment(seg, stats.primary_language))
                    else:
                        # Keep (genuine code-switching)
                        smoothed_segments.append(seg)

    # Add back None-speaker segments
    smoothed_segments.extend(no_speaker_segments)

    # Sort by start_time
    smoothed_segments.sort(key=lambda s: s.start_time)

    # Merge adjacent segments with same language and speaker
    merged = _merge_adjacent_segments(smoothed_segments)

    return merged


def _compute_speaker_stats(segments: list[LanguageSegment]) -> SpeakerLanguageStats:
    """Compute language statistics for a speaker's segments.

    Args:
        segments: List of LanguageSegment objects for one speaker

    Returns:
        SpeakerLanguageStats with primary language and statistics
    """
    language_phoneme_counts: dict[str, int] = {}
    language_confidence: dict[str, float] = {}
    total_phonemes = 0

    for seg in segments:
        phoneme_count = len(seg.phonemes)
        total_phonemes += phoneme_count

        # Exclude "other" from primary language determination
        if seg.language != "other":
            language_phoneme_counts[seg.language] = (
                language_phoneme_counts.get(seg.language, 0) + phoneme_count
            )
            language_confidence[seg.language] = (
                language_confidence.get(seg.language, 0.0) + seg.confidence
            )

    # Primary language = language with most phonemes (excluding "other")
    if not language_phoneme_counts:
        primary_language = ""
    else:
        primary_language = max(language_phoneme_counts.items(), key=lambda x: x[1])[0]

    return SpeakerLanguageStats(
        primary_language=primary_language,
        total_phonemes=total_phonemes,
        language_phoneme_counts=language_phoneme_counts,
        language_confidence=language_confidence
    )


def _reclassify_segment(segment: LanguageSegment, new_language: str) -> LanguageSegment:
    """Create a new segment with updated language, preserving all other fields.

    Args:
        segment: Original LanguageSegment
        new_language: New language code

    Returns:
        New LanguageSegment with updated language
    """
    return LanguageSegment(
        language=new_language,
        phonemes=segment.phonemes,
        start_time=segment.start_time,
        end_time=segment.end_time,
        speaker=segment.speaker,
        confidence=segment.confidence
    )


def _merge_adjacent_segments(segments: list[LanguageSegment]) -> list[LanguageSegment]:
    """Merge adjacent segments with same language and speaker.

    Args:
        segments: List of LanguageSegment objects (sorted by start_time)

    Returns:
        List of LanguageSegment objects after merging
    """
    if not segments:
        return []

    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]

        # Check if this segment can merge with previous
        if seg.language == prev.language and seg.speaker == prev.speaker:
            # Merge: combine phonemes, update end_time, sum confidence
            merged_phonemes = prev.phonemes + seg.phonemes
            merged_confidence = prev.confidence + seg.confidence
            merged[-1] = LanguageSegment(
                language=prev.language,
                phonemes=merged_phonemes,
                start_time=prev.start_time,
                end_time=seg.end_time,
                speaker=prev.speaker,
                confidence=merged_confidence
            )
        else:
            # Can't merge: add as new segment
            merged.append(seg)

    return merged

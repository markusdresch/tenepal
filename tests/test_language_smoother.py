"""Tests for speaker-level language smoothing.

Tests the smooth_by_speaker function which accumulates each speaker's
primary language and reclassifies short/low-confidence outlier segments
to the speaker's dominant language.
"""

import pytest

from tenepal.language.identifier import LanguageSegment
from tenepal.language.smoother import smooth_by_speaker, SpeakerLanguageStats
from tenepal.phoneme import PhonemeSegment


def _make_phonemes(phoneme_list, start=0.0, spacing=0.1):
    """Create sequential PhonemeSegments with customizable timing.

    Helper to avoid repetitive test setup. Each phoneme gets specified
    duration, starting from start time and incrementing by spacing.

    Args:
        phoneme_list: List of IPA phoneme strings
        start: Start time in seconds (default 0.0)
        spacing: Duration and spacing between phonemes (default 0.1)

    Returns:
        List of PhonemeSegment objects
    """
    return [
        PhonemeSegment(phoneme=p, start_time=start + i * spacing, duration=spacing)
        for i, p in enumerate(phoneme_list)
    ]


def _make_segment(language, phoneme_count, speaker=None, confidence=2.0, start_time=0.0):
    """Create a LanguageSegment with dummy phonemes.

    Args:
        language: Language code
        phoneme_count: Number of phonemes to include
        speaker: Speaker label (optional)
        confidence: Confidence score (default 2.0)
        start_time: Starting time in seconds (default 0.0)

    Returns:
        LanguageSegment with generated phonemes
    """
    phonemes = _make_phonemes(["a"] * phoneme_count, start=start_time)
    end_time = phonemes[-1].start_time + phonemes[-1].duration
    return LanguageSegment(
        language=language,
        phonemes=phonemes,
        start_time=start_time,
        end_time=end_time,
        speaker=speaker,
        confidence=confidence
    )


# ============================================================================
# Basic edge cases
# ============================================================================


def test_empty_input():
    """Empty input should return empty output."""
    result = smooth_by_speaker([])
    assert result == []


def test_no_speaker_labels_unchanged():
    """Segments with speaker=None should be returned unchanged."""
    segments = [
        _make_segment("deu", 10, speaker=None, start_time=0.0),
        _make_segment("spa", 5, speaker=None, start_time=1.0),
        _make_segment("deu", 8, speaker=None, start_time=1.5),
    ]
    result = smooth_by_speaker(segments)

    # Should return exactly as-is (no speaker info = no smoothing)
    assert len(result) == 3
    assert result[0].language == "deu"
    assert result[1].language == "spa"
    assert result[2].language == "deu"


def test_single_speaker_consistent_language():
    """Single speaker with all same-language segments returns unchanged."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("deu", 15, speaker="Speaker A", confidence=4.0, start_time=2.0),
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=3.5),
        _make_segment("deu", 22, speaker="Speaker A", confidence=6.0, start_time=5.3),
        _make_segment("deu", 12, speaker="Speaker A", confidence=3.0, start_time=7.5),
    ]
    result = smooth_by_speaker(segments)

    # All DEU, no outliers to smooth
    assert len(result) == 1  # Should be merged into one segment
    assert result[0].language == "deu"
    assert result[0].speaker == "Speaker A"
    assert len(result[0].phonemes) == 20 + 15 + 18 + 22 + 12


# ============================================================================
# Short outlier smoothing
# ============================================================================


def test_single_speaker_short_outlier_smoothed():
    """Short outlier segment (< min_phonemes) gets smoothed to primary language."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=2.0),
        _make_segment("spa", 3, speaker="Speaker A", confidence=0.3, start_time=3.8),  # Short outlier
        _make_segment("deu", 22, speaker="Speaker A", confidence=6.0, start_time=4.1),
        _make_segment("deu", 15, speaker="Speaker A", confidence=4.0, start_time=6.3),
    ]
    result = smooth_by_speaker(segments)

    # SPA outlier (3 phonemes, < min_phonemes=5) should become DEU
    assert len(result) == 1  # All merged into one DEU segment
    assert result[0].language == "deu"
    assert result[0].speaker == "Speaker A"
    assert len(result[0].phonemes) == 20 + 18 + 3 + 22 + 15


def test_single_speaker_long_outlier_preserved():
    """Long, high-confidence outlier is preserved (genuine code-switching)."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=2.0),
        _make_segment("eng", 12, speaker="Speaker A", confidence=3.5, start_time=3.8),  # Long, high-conf
        _make_segment("deu", 22, speaker="Speaker A", confidence=6.0, start_time=5.0),
        _make_segment("deu", 15, speaker="Speaker A", confidence=4.0, start_time=7.2),
    ]
    result = smooth_by_speaker(segments)

    # ENG segment (12 phonemes, confidence 3.5) should be preserved
    assert len(result) == 3
    assert result[0].language == "deu"
    assert result[1].language == "eng"
    assert result[1].speaker == "Speaker A"
    assert result[2].language == "deu"


# ============================================================================
# Multi-speaker independence
# ============================================================================


def test_multi_speaker_independent_smoothing():
    """Each speaker's segments are smoothed independently."""
    segments = [
        # Speaker A: mostly DEU with short SPA outlier
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("spa", 3, speaker="Speaker A", confidence=0.3, start_time=2.0),  # Outlier → DEU
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=2.3),

        # Speaker B: mostly SPA with short DEU outlier
        _make_segment("spa", 22, speaker="Speaker B", confidence=6.0, start_time=4.1),
        _make_segment("deu", 4, speaker="Speaker B", confidence=0.5, start_time=6.3),  # Outlier → SPA
        _make_segment("spa", 15, speaker="Speaker B", confidence=4.0, start_time=6.7),
    ]
    result = smooth_by_speaker(segments)

    # Speaker A: all DEU
    speaker_a = [s for s in result if s.speaker == "Speaker A"]
    assert len(speaker_a) == 1
    assert speaker_a[0].language == "deu"
    assert len(speaker_a[0].phonemes) == 20 + 3 + 18

    # Speaker B: all SPA
    speaker_b = [s for s in result if s.speaker == "Speaker B"]
    assert len(speaker_b) == 1
    assert speaker_b[0].language == "spa"
    assert len(speaker_b[0].phonemes) == 22 + 4 + 15


def test_mixed_speakers_with_no_speaker():
    """Segments with speaker=None mixed in are left untouched."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("spa", 3, speaker="Speaker A", confidence=0.3, start_time=2.0),  # Outlier → DEU
        _make_segment("eng", 10, speaker=None, confidence=2.0, start_time=2.3),  # No speaker
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=3.3),
    ]
    result = smooth_by_speaker(segments)

    # Speaker A segments should be smoothed and adjacent ones merged
    # First two Speaker A segments merge (0.0-2.3), third is separate (3.3-5.1)
    speaker_a = [s for s in result if s.speaker == "Speaker A"]
    assert len(speaker_a) == 2  # Two non-adjacent DEU segments
    assert all(s.language == "deu" for s in speaker_a)
    # First merged segment: 20 + 3 = 23 phonemes
    assert len(speaker_a[0].phonemes) == 23
    # Second segment: 18 phonemes
    assert len(speaker_a[1].phonemes) == 18

    # None-speaker segment should remain unchanged
    no_speaker = [s for s in result if s.speaker is None]
    assert len(no_speaker) == 1
    assert no_speaker[0].language == "eng"
    assert len(no_speaker[0].phonemes) == 10


# ============================================================================
# Primary language determination
# ============================================================================


def test_single_segment_per_speaker_no_smoothing():
    """Speaker with only 1 segment has no primary language to establish."""
    segments = [
        _make_segment("deu", 10, speaker="Speaker A", confidence=2.0, start_time=0.0),
    ]
    result = smooth_by_speaker(segments)

    # Single segment: no smoothing (need at least 2 to establish primary)
    assert len(result) == 1
    assert result[0].language == "deu"


def test_other_segments_not_considered_primary():
    """'other' language segments don't count toward primary language."""
    segments = [
        _make_segment("deu", 15, speaker="Speaker A", confidence=4.0, start_time=0.0),
        _make_segment("other", 20, speaker="Speaker A", confidence=0.0, start_time=1.5),
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=3.5),
        _make_segment("other", 25, speaker="Speaker A", confidence=0.0, start_time=5.3),
        _make_segment("deu", 12, speaker="Speaker A", confidence=3.0, start_time=7.8),
        _make_segment("other", 15, speaker="Speaker A", confidence=0.0, start_time=9.0),
        _make_segment("spa", 3, speaker="Speaker A", confidence=0.3, start_time=10.5),  # Outlier
    ]
    result = smooth_by_speaker(segments)

    # Primary should be DEU (45 phonemes), not "other" (60 phonemes)
    # SPA outlier (3 phonemes) should become DEU
    speaker_a_langs = {s.language for s in result if s.speaker == "Speaker A"}
    assert "deu" in speaker_a_langs
    assert "other" in speaker_a_langs  # other segments remain other
    assert "spa" not in speaker_a_langs  # SPA outlier smoothed to DEU


# ============================================================================
# Smoothing preserves segment structure
# ============================================================================


def test_smoothing_updates_language_only():
    """Smoothed segments keep original phonemes, timing, speaker; only language changes."""
    original_phonemes = _make_phonemes(["a", "e", "i"], start=2.0)
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        LanguageSegment(
            language="spa",
            phonemes=original_phonemes,
            start_time=2.0,
            end_time=2.3,
            speaker="Speaker A",
            confidence=0.3
        ),  # Short outlier
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=2.3),
    ]
    result = smooth_by_speaker(segments)

    # After smoothing and merging
    assert len(result) == 1
    assert result[0].language == "deu"

    # Original phonemes should be preserved in merged result
    all_phonemes = result[0].phonemes
    assert original_phonemes[0] in all_phonemes
    assert original_phonemes[1] in all_phonemes
    assert original_phonemes[2] in all_phonemes


def test_adjacent_smoothed_segments_merged():
    """After smoothing, adjacent same-language same-speaker segments merge."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("spa", 3, speaker="Speaker A", confidence=0.3, start_time=2.0),  # → DEU
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=2.3),
        _make_segment("spa", 4, speaker="Speaker A", confidence=0.4, start_time=4.1),  # → DEU
        _make_segment("deu", 15, speaker="Speaker A", confidence=4.0, start_time=4.5),
    ]
    result = smooth_by_speaker(segments)

    # All should merge into one DEU segment
    assert len(result) == 1
    assert result[0].language == "deu"
    assert result[0].speaker == "Speaker A"
    assert len(result[0].phonemes) == 20 + 3 + 18 + 4 + 15


# ============================================================================
# Low-confidence smoothing
# ============================================================================


def test_low_confidence_segment_smoothed():
    """Segment with low confidence relative to primary language gets smoothed."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("deu", 20, speaker="Speaker A", confidence=6.0, start_time=2.0),
        _make_segment("spa", 10, speaker="Speaker A", confidence=0.5, start_time=4.0),  # Low conf
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.5, start_time=5.0),
    ]
    result = smooth_by_speaker(segments)

    # SPA segment has low confidence compared to DEU average → smooth to DEU
    # DEU avg confidence per phoneme: (5.0 + 6.0 + 5.5) / 60 = 0.275
    # SPA has 10 phonemes, conf 0.5 → 0.05 per phoneme
    # Threshold: 0.3 * 0.275 * 10 = 0.825 > 0.5 → smooth
    assert len(result) == 1
    assert result[0].language == "deu"


def test_high_confidence_segment_preserved():
    """Segment with high confidence is preserved even if not primary language."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("deu", 20, speaker="Speaker A", confidence=6.0, start_time=2.0),
        _make_segment("spa", 10, speaker="Speaker A", confidence=4.0, start_time=4.0),  # High conf
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.5, start_time=5.0),
    ]
    result = smooth_by_speaker(segments)

    # SPA segment has high confidence → preserve
    assert len(result) == 3
    assert result[0].language == "deu"
    assert result[1].language == "spa"
    assert result[2].language == "deu"


# ============================================================================
# OTH reclassification by speaker
# ============================================================================


def test_short_oth_reclassified_to_primary():
    """Short OTH segment (< min_phonemes) gets reclassified to primary language."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("other", 3, speaker="Speaker A", confidence=0.0, start_time=2.0),  # Short OTH → DEU
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=2.3),
    ]
    result = smooth_by_speaker(segments)

    # Short OTH (3 phonemes < 5 min) should become DEU and merge
    assert len(result) == 1
    assert result[0].language == "deu"
    assert result[0].speaker == "Speaker A"
    assert len(result[0].phonemes) == 20 + 3 + 18


def test_long_oth_preserved():
    """Long OTH segment (>= min_phonemes) is preserved as-is."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("other", 10, speaker="Speaker A", confidence=0.0, start_time=2.0),  # Long OTH → kept
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=3.0),
    ]
    result = smooth_by_speaker(segments)

    # Long OTH (10 phonemes >= 5 min) should be preserved
    assert len(result) == 3
    assert result[0].language == "deu"
    assert result[1].language == "other"
    assert result[2].language == "deu"


def test_oth_reclassification_boundary():
    """OTH with exactly min_phonemes is preserved (not reclassified)."""
    segments = [
        _make_segment("spa", 20, speaker="Speaker B", confidence=5.0, start_time=0.0),
        _make_segment("other", 5, speaker="Speaker B", confidence=0.0, start_time=2.0),  # Exactly 5 → kept
        _make_segment("spa", 18, speaker="Speaker B", confidence=4.5, start_time=2.5),
    ]
    result = smooth_by_speaker(segments)

    # Exactly min_phonemes=5: NOT reclassified (condition is <, not <=)
    assert len(result) == 3
    assert result[1].language == "other"


def test_multiple_short_oth_reclassified():
    """Multiple short OTH segments all get reclassified."""
    segments = [
        _make_segment("deu", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("other", 2, speaker="Speaker A", confidence=0.0, start_time=2.0),  # → DEU
        _make_segment("deu", 15, speaker="Speaker A", confidence=4.0, start_time=2.2),
        _make_segment("other", 3, speaker="Speaker A", confidence=0.0, start_time=3.7),  # → DEU
        _make_segment("deu", 18, speaker="Speaker A", confidence=4.5, start_time=4.0),
    ]
    result = smooth_by_speaker(segments)

    # All should merge into one DEU segment
    assert len(result) == 1
    assert result[0].language == "deu"
    assert len(result[0].phonemes) == 20 + 2 + 15 + 3 + 18


def test_short_may_segment_preserved_during_smoothing():
    """Short MAY segment should not be smoothed away to speaker primary language."""
    segments = [
        _make_segment("nah", 20, speaker="Speaker A", confidence=5.0, start_time=0.0),
        _make_segment("may", 3, speaker="Speaker A", confidence=0.2, start_time=2.0),
        _make_segment("nah", 18, speaker="Speaker A", confidence=4.5, start_time=2.3),
    ]
    result = smooth_by_speaker(segments)

    langs = [s.language for s in result if s.speaker == "Speaker A"]
    assert "may" in langs


def test_short_lat_segment_preserved_during_smoothing():
    """Short LAT segment should not be smoothed away to speaker primary language."""
    segments = [
        _make_segment("spa", 20, speaker="Speaker B", confidence=5.0, start_time=0.0),
        _make_segment("lat", 3, speaker="Speaker B", confidence=0.2, start_time=2.0),
        _make_segment("spa", 18, speaker="Speaker B", confidence=4.5, start_time=2.3),
    ]
    result = smooth_by_speaker(segments)

    langs = [s.language for s in result if s.speaker == "Speaker B"]
    assert "lat" in langs

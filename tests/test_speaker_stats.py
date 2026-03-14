"""Tests for speaker statistics formatting."""

import pytest
from tenepal.speaker import format_speaker_stats
from tenepal.language import LanguageSegment
from tenepal.phoneme import PhonemeSegment


def test_single_speaker_stats():
    """One speaker with segments -> 100%."""
    segments = [
        LanguageSegment(
            language="nah",
            phonemes=[
                PhonemeSegment(phoneme="t", start_time=0.0, duration=0.1),
                PhonemeSegment(phoneme="l", start_time=0.1, duration=0.1),
            ],
            start_time=0.0,
            end_time=0.2,
            speaker="Speaker A"
        ),
        LanguageSegment(
            language="nah",
            phonemes=[
                PhonemeSegment(phoneme="k", start_time=0.3, duration=0.2),
            ],
            start_time=0.3,
            end_time=0.5,
            speaker="Speaker A"
        ),
    ]
    
    result = format_speaker_stats(segments)
    assert result == "Detected 1 speaker: Speaker A (100%)"


def test_two_speaker_stats():
    """Two speakers with different speaking times -> correct percentages."""
    segments = [
        LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment(phoneme="t", start_time=0.0, duration=0.1)],
            start_time=0.0,
            end_time=0.6,  # 0.6s duration
            speaker="Speaker A"
        ),
        LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment(phoneme="k", start_time=0.7, duration=0.1)],
            start_time=0.7,
            end_time=1.1,  # 0.4s duration
            speaker="Speaker B"
        ),
    ]
    
    result = format_speaker_stats(segments)
    # 0.6 / 1.0 = 60%, 0.4 / 1.0 = 40%
    assert result == "Detected 2 speakers: Speaker A (60%), Speaker B (40%)"


def test_three_speaker_stats_sorted():
    """Three speakers -> sorted by speaking time descending."""
    segments = [
        LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment(phoneme="t", start_time=0.0, duration=0.1)],
            start_time=0.0,
            end_time=0.2,  # 0.2s
            speaker="Speaker A"
        ),
        LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment(phoneme="k", start_time=0.3, duration=0.1)],
            start_time=0.3,
            end_time=0.8,  # 0.5s (most)
            speaker="Speaker B"
        ),
        LanguageSegment(
            language="eng",
            phonemes=[PhonemeSegment(phoneme="θ", start_time=0.9, duration=0.1)],
            start_time=0.9,
            end_time=1.2,  # 0.3s
            speaker="Speaker C"
        ),
    ]
    
    result = format_speaker_stats(segments)
    # 0.5s (50%), 0.3s (30%), 0.2s (20%)
    assert result == "Detected 3 speakers: Speaker B (50%), Speaker C (30%), Speaker A (20%)"


def test_no_segments():
    """Empty list -> No speakers detected."""
    result = format_speaker_stats([])
    assert result == "No speakers detected"


def test_segments_without_speaker():
    """Segments with speaker=None -> No speakers detected."""
    segments = [
        LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment(phoneme="t", start_time=0.0, duration=0.1)],
            start_time=0.0,
            end_time=0.2,
            speaker=None
        ),
    ]
    
    result = format_speaker_stats(segments)
    assert result == "No speakers detected"


def test_mixed_speakers_and_none():
    """Mix of speaker labels and None -> only count labeled speakers."""
    segments = [
        LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment(phoneme="t", start_time=0.0, duration=0.1)],
            start_time=0.0,
            end_time=0.5,
            speaker="Speaker A"
        ),
        LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment(phoneme="k", start_time=0.6, duration=0.1)],
            start_time=0.6,
            end_time=0.8,
            speaker=None
        ),
    ]
    
    result = format_speaker_stats(segments)
    assert result == "Detected 1 speaker: Speaker A (100%)"

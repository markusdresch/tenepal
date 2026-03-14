"""Tests for speaker label locking algorithm."""

import pytest
from tenepal.speaker import LabelLocker
from tenepal.speaker.diarizer import SpeakerSegment


def test_first_pass_identity():
    """First diarization pass maps labels 1:1."""
    locker = LabelLocker()
    
    segments = [
        SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=2.0),
        SpeakerSegment(speaker="SPEAKER_01", start_time=2.5, end_time=5.0),
    ]
    
    stabilized = locker.stabilize(segments)
    
    assert len(stabilized) == 2
    assert stabilized[0].speaker == "Speaker A"
    assert stabilized[1].speaker == "Speaker B"


def test_relabeled_second_pass():
    """Second pass where pyannote swaps labels -> locker maps back."""
    locker = LabelLocker()
    
    # First pass
    first_segments = [
        SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=2.0),
        SpeakerSegment(speaker="SPEAKER_01", start_time=2.5, end_time=5.0),
    ]
    locker.stabilize(first_segments)
    
    # Second pass: pyannote reassigns (Speaker B speaks in Speaker A's time range)
    second_segments = [
        SpeakerSegment(speaker="SPEAKER_01", start_time=0.1, end_time=2.1),  # was SPEAKER_00
        SpeakerSegment(speaker="SPEAKER_00", start_time=2.6, end_time=5.1),  # was SPEAKER_01
    ]
    
    stabilized = locker.stabilize(second_segments)
    
    assert len(stabilized) == 2
    # Despite new labels, time overlap determines stable label
    assert stabilized[0].speaker == "Speaker A"  # overlaps with original Speaker A time
    assert stabilized[1].speaker == "Speaker B"  # overlaps with original Speaker B time


def test_new_speaker_detected():
    """Third pass introduces Speaker C not in previous mapping."""
    locker = LabelLocker()
    
    # First pass
    first_segments = [
        SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=2.0),
    ]
    locker.stabilize(first_segments)
    
    # Second pass: new speaker appears
    second_segments = [
        SpeakerSegment(speaker="SPEAKER_00", start_time=0.1, end_time=2.1),
        SpeakerSegment(speaker="SPEAKER_02", start_time=5.0, end_time=7.0),  # NEW
    ]
    
    stabilized = locker.stabilize(second_segments)
    
    assert len(stabilized) == 2
    assert stabilized[0].speaker == "Speaker A"
    assert stabilized[1].speaker == "Speaker B"  # Next available letter


def test_empty_segments():
    """Empty segment list returns empty list."""
    locker = LabelLocker()
    stabilized = locker.stabilize([])
    assert stabilized == []


def test_single_speaker():
    """Single speaker always maps to Speaker A."""
    locker = LabelLocker()
    
    segments = [
        SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=5.0),
    ]
    
    stabilized = locker.stabilize(segments)
    
    assert len(stabilized) == 1
    assert stabilized[0].speaker == "Speaker A"


def test_multiple_segments_same_speaker():
    """Multiple segments from same speaker get same label."""
    locker = LabelLocker()
    
    segments = [
        SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=2.0),
        SpeakerSegment(speaker="SPEAKER_01", start_time=2.5, end_time=5.0),
        SpeakerSegment(speaker="SPEAKER_00", start_time=5.5, end_time=7.0),  # Speaker A returns
    ]
    
    stabilized = locker.stabilize(segments)
    
    assert len(stabilized) == 3
    assert stabilized[0].speaker == "Speaker A"
    assert stabilized[1].speaker == "Speaker B"
    assert stabilized[2].speaker == "Speaker A"


def test_three_speakers():
    """Three different speakers get A, B, C labels."""
    locker = LabelLocker()
    
    segments = [
        SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=2.0),
        SpeakerSegment(speaker="SPEAKER_01", start_time=2.5, end_time=5.0),
        SpeakerSegment(speaker="SPEAKER_02", start_time=5.5, end_time=7.0),
    ]
    
    stabilized = locker.stabilize(segments)
    
    assert len(stabilized) == 3
    assert stabilized[0].speaker == "Speaker A"
    assert stabilized[1].speaker == "Speaker B"
    assert stabilized[2].speaker == "Speaker C"

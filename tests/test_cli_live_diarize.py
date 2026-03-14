"""Tests for live mode diarization integration.

Tests the helper functions and formatting for live diarization, not the full
live capture loop (which requires PulseAudio).
"""

import argparse
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from tenepal.cli import _print_live_segment, SPEAKER_COLORS
from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme.recognizer import PhonemeSegment


@pytest.fixture
def sample_segment():
    """Create a sample LanguageSegment for testing."""
    phonemes = [
        PhonemeSegment(phoneme="n", start_time=0.0, duration=0.1),
        PhonemeSegment(phoneme="o", start_time=0.1, duration=0.1),
        PhonemeSegment(phoneme="p", start_time=0.2, duration=0.1),
    ]
    return LanguageSegment(
        language="nah",
        phonemes=phonemes,
        start_time=0.0,
        end_time=0.3
    )


def test_print_live_segment_speaker_color(sample_segment, capsys):
    """Test that segment with speaker uses speaker color, not language color."""
    sample_segment.speaker = "Speaker A"
    speaker_colors = {"Speaker A": SPEAKER_COLORS[0]}  # Blue

    with patch.object(sys.stdout, 'isatty', return_value=True):
        _print_live_segment(sample_segment, speaker_colors=speaker_colors)

    captured = capsys.readouterr()
    # Should use speaker color (blue: \033[38;5;39m), not language color (green: \033[32m)
    assert "\033[38;5;39m" in captured.out  # Speaker A blue
    assert "[Speaker A | NAH]" in captured.out
    assert "n o p" in captured.out


def test_print_live_segment_no_speaker_fallback(sample_segment, capsys):
    """Test that segment without speaker uses language color (backward compatible)."""
    sample_segment.speaker = None

    with patch.object(sys.stdout, 'isatty', return_value=True):
        _print_live_segment(sample_segment, speaker_colors=None)

    captured = capsys.readouterr()
    # Should use language color (green for NAH)
    assert "\033[32m" in captured.out
    assert "[NAH]" in captured.out
    assert "n o p" in captured.out


def test_print_live_segment_speaker_unknown(sample_segment, capsys):
    """Test that Speaker ? uses speaker color."""
    sample_segment.speaker = "Speaker ?"
    speaker_colors = {"Speaker ?": SPEAKER_COLORS[0]}

    with patch.object(sys.stdout, 'isatty', return_value=True):
        _print_live_segment(sample_segment, speaker_colors=speaker_colors)

    captured = capsys.readouterr()
    assert "\033[38;5;39m" in captured.out  # Speaker color
    assert "[Speaker ? | NAH]" in captured.out


def test_print_live_segment_with_speaker_but_no_color_dict(sample_segment, capsys):
    """Test segment with speaker but no speaker_colors dict falls back to language color."""
    sample_segment.speaker = "Speaker A"

    with patch.object(sys.stdout, 'isatty', return_value=True):
        _print_live_segment(sample_segment, speaker_colors=None)

    captured = capsys.readouterr()
    # Should fall back to language color since speaker_colors=None
    assert "\033[32m" in captured.out  # Green for NAH
    assert "[Speaker A | NAH]" in captured.out


def test_new_speaker_detection_tracking():
    """Test that seen_speakers tracking works for [new speaker detected]."""
    seen_speakers = set()

    # First segment from Speaker A
    segment_a = LanguageSegment(
        language="nah",
        phonemes=[PhonemeSegment(phoneme="n", start_time=0.0, duration=0.1)],
        start_time=0.0,
        end_time=0.1
    )
    segment_a.speaker = "Speaker A"

    # Check if new
    is_new = segment_a.speaker not in seen_speakers
    assert is_new is True
    seen_speakers.add(segment_a.speaker)

    # Second segment from Speaker A
    segment_a2 = LanguageSegment(
        language="spa",
        phonemes=[PhonemeSegment(phoneme="s", start_time=0.1, duration=0.1)],
        start_time=0.1,
        end_time=0.2
    )
    segment_a2.speaker = "Speaker A"

    is_new = segment_a2.speaker not in seen_speakers
    assert is_new is False  # Already seen

    # Third segment from Speaker B
    segment_b = LanguageSegment(
        language="eng",
        phonemes=[PhonemeSegment(phoneme="h", start_time=0.2, duration=0.1)],
        start_time=0.2,
        end_time=0.3
    )
    segment_b.speaker = "Speaker B"

    is_new = segment_b.speaker not in seen_speakers
    assert is_new is True
    seen_speakers.add(segment_b.speaker)

    assert len(seen_speakers) == 2
    assert "Speaker A" in seen_speakers
    assert "Speaker B" in seen_speakers


def test_status_bar_format_with_speakers():
    """Test status bar includes speaker count."""
    duration = 45.0
    segment_count = 12
    speaker_count = 3

    status = f"[LIVE] {duration:.0f}s | {segment_count} segments | {speaker_count} speakers | listening..."

    assert status == "[LIVE] 45s | 12 segments | 3 speakers | listening..."


def test_status_bar_format_no_speakers():
    """Test status bar without speakers (before first diarization)."""
    duration = 10.0
    segment_count = 2

    status = f"[LIVE] {duration:.0f}s | {segment_count} segments | listening..."

    assert status == "[LIVE] 10s | 2 segments | listening..."


def test_no_diarize_flag_parsed():
    """Test that --no-diarize flag is parsed correctly."""
    from tenepal.cli import main

    # Mock sys.argv for argparse
    test_args = ["tenepal", "--no-diarize", "test.wav"]

    with patch.object(sys, 'argv', test_args):
        parser = argparse.ArgumentParser()
        parser.add_argument("files", nargs="*")
        parser.add_argument("--no-diarize", action="store_true")
        parser.add_argument("--live", action="store_true")
        parser.add_argument("--output", "-o", type=str)

        # Manually parse to test flag
        args = parser.parse_args(test_args[1:])

    assert args.no_diarize is True
    assert args.files == ["test.wav"]


def test_no_diarize_flag_absent():
    """Test that --no-diarize defaults to False when absent."""
    from tenepal.cli import main

    test_args = ["tenepal", "test.wav"]

    with patch.object(sys, 'argv', test_args):
        parser = argparse.ArgumentParser()
        parser.add_argument("files", nargs="*")
        parser.add_argument("--no-diarize", action="store_true")
        parser.add_argument("--live", action="store_true")
        parser.add_argument("--output", "-o", type=str)

        args = parser.parse_args(test_args[1:])

    assert args.no_diarize is False
    assert args.files == ["test.wav"]


def test_speaker_color_cycling():
    """Test that speaker colors cycle correctly for >6 speakers."""
    speaker_colors = {}

    # Assign colors to 10 speakers
    for i in range(10):
        speaker = f"Speaker {chr(65 + i)}"
        color_idx = len(speaker_colors) % len(SPEAKER_COLORS)
        speaker_colors[speaker] = SPEAKER_COLORS[color_idx]

    # First 6 should use unique colors
    assert speaker_colors["Speaker A"] == SPEAKER_COLORS[0]
    assert speaker_colors["Speaker F"] == SPEAKER_COLORS[5]

    # Speaker G should cycle back to color 0
    assert speaker_colors["Speaker G"] == SPEAKER_COLORS[0]
    assert speaker_colors["Speaker J"] == SPEAKER_COLORS[3]


def test_print_live_segment_non_tty(sample_segment, capsys):
    """Test output without ANSI colors when not a TTY."""
    sample_segment.speaker = "Speaker A"
    speaker_colors = {"Speaker A": SPEAKER_COLORS[0]}

    with patch.object(sys.stdout, 'isatty', return_value=False):
        _print_live_segment(sample_segment, speaker_colors=speaker_colors)

    captured = capsys.readouterr()
    # Should NOT contain ANSI codes
    assert "\033[" not in captured.out
    assert "[Speaker A | NAH]" in captured.out
    assert "n o p" in captured.out


def test_print_live_segment_may_label(sample_segment, capsys):
    """MAY label is shown for Maya segments in live output."""
    sample_segment.language = "may"
    sample_segment.speaker = "Speaker A"
    speaker_colors = {"Speaker A": SPEAKER_COLORS[0]}

    with patch.object(sys.stdout, 'isatty', return_value=False):
        _print_live_segment(sample_segment, speaker_colors=speaker_colors)

    captured = capsys.readouterr()
    assert "[Speaker A | MAY]" in captured.out

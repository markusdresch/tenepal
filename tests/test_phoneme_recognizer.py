"""Tests for phoneme recognition and formatting."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tenepal.phoneme import (
    PhonemeSegment,
    format_phonemes,
    print_phonemes,
    recognize_phonemes,
)


class TestPhonemeSegment:
    """Tests for PhonemeSegment dataclass."""

    def test_create_segment(self):
        """Test creating a PhonemeSegment."""
        seg = PhonemeSegment(phoneme="ae", start_time=0.5, duration=0.1)

        assert seg.phoneme == "ae"
        assert seg.start_time == 0.5
        assert seg.duration == 0.1


class TestFormatter:
    """Tests for phoneme formatting."""

    @pytest.fixture
    def sample_segments(self):
        """Create sample PhonemeSegment list for testing."""
        return [
            PhonemeSegment(phoneme="ae", start_time=0.0, duration=0.05),
            PhonemeSegment(phoneme="l", start_time=0.05, duration=0.03),
            PhonemeSegment(phoneme="u", start_time=0.08, duration=0.04),
        ]

    def test_format_phonemes_with_timestamps(self, sample_segments):
        """Test formatting phonemes with timestamps."""
        result = format_phonemes(sample_segments, show_timestamps=True)

        # Check that output contains expected elements
        assert "Time" in result
        assert "Duration" in result
        assert "Phoneme" in result
        assert "0.000s" in result
        assert "0.050s" in result
        assert "ae" in result
        assert "l" in result
        assert "u" in result

    def test_format_phonemes_without_timestamps(self, sample_segments):
        """Test formatting phonemes without timestamps."""
        result = format_phonemes(sample_segments, show_timestamps=False)

        # Should be space-separated phonemes only
        assert result == "ae l u"
        assert "Time" not in result
        assert "Duration" not in result

    def test_format_empty_segments(self):
        """Test formatting empty segment list."""
        result = format_phonemes([], show_timestamps=True)
        assert result == ""

    def test_print_phonemes(self, sample_segments, capsys):
        """Test printing phonemes to stdout."""
        print_phonemes(sample_segments, show_timestamps=True)

        captured = capsys.readouterr()
        assert "Phoneme Stream" in captured.out
        assert "ae" in captured.out
        assert "Total segments: 3" in captured.out


class TestRecognizer:
    """Tests for phoneme recognition."""

    @pytest.fixture
    def temp_wav_file(self):
        """Create a temporary WAV file with sine wave."""
        # Generate 0.5 second sine wave at 440 Hz
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        sf.write(str(temp_path), samples, sample_rate)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_recognize_phonemes_returns_segments(self, temp_wav_file):
        """Test that recognize_phonemes returns valid PhonemeSegment list."""
        segments = recognize_phonemes(temp_wav_file)

        # Should return a list (might be empty for sine wave - not speech)
        assert isinstance(segments, list)

        # Each segment should have valid structure if any exist
        for seg in segments:
            assert isinstance(seg, PhonemeSegment)
            assert isinstance(seg.phoneme, str)
            assert len(seg.phoneme) > 0
            assert seg.start_time >= 0
            assert seg.duration > 0

    def test_recognize_phonemes_file_not_found(self):
        """Test that recognize_phonemes raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            recognize_phonemes("/nonexistent/path/to/audio.wav")

    def test_recognize_phonemes_with_lang_parameter(self, temp_wav_file):
        """Test recognize_phonemes with language parameter."""
        segments = recognize_phonemes(temp_wav_file, lang="ipa")

        # Should still return valid list structure
        assert isinstance(segments, list)

"""Tests for TranscriptionRouter segment routing logic."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import sys

import numpy as np
import pytest

# Mock faster_whisper before imports
sys.modules["faster_whisper"] = MagicMock()

from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme import PhonemeSegment
from tenepal.transcription.router import TranscriptionRouter, TranscriptionResult


@pytest.fixture
def mock_audio():
    """Mock audio data for testing."""
    # Create 1 second of mock audio at 16kHz
    sample_rate = 16000
    samples = np.zeros(sample_rate, dtype=np.float32)
    return samples, sample_rate


@pytest.fixture
def mock_audio_file(tmp_path, mock_audio):
    """Create a temporary audio file for testing."""
    import soundfile as sf
    audio_path = tmp_path / "test_audio.wav"
    samples, sample_rate = mock_audio
    sf.write(str(audio_path), samples, sample_rate)
    return audio_path


@pytest.fixture
def spa_segment():
    """Spanish language segment."""
    phonemes = [
        PhonemeSegment(phoneme="e", start_time=0.0, duration=0.1),
        PhonemeSegment(phoneme="s", start_time=0.1, duration=0.1),
    ]
    return LanguageSegment(
        language="spa",
        phonemes=phonemes,
        start_time=0.0,
        end_time=0.2,
        confidence=5.0
    )


@pytest.fixture
def eng_segment():
    """English language segment."""
    phonemes = [
        PhonemeSegment(phoneme="ð", start_time=0.2, duration=0.1),
        PhonemeSegment(phoneme="ɪ", start_time=0.3, duration=0.1),
    ]
    return LanguageSegment(
        language="eng",
        phonemes=phonemes,
        start_time=0.2,
        end_time=0.4,
        confidence=4.0
    )


@pytest.fixture
def deu_segment():
    """German language segment."""
    phonemes = [
        PhonemeSegment(phoneme="ç", start_time=0.4, duration=0.1),
        PhonemeSegment(phoneme="i", start_time=0.5, duration=0.1),
    ]
    return LanguageSegment(
        language="deu",
        phonemes=phonemes,
        start_time=0.4,
        end_time=0.6,
        confidence=3.0
    )


@pytest.fixture
def fra_segment():
    """French language segment."""
    phonemes = [
        PhonemeSegment(phoneme="ʁ", start_time=0.6, duration=0.1),
        PhonemeSegment(phoneme="y", start_time=0.7, duration=0.1),
    ]
    return LanguageSegment(
        language="fra",
        phonemes=phonemes,
        start_time=0.6,
        end_time=0.8,
        confidence=3.5
    )


@pytest.fixture
def ita_segment():
    """Italian language segment."""
    phonemes = [
        PhonemeSegment(phoneme="ʎ", start_time=0.8, duration=0.1),
        PhonemeSegment(phoneme="a", start_time=0.9, duration=0.1),
    ]
    return LanguageSegment(
        language="ita",
        phonemes=phonemes,
        start_time=0.8,
        end_time=1.0,
        confidence=4.5
    )


@pytest.fixture
def nah_segment():
    """Nahuatl language segment."""
    phonemes = [
        PhonemeSegment(phoneme="t͡ɬ", start_time=1.0, duration=0.1),
        PhonemeSegment(phoneme="i", start_time=1.1, duration=0.1),
    ]
    return LanguageSegment(
        language="nah",
        phonemes=phonemes,
        start_time=1.0,
        end_time=1.2,
        confidence=6.0
    )


@pytest.fixture
def other_segment():
    """Unidentified language segment."""
    phonemes = [
        PhonemeSegment(phoneme="a", start_time=1.2, duration=0.1),
        PhonemeSegment(phoneme="b", start_time=1.3, duration=0.1),
    ]
    return LanguageSegment(
        language="other",
        phonemes=phonemes,
        start_time=1.2,
        end_time=1.4,
        confidence=0.0
    )


class TestCoreRouting:
    """Test basic routing logic for different languages."""

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_spa_segment_routes_to_whisper(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, spa_segment):
        """Spanish segments should route to Whisper backend with 'es' language code."""
        # Setup mocks
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [
            PhonemeSegment(phoneme="hola", start_time=0.0, duration=0.2)
        ]
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()
        results = router.transcribe_segments([spa_segment], str(mock_audio_file))

        # Verify Whisper backend was called with Spanish language code
        mock_get_backend.assert_called_once_with("whisper", model_size="base", device="auto")
        assert mock_whisper.recognize.call_count == 1
        call_args = mock_whisper.recognize.call_args
        assert call_args[1]["lang"] == "es"

        # Verify result
        assert len(results) == 1
        assert results[0].backend == "whisper"
        assert results[0].language == "spa"

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_eng_segment_routes_to_whisper(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, eng_segment):
        """English segments should route to Whisper with 'en' language code."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [
            PhonemeSegment(phoneme="hello", start_time=0.0, duration=0.2)
        ]
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()
        results = router.transcribe_segments([eng_segment], str(mock_audio_file))

        call_args = mock_whisper.recognize.call_args
        assert call_args[1]["lang"] == "en"
        assert results[0].backend == "whisper"
        assert results[0].language == "eng"

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_deu_segment_routes_to_whisper(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, deu_segment):
        """German segments should route to Whisper with 'de' language code."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [
            PhonemeSegment(phoneme="hallo", start_time=0.0, duration=0.2)
        ]
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()
        results = router.transcribe_segments([deu_segment], str(mock_audio_file))

        call_args = mock_whisper.recognize.call_args
        assert call_args[1]["lang"] == "de"
        assert results[0].backend == "whisper"
        assert results[0].language == "deu"

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_fra_segment_routes_to_whisper(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, fra_segment):
        """French segments should route to Whisper with 'fr' language code."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [
            PhonemeSegment(phoneme="bonjour", start_time=0.0, duration=0.2)
        ]
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()
        results = router.transcribe_segments([fra_segment], str(mock_audio_file))

        call_args = mock_whisper.recognize.call_args
        assert call_args[1]["lang"] == "fr"
        assert results[0].backend == "whisper"
        assert results[0].language == "fra"

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_ita_segment_routes_to_whisper(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, ita_segment):
        """Italian segments should route to Whisper with 'it' language code."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [
            PhonemeSegment(phoneme="ciao", start_time=0.0, duration=0.2)
        ]
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()
        results = router.transcribe_segments([ita_segment], str(mock_audio_file))

        call_args = mock_whisper.recognize.call_args
        assert call_args[1]["lang"] == "it"
        assert results[0].backend == "whisper"
        assert results[0].language == "ita"

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_nah_segment_routes_to_allosaurus(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, nah_segment):
        """Nahuatl segments should route to Allosaurus backend for IPA phonemes."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_allosaurus = MagicMock()
        mock_allosaurus.recognize.return_value = [
            PhonemeSegment(phoneme="t͡ɬ", start_time=0.0, duration=0.1),
            PhonemeSegment(phoneme="i", start_time=0.1, duration=0.1),
        ]
        mock_get_backend.return_value = mock_allosaurus

        router = TranscriptionRouter()
        results = router.transcribe_segments([nah_segment], str(mock_audio_file))

        # Verify Allosaurus backend was called
        mock_get_backend.assert_called_once_with("allosaurus")
        assert mock_allosaurus.recognize.call_count == 1
        call_args = mock_allosaurus.recognize.call_args
        assert call_args[1]["lang"] == "ipa"

        # Verify result
        assert len(results) == 1
        assert results[0].backend == "allosaurus"
        assert results[0].language == "nah"
        assert results[0].is_text is False

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_other_segment_routes_to_allosaurus(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, other_segment):
        """Unidentified language segments should route to Allosaurus."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_allosaurus = MagicMock()
        mock_allosaurus.recognize.return_value = [
            PhonemeSegment(phoneme="a", start_time=0.0, duration=0.1),
            PhonemeSegment(phoneme="b", start_time=0.1, duration=0.1),
        ]
        mock_get_backend.return_value = mock_allosaurus

        router = TranscriptionRouter()
        results = router.transcribe_segments([other_segment], str(mock_audio_file))

        mock_get_backend.assert_called_once_with("allosaurus")
        assert results[0].backend == "allosaurus"
        assert results[0].language == "other"
        assert results[0].is_text is False


class TestResultStructure:
    """Test TranscriptionResult dataclass structure."""

    def test_transcription_result_has_required_fields(self):
        """TranscriptionResult should have all required fields."""
        result = TranscriptionResult(
            text="hello world",
            start_time=0.0,
            end_time=1.0,
            language="eng",
            backend="whisper",
            is_text=True
        )

        assert result.text == "hello world"
        assert result.start_time == 0.0
        assert result.end_time == 1.0
        assert result.language == "eng"
        assert result.backend == "whisper"
        assert result.is_text is True

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_whisper_result_backend_is_whisper(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, spa_segment):
        """Results from Whisper routing should have backend='whisper'."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [
            PhonemeSegment(phoneme="texto", start_time=0.0, duration=0.2)
        ]
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()
        results = router.transcribe_segments([spa_segment], str(mock_audio_file))

        assert results[0].backend == "whisper"
        assert results[0].is_text is True

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_allosaurus_result_backend_is_allosaurus(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, nah_segment):
        """Results from Allosaurus routing should have backend='allosaurus'."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_allosaurus = MagicMock()
        mock_allosaurus.recognize.return_value = [
            PhonemeSegment(phoneme="t͡ɬ", start_time=0.0, duration=0.1),
        ]
        mock_get_backend.return_value = mock_allosaurus

        router = TranscriptionRouter()
        results = router.transcribe_segments([nah_segment], str(mock_audio_file))

        assert results[0].backend == "allosaurus"
        assert results[0].is_text is False

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_result_preserves_original_timing(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, spa_segment):
        """Result timing should come from LanguageSegment, not Whisper internal timing."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        # Whisper returns different internal timing (should be ignored)
        mock_whisper.recognize.return_value = [
            PhonemeSegment(phoneme="word1", start_time=0.0, duration=0.05),
            PhonemeSegment(phoneme="word2", start_time=0.05, duration=0.05),
        ]
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()
        results = router.transcribe_segments([spa_segment], str(mock_audio_file))

        # Result timing should match LanguageSegment, not Whisper's internal timing
        assert results[0].start_time == spa_segment.start_time
        assert results[0].end_time == spa_segment.end_time


class TestMultiSegment:
    """Test handling of multiple segments."""

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_mixed_language_segments(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, spa_segment, nah_segment, eng_segment):
        """Mixed language segments should route to correct backends."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])

        # Setup backends
        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [PhonemeSegment(phoneme="text", start_time=0.0, duration=0.1)]
        mock_allosaurus = MagicMock()
        mock_allosaurus.recognize.return_value = [PhonemeSegment(phoneme="p", start_time=0.0, duration=0.1)]

        def get_backend_side_effect(name, **kwargs):
            if name == "whisper":
                return mock_whisper
            return mock_allosaurus

        mock_get_backend.side_effect = get_backend_side_effect

        router = TranscriptionRouter()
        results = router.transcribe_segments([spa_segment, nah_segment, eng_segment], str(mock_audio_file))

        # Verify correct routing
        assert len(results) == 3
        assert results[0].backend == "whisper"
        assert results[0].language == "spa"
        assert results[1].backend == "allosaurus"
        assert results[1].language == "nah"
        assert results[2].backend == "whisper"
        assert results[2].language == "eng"

    def test_empty_segments_returns_empty(self, mock_audio_file):
        """Empty segment list should return empty results."""
        router = TranscriptionRouter()
        results = router.transcribe_segments([], str(mock_audio_file))

        assert results == []


class TestAudioExtraction:
    """Test segment audio extraction logic."""

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_segment_audio_extraction(self, mock_load_audio, mock_get_backend, mock_audio_file, spa_segment):
        """Router should extract correct audio range based on segment timing."""
        # Create mock audio (1 second at 16kHz)
        sample_rate = 16000
        samples = np.arange(sample_rate, dtype=np.float32) / sample_rate  # 0.0 to 0.999...
        mock_load_audio.return_value = MagicMock(samples=samples, sample_rate=sample_rate)

        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [PhonemeSegment(phoneme="text", start_time=0.0, duration=0.2)]
        mock_get_backend.return_value = mock_whisper

        # Segment from 0.0 to 0.2 seconds should extract samples[0:3200]
        router = TranscriptionRouter()
        results = router.transcribe_segments([spa_segment], str(mock_audio_file))

        # Verify recognize was called (audio extraction happened)
        assert mock_whisper.recognize.call_count == 1

        # Verify result
        assert len(results) == 1
        assert results[0].start_time == 0.0
        assert results[0].end_time == 0.2

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_temp_file_cleanup(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, spa_segment):
        """Temp files should be cleaned up after transcription."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        mock_whisper.recognize.return_value = [PhonemeSegment(phoneme="text", start_time=0.0, duration=0.2)]
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()

        # Track temp file creation
        created_temp_files = []
        original_mkstemp = tempfile.mkstemp
        def track_mkstemp(*args, **kwargs):
            fd, path = original_mkstemp(*args, **kwargs)
            created_temp_files.append(path)
            return fd, path

        with patch("tempfile.mkstemp", side_effect=track_mkstemp):
            results = router.transcribe_segments([spa_segment], str(mock_audio_file))

        # Verify temp files were cleaned up
        for temp_file in created_temp_files:
            assert not Path(temp_file).exists()


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("tenepal.transcription.router.get_backend")
    def test_whisper_unavailable_raises_error(self, mock_get_backend, mock_audio_file, spa_segment):
        """When WhisperBackend is unavailable, router should raise clear error."""
        # Simulate backend unavailable
        mock_get_backend.side_effect = RuntimeError("Backend 'whisper' is not available")

        router = TranscriptionRouter()

        with pytest.raises(RuntimeError, match="Backend 'whisper' is not available"):
            router.transcribe_segments([spa_segment], str(mock_audio_file))

    @patch("tenepal.transcription.router.get_backend")
    @patch("tenepal.transcription.router.load_audio")
    def test_transcription_error_handling(self, mock_load_audio, mock_get_backend, mock_audio_file, mock_audio, spa_segment):
        """When backend.recognize() fails, router should propagate error."""
        mock_load_audio.return_value = MagicMock(samples=mock_audio[0], sample_rate=mock_audio[1])
        mock_whisper = MagicMock()
        mock_whisper.recognize.side_effect = Exception("Transcription failed")
        mock_get_backend.return_value = mock_whisper

        router = TranscriptionRouter()

        with pytest.raises(Exception, match="Transcription failed"):
            router.transcribe_segments([spa_segment], str(mock_audio_file))

"""Tests for the diarize-first pipeline."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from tenepal.audio.loader import AudioData
from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme.recognizer import PhonemeSegment
from tenepal.speaker.diarizer import SpeakerSegment
from tenepal.pipeline import process_audio


def _make_audio(duration=10.0, sr=16000):
    """Create synthetic AudioData for testing."""
    samples = np.zeros(int(duration * sr), dtype=np.float32)
    return AudioData(samples=samples, sample_rate=sr, duration=duration, source_format="wav")


def _make_phonemes(start=0.0, count=5, duration=0.1):
    """Create a list of synthetic phoneme segments."""
    return [
        PhonemeSegment(phoneme="a", start_time=start + i * duration, duration=duration)
        for i in range(count)
    ]


def _make_lang_segments(language="nah", start=0.0, count=1, speaker=None):
    """Create synthetic language segments."""
    phonemes = _make_phonemes(start=start, count=5)
    return [
        LanguageSegment(
            language=language,
            phonemes=phonemes,
            start_time=start,
            end_time=start + 0.5,
            speaker=speaker
        )
        for _ in range(count)
    ]


class TestProcessAudioNoDiarization:
    """Test pipeline without diarization."""

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_no_diarization_returns_segments_without_speaker(
        self, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        phonemes = _make_phonemes()
        mock_recognize.return_value = phonemes
        mock_identify.return_value = _make_lang_segments("nah")
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        result = process_audio(Path("/fake/audio.wav"), enable_diarization=False)

        assert len(result) == 1
        assert result[0].speaker is None

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_no_diarization_calls_recognize_and_identify(
        self, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments()
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        process_audio(Path("/fake/audio.wav"), enable_diarization=False)

        mock_recognize.assert_called_once()
        mock_identify.assert_called_once()

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_default_backend_is_allosaurus(
        self, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments()
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        process_audio(Path("/fake/audio.wav"), enable_diarization=False)

        assert mock_recognize.call_args.kwargs.get("backend") == "allosaurus"

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_explicit_backend_passed_through(
        self, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments()
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        process_audio(Path("/fake/audio.wav"), enable_diarization=False, backend="allosaurus")

        assert mock_recognize.call_args.kwargs.get("backend") == "allosaurus"


class TestProcessAudioWithDiarization:
    """Test pipeline with diarization enabled."""

    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.save_wav")
    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.slice_audio_by_speaker")
    @patch("tenepal.pipeline.diarize")
    def test_two_speakers_produces_labeled_segments(
        self, mock_diarize, mock_slice, mock_load, mock_preprocess, mock_save,
        mock_recognize, mock_identify
    ):
        # Setup: 2 speakers
        spk_a = SpeakerSegment("Speaker A", 0.0, 5.0)
        spk_b = SpeakerSegment("Speaker B", 5.0, 10.0)
        mock_diarize.return_value = [spk_a, spk_b]

        audio = _make_audio(10.0)
        mock_load.return_value = audio
        mock_preprocess.return_value = audio
        mock_slice.return_value = [(spk_a, _make_audio(5.0)), (spk_b, _make_audio(5.0))]

        # Each speaker gets phonemes + language
        mock_recognize.return_value = _make_phonemes(start=0.0)
        mock_identify.side_effect = [
            _make_lang_segments("nah", start=0.0),
            _make_lang_segments("spa", start=5.0),
        ]

        result = process_audio(Path("/fake/audio.wav"), enable_diarization=True)

        assert len(result) == 2
        assert result[0].speaker == "Speaker A"
        assert result[1].speaker == "Speaker B"

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.diarize")
    def test_fallback_uses_single_stream(
        self, mock_diarize, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        # Diarize returns fallback
        mock_diarize.return_value = [SpeakerSegment("Speaker ?", 0.0, 10.0)]
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments("nah")
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        result = process_audio(Path("/fake/audio.wav"), enable_diarization=True)

        assert len(result) == 1
        assert result[0].speaker == "Speaker ?"


class TestProcessAudioOverlap:
    """Test pipeline with overlapping speakers."""

    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.save_wav")
    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.slice_audio_by_speaker")
    @patch("tenepal.pipeline.diarize")
    def test_overlapping_speakers_produces_separate_segments(
        self, mock_diarize, mock_slice, mock_load, mock_preprocess, mock_save,
        mock_recognize, mock_identify
    ):
        spk_a = SpeakerSegment("Speaker A", 0.0, 7.0)
        spk_b = SpeakerSegment("Speaker B", 3.0, 10.0)
        mock_diarize.return_value = [spk_a, spk_b]

        audio = _make_audio(10.0)
        mock_load.return_value = audio
        mock_preprocess.return_value = audio
        mock_slice.return_value = [(spk_a, _make_audio(7.0)), (spk_b, _make_audio(7.0))]

        mock_recognize.return_value = _make_phonemes(start=0.0)
        mock_identify.side_effect = [
            _make_lang_segments("nah", start=0.0),
            _make_lang_segments("spa", start=3.0),
        ]

        result = process_audio(Path("/fake/audio.wav"))

        # Both speakers present
        speakers = {s.speaker for s in result}
        assert "Speaker A" in speakers
        assert "Speaker B" in speakers


class TestProcessAudioCodeSwitching:
    """Test code-switching within a speaker's turn."""

    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.save_wav")
    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.slice_audio_by_speaker")
    @patch("tenepal.pipeline.diarize")
    def test_same_speaker_different_languages(
        self, mock_diarize, mock_slice, mock_load, mock_preprocess, mock_save,
        mock_recognize, mock_identify
    ):
        spk_a = SpeakerSegment("Speaker A", 0.0, 10.0)
        mock_diarize.return_value = [spk_a]

        audio = _make_audio(10.0)
        mock_load.return_value = audio
        mock_preprocess.return_value = audio
        mock_slice.return_value = [(spk_a, _make_audio(10.0))]

        mock_recognize.return_value = _make_phonemes(start=0.0, count=10)
        # Language ID returns 2 segments: NAH then SPA
        mock_identify.return_value = [
            LanguageSegment("nah", _make_phonemes(0.0, 5), 0.0, 0.5, speaker=None),
            LanguageSegment("spa", _make_phonemes(0.5, 5), 0.5, 1.0, speaker=None),
        ]

        result = process_audio(Path("/fake/audio.wav"))

        assert len(result) == 2
        assert result[0].language == "nah"
        assert result[0].speaker == "Speaker A"
        assert result[1].language == "spa"
        assert result[1].speaker == "Speaker A"


class TestProcessAudioDualBackend:
    """Test pipeline with dual backend."""

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_dual_backend_no_diarization(
        self, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments("nah")
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        process_audio(Path("/fake/audio.wav"), enable_diarization=False, backend="dual")

        assert mock_recognize.call_args.kwargs.get("backend") == "dual"
        assert mock_recognize.call_args.kwargs.get("model_size") == "300M"

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_dual_backend_passes_model_size(
        self, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments("nah")
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        process_audio(
            Path("/fake/audio.wav"),
            enable_diarization=False,
            backend="dual",
            model_size="7B",
        )

        assert mock_recognize.call_args.kwargs.get("backend") == "dual"
        assert mock_recognize.call_args.kwargs.get("model_size") == "7B"

    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.save_wav")
    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.slice_audio_by_speaker")
    @patch("tenepal.pipeline.diarize")
    def test_dual_backend_with_diarization(
        self, mock_diarize, mock_slice, mock_load, mock_preprocess, mock_save,
        mock_recognize, mock_identify
    ):
        spk_a = SpeakerSegment("Speaker A", 0.0, 5.0)
        mock_diarize.return_value = [spk_a]

        audio = _make_audio(5.0)
        mock_load.return_value = audio
        mock_preprocess.return_value = audio
        mock_slice.return_value = [(spk_a, _make_audio(5.0))]

        mock_recognize.return_value = _make_phonemes(start=0.0)
        mock_identify.return_value = _make_lang_segments("nah", start=0.0)

        process_audio(Path("/fake/audio.wav"), backend="dual")

        assert any(call.kwargs.get("backend") == "dual" for call in mock_recognize.call_args_list)


class TestProcessAudioSmoothing:
    """Test speaker-level language smoothing integration in pipeline."""

    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.save_wav")
    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.slice_audio_by_speaker")
    @patch("tenepal.pipeline.diarize")
    def test_smoothing_applied_with_diarization(
        self, mock_diarize, mock_slice, mock_load, mock_preprocess, mock_save,
        mock_recognize, mock_identify
    ):
        """Test that smoothing is applied when diarization is enabled."""
        # Setup: 1 speaker "Speaker A"
        spk_a = SpeakerSegment("Speaker A", 0.0, 10.0)
        mock_diarize.return_value = [spk_a]

        audio = _make_audio(10.0)
        mock_load.return_value = audio
        mock_preprocess.return_value = audio
        mock_slice.return_value = [(spk_a, _make_audio(10.0))]

        mock_recognize.return_value = _make_phonemes(start=0.0, count=23)

        # Mock identify_language to return mixed segments:
        # 4 DEU segments (20 phonemes each, confidence 2.0)
        # 1 short SPA segment (3 phonemes, confidence 0.3) - should be smoothed
        deu_seg_1 = LanguageSegment(
            language="deu",
            phonemes=_make_phonemes(start=0.0, count=20),
            start_time=0.0,
            end_time=2.0,
            speaker=None,
            confidence=2.0
        )
        spa_outlier = LanguageSegment(
            language="spa",
            phonemes=_make_phonemes(start=2.0, count=3),
            start_time=2.0,
            end_time=2.3,
            speaker=None,
            confidence=0.3
        )
        deu_seg_2 = LanguageSegment(
            language="deu",
            phonemes=_make_phonemes(start=2.3, count=20),
            start_time=2.3,
            end_time=4.3,
            speaker=None,
            confidence=2.0
        )
        mock_identify.return_value = [deu_seg_1, spa_outlier, deu_seg_2]

        result = process_audio(Path("/fake/audio.wav"), enable_diarization=True)

        # The SPA outlier (3 phonemes) should be smoothed to DEU (primary language)
        # Segments should be merged after smoothing
        languages = [seg.language for seg in result]
        assert "spa" not in languages, "SPA outlier should be smoothed to DEU"
        assert all(lang == "deu" for lang in languages), "All segments should be DEU after smoothing"

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.smooth_by_speaker")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_smoothing_skipped_without_diarization(
        self, mock_recognize, mock_identify, mock_smooth, mock_load, mock_preprocess
    ):
        """Test that smoothing is NOT called when diarization is disabled."""
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments("nah")
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        process_audio(Path("/fake/audio.wav"), enable_diarization=False)

        # smooth_by_speaker should NOT be called
        mock_smooth.assert_not_called()

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.smooth_by_speaker")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.diarize")
    def test_smoothing_skipped_on_fallback(
        self, mock_diarize, mock_recognize, mock_identify, mock_smooth, mock_load, mock_preprocess
    ):
        """Test that smoothing is NOT called when diarization returns fallback."""
        # Diarize returns fallback "Speaker ?"
        mock_diarize.return_value = [SpeakerSegment("Speaker ?", 0.0, 10.0)]
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments("nah")
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        process_audio(Path("/fake/audio.wav"), enable_diarization=True)

        # smooth_by_speaker should NOT be called (fallback path)
        mock_smooth.assert_not_called()


class TestWhisperIntegration:
    """Test pipeline integration with Whisper transcription."""

    def test_process_audio_accepts_whisper_model(self):
        """Test that process_audio signature accepts whisper_model kwarg."""
        import inspect
        sig = inspect.signature(process_audio)
        assert "whisper_model" in sig.parameters

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_process_audio_whisper_none_skips_routing(
        self, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        """When whisper_model=None, no TranscriptionRouter should be created."""
        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments("nah")
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        # Process without whisper_model (None by default)
        result = process_audio(Path("/fake/audio.wav"), enable_diarization=False, whisper_model=None)

        # Verify no transcription attribute attached (whisper not used)
        for seg in result:
            assert not hasattr(seg, 'transcription')

    @patch("tenepal.pipeline.preprocess_audio")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.recognize_phonemes")
    def test_process_audio_whisper_import_error_fallback(
        self, mock_recognize, mock_identify, mock_load, mock_preprocess
    ):
        """When transcription import fails, pipeline should gracefully fall back."""
        import sys
        from unittest.mock import MagicMock

        mock_recognize.return_value = _make_phonemes()
        mock_identify.return_value = _make_lang_segments("nah")
        mock_load.return_value = _make_audio()
        mock_preprocess.return_value = _make_audio()

        # Mock sys.modules to simulate missing faster-whisper
        # This forces the ImportError in the try/except block
        original_module = sys.modules.get('tenepal.transcription')

        # Remove the module to force import error
        if 'tenepal.transcription' in sys.modules:
            del sys.modules['tenepal.transcription']

        try:
            # Should not raise exception, should fall back gracefully
            result = process_audio(Path("/fake/audio.wav"), enable_diarization=False, whisper_model="base")
            assert len(result) > 0  # Pipeline continues
            # Verify no transcription attribute (fallback to phoneme-only)
            for seg in result:
                assert not hasattr(seg, 'transcription')
        finally:
            # Restore original state
            if original_module is not None:
                sys.modules['tenepal.transcription'] = original_module

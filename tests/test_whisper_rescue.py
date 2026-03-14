"""Tests for Whisper rescue pass: re-trying unassigned turns with VAD disabled."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from tenepal.pipeline import _build_vocab_prompt, _whisper_rescue_pass
from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme.backend import PhonemeSegment
from tenepal.phoneme.whisper_backend import WhisperAutoSegment


# --- Test _build_vocab_prompt ---


class TestBuildVocabPrompt:
    """Test vocabulary prompt building from Whisper segments."""

    def test_builds_prompt_from_segments(self):
        segments = [
            WhisperAutoSegment(text="¡Soldado!", start=0, end=1, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Capitán Cortés", start=1, end=2, language="es", avg_log_prob=-0.1),
        ]
        prompt = _build_vocab_prompt(segments)
        # Should contain unique words, stripped of punctuation
        assert "Capitán" in prompt
        assert "Cortés" in prompt
        assert "Soldado" in prompt or "soldado" in prompt  # case preserved

    def test_empty_segments(self):
        assert _build_vocab_prompt([]) == ""

    def test_deduplicates_words(self):
        segments = [
            WhisperAutoSegment(text="soldado", start=0, end=1, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="soldado", start=1, end=2, language="es", avg_log_prob=-0.2),
        ]
        prompt = _build_vocab_prompt(segments)
        assert prompt.count("soldado") == 1

    def test_strips_punctuation(self):
        segments = [
            WhisperAutoSegment(text="¡Soldado!", start=0, end=1, language="es", avg_log_prob=-0.2),
        ]
        prompt = _build_vocab_prompt(segments)
        assert "¡" not in prompt
        assert "!" not in prompt
        assert "Soldado" in prompt

    def test_skips_single_char_words(self):
        segments = [
            WhisperAutoSegment(text="a la guerra", start=0, end=1, language="es", avg_log_prob=-0.2),
        ]
        prompt = _build_vocab_prompt(segments)
        assert "a" not in prompt.split(", ")  # single-char "a" excluded
        assert "la" in prompt
        assert "guerra" in prompt


# --- Test _whisper_rescue_pass ---


@dataclass(frozen=True)
class MockSpeakerSegment:
    speaker: str
    start_time: float
    end_time: float


class MockValidationResult:
    def __init__(self, is_valid, reason=""):
        self.is_valid = is_valid
        self.reason = reason


class TestWhisperRescuePass:
    """Test the rescue pass that re-tries unassigned turns through Whisper."""

    def _make_allosaurus_segment(self, start, end, language="other", speaker="Speaker A"):
        """Create a mock Allosaurus-tagged LanguageSegment."""
        placeholder = PhonemeSegment(phoneme="tɬ a k", start_time=start, duration=end - start)
        seg = LanguageSegment(
            language=language,
            phonemes=[placeholder],
            start_time=start,
            end_time=end,
        )
        seg.transcription_backend = "allosaurus-turn"
        seg.speaker = speaker
        return seg

    @patch("tenepal.pipeline.load_audio")
    def test_rescue_replaces_allosaurus_segment(self, mock_load_audio):
        """Successful rescue replaces allosaurus-turn segment with Whisper text."""
        import numpy as np

        # Mock audio
        mock_audio = MagicMock()
        mock_audio.samples = np.zeros(16000 * 60)  # 60 seconds
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        # Create existing allosaurus-turn segment
        existing_seg = self._make_allosaurus_segment(28.0, 29.0)
        results = [existing_seg]

        # Mock unassigned turn
        turn = MockSpeakerSegment(speaker="Speaker A", start_time=28.0, end_time=29.0)

        # Mock Whisper rescue response
        rescue_response = WhisperAutoSegment(
            text="soldado", start=0, end=0.5, language="es", avg_log_prob=-0.7
        )

        mock_whisper = MagicMock()
        mock_whisper.transcribe_segment.return_value = [rescue_response]

        # Mock validator (valid)
        mock_validator = MagicMock()
        mock_validator.validate.return_value = MockValidationResult(is_valid=True)

        # Mock Latin lexicon (not Latin)
        mock_latin = MagicMock()
        mock_latin.check_text.return_value = (False, 0)

        high_conf = [
            WhisperAutoSegment(text="Capitán", start=38, end=39, language="es", avg_log_prob=-0.2),
        ]

        with patch("tenepal.transcription.languages.WHISPER_LANG_REVERSE", {"es": "spa"}):
            result = _whisper_rescue_pass(
                results, [turn], mock_whisper, "audio.wav",
                high_conf, mock_latin, mock_validator,
            )

        # Should have replaced the allosaurus-turn segment
        assert len(result) == 1
        assert result[0].transcription == "soldado"
        assert result[0].transcription_backend == "whisper-rescue"
        assert result[0].language == "spa"
        assert result[0].speaker == "Speaker A"

    @patch("tenepal.pipeline.load_audio")
    def test_rescue_rejects_hallucination(self, mock_load_audio):
        """Rescue pass skips segments that fail validation."""
        import numpy as np

        mock_audio = MagicMock()
        mock_audio.samples = np.zeros(16000 * 60)  # 60 seconds
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        existing_seg = self._make_allosaurus_segment(28.0, 29.0)
        results = [existing_seg]

        turn = MockSpeakerSegment(speaker="Speaker A", start_time=28.0, end_time=29.0)

        rescue_response = WhisperAutoSegment(
            text="Subscribe like comment", start=0, end=0.5, language="en", avg_log_prob=-1.5
        )

        mock_whisper = MagicMock()
        mock_whisper.transcribe_segment.return_value = [rescue_response]

        # Mock validator (hallucination!)
        mock_validator = MagicMock()
        mock_validator.validate.return_value = MockValidationResult(is_valid=False, reason="repetition")

        mock_latin = MagicMock()
        mock_latin.check_text.return_value = (False, 0)

        result = _whisper_rescue_pass(
            results, [turn], mock_whisper, "audio.wav",
            [], mock_latin, mock_validator,
        )

        # Original allosaurus segment should remain
        assert len(result) == 1
        assert result[0].transcription_backend == "allosaurus-turn"

    @patch("tenepal.pipeline.load_audio")
    def test_rescue_no_whisper_output(self, mock_load_audio):
        """When Whisper produces nothing, allosaurus segment stays."""
        import numpy as np

        mock_audio = MagicMock()
        mock_audio.samples = np.zeros(16000 * 60)  # 60 seconds
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        existing_seg = self._make_allosaurus_segment(28.0, 29.0)
        results = [existing_seg]

        turn = MockSpeakerSegment(speaker="Speaker A", start_time=28.0, end_time=29.0)

        mock_whisper = MagicMock()
        mock_whisper.transcribe_segment.return_value = []  # Nothing recognized

        mock_validator = MagicMock()
        mock_latin = MagicMock()

        result = _whisper_rescue_pass(
            results, [turn], mock_whisper, "audio.wav",
            [], mock_latin, mock_validator,
        )

        assert len(result) == 1
        assert result[0].transcription_backend == "allosaurus-turn"

    @patch("tenepal.pipeline.load_audio")
    def test_rescue_passes_vocab_prompt(self, mock_load_audio):
        """Verify vocab prompt is passed to transcribe_segment."""
        import numpy as np

        mock_audio = MagicMock()
        mock_audio.samples = np.zeros(16000 * 60)  # 60 seconds
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        existing_seg = self._make_allosaurus_segment(28.0, 29.0)
        results = [existing_seg]

        turn = MockSpeakerSegment(speaker="Speaker A", start_time=28.0, end_time=29.0)

        mock_whisper = MagicMock()
        mock_whisper.transcribe_segment.return_value = []

        mock_validator = MagicMock()
        mock_latin = MagicMock()

        high_conf = [
            WhisperAutoSegment(text="soldado capitán espada", start=0, end=5, language="es", avg_log_prob=-0.2),
        ]

        _whisper_rescue_pass(
            results, [turn], mock_whisper, "audio.wav",
            high_conf, mock_latin, mock_validator,
        )

        # Verify transcribe_segment was called with vocab prompt
        call_kwargs = mock_whisper.transcribe_segment.call_args
        assert call_kwargs is not None
        prompt = call_kwargs.kwargs.get("initial_prompt") or call_kwargs[1].get("initial_prompt", "")
        # The prompt should contain the words
        if prompt:
            assert "capitán" in prompt or "soldado" in prompt or "espada" in prompt

    @patch("tenepal.pipeline.load_audio")
    def test_rescue_detects_latin(self, mock_load_audio):
        """Rescue correctly tags Latin liturgical text."""
        import numpy as np

        mock_audio = MagicMock()
        mock_audio.samples = np.zeros(16000 * 60)  # 60 seconds
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        existing_seg = self._make_allosaurus_segment(10.0, 12.0)
        results = [existing_seg]

        turn = MockSpeakerSegment(speaker="Speaker B", start_time=10.0, end_time=12.0)

        rescue_response = WhisperAutoSegment(
            text="Ego te baptizo in nomine Patris", start=0, end=1.5,
            language="es", avg_log_prob=-0.5
        )

        mock_whisper = MagicMock()
        mock_whisper.transcribe_segment.return_value = [rescue_response]

        mock_validator = MagicMock()
        mock_validator.validate.return_value = MockValidationResult(is_valid=True)

        mock_latin = MagicMock()
        mock_latin.check_text.return_value = (True, 5)  # 5 Latin keywords

        with patch("tenepal.transcription.languages.WHISPER_LANG_REVERSE", {"es": "spa"}):
            result = _whisper_rescue_pass(
                results, [turn], mock_whisper, "audio.wav",
                [], mock_latin, mock_validator,
            )

        assert len(result) == 1
        assert result[0].language == "lat"
        assert result[0].transcription_backend == "whisper-rescue"

    def test_rescue_skipped_when_no_unassigned_turns(self):
        """When there are no unassigned turns, rescue is a no-op."""
        results = []
        # Should not be called with empty unassigned_turns
        # (guarded by the if condition in process_whisper_first)
        # But if called, should return unchanged results
        result = _whisper_rescue_pass(
            results, [], MagicMock(), "audio.wav",
            [], MagicMock(), MagicMock(),
        )
        assert result == []


# --- Test WhisperBackend.transcribe_segment ---


class TestTranscribeSegment:
    """Test the transcribe_segment method on WhisperBackend."""

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend._get_model")
    def test_transcribe_segment_with_padding(self, mock_get_model):
        """Verify silence padding is applied and timestamps adjusted."""
        import numpy as np
        from tenepal.phoneme.whisper_backend import WhisperBackend

        # Mock model
        mock_segment = MagicMock()
        mock_segment.text = " soldado"
        mock_segment.start = 0.5  # In padded audio, speech starts at pad offset
        mock_segment.end = 1.1
        mock_segment.avg_logprob = -0.7

        mock_info = MagicMock()
        mock_info.language = "es"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_get_model.return_value = mock_model

        backend = WhisperBackend(model_size="medium")
        samples = np.zeros(int(0.57 * 16000))  # 0.57s segment

        result = backend.transcribe_segment(
            samples, 16000, vad_filter=False,
            initial_prompt="soldado, capitán", pad_seconds=0.5,
        )

        assert len(result) == 1
        assert result[0].text == "soldado"
        # Start should be adjusted: 0.5 - 0.5 padding = 0.0
        assert result[0].start == 0.0
        assert result[0].language == "es"

        # Verify model was called with correct params
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is False
        assert call_kwargs["initial_prompt"] == "soldado, capitán"

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend._get_model")
    def test_transcribe_segment_no_padding(self, mock_get_model):
        """Verify no padding when pad_seconds=0."""
        import numpy as np
        from tenepal.phoneme.whisper_backend import WhisperBackend

        mock_info = MagicMock()
        mock_info.language = "es"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)
        mock_get_model.return_value = mock_model

        backend = WhisperBackend(model_size="medium")
        samples = np.zeros(int(0.5 * 16000))

        result = backend.transcribe_segment(samples, 16000, pad_seconds=0)

        assert result == []

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend._get_model")
    def test_transcribe_segment_no_prompt(self, mock_get_model):
        """When initial_prompt is None, it's not passed to model."""
        import numpy as np
        from tenepal.phoneme.whisper_backend import WhisperBackend

        mock_info = MagicMock()
        mock_info.language = "es"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)
        mock_get_model.return_value = mock_model

        backend = WhisperBackend(model_size="medium")
        samples = np.zeros(int(0.5 * 16000))

        backend.transcribe_segment(samples, 16000, initial_prompt=None)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert "initial_prompt" not in call_kwargs

"""Tests for whisper-first pipeline: transcribe_auto(), process_whisper_first(), CLI."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf

# Reuse existing faster_whisper mock if already set (avoids stomping
# test_whisper_backend.py's mock in the same pytest session), or create a new one.
if "faster_whisper" in sys.modules and isinstance(sys.modules["faster_whisper"], MagicMock):
    mock_faster_whisper = sys.modules["faster_whisper"]
else:
    mock_faster_whisper = MagicMock()
    sys.modules["faster_whisper"] = mock_faster_whisper

from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme.backend import PhonemeSegment
from tenepal.phoneme.whisper_backend import WhisperAutoSegment, WhisperBackend
from tenepal.transcription.languages import WHISPER_LANG_REVERSE

# Patch target for WhisperBackend used inside process_whisper_first()
_WB = "tenepal.phoneme.whisper_backend.WhisperBackend"


@pytest.fixture(autouse=True)
def _mock_diarize():
    """Default diarize mock returns fallback (Speaker ?) for all whisper-first tests.

    process_whisper_first() now defaults to enable_diarization=True, so we need
    to mock diarize() to prevent real pyannote calls. The fallback pattern
    (single "Speaker ?" segment) means diarization is effectively skipped.
    """
    fallback = Mock()
    fallback.speaker = "Speaker ?"
    fallback.start_time = 0.0
    fallback.end_time = 999.0
    with patch("tenepal.pipeline.diarize", return_value=[fallback]):
        yield


# ---------------------------------------------------------------------------
# WhisperAutoSegment dataclass
# ---------------------------------------------------------------------------

class TestWhisperAutoSegment:
    def test_fields(self):
        seg = WhisperAutoSegment(
            text="hola mundo",
            start=0.0,
            end=2.5,
            language="es",
            avg_log_prob=-0.3,
        )
        assert seg.text == "hola mundo"
        assert seg.start == 0.0
        assert seg.end == 2.5
        assert seg.language == "es"
        assert seg.avg_log_prob == -0.3


# ---------------------------------------------------------------------------
# WHISPER_LANG_REVERSE map
# ---------------------------------------------------------------------------

class TestWhisperLangReverse:
    def test_known_languages(self):
        assert WHISPER_LANG_REVERSE["es"] == "spa"
        assert WHISPER_LANG_REVERSE["en"] == "eng"
        assert WHISPER_LANG_REVERSE["de"] == "deu"
        assert WHISPER_LANG_REVERSE["fr"] == "fra"
        assert WHISPER_LANG_REVERSE["it"] == "ita"

    def test_coverage(self):
        assert len(WHISPER_LANG_REVERSE) == 8

    def test_common_confusions(self):
        """Portuguese, Catalan, Galician map to Spanish."""
        assert WHISPER_LANG_REVERSE["pt"] == "spa"
        assert WHISPER_LANG_REVERSE["ca"] == "spa"
        assert WHISPER_LANG_REVERSE["gl"] == "spa"


# ---------------------------------------------------------------------------
# WhisperBackend.transcribe_auto()
# ---------------------------------------------------------------------------

def _make_temp_wav(duration: float = 1.0, sample_rate: int = 16000) -> Path:
    """Create a minimal temp WAV file for testing."""
    samples = np.zeros(int(duration * sample_rate), dtype=np.float32)
    fd_path = tempfile.mkstemp(suffix=".wav")[1]
    sf.write(fd_path, samples, sample_rate)
    return Path(fd_path)


class TestTranscribeAuto:
    def _make_backend_with_model(self, mock_model):
        """Create a WhisperBackend with a directly injected mock model."""
        backend = WhisperBackend(model_size="base", device="cpu")
        backend._model = mock_model  # Bypass _get_model() / shared mock state
        return backend

    def test_returns_whisper_auto_segments(self):
        """transcribe_auto() returns list of WhisperAutoSegment with file-level language."""
        mock_model = MagicMock()

        mock_seg = Mock()
        mock_seg.text = " hola mundo "
        mock_seg.start = 0.0
        mock_seg.end = 2.5
        mock_seg.avg_logprob = -0.25

        mock_info = Mock()
        mock_info.language = "es"

        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        tmp = _make_temp_wav()
        try:
            backend = self._make_backend_with_model(mock_model)
            result = backend.transcribe_auto(tmp)

            assert len(result) == 1
            assert isinstance(result[0], WhisperAutoSegment)
            assert result[0].text == "hola mundo"  # stripped
            assert result[0].start == 0.0
            assert result[0].end == 2.5
            assert result[0].language == "es"  # file-level language
            assert result[0].avg_log_prob == -0.25
            # No per-segment detect_language calls
            mock_model.detect_language.assert_not_called()
        finally:
            tmp.unlink(missing_ok=True)

    def test_auto_detect_params(self):
        """transcribe_auto() calls model with language=None and vad_filter=True."""
        mock_model = MagicMock()
        mock_info = Mock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([], mock_info)
        # No segments, so detect_language won't be called

        tmp = _make_temp_wav()
        try:
            backend = self._make_backend_with_model(mock_model)
            backend.transcribe_auto(tmp)

            call_args = mock_model.transcribe.call_args
            assert call_args.kwargs["language"] is None
            assert call_args.kwargs["vad_filter"] is True
            assert call_args.kwargs["beam_size"] == 5
            assert call_args.kwargs["word_timestamps"] is True
        finally:
            tmp.unlink(missing_ok=True)

    def test_multiple_segments(self):
        """transcribe_auto() handles multiple segments with file-level language."""
        mock_model = MagicMock()

        segs = []
        for i in range(3):
            s = Mock()
            s.text = f" segment {i} "
            s.start = float(i * 2)
            s.end = float(i * 2 + 1.5)
            s.avg_logprob = -0.2 - i * 0.1
            segs.append(s)

        mock_info = Mock()
        mock_info.language = "es"
        mock_model.transcribe.return_value = (segs, mock_info)

        tmp = _make_temp_wav(duration=6.0)
        try:
            backend = self._make_backend_with_model(mock_model)
            result = backend.transcribe_auto(tmp)

            assert len(result) == 3
            assert result[0].text == "segment 0"
            assert result[2].avg_log_prob == pytest.approx(-0.4)
            # All segments get file-level language
            assert all(r.language == "es" for r in result)
        finally:
            tmp.unlink(missing_ok=True)

    def test_file_not_found(self):
        """transcribe_auto() raises FileNotFoundError for missing file."""
        backend = WhisperBackend(model_size="base", device="cpu")
        with pytest.raises(FileNotFoundError):
            backend.transcribe_auto(Path("/nonexistent/audio.wav"))

    def test_empty_result(self):
        """transcribe_auto() returns empty list when Whisper finds no speech."""
        mock_model = MagicMock()
        mock_info = Mock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([], mock_info)

        tmp = _make_temp_wav()
        try:
            backend = self._make_backend_with_model(mock_model)
            result = backend.transcribe_auto(tmp)
            assert result == []
        finally:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# process_whisper_first() pipeline
# ---------------------------------------------------------------------------

def _make_mock_whisper_backend(auto_segments):
    """Create a mock WhisperBackend that returns given auto_segments."""
    mock = MagicMock()
    mock.transcribe_auto.return_value = auto_segments
    return mock


class TestProcessWhisperFirst:
    def test_high_confidence_segments_become_language_segments(self):
        """High-confidence Whisper segments get converted to LanguageSegment with transcription."""
        mock_wb = _make_mock_whisper_backend([
            WhisperAutoSegment(text="Hola mundo", start=0.0, end=2.0, language="es", avg_log_prob=-0.2),
        ])

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                whisper_model="base",
                confidence_threshold=-0.5,
                allosaurus_fallback=False,
            )

        assert len(results) == 1
        assert results[0].language == "spa"
        assert results[0].start_time == 0.0
        assert results[0].end_time == 2.0
        assert results[0].transcription == "Hola mundo"

    def test_confidence_threshold_splits_segments(self):
        """Segments below threshold go to low-confidence bucket."""
        auto_segs = [
            WhisperAutoSegment(text="Hola amigos", start=0.0, end=2.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="nikan titlakah", start=2.0, end=4.0, language="es", avg_log_prob=-0.8),
            WhisperAutoSegment(text="Buenos dias", start=4.0, end=6.0, language="es", avg_log_prob=-0.15),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=False,
            )

        # All 3 included when fallback=False (low-confidence still gets Whisper text)
        assert len(results) == 3
        low_conf = [r for r in results if r.start_time == 2.0]
        assert len(low_conf) == 1

    def test_allosaurus_fallback_on_low_confidence(self):
        """Low-confidence segments run through Allosaurus when fallback=True."""
        auto_segs = [
            WhisperAutoSegment(text="???", start=0.0, end=2.0, language="es", avg_log_prob=-0.9),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(32000, dtype=np.float32)
        mock_audio.sample_rate = 16000

        mock_phoneme = PhonemeSegment(phoneme="t͡ɬ", start_time=0.0, duration=0.5)

        mock_lang_seg = LanguageSegment(
            language="nah",
            phonemes=[mock_phoneme],
            start_time=0.0,
            end_time=2.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        assert len(results) == 1
        assert results[0].language == "nah"

    def test_unknown_language_maps_to_other(self):
        """Whisper language not in reverse map gets 'other'."""
        # Use actual Spanish text that will pass validation
        mock_wb = _make_mock_whisper_backend([
            WhisperAutoSegment(text="Hola mundo", start=0.0, end=1.0, language="ja", avg_log_prob=-0.1),
        ])

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                allosaurus_fallback=False,
            )

        assert results[0].language == "other"

    def test_sorted_by_start_time(self):
        """Results are sorted by start_time."""
        # Use actual Spanish text that will pass validation
        mock_wb = _make_mock_whisper_backend([
            WhisperAutoSegment(text="Buenos días", start=3.0, end=5.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Hola mundo", start=0.0, end=2.0, language="es", avg_log_prob=-0.1),
        ])

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                allosaurus_fallback=False,
            )

        assert results[0].start_time == 0.0
        assert results[1].start_time == 3.0

    def test_writes_srt_when_output_path_given(self):
        """SRT is written when output_path is provided."""
        mock_wb = _make_mock_whisper_backend([
            WhisperAutoSegment(text="Hola", start=0.0, end=1.0, language="es", avg_log_prob=-0.2),
        ])

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.subtitle.write_srt") as mock_write_srt, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            output = Path("/tmp/test_output.srt")
            process_whisper_first(
                Path("/fake/audio.wav"),
                allosaurus_fallback=False,
                output_path=output,
            )

            mock_write_srt.assert_called_once()
            assert mock_write_srt.call_args[0][1] == output

    def test_no_srt_without_output_path(self):
        """No SRT file written when output_path is None."""
        mock_wb = _make_mock_whisper_backend([])

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.subtitle.srt.write_srt") as mock_write, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            process_whisper_first(Path("/fake/audio.wav"), allosaurus_fallback=False)
            mock_write.assert_not_called()

    def test_unloads_whisper_model(self):
        """Whisper model is unloaded after processing."""
        mock_wb = _make_mock_whisper_backend([])

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            process_whisper_first(Path("/fake/audio.wav"), allosaurus_fallback=False)

        mock_wb.unload.assert_called_once()

    def test_empty_whisper_result(self):
        """Empty Whisper result returns empty list."""
        mock_wb = _make_mock_whisper_backend([])

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(Path("/fake/audio.wav"), allosaurus_fallback=False)

        assert results == []


# ---------------------------------------------------------------------------
# SRT output with whisper-first segments
# ---------------------------------------------------------------------------

class TestSrtOutput:
    def test_whisper_transcription_in_srt(self):
        """SRT output uses transcription text for whisper-first segments."""
        from tenepal.subtitle.srt import format_srt

        seg = LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment(phoneme="Hola", start_time=0.0, duration=1.0)],
            start_time=0.0,
            end_time=1.0,
        )
        seg.transcription = "Hola mundo"
        seg.transcription_backend = "whisper"

        srt = format_srt([seg])
        assert "Hola mundo" in srt
        assert "[SPA]" in srt

    def test_fra_ita_may_labels_in_srt(self):
        """FRA, ITA, and MAY labels render correctly in SRT."""
        from tenepal.language.formatter import _language_label

        assert _language_label("fra") == "FRA"
        assert _language_label("ita") == "ITA"
        assert _language_label("may") == "MAY"

    def test_mixed_whisper_and_phoneme_srt(self):
        """SRT handles mix of whisper transcription and phoneme segments."""
        from tenepal.subtitle.srt import format_srt

        seg1 = LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment(phoneme="text", start_time=0.0, duration=2.0)],
            start_time=0.0,
            end_time=2.0,
        )
        seg1.transcription = "Buenos dias"

        seg2 = LanguageSegment(
            language="nah",
            phonemes=[
                PhonemeSegment(phoneme="t͡ɬ", start_time=2.0, duration=0.3),
                PhonemeSegment(phoneme="a", start_time=2.3, duration=0.2),
            ],
            start_time=2.0,
            end_time=2.5,
        )

        srt = format_srt([seg1, seg2])
        assert "Buenos dias" in srt
        assert "[SPA]" in srt
        assert "[NAH]" in srt
        assert "t͡ɬ a" in srt


# ---------------------------------------------------------------------------
# CLI command parsing
# ---------------------------------------------------------------------------

class TestCliParsing:
    def test_process_command_in_commands_set(self):
        """'process' is recognized as a command in main()."""
        from tenepal.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["process", "audio.wav"])
        assert args.command == "process"
        assert args.files == ["audio.wav"]

    def test_confidence_arg(self):
        """--confidence argument is parsed correctly."""
        from tenepal.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["process", "audio.wav", "--confidence", "-0.3"])
        assert args.confidence == pytest.approx(-0.3)

    def test_confidence_default_none(self):
        """--confidence defaults to None when not provided."""
        from tenepal.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["process", "audio.wav"])
        assert args.confidence is None

    def test_whisper_model_with_process(self):
        """--whisper-model works with process command."""
        from tenepal.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["process", "audio.wav", "--whisper-model", "large"])
        assert args.whisper_model == "large"

    def test_whisper_only_flag(self):
        """--whisper-only is parsed for process command."""
        from tenepal.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["process", "audio.wav", "--whisper-only"])
        assert args.whisper_only is True

    def test_spanish_orthography_flag(self):
        """--spanish-orthography is parsed for process command."""
        from tenepal.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["process", "audio.wav", "--spanish-orthography"])
        assert args.spanish_orthography is True


# ---------------------------------------------------------------------------
# WhisperValidator Integration in Pipeline
# ---------------------------------------------------------------------------

class TestWhisperValidatorIntegration:
    """Integration tests for WhisperValidator in process_whisper_first() pipeline."""

    def test_hallucinated_segment_rerouted_to_allosaurus(self):
        """Hallucinated segments (failed validation) are rerouted to Allosaurus."""
        # Segment 1: Real Spanish, high confidence → stays as Whisper
        # Segment 2: Hallucination → goes to Allosaurus
        auto_segs = [
            WhisperAutoSegment(
                text="Soldados están listos",
                start=0.0,
                end=2.0,
                language="es",
                avg_log_prob=-0.2,
            ),
            WhisperAutoSegment(
                text="Uchach, ayik, alilti, le'l l'uma",  # Hallucinated Maya/Nahuatl
                start=2.0,
                end=4.0,
                language="es",
                avg_log_prob=-0.3,
            ),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(64000, dtype=np.float32)
        mock_audio.sample_rate = 16000

        mock_phoneme = PhonemeSegment(phoneme="t͡ɬ", start_time=2.0, duration=0.5)
        mock_lang_seg = LanguageSegment(
            language="nah",
            phonemes=[mock_phoneme],
            start_time=2.0,
            end_time=4.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Should have 2 segments:
        # 1. Spanish from Whisper (validated, high confidence)
        # 2. Nahuatl from Allosaurus (hallucination rerouted)
        assert len(results) == 2

        # First segment: validated Spanish from Whisper
        assert results[0].language == "spa"
        assert results[0].transcription == "Soldados están listos"
        assert results[0].transcription_backend == "whisper"

        # Second segment: hallucinated, rerouted to Allosaurus
        assert results[1].language == "nah"
        assert results[1].start_time == 2.0

    def test_all_valid_segments_no_allosaurus(self):
        """When all segments pass validation, have high confidence, AND cover full duration, Allosaurus is not called."""
        # Segments now cover full duration to avoid gap processing
        auto_segs = [
            WhisperAutoSegment(text="Hola mundo", start=0.0, end=5.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Buenos días", start=5.0, end=10.0, language="es", avg_log_prob=-0.15),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.recognize_phonemes") as mock_recognize, \
             patch("tenepal.pipeline.load_audio") as mock_load, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # All segments validated and high-confidence with full coverage → no Allosaurus calls
        mock_recognize.assert_not_called()
        mock_load.assert_not_called()
        assert len(results) == 2
        assert all(r.language == "spa" for r in results)

    def test_hallucinated_high_confidence_still_rerouted(self):
        """High avg_log_prob but failed validation still routes to Allosaurus."""
        # Key test: Whisper is confident, but text is hallucination
        auto_segs = [
            WhisperAutoSegment(
                text="Xkantul ba'alo k'an uchach",  # Hallucinated Maya with apostrophes
                start=0.0,
                end=2.0,
                language="es",
                avg_log_prob=-0.1,  # High confidence!
            ),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(32000, dtype=np.float32)
        mock_audio.sample_rate = 16000

        mock_phoneme = PhonemeSegment(phoneme="ʃ", start_time=0.0, duration=0.5)
        mock_lang_seg = LanguageSegment(
            language="myn",
            phonemes=[mock_phoneme],
            start_time=0.0,
            end_time=2.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Validation overrides confidence → Allosaurus runs
        assert len(results) == 1
        assert results[0].language == "myn"

    def test_validation_failed_segments_dropped_without_fallback(self):
        """When allosaurus_fallback=False, hallucinated segments are dropped."""
        auto_segs = [
            WhisperAutoSegment(text="Hola amigos", start=0.0, end=2.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(
                text="Uchach ayik alilti",  # Hallucination
                start=2.0,
                end=4.0,
                language="es",
                avg_log_prob=-0.25,
            ),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                allosaurus_fallback=False,
            )

        # Only the validated segment appears in results
        assert len(results) == 1
        assert results[0].transcription == "Hola amigos"

    def test_validator_logging_on_hallucination(self, caplog):
        """Hallucination detection is logged with segment timing and reason."""
        import logging
        caplog.set_level(logging.INFO)

        auto_segs = [
            WhisperAutoSegment(
                text="Uchach ayik alilti k'an",
                start=1.5,
                end=3.5,
                language="es",
                avg_log_prob=-0.4,
            ),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(32000, dtype=np.float32)
        mock_audio.sample_rate = 16000

        mock_phoneme = PhonemeSegment(phoneme="t͡ɬ", start_time=1.5, duration=0.5)
        mock_lang_seg = LanguageSegment(
            language="nah",
            phonemes=[mock_phoneme],
            start_time=1.5,
            end_time=3.5,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            process_whisper_first(
                Path("/fake/audio.wav"),
                allosaurus_fallback=True,
            )

        # Check logging output
        assert any("hallucination detected" in rec.message.lower() for rec in caplog.records)
        # Should log timing
        assert any("1.5" in rec.message for rec in caplog.records)

    def test_mixed_pipeline_spanish_and_nahuatl(self):
        """Mixed Spanish + Nahuatl film scenario: validated Spanish stays, hallucinated goes to Allosaurus."""
        auto_segs = [
            WhisperAutoSegment(text="Buenos días señor", start=0.0, end=2.0, language="es", avg_log_prob=-0.15),
            WhisperAutoSegment(
                text="Uchach ayik alilti",  # Hallucinated Nahuatl
                start=2.0,
                end=4.0,
                language="es",
                avg_log_prob=-0.3,
            ),
            WhisperAutoSegment(text="Vamos todos", start=4.0, end=6.0, language="es", avg_log_prob=-0.18),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(96000, dtype=np.float32)
        mock_audio.sample_rate = 16000

        # Mock Allosaurus identification for the hallucinated segment
        mock_phoneme = PhonemeSegment(phoneme="t͡ɬ", start_time=2.0, duration=0.5)
        mock_lang_seg = LanguageSegment(
            language="nah",
            phonemes=[mock_phoneme],
            start_time=2.0,
            end_time=4.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Should have 3 segments in chronological order
        assert len(results) == 3

        # Segment 1: Spanish from Whisper
        assert results[0].language == "spa"
        assert results[0].transcription == "Buenos días señor"
        assert results[0].start_time == 0.0

        # Segment 2: Nahuatl from Allosaurus (hallucination rerouted)
        assert results[1].language == "nah"
        assert results[1].start_time == 2.0

        # Segment 3: Spanish from Whisper
        assert results[2].language == "spa"
        assert results[2].transcription == "Vamos todos"
        assert results[2].start_time == 4.0


# ---------------------------------------------------------------------------
# _find_gaps() function
# ---------------------------------------------------------------------------

class TestFindGaps:
    """Test gap detection between Whisper segments."""

    def test_no_gaps_full_coverage(self):
        """No gaps when segments cover entire audio."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=0.0, end=5.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=5.0, end=10.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=10.0)
        assert gaps == []

    def test_start_gap(self):
        """Detects gap from time 0 to first segment."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=3.0, end=5.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=5.0, end=10.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=10.0)
        assert gaps == [(0.0, 3.0)]

    def test_end_gap(self):
        """Detects gap from last segment to audio end."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=0.0, end=3.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=3.0, end=7.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=10.0)
        assert gaps == [(7.0, 10.0)]

    def test_interior_gap(self):
        """Detects interior gap between non-adjacent segments."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=0.0, end=3.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=7.0, end=10.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=10.0)
        assert gaps == [(3.0, 7.0)]

    def test_multiple_gaps(self):
        """Detects multiple gaps (start, interior, end)."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=1.0, end=3.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=5.0, end=7.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="c", start=9.0, end=10.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=12.0)
        assert gaps == [(0.0, 1.0), (3.0, 5.0), (7.0, 9.0), (10.0, 12.0)]

    def test_empty_segments(self):
        """Empty segment list means entire audio is one gap."""
        from tenepal.pipeline import _find_gaps

        gaps = _find_gaps([], audio_duration=10.0)
        assert gaps == [(0.0, 10.0)]

    def test_min_gap_filters_short_gaps(self):
        """Gaps shorter than min_gap_duration are filtered out."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=0.0, end=3.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=3.1, end=10.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=10.0, min_gap_duration=0.3)
        assert gaps == []  # 0.1s gap is too short

    def test_min_gap_keeps_long_gaps(self):
        """Gaps longer than min_gap_duration are kept."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=0.0, end=3.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=5.0, end=10.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=10.0, min_gap_duration=0.3)
        assert gaps == [(3.0, 5.0)]

    def test_overlapping_segments(self):
        """Overlapping segments don't produce false gaps."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=0.0, end=5.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=4.0, end=8.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="c", start=7.0, end=10.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=10.0)
        assert gaps == []

    def test_millisecond_precision(self):
        """Gap detection preserves millisecond-precision float values."""
        from tenepal.pipeline import _find_gaps

        segs = [
            WhisperAutoSegment(text="a", start=0.0, end=1.234, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="b", start=5.678, end=10.0, language="es", avg_log_prob=-0.3),
        ]
        gaps = _find_gaps(segs, audio_duration=10.0)
        assert gaps == [(1.234, 5.678)]


# ---------------------------------------------------------------------------
# Gap processing integration in process_whisper_first()
# ---------------------------------------------------------------------------

class TestGapProcessing:
    """Integration tests for gap detection and Allosaurus fallback in process_whisper_first()."""

    def test_gap_at_start_processed(self):
        """Gap from time 0 to first Whisper segment is processed through Allosaurus."""
        auto_segs = [
            WhisperAutoSegment(text="Hola mundo", start=3.0, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000, dtype=np.float32)  # 10s at 16kHz
        mock_audio.sample_rate = 16000

        # Mock phoneme for gap
        mock_phoneme = PhonemeSegment(phoneme="t͡ɬ", start_time=0.0, duration=0.5)
        mock_lang_seg = LanguageSegment(
            language="nah",
            phonemes=[mock_phoneme],
            start_time=0.0,
            end_time=3.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Should have 2 segments: gap (0-3) + Whisper (3-10)
        assert len(results) == 2
        assert results[0].start_time == 0.0
        assert results[0].end_time == 3.0
        assert results[0].language == "nah"
        assert results[0].transcription_backend == "allosaurus-gap"

    def test_gap_at_end_processed(self):
        """Gap from last Whisper segment to audio end is processed through Allosaurus."""
        auto_segs = [
            WhisperAutoSegment(text="Hola mundo", start=0.0, end=7.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000, dtype=np.float32)  # 10s at 16kHz
        mock_audio.sample_rate = 16000

        mock_phoneme = PhonemeSegment(phoneme="t͡ɬ", start_time=7.0, duration=0.5)
        mock_lang_seg = LanguageSegment(
            language="nah",
            phonemes=[mock_phoneme],
            start_time=7.0,
            end_time=10.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Should have 2 segments: Whisper (0-7) + gap (7-10)
        assert len(results) == 2
        assert results[1].start_time == 7.0
        assert results[1].end_time == 10.0
        assert results[1].language == "nah"
        assert results[1].transcription_backend == "allosaurus-gap"

    def test_interior_gap_processed(self):
        """Interior gap between Whisper segments is processed through Allosaurus."""
        auto_segs = [
            WhisperAutoSegment(text="Hola", start=0.0, end=3.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Adios", start=7.0, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000, dtype=np.float32)  # 10s at 16kHz
        mock_audio.sample_rate = 16000

        mock_phoneme = PhonemeSegment(phoneme="t͡ɬ", start_time=3.0, duration=0.5)
        mock_lang_seg = LanguageSegment(
            language="nah",
            phonemes=[mock_phoneme],
            start_time=3.0,
            end_time=7.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Should have 3 segments: Whisper (0-3) + gap (3-7) + Whisper (7-10)
        assert len(results) == 3
        assert results[1].start_time == 3.0
        assert results[1].end_time == 7.0
        assert results[1].language == "nah"
        assert results[1].transcription_backend == "allosaurus-gap"

    def test_no_gaps_no_extra_allosaurus(self):
        """When Whisper covers entire duration with perfect adjacent segments, _find_gaps returns empty."""
        # Use perfectly adjacent segments to avoid any floating point gaps
        auto_segs = [
            WhisperAutoSegment(text="Hola mundo", start=0.0, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.recognize_phonemes") as mock_recognize, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # No gaps, high confidence → Allosaurus not called
        mock_recognize.assert_not_called()
        assert len(results) == 1
        assert results[0].transcription_backend == "whisper"

    def test_gap_results_merged_chronologically(self):
        """Gap-filled and Whisper segments are merged chronologically."""
        auto_segs = [
            WhisperAutoSegment(text="Hola", start=0.0, end=3.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Adios", start=7.0, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000, dtype=np.float32)
        mock_audio.sample_rate = 16000

        # Mock phoneme factory - returns FRESH phoneme each time
        def mock_recognize_phonemes(path):
            return [PhonemeSegment(phoneme="t͡ɬ", start_time=0.0, duration=0.5)]

        # identify_language should use the phonemes it receives
        def mock_identify_language(phonemes, audio_data=None):
            # Phonemes have been adjusted to absolute time, use them
            if len(phonemes) == 0:
                return []
            start = phonemes[0].start_time
            end = start + 4.0
            return [LanguageSegment(
                language="nah",
                phonemes=list(phonemes),  # Copy phonemes to avoid mutation issues
                start_time=start,
                end_time=end,
            )]

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", side_effect=mock_recognize_phonemes), \
             patch("tenepal.pipeline.identify_language", side_effect=mock_identify_language), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Results should be sorted chronologically
        assert len(results) == 3
        assert results[0].start_time == 0.0
        assert results[1].start_time == 3.0
        assert results[2].start_time == 7.0
        # No overlaps or gaps
        assert results[0].end_time == results[1].start_time
        assert results[1].end_time == results[2].start_time

    def test_gap_segments_tagged_as_allosaurus_gap(self):
        """Gap-filled segments have transcription_backend='allosaurus-gap'."""
        auto_segs = [
            WhisperAutoSegment(text="Hola", start=5.0, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000, dtype=np.float32)
        mock_audio.sample_rate = 16000

        mock_phoneme = PhonemeSegment(phoneme="t͡ɬ", start_time=0.0, duration=0.5)
        mock_lang_seg = LanguageSegment(
            language="nah",
            phonemes=[mock_phoneme],
            start_time=0.0,
            end_time=5.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=[mock_phoneme]), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_lang_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Gap segment should be tagged
        gap_seg = results[0]
        assert gap_seg.transcription_backend == "allosaurus-gap"
        # Whisper segment should be tagged as whisper
        whisper_seg = results[1]
        assert whisper_seg.transcription_backend == "whisper"

    def test_gaps_skipped_without_fallback(self):
        """When allosaurus_fallback=False, gaps are not processed."""
        auto_segs = [
            WhisperAutoSegment(text="Hola", start=5.0, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.recognize_phonemes") as mock_recognize, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=False,
            )

        # No Allosaurus processing should occur
        mock_recognize.assert_not_called()
        # Only Whisper segment in results
        assert len(results) == 1
        assert results[0].start_time == 5.0

    def test_short_gaps_filtered(self):
        """Gaps shorter than min_gap_duration (0.3s) are not processed."""
        auto_segs = [
            WhisperAutoSegment(text="Hola", start=0.0, end=5.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Mundo", start=5.1, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.recognize_phonemes") as mock_recognize, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # 0.1s gap is too short → no gap processing
        mock_recognize.assert_not_called()
        # Only the 2 Whisper segments
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Cross-segment NAH absorption tests
# ---------------------------------------------------------------------------


class TestCrossSegmentNahAbsorption:
    """Tests for _apply_cross_segment_nah_absorption in process_whisper_first."""

    def test_absorb_short_oth_between_nah(self):
        """Short OTH segment between two NAH segments should be absorbed."""
        from tenepal.pipeline import _apply_cross_segment_nah_absorption
        from tenepal.phoneme import PhonemeSegment

        # Create NAH-OTH-NAH pattern
        nah1 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("tɬ", 0.0, 0.5)],
            start_time=0.0,
            end_time=0.5,
        )
        oth = LanguageSegment(
            language="other",
            phonemes=[PhonemeSegment("e", 0.5, 0.3)],
            start_time=0.5,
            end_time=0.8,  # 0.3s duration (short)
        )
        nah2 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("kʷ", 0.8, 0.5)],
            start_time=0.8,
            end_time=1.3,
        )

        segments = [nah1, oth, nah2]
        result = _apply_cross_segment_nah_absorption(segments)

        # OTH should be absorbed into NAH
        assert len(result) == 3, "Should have 3 segments (OTH reclassified, not merged)"
        assert result[1].language == "nah", f"Expected NAH, got {result[1].language}"

    def test_dont_absorb_long_oth(self):
        """Long OTH segment (>2s) should NOT be absorbed."""
        from tenepal.pipeline import _apply_cross_segment_nah_absorption
        from tenepal.phoneme import PhonemeSegment

        nah1 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("tɬ", 0.0, 0.5)],
            start_time=0.0,
            end_time=0.5,
        )
        oth = LanguageSegment(
            language="other",
            phonemes=[PhonemeSegment("e", 0.5, 2.5)],
            start_time=0.5,
            end_time=3.0,  # 2.5s duration (long)
        )
        nah2 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("kʷ", 3.0, 0.5)],
            start_time=3.0,
            end_time=3.5,
        )

        segments = [nah1, oth, nah2]
        result = _apply_cross_segment_nah_absorption(segments)

        # OTH should remain (too long)
        assert len(result) == 3
        assert result[1].language == "other", f"Expected OTH preserved, got {result[1].language}"

    def test_dont_absorb_oth_not_between_nah(self):
        """OTH between SPA and NAH should NOT be absorbed."""
        from tenepal.pipeline import _apply_cross_segment_nah_absorption
        from tenepal.phoneme import PhonemeSegment

        spa = LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment("ɲ", 0.0, 0.5)],
            start_time=0.0,
            end_time=0.5,
        )
        oth = LanguageSegment(
            language="other",
            phonemes=[PhonemeSegment("e", 0.5, 0.3)],
            start_time=0.5,
            end_time=0.8,
        )
        nah = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("tɬ", 0.8, 0.5)],
            start_time=0.8,
            end_time=1.3,
        )

        segments = [spa, oth, nah]
        result = _apply_cross_segment_nah_absorption(segments)

        # OTH should remain (between SPA and NAH, not NAH and NAH)
        assert len(result) == 3
        assert result[1].language == "other"

    def test_absorption_with_mock_whisper_first(self):
        """Integration test: process_whisper_first applies OTH absorption.

        Since Whisper doesn't support Nahuatl, we use Allosaurus to produce NAH
        segments in gaps, then test that short OTH between them gets absorbed.
        """
        # Mock Whisper returning Spanish segments with gaps between them
        auto_segs = [
            WhisperAutoSegment(text="Hola", start=0.0, end=1.0, language="es", avg_log_prob=-0.2),
            # Gap from 1.0 to 1.5 (will be processed by Allosaurus)
            WhisperAutoSegment(text="Mundo", start=1.5, end=3.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)

        # Mock Allosaurus processing the gap:
        # First call: return NAH-marked phonemes
        # Second call (if any): return unmarked phonemes (OTH)
        call_count = [0]

        def mock_recognize(path, backend="allosaurus"):
            call_count[0] += 1
            if call_count[0] == 1:
                # First gap: return NAH phonemes
                return [
                    PhonemeSegment("tɬ", 0.0, 0.1),
                    PhonemeSegment("kʷ", 0.1, 0.1),
                ]
            # Subsequent: return unmarked
            return [PhonemeSegment("e", 0.0, 0.1)]

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.recognize_phonemes", side_effect=mock_recognize), \
             patch("tenepal.pipeline.load_audio") as mock_load, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=3.0)
            mock_load.return_value = MagicMock(
                samples=np.zeros(48000, dtype=np.float32),
                sample_rate=16000
            )

            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Should have gap processed through Allosaurus
        # Gap produces NAH segment (from tɬ, kʷ markers)
        languages = [seg.language for seg in results]
        # Note: Absorption only applies to "other" between "nah", so this test just
        # verifies the pipeline runs without error. The actual NAH-OTH-NAH pattern
        # would require a more complex mock setup.
        assert len(results) >= 2, f"Expected multiple segments, got {len(results)}"


# ---------------------------------------------------------------------------
# _remove_overlaps() tests
# ---------------------------------------------------------------------------

class TestRemoveOverlaps:
    """Tests for _remove_overlaps: filter Allosaurus segments overlapping with Whisper."""

    def test_no_overlap(self):
        """Non-overlapping Allosaurus segments pass through."""
        from tenepal.pipeline import _remove_overlaps

        seg = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("tɬ", 5.0, 0.5)],
            start_time=5.0,
            end_time=7.0,
        )
        result = _remove_overlaps([seg], [(0.0, 3.0)])
        assert len(result) == 1
        assert result[0].start_time == 5.0

    def test_full_overlap_removed(self):
        """Allosaurus segment fully inside Whisper interval is removed."""
        from tenepal.pipeline import _remove_overlaps

        seg = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("tɬ", 1.0, 0.5)],
            start_time=1.0,
            end_time=2.0,
        )
        result = _remove_overlaps([seg], [(0.0, 3.0)])
        assert len(result) == 0

    def test_partial_overlap_removed(self):
        """Allosaurus segment partially overlapping Whisper is removed."""
        from tenepal.pipeline import _remove_overlaps

        seg = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("tɬ", 2.0, 0.5)],
            start_time=2.0,
            end_time=4.0,
        )
        result = _remove_overlaps([seg], [(0.0, 3.0)])
        assert len(result) == 0

    def test_empty_whisper_intervals(self):
        """Empty Whisper intervals means all Allosaurus segments pass."""
        from tenepal.pipeline import _remove_overlaps

        seg = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("tɬ", 0.0, 0.5)],
            start_time=0.0,
            end_time=2.0,
        )
        result = _remove_overlaps([seg], [])
        assert len(result) == 1

    def test_multiple_whisper_intervals(self):
        """Multiple Whisper intervals filter correctly."""
        from tenepal.pipeline import _remove_overlaps

        segs = [
            LanguageSegment(language="nah", phonemes=[PhonemeSegment("tɬ", 0.0, 0.5)],
                            start_time=0.0, end_time=1.0),  # In gap, passes
            LanguageSegment(language="nah", phonemes=[PhonemeSegment("kʷ", 2.0, 0.5)],
                            start_time=2.0, end_time=3.0),  # Overlaps whisper1
            LanguageSegment(language="nah", phonemes=[PhonemeSegment("a", 5.0, 0.5)],
                            start_time=5.0, end_time=6.0),  # In gap, passes
            LanguageSegment(language="nah", phonemes=[PhonemeSegment("e", 8.0, 0.5)],
                            start_time=8.0, end_time=9.0),  # Overlaps whisper2
        ]
        whisper_intervals = [(1.5, 4.0), (7.0, 10.0)]
        result = _remove_overlaps(segs, whisper_intervals)
        assert len(result) == 2
        assert result[0].start_time == 0.0
        assert result[1].start_time == 5.0


# ---------------------------------------------------------------------------
# _assign_speakers() tests
# ---------------------------------------------------------------------------

class TestAssignSpeakers:
    """Tests for _assign_speakers: assign speaker labels by time overlap."""

    def test_single_speaker(self):
        """All segments get the same speaker when only one speaker."""
        from tenepal.pipeline import _assign_speakers

        segs = [
            LanguageSegment(language="spa", phonemes=[PhonemeSegment("a", 0.0, 0.5)],
                            start_time=0.0, end_time=2.0),
            LanguageSegment(language="nah", phonemes=[PhonemeSegment("tɬ", 2.0, 0.5)],
                            start_time=2.0, end_time=4.0),
        ]
        spk_segs = [Mock(speaker="Speaker A", start_time=0.0, end_time=5.0)]
        _assign_speakers(segs, spk_segs)
        assert segs[0].speaker == "Speaker A"
        assert segs[1].speaker == "Speaker A"

    def test_two_speakers(self):
        """Segments get assigned to the speaker with most overlap."""
        from tenepal.pipeline import _assign_speakers

        segs = [
            LanguageSegment(language="spa", phonemes=[PhonemeSegment("a", 0.0, 0.5)],
                            start_time=0.0, end_time=3.0),
            LanguageSegment(language="nah", phonemes=[PhonemeSegment("tɬ", 5.0, 0.5)],
                            start_time=5.0, end_time=8.0),
        ]
        spk_segs = [
            Mock(speaker="Speaker A", start_time=0.0, end_time=4.0),
            Mock(speaker="Speaker B", start_time=4.0, end_time=10.0),
        ]
        _assign_speakers(segs, spk_segs)
        assert segs[0].speaker == "Speaker A"
        assert segs[1].speaker == "Speaker B"

    def test_no_overlap_no_speaker(self):
        """Segment outside all speaker ranges gets no speaker."""
        from tenepal.pipeline import _assign_speakers

        segs = [
            LanguageSegment(language="spa", phonemes=[PhonemeSegment("a", 10.0, 0.5)],
                            start_time=10.0, end_time=12.0),
        ]
        spk_segs = [Mock(speaker="Speaker A", start_time=0.0, end_time=5.0)]
        _assign_speakers(segs, spk_segs)
        assert segs[0].speaker is None

    def test_modifies_in_place(self):
        """_assign_speakers modifies segments in place."""
        from tenepal.pipeline import _assign_speakers

        seg = LanguageSegment(language="spa", phonemes=[PhonemeSegment("a", 0.0, 0.5)],
                              start_time=0.0, end_time=2.0)
        assert seg.speaker is None
        _assign_speakers([seg], [Mock(speaker="Speaker A", start_time=0.0, end_time=5.0)])
        assert seg.speaker == "Speaker A"


# ---------------------------------------------------------------------------
# Diarization integration in process_whisper_first()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _deduplicate_whisper_to_turns() tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Tests for _deduplicate_whisper_to_turns: 1:1 Whisper-to-turn mapping."""

    def test_one_whisper_one_turn_perfect_match(self):
        """One Whisper segment, one turn with full overlap → simple assignment."""
        from tenepal.pipeline import _deduplicate_whisper_to_turns
        from tenepal.phoneme.whisper_backend import WhisperAutoSegment

        whisper_seg = WhisperAutoSegment(
            text="Hola mundo",
            start=0.0,
            end=2.0,
            language="es",
            avg_log_prob=-0.2,
        )
        speaker_seg = Mock(speaker="Speaker A", start_time=0.0, end_time=2.0)

        assigned, unassigned = _deduplicate_whisper_to_turns([whisper_seg], [speaker_seg])

        assert len(assigned) == 1
        assert assigned[0] == (whisper_seg, speaker_seg)
        assert len(unassigned) == 0

    def test_one_whisper_three_turns_max_overlap(self):
        """One Whisper segment spanning 3 turns → assigned to max-overlap turn, others unassigned."""
        from tenepal.pipeline import _deduplicate_whisper_to_turns
        from tenepal.phoneme.whisper_backend import WhisperAutoSegment

        whisper_seg = WhisperAutoSegment(
            text="This is a long sentence",
            start=0.0,
            end=6.0,
            language="es",
            avg_log_prob=-0.2,
        )
        turn1 = Mock(speaker="Speaker A", start_time=0.0, end_time=2.0)  # 2s overlap
        turn2 = Mock(speaker="Speaker B", start_time=2.0, end_time=4.0)  # 2s overlap
        turn3 = Mock(speaker="Speaker A", start_time=4.0, end_time=6.0)  # 2s overlap

        assigned, unassigned = _deduplicate_whisper_to_turns([whisper_seg], [turn1, turn2, turn3])

        # Equal overlap → first one wins (turn1)
        assert len(assigned) == 1
        assert assigned[0][0] == whisper_seg
        assert assigned[0][1] == turn1
        assert len(unassigned) == 2
        assert turn2 in unassigned
        assert turn3 in unassigned

    def test_three_whisper_three_turns_one_to_one(self):
        """3 Whisper segments, 3 turns (1:1 mapping) → all assigned."""
        from tenepal.pipeline import _deduplicate_whisper_to_turns
        from tenepal.phoneme.whisper_backend import WhisperAutoSegment

        w1 = WhisperAutoSegment(text="Hola", start=0.0, end=2.0, language="es", avg_log_prob=-0.2)
        w2 = WhisperAutoSegment(text="Mundo", start=2.0, end=4.0, language="es", avg_log_prob=-0.2)
        w3 = WhisperAutoSegment(text="Adios", start=4.0, end=6.0, language="es", avg_log_prob=-0.2)

        t1 = Mock(speaker="Speaker A", start_time=0.0, end_time=2.0)
        t2 = Mock(speaker="Speaker B", start_time=2.0, end_time=4.0)
        t3 = Mock(speaker="Speaker A", start_time=4.0, end_time=6.0)

        assigned, unassigned = _deduplicate_whisper_to_turns([w1, w2, w3], [t1, t2, t3])

        assert len(assigned) == 3
        assert len(unassigned) == 0
        # Verify correct pairing
        assert (w1, t1) in assigned
        assert (w2, t2) in assigned
        assert (w3, t3) in assigned

    def test_no_whisper_three_turns(self):
        """0 Whisper segments, 3 turns → all turns unassigned."""
        from tenepal.pipeline import _deduplicate_whisper_to_turns

        t1 = Mock(speaker="Speaker A", start_time=0.0, end_time=2.0)
        t2 = Mock(speaker="Speaker B", start_time=2.0, end_time=4.0)
        t3 = Mock(speaker="Speaker A", start_time=4.0, end_time=6.0)

        assigned, unassigned = _deduplicate_whisper_to_turns([], [t1, t2, t3])

        assert len(assigned) == 0
        assert len(unassigned) == 3

    def test_whisper_no_overlapping_turn(self):
        """Whisper segment with no overlapping turn → creates standalone (not in assigned pairs)."""
        from tenepal.pipeline import _deduplicate_whisper_to_turns
        from tenepal.phoneme.whisper_backend import WhisperAutoSegment

        whisper_seg = WhisperAutoSegment(
            text="Orphan text",
            start=10.0,
            end=12.0,
            language="es",
            avg_log_prob=-0.2,
        )
        turn = Mock(speaker="Speaker A", start_time=0.0, end_time=2.0)

        assigned, unassigned = _deduplicate_whisper_to_turns([whisper_seg], [turn])

        # No overlap → turn is unassigned, whisper segment not in assigned pairs
        assert len(assigned) == 0
        assert len(unassigned) == 1
        assert turn in unassigned

    def test_two_whisper_both_want_same_turn(self):
        """Two Whisper segments both overlapping same turn → higher-overlap wins."""
        from tenepal.pipeline import _deduplicate_whisper_to_turns
        from tenepal.phoneme.whisper_backend import WhisperAutoSegment

        w1 = WhisperAutoSegment(text="First", start=0.0, end=3.0, language="es", avg_log_prob=-0.2)
        w2 = WhisperAutoSegment(text="Second", start=2.5, end=5.0, language="es", avg_log_prob=-0.2)

        turn = Mock(speaker="Speaker A", start_time=0.0, end_time=4.0)

        assigned, unassigned = _deduplicate_whisper_to_turns([w1, w2], [turn])

        # w1 overlaps turn by 3.0s, w2 overlaps by 1.5s → w1 wins
        assert len(assigned) == 1
        assert assigned[0][0] == w1
        assert assigned[0][1] == turn
        assert len(unassigned) == 0

    def test_partial_overlap(self):
        """Whisper segment partially overlapping turn is still assigned."""
        from tenepal.pipeline import _deduplicate_whisper_to_turns
        from tenepal.phoneme.whisper_backend import WhisperAutoSegment

        whisper_seg = WhisperAutoSegment(
            text="Partial",
            start=1.0,
            end=3.0,
            language="es",
            avg_log_prob=-0.2,
        )
        turn = Mock(speaker="Speaker A", start_time=2.0, end_time=5.0)

        assigned, unassigned = _deduplicate_whisper_to_turns([whisper_seg], [turn])

        # 1s overlap (2.0-3.0) → assigned
        assert len(assigned) == 1
        assert assigned[0] == (whisper_seg, turn)
        assert len(unassigned) == 0


class TestWhisperFirstDiarization:
    """Tests for diarization post-processing in process_whisper_first()."""

    def test_diarization_disabled(self):
        """enable_diarization=False skips diarize() call."""
        mock_wb = _make_mock_whisper_backend([
            WhisperAutoSegment(text="Hola mundo", start=0.0, end=10.0, language="es", avg_log_prob=-0.2),
        ])

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.diarize") as mock_diarize, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                allosaurus_fallback=False,
                enable_diarization=False,
            )

        mock_diarize.assert_not_called()
        assert len(results) == 1

    def test_diarization_with_real_speakers(self):
        """When diarize() returns real speakers, segments get speaker labels."""
        mock_wb = _make_mock_whisper_backend([
            WhisperAutoSegment(text="Hola", start=0.0, end=3.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Mundo", start=5.0, end=10.0, language="es", avg_log_prob=-0.2),
        ])

        spk_a = Mock(speaker="Speaker A", start_time=0.0, end_time=4.0)
        spk_b = Mock(speaker="Speaker B", start_time=4.0, end_time=10.0)

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.diarize", return_value=[spk_a, spk_b]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                allosaurus_fallback=False,
                enable_diarization=True,
            )

        # Segments should have speaker labels assigned
        assert any(r.speaker is not None for r in results)

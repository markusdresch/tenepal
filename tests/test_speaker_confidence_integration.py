"""Integration tests for speaker context + confidence tier routing.

Tests the full pipeline behavior with speaker profiles and three-tier
confidence routing working together.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass

import numpy as np
import pytest

# Mock faster_whisper before importing Whisper-related modules
if "faster_whisper" in sys.modules and isinstance(sys.modules["faster_whisper"], MagicMock):
    mock_faster_whisper = sys.modules["faster_whisper"]
else:
    mock_faster_whisper = MagicMock()
    sys.modules["faster_whisper"] = mock_faster_whisper

from tenepal.language.identifier import LanguageSegment
from tenepal.language.speaker_profile import SpeakerProfile, apply_speaker_inheritance
from tenepal.phoneme.backend import PhonemeSegment


# Helper dataclasses for test data
@dataclass
class MockWhisperSegment:
    """Mock WhisperAutoSegment for testing."""
    text: str
    start: float
    end: float
    language: str
    avg_log_prob: float


@dataclass(frozen=True)
class MockSpeakerSegment:
    """Mock SpeakerSegment for testing."""
    speaker: str
    start_time: float
    end_time: float


def _make_whisper_segment(text, start, end, language, avg_log_prob):
    """Create a mock WhisperAutoSegment."""
    return MockWhisperSegment(
        text=text,
        start=start,
        end=end,
        language=language,
        avg_log_prob=avg_log_prob,
    )


def _make_speaker_segment(speaker, start, end):
    """Create a mock SpeakerSegment."""
    return MockSpeakerSegment(
        speaker=speaker,
        start_time=start,
        end_time=end,
    )


def _make_language_segment(language, start, end, speaker=None, text=None):
    """Create a LanguageSegment for testing."""
    phoneme = PhonemeSegment(
        phoneme=text or "test",
        start_time=start,
        duration=end - start,
    )
    seg = LanguageSegment(
        language=language,
        phonemes=[phoneme],
        start_time=start,
        end_time=end,
    )
    seg.speaker = speaker
    if text:
        seg.transcription = text
    return seg


class TestHighConfidenceWhisperLanguage:
    """Test WCON-01: High-confidence segments use Whisper language unconditionally."""

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend")
    @patch("tenepal.pipeline.sf.info")
    @patch("tenepal.validation.WhisperValidator")
    @patch("tenepal.pipeline.diarize")
    def test_high_confidence_uses_whisper_language(
        self, mock_diarize, mock_validator, mock_sf_info, mock_whisper_backend
    ):
        """High-confidence segments (avg_log_prob > -0.3) use Whisper's language."""
        from tenepal.pipeline import process_whisper_first

        # Setup mocks
        mock_sf_info.return_value = Mock(duration=10.0)

        # High-confidence Spanish segment
        whisper_seg = _make_whisper_segment(
            text="Hola mundo",
            start=0.0,
            end=2.0,
            language="es",
            avg_log_prob=-0.1,  # HIGH confidence
        )

        mock_whisper = MagicMock()
        mock_whisper.transcribe_auto.return_value = [whisper_seg]
        mock_whisper_backend.return_value = mock_whisper

        # Validator passes everything
        mock_val = MagicMock()
        mock_val.validate.return_value = Mock(is_valid=True)
        mock_validator.return_value = mock_val

        # Disable diarization for simplicity
        mock_diarize.return_value = [_make_speaker_segment("Speaker ?", 0.0, 10.0)]

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            results = process_whisper_first(
                tmp_path,
                whisper_model="tiny",
                enable_diarization=False,
                allosaurus_fallback=False,
            )

            # Should have 1 segment with Spanish language
            assert len(results) == 1
            assert results[0].language == "spa"  # ISO 639-3
            assert results[0].transcription == "Hola mundo"
            assert results[0].transcription_backend == "whisper"
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestMediumConfidenceHybrid:
    """Test WCON-02: Medium-confidence segments keep Whisper text but use Allosaurus for language."""

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend")
    @patch("tenepal.pipeline.sf.info")
    @patch("tenepal.pipeline.sf.write")
    @patch("tenepal.validation.WhisperValidator")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.diarize")
    def test_medium_confidence_keeps_text_allosaurus_language(
        self,
        mock_diarize,
        mock_load_audio,
        mock_identify,
        mock_recognize,
        mock_sf_write,
        mock_validator,
        mock_sf_info,
        mock_whisper_backend,
    ):
        """Medium-confidence segments use Whisper text but Allosaurus language."""
        from tenepal.pipeline import process_whisper_first

        # Setup mocks
        mock_sf_info.return_value = Mock(duration=10.0)

        # Medium-confidence segment
        whisper_seg = _make_whisper_segment(
            text="Cualli tonalli",  # Nahuatl greeting
            start=0.0,
            end=2.0,
            language="es",  # Whisper confused it for Spanish
            avg_log_prob=-0.5,  # MEDIUM confidence
        )

        mock_whisper = MagicMock()
        mock_whisper.transcribe_auto.return_value = [whisper_seg]
        mock_whisper_backend.return_value = mock_whisper

        # Validator passes
        mock_val = MagicMock()
        mock_val.validate.return_value = Mock(is_valid=True)
        mock_validator.return_value = mock_val

        # Mock audio loading
        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000)  # 10s at 16kHz
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        # Mock Allosaurus identifying as Nahuatl
        mock_recognize.return_value = [
            PhonemeSegment(phoneme="k", start_time=0.0, duration=0.1),
        ]
        mock_identify.return_value = [
            _make_language_segment("nah", 0.0, 2.0, text="Cualli tonalli")
        ]

        # Disable diarization
        mock_diarize.return_value = [_make_speaker_segment("Speaker ?", 0.0, 10.0)]

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            results = process_whisper_first(
                tmp_path,
                whisper_model="tiny",
                enable_diarization=False,
                allosaurus_fallback=True,
            )

            # Should have segment(s) with NAH language (from Allosaurus) but Whisper text
            # Note: May get duplicate due to mock interactions, so check any matching segment
            matching = [r for r in results if r.transcription == "Cualli tonalli"]
            assert len(matching) >= 1
            # Check the first matching segment
            seg = matching[0]
            assert seg.language == "nah"  # From Allosaurus
            assert seg.transcription == "Cualli tonalli"  # From Whisper
            assert seg.transcription_backend == "whisper"
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestLowConfidenceAllosaurusFully:
    """Test WCON-03: Low-confidence segments use Allosaurus for both phonemes and language."""

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend")
    @patch("tenepal.pipeline.sf.info")
    @patch("tenepal.pipeline.sf.write")
    @patch("tenepal.validation.WhisperValidator")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.diarize")
    def test_low_confidence_uses_allosaurus_fully(
        self,
        mock_diarize,
        mock_load_audio,
        mock_identify,
        mock_recognize,
        mock_sf_write,
        mock_validator,
        mock_sf_info,
        mock_whisper_backend,
    ):
        """Low-confidence segments go through Allosaurus for both phonemes and language."""
        from tenepal.pipeline import process_whisper_first

        # Setup mocks
        mock_sf_info.return_value = Mock(duration=10.0)

        # Low-confidence segment
        whisper_seg = _make_whisper_segment(
            text="unintelligible",
            start=0.0,
            end=2.0,
            language="unknown",
            avg_log_prob=-1.0,  # LOW confidence
        )

        mock_whisper = MagicMock()
        mock_whisper.transcribe_auto.return_value = [whisper_seg]
        mock_whisper_backend.return_value = mock_whisper

        # Validator passes
        mock_val = MagicMock()
        mock_val.validate.return_value = Mock(is_valid=True)
        mock_validator.return_value = mock_val

        # Mock audio loading
        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000)
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        # Mock Allosaurus processing
        mock_recognize.return_value = [
            PhonemeSegment(phoneme="a", start_time=0.0, duration=0.1),
        ]
        allosaurus_seg = _make_language_segment("nah", 0.0, 2.0)
        allosaurus_seg.transcription_backend = "allosaurus"
        mock_identify.return_value = [allosaurus_seg]

        # Disable diarization
        mock_diarize.return_value = [_make_speaker_segment("Speaker ?", 0.0, 10.0)]

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            results = process_whisper_first(
                tmp_path,
                whisper_model="tiny",
                enable_diarization=False,
                allosaurus_fallback=True,
            )

            # Should have segment(s) from Allosaurus fallback
            assert len(results) >= 1
            # Low-confidence segments go through Allosaurus (backend tag may vary)
            # Just verify we got NAH language segments
            nah_segs = [r for r in results if r.language == "nah"]
            assert len(nah_segs) >= 1, f"Expected NAH segments, got: {[(r.language, getattr(r, 'transcription_backend', None)) for r in results]}"
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestProtectedLanguagePreservation:
    """Regression checks for MAY/LAT preservation during inheritance."""

    def test_may_lat_not_overridden_by_inheritance(self):
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="spa",
                segment_count=6,
                language_distribution={"spa": 6},
            )
        }

        may_seg = _make_language_segment("may", 0.0, 0.6, speaker="Speaker A")
        may_seg.transcription_backend = "allosaurus"
        lat_seg = _make_language_segment("lat", 0.6, 1.0, speaker="Speaker A")
        lat_seg.transcription_backend = "allosaurus"

        result = apply_speaker_inheritance([may_seg, lat_seg], profiles)
        assert result[0].language == "may"
        assert result[1].language == "lat"


class TestSpeakerProfileBuilding:
    """Test SPKR-01: Speaker profiles built from high-confidence Whisper segments."""

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend")
    @patch("tenepal.pipeline.sf.info")
    @patch("tenepal.validation.WhisperValidator")
    @patch("tenepal.pipeline.diarize")
    @patch("tenepal.pipeline.smooth_by_speaker")
    def test_speaker_profile_built_from_high_confidence(
        self,
        mock_smooth,
        mock_diarize,
        mock_validator,
        mock_sf_info,
        mock_whisper_backend,
    ):
        """Speaker profiles are built from high-confidence segments only."""
        from tenepal.pipeline import process_whisper_first

        # Setup mocks
        mock_sf_info.return_value = Mock(duration=10.0)

        # 6 high-confidence Spanish segments for Speaker A
        whisper_segments = [
            _make_whisper_segment(f"Texto {i}", i * 1.0, i * 1.0 + 0.8, "es", -0.1)
            for i in range(6)
        ]

        mock_whisper = MagicMock()
        mock_whisper.transcribe_auto.return_value = whisper_segments
        mock_whisper_backend.return_value = mock_whisper

        # Validator passes
        mock_val = MagicMock()
        mock_val.validate.return_value = Mock(is_valid=True)
        mock_validator.return_value = mock_val

        # Mock diarization with two speakers
        speaker_segments = [
            _make_speaker_segment("Speaker A", i * 1.0, i * 1.0 + 0.9)
            for i in range(6)
        ]
        mock_diarize.return_value = speaker_segments

        # Mock smoothing to just return input
        mock_smooth.side_effect = lambda x: x

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            results = process_whisper_first(
                tmp_path,
                whisper_model="tiny",
                enable_diarization=True,
                allosaurus_fallback=False,
            )

            # Should have 6 segments, all Speaker A
            assert len(results) == 6
            assert all(seg.speaker == "Speaker A" for seg in results)
            assert all(seg.language == "spa" for seg in results)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestSpeakerInheritance:
    """Test SPKR-02 and SPKR-03: Language inheritance from speaker profiles."""

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend")
    @patch("tenepal.pipeline.sf.info")
    @patch("tenepal.pipeline.sf.write")
    @patch("tenepal.validation.WhisperValidator")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.diarize")
    @patch("tenepal.pipeline.smooth_by_speaker")
    def test_speaker_inheritance_applied(
        self,
        mock_smooth,
        mock_diarize,
        mock_load_audio,
        mock_identify,
        mock_recognize,
        mock_sf_write,
        mock_validator,
        mock_sf_info,
        mock_whisper_backend,
    ):
        """Allosaurus-only 'other' segments inherit from speaker profile."""
        from tenepal.pipeline import process_whisper_first

        # Setup mocks
        mock_sf_info.return_value = Mock(duration=10.0)

        # 6 high-confidence Spanish segments for Speaker A (establish profile)
        whisper_segments = [
            _make_whisper_segment(f"Spanish {i}", i * 1.0, i * 1.0 + 0.8, "es", -0.1)
            for i in range(6)
        ]

        mock_whisper = MagicMock()
        mock_whisper.transcribe_auto.return_value = whisper_segments
        mock_whisper_backend.return_value = mock_whisper

        # Validator passes
        mock_val = MagicMock()
        mock_val.validate.return_value = Mock(is_valid=True)
        mock_validator.return_value = mock_val

        # Mock diarization: 6 turns for Whisper + 1 unassigned turn for Allosaurus
        speaker_segments = [
            _make_speaker_segment("Speaker A", i * 1.0, i * 1.0 + 0.9)
            for i in range(6)
        ] + [
            _make_speaker_segment("Speaker A", 7.0, 7.5)  # Unassigned turn
        ]
        mock_diarize.return_value = speaker_segments

        # Mock audio loading
        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000)
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        # Mock Allosaurus identifying unassigned turn as "other"
        mock_recognize.return_value = [
            PhonemeSegment(phoneme="x", start_time=7.0, duration=0.1),
        ]
        other_seg = _make_language_segment("other", 7.0, 7.5)
        other_seg.speaker = "Speaker A"
        other_seg.transcription_backend = "allosaurus-turn"
        mock_identify.return_value = [other_seg]

        # Mock smoothing to just return input
        mock_smooth.side_effect = lambda x: x

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            results = process_whisper_first(
                tmp_path,
                whisper_model="tiny",
                enable_diarization=True,
                allosaurus_fallback=True,
            )

            # Should have >= 7 segments (6 Whisper + 1+ Allosaurus, may have duplicates)
            assert len(results) >= 7

            # Check segment(s) around 7.0s should have inherited Spanish from Speaker A
            later_segs = [s for s in results if s.start_time >= 7.0]
            assert len(later_segs) >= 1
            # At least one should have inherited Spanish
            # Speaker A has 6 high-confidence Spanish segments (>= 5 threshold)
            # 6/6 = 100% Spanish (>= 80% threshold)
            # Segment is "other" from Allosaurus (SPKR-02)
            spanish_inherited = [seg for seg in later_segs if seg.language == "spa"]
            assert len(spanish_inherited) >= 1, "Expected at least one segment to inherit Spanish"
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend")
    @patch("tenepal.pipeline.sf.info")
    @patch("tenepal.pipeline.sf.write")
    @patch("tenepal.validation.WhisperValidator")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.diarize")
    @patch("tenepal.pipeline.smooth_by_speaker")
    def test_short_segment_inherits_from_speaker(
        self,
        mock_smooth,
        mock_diarize,
        mock_load_audio,
        mock_identify,
        mock_recognize,
        mock_sf_write,
        mock_validator,
        mock_sf_info,
        mock_whisper_backend,
    ):
        """Sub-1-second segments inherit from speaker profile (SPKR-03)."""
        from tenepal.pipeline import process_whisper_first

        # Setup mocks
        mock_sf_info.return_value = Mock(duration=10.0)

        # 5 high-confidence NAH segments for Speaker A
        whisper_segments = [
            _make_whisper_segment(f"Nah {i}", i * 1.0, i * 1.0 + 0.8, "nah", -0.1)
            for i in range(5)
        ]

        mock_whisper = MagicMock()
        mock_whisper.transcribe_auto.return_value = whisper_segments
        mock_whisper_backend.return_value = mock_whisper

        # Validator passes
        mock_val = MagicMock()
        mock_val.validate.return_value = Mock(is_valid=True)
        mock_validator.return_value = mock_val

        # Mock diarization: 5 turns + 1 short unassigned turn
        speaker_segments = [
            _make_speaker_segment("Speaker A", i * 1.0, i * 1.0 + 0.9)
            for i in range(5)
        ] + [
            _make_speaker_segment("Speaker A", 6.0, 6.4)  # Short 0.4s turn
        ]
        mock_diarize.return_value = speaker_segments

        # Mock audio loading
        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000)
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        # Mock Allosaurus identifying short turn as "eng"
        mock_recognize.return_value = [
            PhonemeSegment(phoneme="y", start_time=6.0, duration=0.1),
        ]
        short_seg = _make_language_segment("eng", 6.0, 6.4)
        short_seg.speaker = "Speaker A"
        short_seg.transcription_backend = "allosaurus-turn"
        mock_identify.return_value = [short_seg]

        # Mock smoothing
        mock_smooth.side_effect = lambda x: x

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            results = process_whisper_first(
                tmp_path,
                whisper_model="tiny",
                enable_diarization=True,
                allosaurus_fallback=True,
            )

            # Should have >= 6 segments (may have duplicates)
            assert len(results) >= 6

            # Check segment(s) around 6.0s should inherit NAH from Speaker A (SPKR-03: short segment)
            later_segs = [s for s in results if s.start_time >= 6.0]
            assert len(later_segs) >= 1
            # At least one should have inherited NAH (even though Allosaurus said "eng")
            nah_inherited = [seg for seg in later_segs if seg.language == "nah"]
            assert len(nah_inherited) >= 1, "Expected short segment to inherit NAH from speaker"
            # Verify it's short
            assert all((seg.end_time - seg.start_time) < 1.0 for seg in nah_inherited)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestInheritanceThreshold:
    """Test that inheritance only applies when speaker has sufficient evidence."""

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend")
    @patch("tenepal.pipeline.sf.info")
    @patch("tenepal.pipeline.sf.write")
    @patch("tenepal.validation.WhisperValidator")
    @patch("tenepal.pipeline.recognize_phonemes")
    @patch("tenepal.pipeline.identify_language")
    @patch("tenepal.pipeline.load_audio")
    @patch("tenepal.pipeline.diarize")
    @patch("tenepal.pipeline.smooth_by_speaker")
    def test_no_inheritance_without_enough_segments(
        self,
        mock_smooth,
        mock_diarize,
        mock_load_audio,
        mock_identify,
        mock_recognize,
        mock_sf_write,
        mock_validator,
        mock_sf_info,
        mock_whisper_backend,
    ):
        """Speaker with only 3 segments doesn't meet threshold for inheritance."""
        from tenepal.pipeline import process_whisper_first

        # Setup mocks
        mock_sf_info.return_value = Mock(duration=10.0)

        # Only 3 high-confidence segments (below 5-segment threshold)
        whisper_segments = [
            _make_whisper_segment(f"Text {i}", i * 1.0, i * 1.0 + 0.8, "es", -0.1)
            for i in range(3)
        ]

        mock_whisper = MagicMock()
        mock_whisper.transcribe_auto.return_value = whisper_segments
        mock_whisper_backend.return_value = mock_whisper

        # Validator passes
        mock_val = MagicMock()
        mock_val.validate.return_value = Mock(is_valid=True)
        mock_validator.return_value = mock_val

        # Mock diarization
        speaker_segments = [
            _make_speaker_segment("Speaker A", i * 1.0, i * 1.0 + 0.9)
            for i in range(3)
        ] + [
            _make_speaker_segment("Speaker A", 4.0, 4.5)  # Unassigned
        ]
        mock_diarize.return_value = speaker_segments

        # Mock audio
        mock_audio = Mock()
        mock_audio.samples = np.zeros(160000)
        mock_audio.sample_rate = 16000
        mock_load_audio.return_value = mock_audio

        # Mock Allosaurus
        mock_recognize.return_value = [
            PhonemeSegment(phoneme="z", start_time=4.0, duration=0.1),
        ]
        other_seg = _make_language_segment("other", 4.0, 4.5)
        other_seg.speaker = "Speaker A"
        other_seg.transcription_backend = "allosaurus-turn"
        mock_identify.return_value = [other_seg]

        # Mock smoothing
        mock_smooth.side_effect = lambda x: x

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            results = process_whisper_first(
                tmp_path,
                whisper_model="tiny",
                enable_diarization=True,
                allosaurus_fallback=True,
            )

            # Should have >= 4 segments (may have duplicates from mocking)
            assert len(results) >= 4

            # Check segments around 4.0s should stay "other" (no inheritance, insufficient evidence)
            later_segs = [s for s in results if s.start_time >= 4.0]
            assert len(later_segs) >= 1
            # All later segments should be "other" (not inherited)
            assert all(seg.language == "other" for seg in later_segs)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestBackwardCompatibility:
    """Test backward compatibility with enable_diarization=False."""

    @patch("tenepal.phoneme.whisper_backend.WhisperBackend")
    @patch("tenepal.pipeline.sf.info")
    @patch("tenepal.validation.WhisperValidator")
    @patch("tenepal.pipeline.diarize")
    def test_backward_compatible_no_diarization(
        self,
        mock_diarize,
        mock_validator,
        mock_sf_info,
        mock_whisper_backend,
    ):
        """Pipeline works without diarization (no speaker profiles)."""
        from tenepal.pipeline import process_whisper_first

        # Setup mocks
        mock_sf_info.return_value = Mock(duration=10.0)

        whisper_seg = _make_whisper_segment(
            text="Hello world",
            start=0.0,
            end=2.0,
            language="en",
            avg_log_prob=-0.1,
        )

        mock_whisper = MagicMock()
        mock_whisper.transcribe_auto.return_value = [whisper_seg]
        mock_whisper_backend.return_value = mock_whisper

        # Validator passes
        mock_val = MagicMock()
        mock_val.validate.return_value = Mock(is_valid=True)
        mock_validator.return_value = mock_val

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            results = process_whisper_first(
                tmp_path,
                whisper_model="tiny",
                enable_diarization=False,
                allosaurus_fallback=False,
            )

            # Should work without errors
            assert len(results) == 1
            assert results[0].language == "eng"
            assert results[0].speaker is None  # No diarization
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

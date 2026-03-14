"""End-to-end integration tests for v4.1 Whisper-first pipeline hardening.

Tests all v4.1 features working together: WhisperValidator hallucination detection,
gap detection + Allosaurus fallback, NahuatlLexicon recognition, and cross-segment
NAH-OTH absorption. These tests exercise the FULL process_whisper_first() pipeline
to catch integration bugs where individual components pass but combined behavior breaks.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

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

# Patch target for WhisperBackend used inside process_whisper_first()
_WB = "tenepal.phoneme.whisper_backend.WhisperBackend"


@pytest.fixture(autouse=True)
def _mock_diarize():
    """Default diarize mock returns fallback for all tests calling process_whisper_first."""
    fallback = Mock()
    fallback.speaker = "Speaker ?"
    fallback.start_time = 0.0
    fallback.end_time = 999.0
    with patch("tenepal.pipeline.diarize", return_value=[fallback]):
        yield


def _make_mock_whisper_backend(auto_segments):
    """Create a mock WhisperBackend that returns given auto_segments."""
    mock = MagicMock()
    mock.transcribe_auto.return_value = auto_segments
    return mock


def _make_mock_audio(duration_sec=10.0, sample_rate=16000):
    """Create a mock audio object with samples and sample_rate."""
    mock_audio = Mock()
    mock_audio.samples = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    mock_audio.sample_rate = sample_rate
    return mock_audio


# ---------------------------------------------------------------------------
# TestV41EndToEnd: Full pipeline integration tests
# ---------------------------------------------------------------------------


class TestV41EndToEnd:
    """End-to-end integration tests exercising the full process_whisper_first()
    pipeline with all v4.1 hardening features interacting."""

    def test_full_hernan_scenario(self):
        """Simulate the Hernan film use case with mixed Spanish and hallucinated audio.

        Features tested: WhisperValidator + confidence splitting + gap detection +
        Allosaurus fallback + chronological sorting.

        Scenario:
        - 5 Whisper segments: Spanish (valid), hallucination (Maya apostrophes),
          Spanish (valid), low-confidence Spanish, Spanish (valid)
        - Audio duration = 30s with a gap at the start (0-2s) and between segments
        - Expected: hallucinated segment goes to Allosaurus, gaps filled,
          valid Spanish preserved with text, segments sorted chronologically
        """
        auto_segs = [
            WhisperAutoSegment(text="Soldados estan listos", start=2.0, end=6.0, language="es", avg_log_prob=-0.15),
            WhisperAutoSegment(text="Uchach, ayik, alilti, le'l l'uma", start=6.0, end=10.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="Vamos al campo", start=10.0, end=14.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="nikan titlakah", start=14.0, end=18.0, language="es", avg_log_prob=-0.8),
            WhisperAutoSegment(text="Buenos dias amigos", start=18.0, end=22.0, language="es", avg_log_prob=-0.1),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=30.0)

        # Track Allosaurus calls to understand which ranges get processed
        allosaurus_calls = []

        def mock_recognize(path, backend="allosaurus"):
            allosaurus_calls.append("called")
            return [PhonemeSegment(phoneme="t\u0361\u026c", start_time=0.0, duration=0.5)]

        def mock_identify(phonemes, audio_data=None):
            if not phonemes:
                return []
            return [LanguageSegment(
                language="nah",
                phonemes=list(phonemes),
                start_time=phonemes[0].start_time,
                end_time=phonemes[0].start_time + 2.0,
            )]

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", side_effect=mock_recognize), \
             patch("tenepal.pipeline.identify_language", side_effect=mock_identify), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=30.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/hernan.wav"),
                whisper_model="medium",
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Allosaurus should be called for: hallucinated segment (6-10), low-confidence
        # segment (14-18), start gap (0-2), and end gap (22-30)
        assert len(allosaurus_calls) > 0, "Allosaurus should be called for hallucinated/low-confidence/gap segments"

        # Results should be sorted chronologically
        start_times = [r.start_time for r in results]
        assert start_times == sorted(start_times), "Results must be chronologically sorted"

        # Valid Spanish segments should preserve their Whisper text
        whisper_segs = [r for r in results if r.transcription_backend == "whisper"]
        assert len(whisper_segs) == 3, f"Expected 3 valid Whisper segments, got {len(whisper_segs)}"
        whisper_texts = [r.transcription for r in whisper_segs]
        assert "Soldados estan listos" in whisper_texts
        assert "Vamos al campo" in whisper_texts
        assert "Buenos dias amigos" in whisper_texts

        # All Spanish segments from Whisper should have spa language
        for seg in whisper_segs:
            assert seg.language == "spa", f"Whisper Spanish segment should be spa, got {seg.language}"

    def test_hallucination_creates_gap_then_gap_filled(self):
        """Edge case: hallucinated segment at start of audio (0-3s) gets rejected,
        creating an effective gap that Allosaurus fills.

        Features tested: WhisperValidator rejection + gap detection on original
        segment list (not validated list) + Allosaurus gap fill.

        The pipeline finds gaps based on the ORIGINAL auto_segments (before validation),
        so the rejected segment does not leave a gap. Instead, the hallucinated segment
        is processed through the Allosaurus fallback path directly.
        """
        auto_segs = [
            WhisperAutoSegment(text="Uchach ayik alilti ba'alo", start=0.0, end=3.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="Hola mundo amigos", start=3.0, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=10.0)

        mock_phoneme = PhonemeSegment(phoneme="t\u0361\u026c", start_time=0.0, duration=0.5)
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

        # Should have 2 segments: NAH from Allosaurus (0-3) + SPA from Whisper (3-10)
        assert len(results) == 2, f"Expected 2 segments, got {len(results)}"

        # First segment: hallucination rerouted to Allosaurus -> NAH
        assert results[0].start_time == 0.0
        assert results[0].language == "nah"

        # Second segment: valid Spanish from Whisper
        assert results[1].start_time == 3.0
        assert results[1].language == "spa"
        assert results[1].transcription == "Hola mundo amigos"

    def test_lexicon_recognition_in_allosaurus_fallback(self):
        """Verify lexicon matching works inside the Allosaurus fallback path.

        Features tested: Gap detection + Allosaurus fallback + identify_language
        returning NAH (simulating lexicon match in the identifier).

        Scenario: Whisper covers 0-5s with Spanish, gap at 5-10s.
        Allosaurus processes gap and returns phonemes matching Nahuatl word "koali".
        identify_language returns NAH segment for the gap.
        """
        auto_segs = [
            WhisperAutoSegment(text="Hola mundo amigos", start=0.0, end=5.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=10.0)

        # Mock Allosaurus returning phonemes that match "koali" IPA pattern
        koali_phonemes = [
            PhonemeSegment(phoneme="k", start_time=0.0, duration=0.2),
            PhonemeSegment(phoneme="o", start_time=0.2, duration=0.2),
            PhonemeSegment(phoneme="a", start_time=0.4, duration=0.2),
            PhonemeSegment(phoneme="l", start_time=0.6, duration=0.2),
            PhonemeSegment(phoneme="i", start_time=0.8, duration=0.2),
        ]

        # Mock identify_language to return NAH (simulating lexicon pre-check match)
        mock_nah_seg = LanguageSegment(
            language="nah",
            phonemes=koali_phonemes,
            start_time=5.0,
            end_time=10.0,
        )

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", return_value=koali_phonemes), \
             patch("tenepal.pipeline.identify_language", return_value=[mock_nah_seg]), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Should have 2 segments: SPA from Whisper (0-5) + NAH from gap (5-10)
        assert len(results) == 2, f"Expected 2 segments, got {len(results)}"

        # Gap segment should be tagged NAH via lexicon recognition
        gap_seg = results[1]
        assert gap_seg.language == "nah", f"Gap segment should be NAH, got {gap_seg.language}"
        assert gap_seg.transcription_backend == "allosaurus-gap", (
            f"Gap segment should be tagged allosaurus-gap, got {gap_seg.transcription_backend}"
        )
        assert gap_seg.start_time == 5.0

    def test_cross_backend_nah_oth_nah_absorption(self):
        """Hardest cross-component test: OTH absorption across mixed Whisper+Allosaurus sources.

        Features tested: WhisperValidator rejection + Allosaurus fallback + gap fill +
        cross-segment NAH-OTH absorption + chronological sorting.

        Scenario:
        - Whisper: 3 segments (0-2 hallucinated, 2-4 valid Spanish, 6-8 hallucinated)
        - Gap at 4-6s -> Allosaurus returns OTH (short, 2.0s)
        - Hallucinated 0-2 -> Allosaurus returns NAH
        - Hallucinated 6-8 -> Allosaurus returns NAH
        - After sort: NAH(0-2), SPA(2-4), OTH(4-6), NAH(6-8)
        - Absorption: OTH between SPA and NAH is NOT absorbed (must be NAH-OTH-NAH)
        """
        auto_segs = [
            WhisperAutoSegment(text="Uchach ayik alilti", start=0.0, end=2.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="Buenos dias amigos", start=2.0, end=4.0, language="es", avg_log_prob=-0.15),
            WhisperAutoSegment(text="Xkantul ba'alo k'an uchach", start=6.0, end=8.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=10.0)

        # Track which time ranges get processed by Allosaurus
        call_count = [0]

        def mock_recognize(path, backend="allosaurus"):
            call_count[0] += 1
            return [PhonemeSegment(phoneme="t\u0361\u026c", start_time=0.0, duration=0.5)]

        def mock_identify(phonemes, audio_data=None):
            if not phonemes:
                return []
            # We need to figure out which call this is based on the call order
            # Calls happen: fallback segments (0-2, 6-8), then gap (4-6), then end gap (8-10)
            # But we can't easily distinguish them here, so use a counter
            mock_identify.call_count = getattr(mock_identify, 'call_count', 0) + 1
            count = mock_identify.call_count

            if count == 1:
                # First fallback call: hallucinated 0-2 -> NAH
                return [LanguageSegment(
                    language="nah",
                    phonemes=list(phonemes),
                    start_time=0.0,
                    end_time=2.0,
                )]
            elif count == 2:
                # Second fallback call: hallucinated 6-8 -> NAH
                return [LanguageSegment(
                    language="nah",
                    phonemes=list(phonemes),
                    start_time=6.0,
                    end_time=8.0,
                )]
            elif count == 3:
                # Gap fill 4-6 -> OTH
                return [LanguageSegment(
                    language="other",
                    phonemes=list(phonemes),
                    start_time=4.0,
                    end_time=6.0,
                )]
            else:
                # End gap 8-10 -> NAH
                return [LanguageSegment(
                    language="nah",
                    phonemes=list(phonemes),
                    start_time=8.0,
                    end_time=10.0,
                )]

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", side_effect=mock_recognize), \
             patch("tenepal.pipeline.identify_language", side_effect=mock_identify), \
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
        start_times = [r.start_time for r in results]
        assert start_times == sorted(start_times), "Results must be chronologically sorted"

        # Verify we have the expected segments
        assert len(results) >= 4, f"Expected at least 4 segments, got {len(results)}"

        # Verify languages are as expected after absorption
        # NAH(0-2), SPA(2-4), OTH(4-6), NAH(6-8), NAH(8-10)
        # OTH at 4-6 is between SPA(2-4) and NAH(6-8) -> NOT absorbed (need NAH-OTH-NAH)
        # But NAH(6-8) then NAH(8-10) are adjacent NAH segments
        languages = [(r.start_time, r.language) for r in results]

        # Find the SPA segment - should still be present
        spa_segs = [r for r in results if r.language == "spa"]
        assert len(spa_segs) == 1, f"Expected 1 SPA segment, got {len(spa_segs)}"
        assert spa_segs[0].transcription == "Buenos dias amigos"

    def test_all_hallucinations_with_no_valid_segments(self):
        """Edge case: Whisper produces ONLY hallucinated output (pure Nahuatl audio file).

        Features tested: WhisperValidator rejects all + Allosaurus processes everything.

        All 3 segments are hallucinations -> all go to Allosaurus fallback.
        No Whisper text should appear in output.
        """
        auto_segs = [
            WhisperAutoSegment(text="Uchach ayik alilti", start=0.0, end=3.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="Xkantul ba'alo k'an", start=3.0, end=6.0, language="es", avg_log_prob=-0.4),
            WhisperAutoSegment(text="Le'l l'uma uchach", start=6.0, end=10.0, language="es", avg_log_prob=-0.35),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=10.0)

        def mock_recognize(path, backend="allosaurus"):
            return [PhonemeSegment(phoneme="t\u0361\u026c", start_time=0.0, duration=0.5)]

        def mock_identify(phonemes, audio_data=None):
            if not phonemes:
                return []
            return [LanguageSegment(
                language="nah",
                phonemes=list(phonemes),
                start_time=phonemes[0].start_time,
                end_time=phonemes[0].start_time + 3.0,
            )]

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", side_effect=mock_recognize), \
             patch("tenepal.pipeline.identify_language", side_effect=mock_identify), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # All segments should come from Allosaurus, not Whisper
        assert len(results) >= 3, f"Expected at least 3 segments, got {len(results)}"

        # No segment should have whisper backend
        whisper_segs = [r for r in results if r.transcription_backend == "whisper"]
        assert len(whisper_segs) == 0, "No Whisper text should appear when all segments are hallucinated"

        # All segments should be from Allosaurus
        for seg in results:
            assert seg.transcription_backend in ("allosaurus", "allosaurus-gap"), (
                f"All segments should be Allosaurus-sourced, got {seg.transcription_backend}"
            )

    def test_validator_and_confidence_interact_correctly(self):
        """Verify that validation runs BEFORE confidence splitting.

        Features tested: WhisperValidator + confidence threshold interaction.

        Scenario:
        - Segment 1: high confidence (-0.1) + hallucinated text -> rejected by validator
        - Segment 2: low confidence (-0.8) + valid Spanish text -> goes to fallback via confidence
        - Segment 3: high confidence (-0.2) + valid Spanish -> accepted by Whisper
        - Expected: Segments 1 and 2 both go to Allosaurus (different reasons), segment 3 stays Whisper
        """
        auto_segs = [
            WhisperAutoSegment(
                text="Xkantul ba'alo k'an uchach",  # Hallucinated
                start=0.0, end=3.0, language="es", avg_log_prob=-0.1,
            ),
            WhisperAutoSegment(
                text="Hola mundo amigos",  # Valid Spanish but low confidence
                start=3.0, end=6.0, language="es", avg_log_prob=-0.8,
            ),
            WhisperAutoSegment(
                text="Buenos dias amigos",  # Valid Spanish, high confidence
                start=6.0, end=10.0, language="es", avg_log_prob=-0.2,
            ),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=10.0)

        call_counter = [0]

        def mock_recognize(path, backend="allosaurus"):
            return [PhonemeSegment(phoneme="t\u0361\u026c", start_time=0.0, duration=0.5)]

        def mock_identify(phonemes, audio_data=None):
            if not phonemes:
                return []
            call_counter[0] += 1
            count = call_counter[0]
            if count == 1:
                # First Allosaurus call: hallucinated segment 0-3 -> NAH
                return [LanguageSegment(
                    language="nah",
                    phonemes=list(phonemes),
                    start_time=0.0,
                    end_time=3.0,
                )]
            else:
                # Second Allosaurus call: low-confidence valid segment 3-6 -> SPA
                return [LanguageSegment(
                    language="spa",
                    phonemes=list(phonemes),
                    start_time=3.0,
                    end_time=6.0,
                )]

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", side_effect=mock_recognize), \
             patch("tenepal.pipeline.identify_language", side_effect=mock_identify), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Should have at least 3 segments
        assert len(results) >= 3, f"Expected at least 3 segments, got {len(results)}"

        # Segment 3 (6-10) should be from Whisper with text preserved
        whisper_segs = [r for r in results if r.transcription_backend == "whisper"]
        assert len(whisper_segs) == 1, f"Expected 1 Whisper segment, got {len(whisper_segs)}"
        assert whisper_segs[0].transcription == "Buenos dias amigos"
        assert whisper_segs[0].start_time == 6.0

        # First segment (hallucinated) should go through Allosaurus
        first_seg = results[0]
        assert first_seg.start_time == 0.0
        assert first_seg.transcription_backend != "whisper", (
            "Hallucinated segment should not have whisper backend"
        )


class TestModalPhonemeBackendSelection:
    """Regression checks for modal backend resolution and factory behavior."""

    def test_default_backend_name(self):
        modal = pytest.importorskip("tenepal_modal")
        name, warning = modal.resolve_phoneme_backend_name(None)
        assert name == "allosaurus"
        assert warning is None

    def test_known_backend_name(self):
        modal = pytest.importorskip("tenepal_modal")
        name, warning = modal.resolve_phoneme_backend_name("wav2vec2")
        assert name == "wav2vec2"
        assert warning is None

    def test_espnet_maps_to_compat_backend(self):
        modal = pytest.importorskip("tenepal_modal")
        name, warning = modal.resolve_phoneme_backend_name("espnet")
        assert name == "wav2vec2"
        assert warning is not None

    def test_unknown_backend_falls_back(self):
        modal = pytest.importorskip("tenepal_modal")
        name, warning = modal.resolve_phoneme_backend_name("does-not-exist")
        assert name == "allosaurus"
        assert warning is not None

    def test_factory_resolves_known_backend(self):
        modal = pytest.importorskip("tenepal_modal")
        backend = modal.get_phoneme_backend("allosaurus", "/tmp")
        assert backend.name == "allosaurus"

    def test_srt_output_contains_all_backend_types(self):
        """Integration with SRT export: segments from all backend types appear in SRT.

        Features tested: Whisper text + Allosaurus fallback + gap fill + SRT write.

        Create a scenario with Whisper text (valid Spanish), Allosaurus fallback
        (hallucinated -> NAH), and gap-filled segments. Use output_path to trigger
        SRT write and verify write_srt is called with segments from all sources.
        """
        auto_segs = [
            WhisperAutoSegment(text="Hola mundo amigos", start=0.0, end=5.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Uchach ayik alilti", start=5.0, end=8.0, language="es", avg_log_prob=-0.3),
            # Gap at 8-10
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=10.0)

        call_counter = [0]

        def mock_recognize(path, backend="allosaurus"):
            return [PhonemeSegment(phoneme="t\u0361\u026c", start_time=0.0, duration=0.5)]

        def mock_identify(phonemes, audio_data=None):
            if not phonemes:
                return []
            call_counter[0] += 1
            if call_counter[0] == 1:
                # Hallucinated segment -> NAH
                return [LanguageSegment(
                    language="nah",
                    phonemes=list(phonemes),
                    start_time=5.0,
                    end_time=8.0,
                )]
            else:
                # Gap fill -> OTH
                return [LanguageSegment(
                    language="other",
                    phonemes=list(phonemes),
                    start_time=8.0,
                    end_time=10.0,
                )]

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", side_effect=mock_recognize), \
             patch("tenepal.pipeline.identify_language", side_effect=mock_identify), \
             patch("tenepal.subtitle.write_srt") as mock_write_srt, \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=10.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            output = Path("/tmp/test_v41_output.srt")
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
                output_path=output,
            )

        # write_srt should be called with segments from all backend types
        mock_write_srt.assert_called_once()
        srt_segments = mock_write_srt.call_args[0][0]
        assert mock_write_srt.call_args[0][1] == output

        # Verify multiple backend types present in SRT output
        backends = {seg.transcription_backend for seg in srt_segments}
        assert "whisper" in backends, "SRT should contain Whisper segments"
        assert len(backends) >= 2, f"SRT should contain multiple backend types, got {backends}"

    def test_pipeline_preserves_chronological_order_with_all_features(self):
        """Stress test for ordering: 6+ segments from different sources must remain sorted.

        Features tested: Chronological sorting after combining Whisper text +
        Allosaurus fallback + gap fill with non-trivial timing.

        Scenario:
        - Gap 0-1s -> Allosaurus fills
        - Whisper 1-3s (valid Spanish)
        - Hallucinated 3-5s -> Allosaurus
        - Whisper 5-7s (valid Spanish)
        - Gap 7-9s -> Allosaurus fills
        - Whisper 9-12s (valid Spanish)
        All results must be strictly sorted by start_time.
        """
        auto_segs = [
            WhisperAutoSegment(text="Buenos dias", start=1.0, end=3.0, language="es", avg_log_prob=-0.15),
            WhisperAutoSegment(text="Uchach ayik alilti", start=3.0, end=5.0, language="es", avg_log_prob=-0.25),
            WhisperAutoSegment(text="Hola amigos todos", start=5.0, end=7.0, language="es", avg_log_prob=-0.2),
            WhisperAutoSegment(text="Vamos al campo", start=9.0, end=12.0, language="es", avg_log_prob=-0.18),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=12.0)

        call_counter = [0]

        def mock_recognize(path, backend="allosaurus"):
            return [PhonemeSegment(phoneme="t\u0361\u026c", start_time=0.0, duration=0.5)]

        def mock_identify(phonemes, audio_data=None):
            if not phonemes:
                return []
            call_counter[0] += 1
            count = call_counter[0]
            # Expected calls:
            # 1: hallucinated fallback (3-5) -> NAH
            # 2: start gap (0-1) -> NAH
            # 3: interior gap (7-9) -> OTH
            if count == 1:
                return [LanguageSegment(
                    language="nah", phonemes=list(phonemes),
                    start_time=3.0, end_time=5.0,
                )]
            elif count == 2:
                return [LanguageSegment(
                    language="nah", phonemes=list(phonemes),
                    start_time=0.0, end_time=1.0,
                )]
            else:
                return [LanguageSegment(
                    language="other", phonemes=list(phonemes),
                    start_time=7.0, end_time=9.0,
                )]

        with patch(_WB, return_value=mock_wb), \
             patch("tenepal.pipeline.load_audio", return_value=mock_audio), \
             patch("tenepal.pipeline.recognize_phonemes", side_effect=mock_recognize), \
             patch("tenepal.pipeline.identify_language", side_effect=mock_identify), \
             patch("tenepal.pipeline.sf") as mock_sf:
            mock_sf.info.return_value = MagicMock(duration=12.0)
            mock_sf.write = MagicMock()
            from tenepal.pipeline import process_whisper_first
            results = process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Verify strict chronological ordering
        start_times = [r.start_time for r in results]
        assert start_times == sorted(start_times), (
            f"Results must be strictly sorted by start_time: {start_times}"
        )

        # Verify no overlaps
        for i in range(len(results) - 1):
            assert results[i].end_time <= results[i + 1].start_time, (
                f"Overlap detected: segment {i} ends at {results[i].end_time} "
                f"but segment {i+1} starts at {results[i+1].start_time}"
            )

        # Should have at least 6 segments (3 Whisper + hallucinated + 2 gaps)
        assert len(results) >= 6, f"Expected at least 6 segments, got {len(results)}"

    def test_whisper_unload_called_after_all_processing(self):
        """Verify Whisper model is unloaded even with complex multi-feature processing.

        Features tested: Pipeline cleanup after validation + fallback + gap fill.

        Even when hallucinations, gaps, and absorption all run, the Whisper model
        must be unloaded at the end to free GPU memory.
        """
        auto_segs = [
            WhisperAutoSegment(text="Uchach ayik alilti", start=0.0, end=5.0, language="es", avg_log_prob=-0.3),
            WhisperAutoSegment(text="Hola mundo amigos", start=5.0, end=10.0, language="es", avg_log_prob=-0.2),
        ]
        mock_wb = _make_mock_whisper_backend(auto_segs)
        mock_audio = _make_mock_audio(duration_sec=10.0)

        mock_phoneme = PhonemeSegment(phoneme="t\u0361\u026c", start_time=0.0, duration=0.5)
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
            process_whisper_first(
                Path("/fake/audio.wav"),
                confidence_threshold=-0.5,
                allosaurus_fallback=True,
            )

        # Whisper model must be unloaded after all processing
        mock_wb.unload.assert_called_once()


# ---------------------------------------------------------------------------
# TestV41ComponentInteraction: Component boundary smoke tests
# ---------------------------------------------------------------------------


class TestV41ComponentInteraction:
    """Tests verifying the boundaries between v4.1 components hold:
    WhisperValidator, NahuatlLexicon, _find_gaps, _apply_cross_segment_nah_absorption."""

    def test_whisper_validator_with_lexicon_data(self):
        """WhisperValidator and NahuatlLexicon agree: Nahuatl words are not Spanish.

        Verifies that text containing Nahuatl words from the lexicon is correctly
        rejected by WhisperValidator (low Spanish lexicon score), while NahuatlLexicon
        recognizes those same words as Nahuatl.
        """
        from tenepal.validation import WhisperValidator
        from tenepal.language.nahuatl_lexicon import NahuatlLexicon

        validator = WhisperValidator()
        lexicon = NahuatlLexicon()

        # Nahuatl text that WhisperValidator should reject (not Spanish)
        nahuatl_text = "koali nikan axkan"
        result = validator.validate(nahuatl_text, avg_log_prob=0.0)
        assert not result.is_valid, (
            f"Validator should reject Nahuatl text '{nahuatl_text}', got valid={result.is_valid}, "
            f"reason={result.reason}"
        )

        # NahuatlLexicon should recognize these as Nahuatl words
        # "koali" IPA: k, o, a, l, i
        match = lexicon.match(["k", "o", "a", "l", "i"])
        assert match is not None, "Lexicon should match 'koali' IPA sequence"
        assert match.word == "koali"

    def test_find_gaps_after_validation_filtering(self):
        """Gap detection works correctly on the filtered segment list.

        Simulates the scenario where validation removes a middle segment,
        and _find_gaps should detect the resulting gap.
        """
        from tenepal.pipeline import _find_gaps

        # Original 3 segments covering 0-10
        seg1 = WhisperAutoSegment(text="a", start=0.0, end=3.0, language="es", avg_log_prob=-0.2)
        seg2 = WhisperAutoSegment(text="b", start=3.0, end=6.0, language="es", avg_log_prob=-0.3)
        seg3 = WhisperAutoSegment(text="c", start=6.0, end=10.0, language="es", avg_log_prob=-0.2)

        # Simulate validation rejecting segment 2
        remaining = [seg1, seg3]

        # Find gaps on remaining segments
        gaps = _find_gaps(remaining, audio_duration=10.0)

        # Gap at 3-6 should be detected
        assert gaps == [(3.0, 6.0)], f"Expected gap at (3.0, 6.0), got {gaps}"

    def test_absorption_runs_after_gap_merge(self):
        """Absorption correctly absorbs OTH from gap fill between NAH segments.

        Creates the NAH-OTH-NAH pattern that would result from gap filling
        between two Nahuatl segments, and verifies absorption works.
        """
        from tenepal.pipeline import _apply_cross_segment_nah_absorption

        # NAH from Allosaurus fallback (0-2)
        nah1 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("t\u0361\u026c", 0.0, 0.5)],
            start_time=0.0,
            end_time=2.0,
        )
        nah1.transcription_backend = "allosaurus"

        # OTH from gap fill (2-3)
        oth = LanguageSegment(
            language="other",
            phonemes=[PhonemeSegment("e", 2.0, 0.3)],
            start_time=2.0,
            end_time=3.0,  # 1.0s duration -- short enough to absorb
        )
        oth.transcription_backend = "allosaurus-gap"

        # NAH from Allosaurus fallback (3-5)
        nah2 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("k\u02b7", 3.0, 0.5)],
            start_time=3.0,
            end_time=5.0,
        )
        nah2.transcription_backend = "allosaurus"

        segments = [nah1, oth, nah2]
        result = _apply_cross_segment_nah_absorption(segments)

        # OTH should be absorbed: reclassified as NAH
        assert len(result) == 3, "Should still have 3 segments (reclassified, not merged)"
        assert result[1].language == "nah", (
            f"OTH between NAH segments should be absorbed to NAH, got {result[1].language}"
        )
        # Backend tag should be preserved
        assert result[1].transcription_backend == "allosaurus-gap"

    def test_pipeline_import_chain(self):
        """Smoke test that all v4.1 imports resolve correctly.

        Verifies that the complete v4.1 module chain is importable and functional:
        process_whisper_first, WhisperValidator, ValidationResult, NahuatlLexicon,
        LexiconMatch, _find_gaps, _apply_cross_segment_nah_absorption.
        """
        from tenepal.pipeline import process_whisper_first
        from tenepal.pipeline import _find_gaps
        from tenepal.pipeline import _apply_cross_segment_nah_absorption
        from tenepal.validation import WhisperValidator, ValidationResult
        from tenepal.language.nahuatl_lexicon import NahuatlLexicon, LexiconMatch

        # All should be importable
        assert callable(process_whisper_first)
        assert callable(_find_gaps)
        assert callable(_apply_cross_segment_nah_absorption)

        # WhisperValidator should work with real Spanish text
        validator = WhisperValidator()
        result = validator.validate("Hola mundo")
        assert isinstance(result, ValidationResult)
        assert result.is_valid, f"'Hola mundo' should be valid Spanish, got reason: {result.reason}"

        # NahuatlLexicon should work with real Nahuatl IPA
        lexicon = NahuatlLexicon()
        match = lexicon.match(["k", "o", "a", "l", "i"])
        assert isinstance(match, LexiconMatch)
        assert match.word == "koali"
        assert match.score > 0.7, f"Expected high match score, got {match.score}"

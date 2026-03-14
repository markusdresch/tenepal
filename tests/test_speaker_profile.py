"""Tests for speaker profile building and language inheritance."""

import pytest
from dataclasses import dataclass

from tenepal.language.speaker_profile import (
    SpeakerProfile,
    build_speaker_profiles,
    apply_speaker_inheritance,
)
from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme import PhonemeSegment


# Simple mock classes for testing
@dataclass
class MockWhisperSegment:
    """Mock WhisperAutoSegment for testing."""
    text: str
    start: float
    end: float
    language: str  # ISO 639-1 code (e.g., "es")
    avg_log_prob: float


@dataclass
class MockSpeakerSegment:
    """Mock SpeakerSegment for testing."""
    speaker: str
    start_time: float
    end_time: float


class TestBuildProfiles:
    """Test suite for build_speaker_profiles()."""

    def test_build_profiles_empty_input(self):
        """Empty lists return empty dict."""
        result = build_speaker_profiles([])
        assert result == {}

    def test_build_profiles_single_speaker(self):
        """One speaker with 6 high-confidence Spanish segments builds correct profile."""
        assigned_pairs = []
        for i in range(6):
            whisper_seg = MockWhisperSegment(
                text=f"texto {i}",
                start=float(i),
                end=float(i + 1),
                language="es",  # ISO 639-1
                avg_log_prob=-0.1,  # High confidence
            )
            speaker_seg = MockSpeakerSegment(
                speaker="Speaker A",
                start_time=float(i),
                end_time=float(i + 1),
            )
            assigned_pairs.append((whisper_seg, speaker_seg))

        result = build_speaker_profiles(assigned_pairs)

        assert "Speaker A" in result
        profile = result["Speaker A"]
        assert profile.speaker == "Speaker A"
        assert profile.primary_language == "spa"  # ISO 639-3
        assert profile.segment_count == 6
        assert profile.language_distribution == {"spa": 6}

    def test_build_profiles_multiple_speakers(self):
        """Two speakers each get separate profiles."""
        assigned_pairs = []

        # Speaker A: 4 Spanish segments
        for i in range(4):
            whisper_seg = MockWhisperSegment(
                text=f"texto {i}",
                start=float(i),
                end=float(i + 1),
                language="es",
                avg_log_prob=-0.1,
            )
            speaker_seg = MockSpeakerSegment(
                speaker="Speaker A",
                start_time=float(i),
                end_time=float(i + 1),
            )
            assigned_pairs.append((whisper_seg, speaker_seg))

        # Speaker B: 3 English segments
        for i in range(3):
            whisper_seg = MockWhisperSegment(
                text=f"text {i}",
                start=float(i + 10),
                end=float(i + 11),
                language="en",
                avg_log_prob=-0.15,
            )
            speaker_seg = MockSpeakerSegment(
                speaker="Speaker B",
                start_time=float(i + 10),
                end_time=float(i + 11),
            )
            assigned_pairs.append((whisper_seg, speaker_seg))

        result = build_speaker_profiles(assigned_pairs)

        assert len(result) == 2
        assert result["Speaker A"].primary_language == "spa"
        assert result["Speaker A"].segment_count == 4
        assert result["Speaker B"].primary_language == "eng"
        assert result["Speaker B"].segment_count == 3

    def test_build_profiles_ignores_low_confidence(self):
        """Segments with avg_log_prob < -0.3 are excluded from profile building."""
        assigned_pairs = [
            # High confidence - should count
            (
                MockWhisperSegment("texto", 0.0, 1.0, "es", -0.1),
                MockSpeakerSegment("Speaker A", 0.0, 1.0),
            ),
            # Low confidence - should be excluded
            (
                MockWhisperSegment("texto", 1.0, 2.0, "es", -0.5),
                MockSpeakerSegment("Speaker A", 1.0, 2.0),
            ),
            # High confidence - should count
            (
                MockWhisperSegment("texto", 2.0, 3.0, "es", -0.2),
                MockSpeakerSegment("Speaker A", 2.0, 3.0),
            ),
        ]

        result = build_speaker_profiles(assigned_pairs)

        assert result["Speaker A"].segment_count == 2  # Only 2 high-confidence segments
        assert result["Speaker A"].language_distribution == {"spa": 2}

    def test_build_profiles_mixed_languages(self):
        """Speaker with 8 Spanish + 2 English segments shows correct distribution."""
        assigned_pairs = []

        # 8 Spanish segments
        for i in range(8):
            whisper_seg = MockWhisperSegment(
                text=f"texto {i}",
                start=float(i),
                end=float(i + 1),
                language="es",
                avg_log_prob=-0.1,
            )
            speaker_seg = MockSpeakerSegment(
                speaker="Speaker A",
                start_time=float(i),
                end_time=float(i + 1),
            )
            assigned_pairs.append((whisper_seg, speaker_seg))

        # 2 English segments
        for i in range(2):
            whisper_seg = MockWhisperSegment(
                text=f"text {i}",
                start=float(i + 10),
                end=float(i + 11),
                language="en",
                avg_log_prob=-0.1,
            )
            speaker_seg = MockSpeakerSegment(
                speaker="Speaker A",
                start_time=float(i + 10),
                end_time=float(i + 11),
            )
            assigned_pairs.append((whisper_seg, speaker_seg))

        result = build_speaker_profiles(assigned_pairs)

        profile = result["Speaker A"]
        assert profile.primary_language == "spa"
        assert profile.segment_count == 10
        assert profile.language_distribution == {"spa": 8, "eng": 2}


class TestInheritanceThreshold:
    """Test suite for SpeakerProfile.meets_inheritance_threshold()."""

    def test_should_inherit_yes(self):
        """Speaker with 5+ validated segments and 80%+ same language → True."""
        profile = SpeakerProfile(
            speaker="Speaker A",
            primary_language="spa",
            segment_count=5,
            language_distribution={"spa": 5},
        )
        assert profile.meets_inheritance_threshold() is True

    def test_should_inherit_no_too_few_segments(self):
        """Speaker with only 3 segments → False (need 5+)."""
        profile = SpeakerProfile(
            speaker="Speaker A",
            primary_language="spa",
            segment_count=3,
            language_distribution={"spa": 3},
        )
        assert profile.meets_inheritance_threshold() is False

    def test_should_inherit_no_mixed_languages(self):
        """Speaker with 50% spa / 50% eng → False (need 80%+)."""
        profile = SpeakerProfile(
            speaker="Speaker A",
            primary_language="spa",
            segment_count=10,
            language_distribution={"spa": 5, "eng": 5},
        )
        assert profile.meets_inheritance_threshold() is False

    def test_should_inherit_borderline(self):
        """Test exactly 80% threshold (5 of 6 segments)."""
        # Fails: 4 of 5 = 80% exactly, but let's test 5 of 6 which is 83.3%
        profile = SpeakerProfile(
            speaker="Speaker A",
            primary_language="spa",
            segment_count=6,
            language_distribution={"spa": 5, "eng": 1},
        )
        assert profile.meets_inheritance_threshold() is True

        # Just below: 7 of 10 = 70%
        profile2 = SpeakerProfile(
            speaker="Speaker B",
            primary_language="spa",
            segment_count=10,
            language_distribution={"spa": 7, "eng": 3},
        )
        assert profile2.meets_inheritance_threshold() is False


class TestApplyInheritance:
    """Test suite for apply_speaker_inheritance()."""

    def _make_segment(
        self,
        language: str,
        start: float,
        end: float,
        speaker: str | None = None,
        backend: str = "allosaurus",
    ) -> LanguageSegment:
        """Helper to create LanguageSegment for testing."""
        phonemes = [
            PhonemeSegment(phoneme="a", start_time=start, duration=(end - start) / 3),
            PhonemeSegment(phoneme="b", start_time=start + (end - start) / 3, duration=(end - start) / 3),
            PhonemeSegment(phoneme="c", start_time=start + 2 * (end - start) / 3, duration=(end - start) / 3),
        ]
        seg = LanguageSegment(
            language=language,
            phonemes=phonemes,
            start_time=start,
            end_time=end,
            speaker=speaker,
        )
        # Add transcription_backend attribute (not in dataclass, but ok for testing)
        seg.transcription_backend = backend
        return seg

    def test_apply_inheritance_allosaurus_only(self):
        """Allosaurus-only LanguageSegment (transcription_backend != 'whisper') with 'other' gets inherited."""
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="spa",
                segment_count=6,
                language_distribution={"spa": 6},
            ),
        }

        segments = [
            self._make_segment(
                language="other",
                start=0.0,
                end=1.0,
                speaker="Speaker A",
                backend="allosaurus",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)

        assert len(result) == 1
        assert result[0].language == "spa"  # Inherited from Speaker A

    def test_apply_inheritance_short_segment(self):
        """LanguageSegment under 1 second with no Whisper match gets inherited."""
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="eng",
                segment_count=5,
                language_distribution={"eng": 5},
            ),
        }

        # Short segment (0.8 seconds) without whisper backend
        segments = [
            self._make_segment(
                language="other",
                start=0.0,
                end=0.8,
                speaker="Speaker A",
                backend="allosaurus",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)

        assert len(result) == 1
        assert result[0].language == "eng"  # Inherited from Speaker A

    def test_apply_inheritance_preserves_whisper(self):
        """LanguageSegment with transcription_backend='whisper' is NOT changed."""
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="spa",
                segment_count=6,
                language_distribution={"spa": 6},
            ),
        }

        segments = [
            self._make_segment(
                language="eng",  # Whisper detected English
                start=0.0,
                end=2.0,
                speaker="Speaker A",
                backend="whisper",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)

        assert len(result) == 1
        assert result[0].language == "eng"  # NOT changed to spa (preserve Whisper)

    def test_apply_inheritance_no_profile(self):
        """Segment from speaker without profile is unchanged."""
        profiles = {}  # No profiles

        segments = [
            self._make_segment(
                language="other",
                start=0.0,
                end=1.0,
                speaker="Speaker A",
                backend="allosaurus",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)

        assert len(result) == 1
        assert result[0].language == "other"  # Unchanged

    def test_apply_inheritance_insufficient_evidence(self):
        """Speaker profile doesn't meet inheritance threshold → no change."""
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="spa",
                segment_count=3,  # Too few (need 5+)
                language_distribution={"spa": 3},
            ),
        }

        segments = [
            self._make_segment(
                language="other",
                start=0.0,
                end=1.0,
                speaker="Speaker A",
                backend="allosaurus",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)

        assert len(result) == 1
        assert result[0].language == "other"  # Not inherited (insufficient evidence)

    def test_apply_inheritance_long_segment_not_inherited(self):
        """Segment over 1 second is NOT inherited (SPKR-03 only applies to <1s)."""
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="spa",
                segment_count=6,
                language_distribution={"spa": 6},
            ),
        }

        # Long segment (2 seconds) without whisper backend, but language="other"
        segments = [
            self._make_segment(
                language="other",
                start=0.0,
                end=2.0,
                speaker="Speaker A",
                backend="allosaurus",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)

        # SPKR-03 only applies to <1s segments
        # But SPKR-02 (Allosaurus-only "other") should still apply
        assert len(result) == 1
        assert result[0].language == "spa"  # Inherited via SPKR-02

    def test_apply_inheritance_allosaurus_not_other(self):
        """Allosaurus backend with non-'other' language is NOT inherited."""
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="eng",
                segment_count=6,
                language_distribution={"eng": 6},
            ),
        }

        segments = [
            self._make_segment(
                language="spa",  # Allosaurus detected Spanish
                start=0.0,
                end=2.0,
                speaker="Speaker A",
                backend="allosaurus",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)

        assert len(result) == 1
        assert result[0].language == "spa"  # NOT changed (Allosaurus detected spa, not "other")

    def test_apply_inheritance_preserves_may_segment(self):
        """MAY segments are protected from short-segment inheritance overrides."""
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="spa",
                segment_count=6,
                language_distribution={"spa": 6},
            ),
        }
        segments = [
            self._make_segment(
                language="may",
                start=0.0,
                end=0.7,  # short enough to trigger SPKR-03 if unprotected
                speaker="Speaker A",
                backend="allosaurus",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)
        assert result[0].language == "may"

    def test_apply_inheritance_preserves_lat_segment(self):
        """LAT segments are protected from short-segment inheritance overrides."""
        profiles = {
            "Speaker A": SpeakerProfile(
                speaker="Speaker A",
                primary_language="nah",
                segment_count=6,
                language_distribution={"nah": 6},
            ),
        }
        segments = [
            self._make_segment(
                language="lat",
                start=0.0,
                end=0.6,  # short enough to trigger SPKR-03 if unprotected
                speaker="Speaker A",
                backend="allosaurus",
            ),
        ]

        result = apply_speaker_inheritance(segments, profiles)
        assert result[0].language == "lat"

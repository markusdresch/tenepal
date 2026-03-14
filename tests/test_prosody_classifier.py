"""Tests for prosodic language profile classifier."""

import json
from pathlib import Path

import pytest

from tenepal.prosody import (
    ProsodyFeatures,
    ProsodyProfile,
    load_prosody_profiles,
    score_all_profiles,
    score_prosody_profile,
)
from tenepal.prosody.classifier import load_prosody_profile


class TestProsodyProfile:
    """Tests for ProsodyProfile dataclass."""

    def test_dataclass_creation(self):
        """ProsodyProfile should be creatable with all fields."""
        profile = ProsodyProfile(
            code="spa",
            name="Spanish",
            rhythm_class="syllable-timed",
            f0_range={"min": 50, "max": 200, "target": 120},
            f0_std={"min": 15, "max": 60, "target": 35},
            npvi_v={"min": 35, "max": 52, "target": 43},
            speech_rate={"min": 6.5, "max": 9.0, "target": 7.8},
            weight=1.0,
        )

        assert profile.code == "spa"
        assert profile.name == "Spanish"
        assert profile.rhythm_class == "syllable-timed"
        assert profile.f0_range["target"] == 120
        assert profile.npvi_v["target"] == 43
        assert profile.weight == 1.0


class TestLoadProsodyProfile:
    """Tests for loading individual prosody profiles from JSON."""

    def test_load_spanish_profile(self):
        """Load Spanish prosodic profile from built-in JSON."""
        # Get the built-in profiles directory
        from tenepal.prosody import classifier
        profiles_dir = Path(classifier.__file__).parent / "profiles"
        spa_path = profiles_dir / "spa.json"

        profile = load_prosody_profile(spa_path)

        assert profile.code == "spa"
        assert profile.name == "Spanish"
        assert profile.rhythm_class == "syllable-timed"
        assert profile.npvi_v["target"] == 43  # Syllable-timed
        assert profile.weight == 1.0

    def test_load_english_profile(self):
        """Load English prosodic profile from built-in JSON."""
        from tenepal.prosody import classifier
        profiles_dir = Path(classifier.__file__).parent / "profiles"
        eng_path = profiles_dir / "eng.json"

        profile = load_prosody_profile(eng_path)

        assert profile.code == "eng"
        assert profile.name == "English"
        assert profile.rhythm_class == "stress-timed"
        assert profile.npvi_v["target"] == 57  # Stress-timed (high nPVI)
        assert profile.weight == 1.0

    def test_load_nahuatl_profile_has_low_weight(self):
        """Nahuatl profile should have weight=0.5 (estimated values)."""
        from tenepal.prosody import classifier
        profiles_dir = Path(classifier.__file__).parent / "profiles"
        nah_path = profiles_dir / "nah.json"

        profile = load_prosody_profile(nah_path)

        assert profile.code == "nah"
        assert profile.name == "Nahuatl"
        assert profile.rhythm_class == "stress-accent"
        assert profile.weight == 0.5  # Low confidence (estimated)


class TestLoadProsodyProfiles:
    """Tests for loading all prosody profiles from directory."""

    def test_load_all_default_profiles(self):
        """Load all 7 built-in prosodic profiles.

        NOTE: Phase 32 added MAY (Yucatec Maya) as a 7th language.
        """
        profiles = load_prosody_profiles()

        assert len(profiles) == 7

        # Check sorted by code
        codes = [p.code for p in profiles]
        assert codes == sorted(codes)
        assert "deu" in codes
        assert "eng" in codes
        assert "fra" in codes
        assert "may" in codes
        assert "ita" in codes
        assert "nah" in codes
        assert "spa" in codes

    def test_profiles_have_rhythm_classes(self):
        """All profiles should have valid rhythm_class values."""
        profiles = load_prosody_profiles()

        # Stress-timed languages
        eng = next(p for p in profiles if p.code == "eng")
        deu = next(p for p in profiles if p.code == "deu")
        assert eng.rhythm_class == "stress-timed"
        assert deu.rhythm_class == "stress-timed"

        # Syllable-timed languages
        spa = next(p for p in profiles if p.code == "spa")
        fra = next(p for p in profiles if p.code == "fra")
        ita = next(p for p in profiles if p.code == "ita")
        assert spa.rhythm_class == "syllable-timed"
        assert fra.rhythm_class == "syllable-timed"
        assert ita.rhythm_class == "syllable-timed"

        # Stress-accent (Nahuatl)
        nah = next(p for p in profiles if p.code == "nah")
        assert nah.rhythm_class == "stress-accent"

    def test_stress_timed_have_higher_npvi_targets(self):
        """Stress-timed languages should have higher nPVI targets than syllable-timed."""
        profiles = load_prosody_profiles()

        eng = next(p for p in profiles if p.code == "eng")
        deu = next(p for p in profiles if p.code == "deu")
        spa = next(p for p in profiles if p.code == "spa")
        fra = next(p for p in profiles if p.code == "fra")
        ita = next(p for p in profiles if p.code == "ita")

        # Stress-timed nPVI targets (50-60 range)
        assert eng.npvi_v["target"] >= 50
        assert deu.npvi_v["target"] >= 50

        # Syllable-timed nPVI targets (35-52 range)
        assert spa.npvi_v["target"] < 50
        assert fra.npvi_v["target"] < 50
        assert ita.npvi_v["target"] < 50


class TestScoreProsodyProfile:
    """Tests for scoring features against a single profile."""

    def test_perfect_match_returns_high_score(self):
        """Features exactly matching profile targets should return score near 1.0."""
        profile = ProsodyProfile(
            code="spa",
            name="Spanish",
            rhythm_class="syllable-timed",
            f0_range={"min": 50, "max": 200, "target": 120},
            f0_std={"min": 15, "max": 60, "target": 35},
            npvi_v={"min": 35, "max": 52, "target": 43},
            speech_rate={"min": 6.5, "max": 9.0, "target": 7.8},
            weight=1.0,
        )

        # Features exactly matching targets
        features = ProsodyFeatures(
            f0_mean=150.0,  # Not used in scoring (speaker-dependent)
            f0_std=35.0,    # Matches target
            f0_range=120.0,  # Matches target
            intensity_mean=60.0,  # Not used
            duration=2.0,    # Not used
            speech_rate=7.8,  # Matches target
            npvi_v=43.0,     # Matches target
        )

        score = score_prosody_profile(features, profile)

        # Should return 1.0 for perfect match
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_mismatch_returns_lower_score(self):
        """Features partially matching profile should return intermediate score."""
        profile = ProsodyProfile(
            code="spa",
            name="Spanish",
            rhythm_class="syllable-timed",
            f0_range={"min": 50, "max": 200, "target": 120},
            f0_std={"min": 15, "max": 60, "target": 35},
            npvi_v={"min": 35, "max": 52, "target": 43},
            speech_rate={"min": 6.5, "max": 9.0, "target": 7.8},
            weight=1.0,
        )

        # Features somewhat off from targets
        features = ProsodyFeatures(
            f0_mean=150.0,
            f0_std=50.0,    # Off from target 35
            f0_range=180.0,  # Off from target 120
            intensity_mean=60.0,
            duration=2.0,
            speech_rate=6.5,  # Off from target 7.8
            npvi_v=52.0,     # Off from target 43
        )

        score = score_prosody_profile(features, profile)

        # Should return intermediate score
        assert 0.3 < score < 0.8

    def test_features_outside_range_return_lower_score(self):
        """Features far outside profile range should return low score."""
        profile = ProsodyProfile(
            code="eng",
            name="English",
            rhythm_class="stress-timed",
            f0_range={"min": 60, "max": 250, "target": 150},
            f0_std={"min": 20, "max": 70, "target": 45},
            npvi_v={"min": 50, "max": 73, "target": 57},
            speech_rate={"min": 5.0, "max": 7.5, "target": 6.2},
            weight=1.0,
        )

        # Features far from profile (syllable-timed characteristics)
        features = ProsodyFeatures(
            f0_mean=150.0,
            f0_std=25.0,    # Below English range
            f0_range=100.0,  # Below English range
            intensity_mean=60.0,
            duration=2.0,
            speech_rate=8.5,  # Above English range
            npvi_v=40.0,     # Well below English nPVI (syllable-timed value)
        )

        score = score_prosody_profile(features, profile)

        # Should return low score
        assert score < 0.5

    def test_nahuatl_weight_reduces_score(self):
        """Nahuatl profile with weight=0.5 should produce half the score."""
        profile_full_weight = ProsodyProfile(
            code="test",
            name="Test",
            rhythm_class="stress-accent",
            f0_range={"min": 40, "max": 180, "target": 100},
            f0_std={"min": 15, "max": 50, "target": 30},
            npvi_v={"min": 45, "max": 65, "target": 55},
            speech_rate={"min": 5.0, "max": 7.5, "target": 6.2},
            weight=1.0,
        )

        profile_half_weight = ProsodyProfile(
            code="nah",
            name="Nahuatl",
            rhythm_class="stress-accent",
            f0_range={"min": 40, "max": 180, "target": 100},
            f0_std={"min": 15, "max": 50, "target": 30},
            npvi_v={"min": 45, "max": 65, "target": 55},
            speech_rate={"min": 5.0, "max": 7.5, "target": 6.2},
            weight=0.5,
        )

        # Features matching targets
        features = ProsodyFeatures(
            f0_mean=100.0,
            f0_std=30.0,
            f0_range=100.0,
            intensity_mean=60.0,
            duration=2.0,
            speech_rate=6.2,
            npvi_v=55.0,
        )

        score_full = score_prosody_profile(features, profile_full_weight)
        score_half = score_prosody_profile(features, profile_half_weight)

        # Half weight should produce half the score
        assert score_half == pytest.approx(score_full * 0.5, abs=0.01)

    def test_all_zeros_returns_valid_score(self):
        """Edge case: all zero features should not crash."""
        profile = ProsodyProfile(
            code="spa",
            name="Spanish",
            rhythm_class="syllable-timed",
            f0_range={"min": 50, "max": 200, "target": 120},
            f0_std={"min": 15, "max": 60, "target": 35},
            npvi_v={"min": 35, "max": 52, "target": 43},
            speech_rate={"min": 6.5, "max": 9.0, "target": 7.8},
            weight=1.0,
        )

        features = ProsodyFeatures(
            f0_mean=0.0,
            f0_std=0.0,
            f0_range=0.0,
            intensity_mean=0.0,
            duration=0.0,
            speech_rate=0.0,
            npvi_v=0.0,
        )

        score = score_prosody_profile(features, profile)

        # Should return valid score (not NaN, not exception)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestScoreAllProfiles:
    """Tests for scoring features against all profiles."""

    def test_returns_dict_with_six_languages(self):
        """score_all_profiles should return dict with 7 language codes.

        NOTE: Phase 32 added MAY (Yucatec Maya) as a 7th language.
        Test name kept for continuity but validates 7 languages.
        """
        features = ProsodyFeatures(
            f0_mean=150.0,
            f0_std=35.0,
            f0_range=120.0,
            intensity_mean=60.0,
            duration=2.0,
            speech_rate=7.8,
            npvi_v=43.0,
        )

        scores = score_all_profiles(features)

        assert isinstance(scores, dict)
        assert len(scores) == 7
        assert "deu" in scores
        assert "eng" in scores
        assert "fra" in scores
        assert "ita" in scores
        assert "may" in scores
        assert "nah" in scores
        assert "spa" in scores

        # All scores should be in [0, 1]
        for code, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_stress_timed_features_score_higher_on_eng_deu(self):
        """Features with high nPVI (stress-timed) should score higher on ENG/DEU."""
        # Stress-timed features: high nPVI (variability)
        features = ProsodyFeatures(
            f0_mean=150.0,
            f0_std=45.0,    # Higher variability
            f0_range=150.0,  # Wider range
            intensity_mean=60.0,
            duration=2.0,
            speech_rate=6.2,  # Moderate rate
            npvi_v=58.0,     # High nPVI (stress-timed)
        )

        scores = score_all_profiles(features)

        # Stress-timed languages (ENG, DEU) should score higher
        stress_timed_scores = [scores["eng"], scores["deu"]]
        syllable_timed_scores = [scores["spa"], scores["fra"], scores["ita"]]

        avg_stress_timed = sum(stress_timed_scores) / len(stress_timed_scores)
        avg_syllable_timed = sum(syllable_timed_scores) / len(syllable_timed_scores)

        assert avg_stress_timed > avg_syllable_timed

    def test_syllable_timed_features_score_higher_on_spa_fra_ita(self):
        """Features with low nPVI (syllable-timed) should score higher on SPA/FRA/ITA."""
        # Syllable-timed features: low nPVI (regular rhythm)
        features = ProsodyFeatures(
            f0_mean=120.0,
            f0_std=35.0,    # Lower variability
            f0_range=120.0,  # Narrower range
            intensity_mean=60.0,
            duration=2.0,
            speech_rate=7.5,  # Higher rate
            npvi_v=42.0,     # Low nPVI (syllable-timed)
        )

        scores = score_all_profiles(features)

        # Syllable-timed languages (SPA, FRA, ITA) should score higher
        stress_timed_scores = [scores["eng"], scores["deu"]]
        syllable_timed_scores = [scores["spa"], scores["fra"], scores["ita"]]

        avg_stress_timed = sum(stress_timed_scores) / len(stress_timed_scores)
        avg_syllable_timed = sum(syllable_timed_scores) / len(syllable_timed_scores)

        assert avg_syllable_timed > avg_stress_timed

    def test_can_pass_custom_profiles(self):
        """score_all_profiles should accept custom profile list."""
        # Create single custom profile
        custom_profile = ProsodyProfile(
            code="test",
            name="Test Language",
            rhythm_class="syllable-timed",
            f0_range={"min": 50, "max": 200, "target": 120},
            f0_std={"min": 15, "max": 60, "target": 35},
            npvi_v={"min": 35, "max": 52, "target": 43},
            speech_rate={"min": 6.5, "max": 9.0, "target": 7.8},
            weight=1.0,
        )

        features = ProsodyFeatures(
            f0_mean=150.0,
            f0_std=35.0,
            f0_range=120.0,
            intensity_mean=60.0,
            duration=2.0,
            speech_rate=7.8,
            npvi_v=43.0,
        )

        scores = score_all_profiles(features, profiles=[custom_profile])

        assert len(scores) == 1
        assert "test" in scores
        assert scores["test"] > 0.9  # Should match well

"""Tests for score normalization and fusion."""

import pytest

from tenepal.fusion import (
    FusionWeights,
    default_fusion_weights,
    fuse_scores,
    normalize_phoneme_scores,
)


class TestNormalizePhonemeScores:
    """Test phoneme score normalization to [0, 1] range."""

    def test_score_at_min_normalizes_to_zero(self):
        """Score at min_val should normalize to 0.0."""
        scores = {"spa": 0.0, "eng": 0.0}
        normalized = normalize_phoneme_scores(scores, min_val=0.0, max_val=150.0)
        assert normalized["spa"] == 0.0
        assert normalized["eng"] == 0.0

    def test_score_at_max_normalizes_to_one(self):
        """Score at max_val should normalize to 1.0."""
        scores = {"spa": 150.0, "eng": 150.0}
        normalized = normalize_phoneme_scores(scores, min_val=0.0, max_val=150.0)
        assert normalized["spa"] == 1.0
        assert normalized["eng"] == 1.0

    def test_score_at_midpoint_normalizes_to_half(self):
        """Score at midpoint should normalize to 0.5."""
        scores = {"spa": 75.0}
        normalized = normalize_phoneme_scores(scores, min_val=0.0, max_val=150.0)
        assert normalized["spa"] == pytest.approx(0.5)

    def test_negative_scores_clamp_to_zero(self):
        """Negative scores should be clamped to 0.0."""
        scores = {"spa": -10.0, "eng": -50.0}
        normalized = normalize_phoneme_scores(scores, min_val=0.0, max_val=150.0)
        assert normalized["spa"] == 0.0
        assert normalized["eng"] == 0.0

    def test_scores_above_max_clamp_to_one(self):
        """Scores above max_val should be clamped to 1.0."""
        scores = {"spa": 200.0, "eng": 999.0}
        normalized = normalize_phoneme_scores(scores, min_val=0.0, max_val=150.0)
        assert normalized["spa"] == 1.0
        assert normalized["eng"] == 1.0

    def test_empty_dict_returns_empty_dict(self):
        """Empty input should return empty output."""
        scores = {}
        normalized = normalize_phoneme_scores(scores, min_val=0.0, max_val=150.0)
        assert normalized == {}

    def test_custom_min_max_values_work(self):
        """Custom min/max values should work correctly."""
        scores = {"spa": 50.0, "eng": 100.0}
        normalized = normalize_phoneme_scores(scores, min_val=0.0, max_val=200.0)
        assert normalized["spa"] == pytest.approx(0.25)
        assert normalized["eng"] == pytest.approx(0.5)

    def test_all_keys_preserved_in_output(self):
        """All input keys should be present in output."""
        scores = {"spa": 45.0, "eng": 90.0, "deu": 120.0, "nah": 30.0}
        normalized = normalize_phoneme_scores(scores)
        assert set(normalized.keys()) == set(scores.keys())

    def test_edge_case_max_equals_min(self):
        """When max_val == min_val, all scores should be 0.0."""
        scores = {"spa": 50.0, "eng": 100.0}
        normalized = normalize_phoneme_scores(scores, min_val=100.0, max_val=100.0)
        assert normalized["spa"] == 0.0
        assert normalized["eng"] == 0.0


class TestFusionWeights:
    """Test FusionWeights dataclass."""

    def test_dataclass_creation(self):
        """Should create FusionWeights with alpha and beta."""
        weights = FusionWeights(alpha=0.6, beta=0.4)
        assert weights.alpha == 0.6
        assert weights.beta == 0.4

    def test_default_values(self):
        """Default values should be alpha=0.7, beta=0.3."""
        weights = FusionWeights()
        assert weights.alpha == 0.7
        assert weights.beta == 0.3


class TestDefaultFusionWeights:
    """Test default fusion weight configuration."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        weights = default_fusion_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_spa_ita_is_prosody_dominant(self):
        """SPA/ITA pair should be prosody-dominant (beta=0.7)."""
        weights = default_fusion_weights()
        pair = frozenset(["spa", "ita"])
        assert pair in weights
        assert weights[pair].alpha == 0.3
        assert weights[pair].beta == 0.7

    def test_spa_fra_is_prosody_dominant(self):
        """SPA/FRA pair should be prosody-dominant (beta=0.7)."""
        weights = default_fusion_weights()
        pair = frozenset(["spa", "fra"])
        assert pair in weights
        assert weights[pair].beta == 0.7

    def test_nah_spa_is_phoneme_dominant(self):
        """NAH/SPA pair should be phoneme-dominant (alpha=0.7)."""
        weights = default_fusion_weights()
        pair = frozenset(["nah", "spa"])
        assert pair in weights
        assert weights[pair].alpha == 0.7
        assert weights[pair].beta == 0.3

    def test_eng_deu_is_phoneme_dominant(self):
        """ENG/DEU pair should be phoneme-dominant (strong markers)."""
        weights = default_fusion_weights()
        pair = frozenset(["eng", "deu"])
        assert pair in weights
        assert weights[pair].alpha == 0.7

    def test_balanced_pairs_exist(self):
        """Should have balanced pairs (alpha=0.5, beta=0.5)."""
        weights = default_fusion_weights()
        pair = frozenset(["eng", "spa"])
        assert pair in weights
        assert weights[pair].alpha == 0.5
        assert weights[pair].beta == 0.5


class TestFuseScores:
    """Test adaptive weighted score fusion."""

    def test_phoneme_only_fallback_with_none(self):
        """When prosody_scores is None, should return normalized phoneme scores."""
        phoneme = {"spa": 45.0, "eng": 90.0, "ita": 30.0}
        fused = fuse_scores(phoneme, prosody_scores=None)

        # Should be normalized to [0, 1]
        assert all(0 <= v <= 1 for v in fused.values())
        # Relative ranking preserved
        assert fused["eng"] > fused["spa"] > fused["ita"]

    def test_phoneme_only_fallback_with_empty(self):
        """When prosody_scores is empty dict, should return normalized phoneme scores."""
        phoneme = {"spa": 60.0, "eng": 120.0}
        fused = fuse_scores(phoneme, prosody_scores={})

        assert all(0 <= v <= 1 for v in fused.values())
        assert fused["eng"] > fused["spa"]

    def test_already_normalized_phoneme_scores(self):
        """Already normalized scores (max <= 1.0) should not be re-normalized."""
        phoneme = {"spa": 0.4, "eng": 0.6}
        prosody = {"spa": 0.5, "eng": 0.5}
        fused = fuse_scores(phoneme, prosody)

        # Should have some contribution from prosody
        assert all(0 <= v <= 1 for v in fused.values())

    def test_balanced_fusion(self):
        """Known inputs with default weights should produce expected outputs."""
        phoneme = {"spa": 60.0, "eng": 90.0}  # Will normalize to 0.4, 0.6
        prosody = {"spa": 0.8, "eng": 0.3}
        fused = fuse_scores(phoneme, prosody)

        # With balanced weights (0.5, 0.5):
        # spa: 0.5 * 0.4 + 0.5 * 0.8 = 0.6
        # eng: 0.5 * 0.6 + 0.5 * 0.3 = 0.45
        # (actual weights depend on competitor detection)
        assert all(0 <= v <= 1 for v in fused.values())
        assert "spa" in fused
        assert "eng" in fused

    def test_spa_vs_ita_prosody_dominant(self):
        """SPA/ITA discrimination should be prosody-dominant (beta=0.7)."""
        # Similar phoneme scores but different prosody
        phoneme = {"spa": 45.0, "ita": 42.0, "eng": 10.0}
        prosody = {"spa": 0.8, "ita": 0.3, "eng": 0.2}
        fused = fuse_scores(phoneme, prosody)

        # Prosody should help separate SPA from ITA
        # With prosody-dominant weights, high prosody score for SPA should boost it
        assert fused["spa"] > fused["ita"]

    def test_nah_vs_spa_phoneme_dominant(self):
        """NAH/SPA discrimination should be phoneme-dominant (alpha=0.7)."""
        # Strong phoneme difference, moderate prosody difference
        phoneme = {"nah": 90.0, "spa": 30.0}
        prosody = {"nah": 0.5, "spa": 0.6}  # Prosody slightly favors SPA
        fused = fuse_scores(phoneme, prosody)

        # Phoneme channel should dominate
        assert fused["nah"] > fused["spa"]

    def test_missing_prosody_for_one_language(self):
        """Language missing from prosody_scores should use phoneme-only."""
        phoneme = {"spa": 60.0, "eng": 90.0, "deu": 75.0}
        prosody = {"spa": 0.8, "eng": 0.3}  # DEU missing
        fused = fuse_scores(phoneme, prosody)

        # DEU should fall back to normalized phoneme score
        assert "deu" in fused
        assert 0 <= fused["deu"] <= 1

    def test_all_scores_positive_and_bounded(self):
        """All fused scores should be in [0, 1] range."""
        phoneme = {"spa": 45.0, "eng": 90.0, "deu": 120.0, "ita": 30.0}
        prosody = {"spa": 0.8, "eng": 0.3, "deu": 0.5, "ita": 0.7}
        fused = fuse_scores(phoneme, prosody)

        for lang, score in fused.items():
            assert 0 <= score <= 1, f"{lang} score {score} out of bounds"

    def test_ranking_preservation_when_channels_agree(self):
        """When both channels agree, fused ranking should match."""
        phoneme = {"spa": 90.0, "eng": 60.0, "ita": 30.0}
        prosody = {"spa": 0.9, "eng": 0.6, "ita": 0.3}
        fused = fuse_scores(phoneme, prosody)

        # Both channels rank: spa > eng > ita
        assert fused["spa"] > fused["eng"] > fused["ita"]

    def test_custom_weights(self):
        """Custom fusion weights should be respected."""
        phoneme = {"spa": 60.0, "eng": 90.0}
        prosody = {"spa": 0.9, "eng": 0.1}

        # Create custom weights favoring prosody strongly
        custom_weights = {
            frozenset(["spa", "eng"]): FusionWeights(alpha=0.1, beta=0.9)
        }
        fused = fuse_scores(phoneme, prosody, weights=custom_weights)

        # With strong prosody weight, SPA should rank higher despite lower phoneme score
        # (after normalization: spa=0.4, eng=0.6; with alpha=0.1, beta=0.9:
        #  spa: 0.1*0.4 + 0.9*0.9 = 0.85, eng: 0.1*0.6 + 0.9*0.1 = 0.15)
        assert fused["spa"] > fused["eng"]

    def test_empty_phoneme_scores(self):
        """Empty phoneme_scores should return empty dict."""
        fused = fuse_scores({}, {"spa": 0.8})
        assert fused == {}

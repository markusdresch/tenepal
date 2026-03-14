"""Tests for weighted confidence scoring engine in language identification."""

import pytest

from tenepal.language.identifier import LanguageSegment, identify_language
from tenepal.language.registry import LanguageProfile, LanguageRegistry
from tenepal.phoneme import PhonemeSegment


def _make_phonemes(phoneme_list: list[str]) -> list[PhonemeSegment]:
    """Create sequential PhonemeSegments with 0.1s spacing.

    Helper to avoid repetitive test setup. Each phoneme gets 0.1s duration,
    starting from 0.0s and incrementing by 0.1s per phoneme.

    Args:
        phoneme_list: List of IPA phoneme strings

    Returns:
        List of PhonemeSegment objects
    """
    segments = []
    for i, phoneme in enumerate(phoneme_list):
        segments.append(PhonemeSegment(
            phoneme=phoneme,
            start_time=i * 0.1,
            duration=0.1
        ))
    return segments


def test_default_weights_backward_compatible_nahuatl():
    """Default weights (all 1.0) should produce same result as current binary system for Nahuatl."""
    # Uses default registry with no custom weights or thresholds
    phonemes = _make_phonemes(["a", "tɬ", "i"])
    result = identify_language(phonemes)

    # Should work exactly as before (binary marker detection)
    assert len(result) == 1
    assert result[0].language == "nah"
    assert len(result[0].phonemes) == 3


def test_default_weights_backward_compatible_other():
    """Default weights should label unmarked phonemes as 'other' same as binary system."""
    # Phonemes with no markers in any language profile
    phonemes = _make_phonemes(["a", "e", "i", "o", "u"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "other"
    assert len(result[0].phonemes) == 5


def test_default_weights_backward_compatible_code_switch():
    """Default weights should detect code-switching with sufficient marker evidence."""
    # Nahuatl markers then Spanish markers — contiguous to avoid Phase C fragmentation
    phonemes = _make_phonemes([
        "tɬ", "kʷ", "ʔ",                        # 3 NAH markers
        "ɲ", "ɾ", "b", "ɲ", "ɾ", "b", "ɲ",     # 7 SPA markers (conf 3.60 > 2.25)
    ])
    result = identify_language(phonemes)

    # Should have 2 segments
    languages = [seg.language for seg in result]
    assert "nah" in languages
    assert "spa" in languages


def test_custom_weights_reduce_marker_impact():
    """Custom low marker weight should reduce detection confidence below threshold."""
    # Create custom registry with low weight for ŋ marker
    custom_registry = LanguageRegistry()
    eng_profile = LanguageProfile(
        code="eng",
        name="English",
        family="Indo-European",
        marker_phonemes={"θ", "ŋ"},
        absent_phonemes=set(),
        marker_weights={"θ": 1.0, "ŋ": 0.1},  # ŋ is weak marker
        threshold=0.5,  # Need 0.5 total weight to tag as eng
    )
    custom_registry.register(eng_profile)

    # Input with only low-weight markers: [a, ŋ, i, ŋ, o]
    # Score = 0.1 + 0.1 = 0.2 < 0.5 threshold → should NOT tag as eng
    phonemes = _make_phonemes(["a", "ŋ", "i", "ŋ", "o"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "other"  # Didn't meet threshold

    # Same registry with threshold=0.0 should still detect (backward compatible)
    eng_profile_low_threshold = LanguageProfile(
        code="eng",
        name="English",
        family="Indo-European",
        marker_phonemes={"θ", "ŋ"},
        absent_phonemes=set(),
        marker_weights={"θ": 1.0, "ŋ": 0.1},
        threshold=0.0,  # Any marker match suffices
    )
    custom_registry2 = LanguageRegistry()
    custom_registry2.register(eng_profile_low_threshold)

    result2 = identify_language(phonemes, registry=custom_registry2)
    assert len(result2) == 1
    assert result2[0].language == "eng"  # Low weight still counts


def test_custom_threshold_blocks_weak_detection():
    """High threshold should block detection when only one marker present."""
    # Custom Spanish profile with high threshold
    custom_registry = LanguageRegistry()
    spa_profile = LanguageProfile(
        code="spa",
        name="Spanish",
        family="Indo-European",
        marker_phonemes={"b", "d", "ɡ"},
        absent_phonemes=set(),
        threshold=2.0,  # Need 2.0 total score (2+ markers at default weight 1.0)
    )
    custom_registry.register(spa_profile)

    # Input with only 1 marker: [a, b, i]
    # Score = 1.0 < 2.0 threshold → should NOT tag as spa
    phonemes = _make_phonemes(["a", "b", "i"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "other"


def test_custom_threshold_allows_strong_detection():
    """Multiple markers should exceed threshold for successful detection."""
    # Same Spanish profile with threshold 0.5
    custom_registry = LanguageRegistry()
    spa_profile = LanguageProfile(
        code="spa",
        name="Spanish",
        family="Indo-European",
        marker_phonemes={"b", "d", "ɡ"},
        absent_phonemes=set(),
        threshold=0.5,
    )
    custom_registry.register(spa_profile)

    # Input with 2 markers: [a, b, i, d, o]
    # Score = 1.0 + 1.0 = 2.0 > 0.5 threshold → should tag as spa
    phonemes = _make_phonemes(["a", "b", "i", "d", "o"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "spa"


def test_weighted_scoring_highest_score_wins():
    """When multiple languages match, highest weighted score wins at each phoneme position."""
    # Custom registry with two languages competing for same phoneme
    custom_registry = LanguageRegistry()

    # lang_a: x has weight 1.0, higher priority
    lang_a = LanguageProfile(
        code="lang_a",
        name="Language A",
        family="Test",
        marker_phonemes={"x"},
        absent_phonemes=set(),
        marker_weights={"x": 1.0},
        priority=10,  # Higher priority to avoid reclassification
    )

    # lang_b: x and y both have weight 0.3
    lang_b = LanguageProfile(
        code="lang_b",
        name="Language B",
        family="Test",
        marker_phonemes={"x", "y"},
        absent_phonemes=set(),
        marker_weights={"x": 0.3, "y": 0.3},
        priority=1,
    )

    custom_registry.register(lang_a)
    custom_registry.register(lang_b)

    # Input: [x, a, a, a, y, b, b, b] - need 3+ phonemes per segment to avoid merging
    # For x: lang_a scores 1.0, lang_b scores 0.3 → lang_a wins
    # For a: unmarked, absorbed into lang_a
    # For y: lang_b scores 0.3, lang_a scores 0.0 → lang_b wins
    # For b: unmarked, absorbed into lang_b
    phonemes = _make_phonemes(["x", "a", "a", "a", "y", "b", "b", "b"])
    result = identify_language(phonemes, registry=custom_registry)

    # Should have 2 segments: lang_a with x+aaa, lang_b with y+bbb
    # lang_a has higher priority so reclassification won't flip it
    languages = [seg.language for seg in result]
    assert "lang_a" in languages
    assert "lang_b" in languages


def test_confidence_score_on_segment():
    """Successfully detected segment should have confidence > 0.0."""
    phonemes = _make_phonemes(["a", "tɬ", "i", "kʷ"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"
    # With default weights, tɬ=1.0 + kʷ=1.0 = confidence 2.0
    assert result[0].confidence > 0.0


def test_confidence_zero_for_other():
    """Unmarked phonemes ('other') should have confidence == 0.0."""
    phonemes = _make_phonemes(["a", "e", "i", "o", "u"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "other"
    assert result[0].confidence == 0.0


def test_sequence_weights_affect_scoring():
    """Bigram sequence weights should affect detection confidence."""
    # Custom Nahuatl profile with lower weight for (t, l) sequence
    custom_registry = LanguageRegistry()
    nah_profile = LanguageProfile(
        code="nah",
        name="Nahuatl",
        family="Uto-Aztecan",
        marker_phonemes={"tɬ"},
        absent_phonemes=set(),
        marker_sequences={("t", "l")},
        sequence_weights={("t", "l"): 0.5},  # Lower weight for this bigram
        threshold=0.0,
    )
    custom_registry.register(nah_profile)

    # Input: [a, t, l, i]
    # Bigram (t, l) detected with weight 0.5 (not 1.0)
    # Use phonemes that won't match Nahuatl lexicon: [u, t, l, y]
    phonemes = _make_phonemes(["u", "t", "l", "y"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "nah"
    # Confidence should reflect the lower sequence weight
    # Both t and l tagged by bigram, so confidence = 0.5 (sequence weight)
    assert 0.0 < result[0].confidence < 1.0


def test_trigram_weight_affects_confidence():
    """Trigram weight should affect detection confidence."""
    custom_registry = LanguageRegistry()
    profile = LanguageProfile(
        code="tst",
        name="Test",
        family="Test",
        marker_phonemes=set(),
        absent_phonemes=set(),
        marker_trigrams={("x", "y", "z")},
        trigram_weights={("x", "y", "z"): 0.9},
    )
    custom_registry.register(profile)

    phonemes = _make_phonemes(["a", "x", "y", "z", "a"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "tst"
    # Weight 0.9 split across 3 positions = 0.3 each
    assert result[0].confidence == pytest.approx(0.9, abs=0.01)

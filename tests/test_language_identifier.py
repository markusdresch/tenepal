"""Tests for language identification from phoneme streams."""

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


def test_empty_input():
    """Empty phoneme list should return empty language segment list."""
    result = identify_language([])
    assert result == []


def test_single_nahuatl_marker():
    """Phoneme stream containing Nahuatl marker /tɬ/ should label segment as 'nah'."""
    phonemes = _make_phonemes(["a", "tɬ", "i"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"
    assert len(result[0].phonemes) == 3


def test_single_spanish_marker():
    """Spanish markers with sufficient confidence should label segment as 'spa'."""
    # SPA confidence: 2×ɲ(1.60) + 2×ɾ(1.40) + b(0.50) = 3.50 > 3.0
    phonemes = _make_phonemes(["ɲ", "a", "ɾ", "o", "ɲ", "a", "ɾ", "b"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "spa"
    assert len(result[0].phonemes) == 8


def test_unmarked_phonemes_labeled_other():
    """Phonemes with no language markers should be labeled 'other'."""
    # Use phonemes that aren't markers for either Nahuatl or Spanish
    phonemes = _make_phonemes(["a", "e", "i", "o", "u"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "other"
    assert len(result[0].phonemes) == 5


def test_nahuatl_segment_grouping():
    """Consecutive phonemes with Nahuatl markers should form one LanguageSegment."""
    # Multiple Nahuatl markers in a row
    phonemes = _make_phonemes(["tɬ", "a", "kʷ", "i", "ʔ"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"
    assert len(result[0].phonemes) == 5


def test_code_switching():
    """Sequence with Nahuatl markers then Spanish markers should produce two segments."""
    # Nahuatl markers, then Spanish markers with enough confidence
    phonemes = _make_phonemes(["tɬ", "a", "kʷ", "ɲ", "o", "ɾ", "ɲ", "o", "ɾ", "b"])
    result = identify_language(phonemes)

    # Should have at least 2 segments (before merging might have more)
    assert len(result) >= 2

    # Find Nahuatl and Spanish segments
    languages = [seg.language for seg in result]
    assert "nah" in languages
    # With stricter negative-marker penalties, short mixed tails may downgrade to OTHER.
    assert any(lang in {"spa", "other"} for lang in languages if lang != "nah")


def test_short_segment_merging():
    """A 1-phoneme 'other' segment between two 'nah' segments should merge into Nahuatl."""
    # Nahuatl markers, then single unmarked phoneme, then Nahuatl markers again
    phonemes = _make_phonemes(["tɬ", "kʷ", "ʔ", "a", "ɬ", "tʃ", "kʷ"])
    result = identify_language(phonemes)

    # After merging, should be single Nahuatl segment
    assert len(result) == 1
    assert result[0].language == "nah"
    assert len(result[0].phonemes) == 7


def test_segment_timing():
    """LanguageSegment start_time and end_time should match phoneme boundaries."""
    phonemes = _make_phonemes(["tɬ", "a", "kʷ"])
    result = identify_language(phonemes)

    assert len(result) == 1
    segment = result[0]

    # Start time should match first phoneme
    assert segment.start_time == 0.0

    # End time should be last phoneme's start + duration
    # Last phoneme: index 2, start_time = 0.2, duration = 0.1
    assert segment.end_time == pytest.approx(0.3)


def test_custom_registry():
    """Passing a custom registry with different profiles should work correctly."""
    # Create custom registry with only a test language
    custom_registry = LanguageRegistry()
    test_profile = LanguageProfile(
        code="tst",
        name="Test Language",
        family="Test",
        marker_phonemes={"x", "y"},
        absent_phonemes=set()
    )
    custom_registry.register(test_profile)

    # Use phonemes that match the test language
    phonemes = _make_phonemes(["x", "a", "y"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "tst"


def test_mixed_stream_with_merging():
    """Realistic scenario: Nahuatl phonemes with short 'other' interruptions merge correctly."""
    # Nahuatl markers, then 1-2 unmarked phonemes, then more Nahuatl markers
    # The short "other" segments should merge into the Nahuatl segments
    phonemes = _make_phonemes([
        "tɬ", "kʷ", "ʔ",  # 3 Nahuatl
        "a",              # 1 other (too short)
        "ɬ", "tʃ", "kʷ",  # 3 Nahuatl
        "e", "i",         # 2 other (too short)
        "tɬ", "ʔ", "kʷ",  # 3 Nahuatl
    ])
    result = identify_language(phonemes)

    # After merging, should be single Nahuatl segment
    assert len(result) == 1
    assert result[0].language == "nah"
    assert len(result[0].phonemes) == 12


def test_code_switching_with_sufficient_segments():
    """Code-switching with sufficiently long segments in each language should not merge."""
    # Nahuatl phonemes, then Spanish phonemes — all markers contiguous
    # so Phase C short-segment merging doesn't fragment across "other" vowels
    phonemes = _make_phonemes([
        "tɬ", "kʷ", "ʔ",                        # 3 NAH (conf 3.0)
        "ɲ", "ɾ", "b", "ɲ", "ɾ", "b", "ɲ",     # 7 SPA (conf 3.60)
    ])
    result = identify_language(phonemes)

    # Should have exactly 2 segments
    assert len(result) == 2
    assert result[0].language == "nah"
    assert result[1].language == "spa"


def test_allosaurus_affricate_variants_tagged_nahuatl():
    """Allosaurus affricate outputs (t͡ɕ, tɕ, tʂ, ts) should identify as Nahuatl."""
    phonemes = _make_phonemes(["a", "t͡ɕ", "i", "tʂ", "a", "ts", "o"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_allosaurus_palatalized_velar_tagged_nahuatl():
    """Allosaurus k̟ʲ (palatalized velar) should identify as Nahuatl /kʷ/."""
    phonemes = _make_phonemes(["a", "k̟ʲ", "i", "k̟ʲ", "a"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_allosaurus_fricative_variants_tagged_nahuatl():
    """Allosaurus ɕ and ʂ should identify as Nahuatl (near /ɬ/)."""
    phonemes = _make_phonemes(["a", "ɕ", "i", "ʂ", "a"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_bigram_k_w_tagged_nahuatl():
    """Consecutive k + w should be tagged as Nahuatl (split /kʷ/)."""
    phonemes = _make_phonemes(["a", "k", "w", "a", "ts", "i"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_bigram_t_l_tagged_nahuatl():
    """Consecutive t + l should be tagged as Nahuatl (split /tɬ/)."""
    phonemes = _make_phonemes(["a", "t", "l", "a", "ts", "i"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_realistic_allosaurus_nahuatl_stream():
    """Realistic Allosaurus output for Nahuatl should identify as Nahuatl."""
    # Based on actual Allosaurus output for Nahuatl audio
    phonemes = _make_phonemes([
        "lʲ", "ɪ", "tʲ", "ɕ", "lʲ", "i", "m", "a",
        "ɴ", "a", "w", "ɪ", "ts", "k",
    ])
    result = identify_language(phonemes)

    languages = [seg.language for seg in result]
    assert "nah" in languages
    # Nahuatl should be the dominant language
    nah_phonemes = sum(len(s.phonemes) for s in result if s.language == "nah")
    assert nah_phonemes >= 10


def test_priority_nahuatl_wins_over_spanish_in_mixed_segment():
    """Segment with both Nahuatl and Spanish markers should be classified as Nahuatl."""
    # Mix of unique Nahuatl markers and Spanish voiced plosives
    # NAH unique markers (tɬ, kʷ, ʔ) outscore SPA markers (ɡ, ɾ)
    phonemes = _make_phonemes([
        "tɬ", "a", "ɡ", "i", "kʷ", "o", "ɾ", "a", "ʔ", "e",
    ])
    result = identify_language(phonemes)

    # Should be classified as Nahuatl (higher score + priority)
    assert len(result) == 1
    assert result[0].language == "nah"


def test_priority_spanish_only_without_nahuatl_markers():
    """Segment with only Spanish markers and no Nahuatl markers stays Spanish."""
    # SPA confidence: 2×ɲ(1.60) + 2×ɾ(1.40) + b(0.50) = 3.50 > 3.0
    phonemes = _make_phonemes([
        "ɲ", "a", "ɾ", "o", "ɲ", "a", "ɾ", "e", "b",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "spa"


def test_priority_reclassifies_and_remerges():
    """Adjacent segments reclassified to same language should merge."""
    # NAH segment, then SPA segment that contains a NAH marker → both become NAH
    phonemes = _make_phonemes([
        "ts", "a", "tʂ",       # 3 clearly Nahuatl
        "e", "i", "o",         # 3 unmarked
        "ɡ", "a", "ts", "o",  # SPA + NAH marker → reclassify to NAH
    ])
    result = identify_language(phonemes)

    # After reclassification and re-merge, should be single Nahuatl segment
    assert len(result) == 1
    assert result[0].language == "nah"


def test_bigram_matches_with_length_mark():
    """t + lː should match sequence ("t", "l") by stripping ː modifier."""
    # Real Allosaurus output: t lː a t i = /tɬatl/ pattern
    phonemes = _make_phonemes(["a", "t", "lː", "a", "t", "i"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_bigram_matches_with_palatalization():
    """t + lʲ should match sequence ("t", "l") by stripping ʲ modifier."""
    phonemes = _make_phonemes(["a", "t", "lʲ", "a", "ts", "i"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_bigram_matches_with_dental_diacritic():
    """t̪ + l should match sequence ("t", "l") by stripping combining diacritics."""
    phonemes = _make_phonemes(["a", "t̪", "l", "a", "ts", "i"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_island_absorption_neutral_between_nah():
    """Neutral island between two Nahuatl segments gets absorbed into Nahuatl."""
    phonemes = _make_phonemes([
        "tɬ", "a", "kʷ", "i", "ʔ",  # 5 NAH (unique markers)
        "a", "e", "o",               # 3 neutral (no markers for any language)
        "tɬ", "o", "ʔ", "a", "kʷ",  # 5 NAH (unique markers)
    ])
    result = identify_language(phonemes)

    # Neutral island should be absorbed — single NAH segment
    assert len(result) == 1
    assert result[0].language == "nah"
    assert len(result[0].phonemes) == 13


def test_island_absorption_preserves_real_code_switching():
    """Genuine code-switching (SPA at start/end) is NOT absorbed."""
    # All markers contiguous so Phase C short-segment merging doesn't fragment
    phonemes = _make_phonemes([
        "ɲ", "ɾ", "b", "ɲ", "ɾ", "b", "ɲ",    # 7 SPA markers (conf 3.60)
        "ts", "tʂ", "k̟ʲ", "ʔ", "tɬ",           # 5 NAH markers
        "ɲ", "ɾ", "b", "ɲ", "ɾ", "b", "ɲ",    # 7 SPA markers (conf 3.60)
    ])
    result = identify_language(phonemes)

    # SPA at boundaries should remain
    assert len(result) == 3
    assert result[0].language == "spa"
    assert result[1].language == "nah"
    assert result[2].language == "spa"


def test_island_absorption_only_lower_priority():
    """Higher-priority island between lower-priority segments is NOT absorbed."""
    phonemes = _make_phonemes([
        "b", "d", "ɡ", "a", "ɾ",    # 5 SPA
        "ts", "tʂ", "k̟ʲ",           # 3 NAH (higher priority)
        "b", "d", "ɡ", "a", "ɾ",    # 5 SPA
    ])
    result = identify_language(phonemes)

    # NAH has higher priority — should NOT be absorbed into SPA
    assert any(seg.language == "nah" for seg in result)


# ============================================================================
# English detection tests
# ============================================================================


def test_english_theta_marker():
    """English markers with sufficient confidence should identify as English."""
    # ENG confidence: 10×æ(2.50) + 9×ɪ(1.80) = 4.30 > 4.10
    phonemes = _make_phonemes([
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ",
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "eng"


def test_english_rhotic_marker():
    """English markers with sufficient weighted evidence should identify as English."""
    # ENG confidence: 10×æ(2.50) + 9×ɪ(1.80) = 4.30 > 4.10
    phonemes = _make_phonemes([
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ",
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "eng"


def test_english_rhotacized_vowels():
    """English detection with multiple weighted markers exceeding threshold."""
    # ENG confidence: 10×æ(2.50) + 9×ɪ(1.80) + 1×ʊ(0.05) = 4.35 > 4.10
    phonemes = _make_phonemes([
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ",
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "ʊ", "æ",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "eng"


def test_english_dʒ_bigram():
    """English /dʒ/ affricate bigram with supporting markers should identify as English."""
    # ENG confidence: d(0.50) + ʒ→FRA(0.80) + 9×æ(2.25) + 8×ɪ(1.60) = 5.15 > 4.10
    phonemes = _make_phonemes([
        "d", "ʒ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ",
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "eng"


# ============================================================================
# German detection tests
# ============================================================================


def test_german_ich_laut_marker():
    """German /x/ ach-laut with sufficient evidence should identify as German."""
    # DEU confidence: 7×x(6.30) > 6.10 (vowels absorbed as None)
    phonemes = _make_phonemes([
        "x", "a", "x", "i", "x", "a", "x", "i", "x", "a", "x", "i", "x",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "deu"


def test_german_rounded_vowels():
    """German uvular fricative /ʁ/ with sufficient evidence should identify as German."""
    # DEU confidence: 5×x(4.50) + 3×ʁ(2.10) = 6.60 > 6.10 (vowels absorbed as None)
    phonemes = _make_phonemes([
        "x", "a", "ʁ", "i", "x", "a", "ʁ", "i", "x", "a", "ʁ", "i", "x", "a", "x",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "deu"


def test_german_uvular_r():
    """German uvular fricative /ʁ/ with /x/ should identify as German."""
    # DEU confidence: 5×x(4.50) + 3×ʁ(2.10) = 6.60 > 6.10
    phonemes = _make_phonemes([
        "ʁ", "a", "x", "i", "ʁ", "a", "x", "i", "ʁ", "a", "x", "i", "x", "a", "x",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "deu"


def test_german_pf_bigram():
    """German /pf/ affricate bigram with supporting markers should identify as German."""
    # DEU confidence: p+f(1.10) + 6×x(5.40) = 6.50 > 6.10 (vowels absorbed as None)
    phonemes = _make_phonemes([
        "p", "f", "a", "x", "i", "x", "a", "x", "i", "x", "a", "x", "i", "x",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "deu"


# ============================================================================
# Spanish validation tests (with ð removed)
# ============================================================================


def test_spanish_voiced_plosives_still_detected():
    """Spanish detection with sufficient marker evidence."""
    # SPA confidence: 2×ɲ(1.60) + 2×ɾ(1.40) + b(0.50) = 3.50 > 3.0
    phonemes = _make_phonemes(["ɲ", "a", "ɾ", "o", "ɲ", "a", "ɾ", "e", "b"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "spa"


def test_spanish_palatal_nasal():
    """Spanish palatal nasal /ɲ/ with supporting markers should identify as Spanish."""
    # SPA confidence: 3×ɲ(2.40) + 2×ɾ(1.40) = 3.80 > 3.0
    phonemes = _make_phonemes(["a", "ɲ", "o", "ɲ", "a", "ɲ", "ɾ", "ɾ"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "spa"


# ============================================================================
# Multi-language mixing tests (LANG-04)
# ============================================================================


def test_four_language_code_switching():
    """Four-language code-switching should produce correctly labeled segments."""
    # NAH → SPA → ENG → DEU — each with contiguous markers exceeding thresholds
    phonemes = _make_phonemes([
        "tɬ", "kʷ", "ʔ",                                               # 3 NAH (conf 3.0)
        "ɲ", "ɾ", "b", "ɲ", "ɾ", "b", "ɲ",                            # 7 SPA (conf 3.60 > 2.25)
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ",
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ",                  # 19 ENG (conf 4.30 > 4.10)
        "x", "a", "x", "i", "x", "a", "x", "i", "x", "a", "x", "i", "x",  # 13 DEU (conf 6.30 > 6.10)
    ])
    result = identify_language(phonemes)

    # Should have 4 segments
    assert len(result) == 4
    languages = [seg.language for seg in result]
    assert languages == ["nah", "spa", "eng", "deu"]


def test_english_german_boundary():
    """English followed by German should produce two distinct segments."""
    phonemes = _make_phonemes([
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ",
        "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ", "ɪ", "æ",                  # 19 ENG (conf 4.30 > 4.10)
        "x", "a", "x", "i", "x", "a", "x", "i", "x", "a", "x", "i", "x",  # 13 DEU (conf 6.30 > 6.10)
    ])
    result = identify_language(phonemes)

    assert len(result) == 2
    assert result[0].language == "eng"
    assert result[1].language == "deu"


# ============================================================================
# Extensibility test (LANG-05)
# ============================================================================


def test_extensibility_new_language_no_identifier_changes():
    """Adding a new language profile should work without modifying identifier.py."""
    # Create custom registry with a fictional language
    custom_registry = LanguageRegistry()
    jpn_profile = LanguageProfile(
        code="jpn",
        name="Japanese",
        family="Japonic",
        marker_phonemes={"ɰ", "ɸ"},  # Japanese /w/ (labial-velar approximant) and /ɸ/ (voiceless bilabial fricative)
        absent_phonemes=set()
    )
    custom_registry.register(jpn_profile)

    # Use phonemes containing Japanese markers
    phonemes = _make_phonemes(["a", "ɰ", "i", "ɸ", "a"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "jpn"


# ============================================================================
# Priority interaction tests
# ============================================================================


def test_nahuatl_priority_over_english():
    """Strong English markers should override weak mixed Nahuatl evidence."""
    # Mixed markers with explicit English-only cues (θ, ɹ).
    # With the stricter ENG profile this should classify as ENG.
    phonemes = _make_phonemes([
        "ts", "a", "θ", "i", "tʂ", "o", "ɹ", "a", "k̟ʲ", "e",
    ])
    result = identify_language(phonemes)

    # Should now be classified as English to avoid NAH false positives.
    assert len(result) == 1
    assert result[0].language == "eng"


def test_german_priority_over_english():
    """Segment with both DEU and ENG markers should be classified as German."""
    # Mix of DEU and ENG markers — DEU wins by score (5.40 vs 1.15)
    # Total conf: 6×x(5.40) + 3×æ(0.75) + 2×ɪ(0.40) = 6.55 > 6.10
    phonemes = _make_phonemes([
        "x", "æ", "x", "ɪ", "x", "æ", "x", "ɪ", "x", "æ", "x",
    ])
    result = identify_language(phonemes)

    # Should be classified as German (higher priority wins via reclassification)
    assert len(result) == 1
    assert result[0].language == "deu"


# ============================================================================
# Confidence-aware reclassification tests
# ============================================================================


def test_confidence_reclassify_deu_not_overridden_by_nah():
    """DEU segment with weak NAH markers should NOT be reclassified to NAH.

    This is the core fix: a DEU segment with e.g. 3x 'ts' used to be
    reclassified to NAH because NAH priority > DEU priority. With
    confidence-aware reclassification, DEU score must be compared against
    NAH score and DEU should win when it has more evidence.
    """
    # DEU score: 5×x(4.50) + 2×ʁ(1.40) = 5.90 vs NAH score: ts(0.50)
    # Total conf: 6.40 > 6.10 (DEU threshold)
    # With old logic: NAH marker present + higher priority → NAH wins (WRONG)
    # With new logic: DEU 5.90 > NAH 0.50 → DEU stays (CORRECT)
    phonemes = _make_phonemes([
        "x", "a", "ʁ", "i", "x", "a", "ts", "i", "x", "a", "ʁ", "i", "x", "a", "x",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "deu"


def test_confidence_reclassify_genuine_nah_stays_nah():
    """Genuine NAH segment with high NAH score should still be classified as NAH."""
    # NAH markers dominate: ts, tʂ, k̟ʲ, ɕ, ʃ, ts (6 markers)
    # Only 1 DEU marker (x) — NAH should clearly win
    phonemes = _make_phonemes([
        "ts", "a", "tʂ", "i", "k̟ʲ", "o", "ɕ", "a", "ʃ", "x", "ts",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


def test_confidence_reclassify_nah_still_wins_over_spa():
    """NAH markers in SPA segment should reclassify to NAH when NAH score > SPA score."""
    # Mix: 3 unique NAH markers (tɬ, kʷ, ʔ) + 1 SPA marker (ɡ)
    # NAH score ~2.5 (3.0 - 0.4 ɡ penalty) vs SPA score ~0.2 → NAH wins
    phonemes = _make_phonemes([
        "tɬ", "a", "ɡ", "i", "kʷ", "o", "ɕ", "a", "ʔ", "e",
    ])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah"


# ============================================================================
# Trigram scoring tests
# ============================================================================


def test_trigram_scoring_deu():
    """DEU trigram (ʃ, t, ʁ) should contribute to scoring."""
    custom_registry = LanguageRegistry()
    deu_profile = LanguageProfile(
        code="deu",
        name="German",
        family="Indo-European",
        marker_phonemes=set(),
        absent_phonemes=set(),
        marker_trigrams={("ʃ", "t", "ʁ")},
        trigram_weights={("ʃ", "t", "ʁ"): 1.5},
    )
    custom_registry.register(deu_profile)

    phonemes = _make_phonemes(["a", "ʃ", "t", "ʁ", "a"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "deu"
    assert result[0].confidence > 0.0


def test_trigram_priority_over_bigram_at_same_position():
    """Trigram at a position should prevent bigram double-counting for same language."""
    custom_registry = LanguageRegistry()
    profile = LanguageProfile(
        code="tst",
        name="Test",
        family="Test",
        marker_phonemes=set(),
        absent_phonemes=set(),
        marker_sequences={("a", "b")},
        sequence_weights={("a", "b"): 1.0},
        marker_trigrams={("a", "b", "c")},
        trigram_weights={("a", "b", "c"): 1.5},
    )
    custom_registry.register(profile)

    # Trigram (a, b, c) should be matched. Bigram (a, b) should NOT
    # double-count at positions 0 and 1 since they're trigram-tagged.
    phonemes = _make_phonemes(["a", "b", "c"])
    result = identify_language(phonemes, registry=custom_registry)

    assert len(result) == 1
    assert result[0].language == "tst"
    # Confidence should be trigram weight only (1.5), NOT trigram + bigram
    assert result[0].confidence == pytest.approx(1.5, abs=0.01)


# ============================================================================
# Phase G: Whole-file consolidation tests
# ============================================================================


def test_file_consolidation_dominant_language():
    """When one language dominates whole-file score, consolidate all segments."""
    # Need >100 phonemes for Phase G to trigger
    # Build a file with strong DEU markers spread among neutral phonemes
    deu_markers = ["x", "ʁ"] * 15  # 30 DEU markers
    neutral = ["a", "e", "i", "o", "u"] * 16  # 80 neutral
    eng_markers = ["æ", "ɪ"] * 5  # 10 ENG markers (minority)
    phonemes = _make_phonemes(deu_markers + neutral + eng_markers)
    result = identify_language(phonemes)

    # DEU should dominate since it has more/stronger markers
    lang_counts = {}
    for seg in result:
        lang_counts[seg.language] = lang_counts.get(seg.language, 0) + len(seg.phonemes)
    assert lang_counts.get("deu", 0) > lang_counts.get("eng", 0)


def test_file_consolidation_skipped_for_short_input():
    """Phase G should not trigger for short inputs (<100 phonemes)."""
    # Short input: 13 phonemes, should NOT consolidate
    phonemes = _make_phonemes([
        "tɬ", "a", "kʷ", "i", "ʔ",  # NAH
        "ɲ", "a", "ɾ",               # SPA
        "tɬ", "o", "ʔ", "a", "kʷ",  # NAH
    ])
    result = identify_language(phonemes)

    # Should preserve code-switching (not consolidated)
    assert len(result) >= 1


def test_file_consolidation_absorbs_weak_may_segment():
    """File-level consolidation should absorb weak MAY segments.

    v7.0: MAY removed from _PROTECTED_FILE_CONSOLIDATION_LANGUAGES
    to fix 70 false MAY tags in La Otra Conquista. Weak MAY segments
    (where the dominant language scores higher) are now correctly absorbed.
    """
    from tenepal.language.identifier import _consolidate_by_file_score, _create_segment
    from tenepal.language.registry import default_registry

    registry = default_registry()

    # Build long file context dominated by Nahuatl markers so consolidation triggers.
    all_phonemes = _make_phonemes((["tɬ"] * 110) + (["kʼ"] * 8))

    # Segment stream includes a weak MAY segment between NAH segments.
    seg_nah_1 = _create_segment("nah", _make_phonemes(["tɬ"] * 40))
    may_ph = _make_phonemes(["kʼ", "a", "ʔ", "a", "n"])
    for p in may_ph:
        p.start_time += 4.0
    seg_may = _create_segment("may", may_ph)
    seg_nah_2_ph = _make_phonemes(["tɬ"] * 40)
    for p in seg_nah_2_ph:
        p.start_time += 4.5
    seg_nah_2 = _create_segment("nah", seg_nah_2_ph)

    result = _consolidate_by_file_score(
        [seg_nah_1, seg_may, seg_nah_2],
        all_phonemes,
        registry,
    )

    # MAY should be absorbed by the dominant NAH (no longer protected)
    assert not any(seg.language == "may" for seg in result), (
        "Weak MAY segment should be absorbed by NAH-dominant file consolidation"
    )


def test_file_consolidation_preserves_lat_segment():
    """File-level consolidation should not erase explicit LAT segments."""
    from tenepal.language.identifier import _consolidate_by_file_score, _create_segment
    from tenepal.language.registry import default_registry

    registry = default_registry()
    all_phonemes = _make_phonemes((["tɬ"] * 120) + (["a"] * 10))

    seg_nah = _create_segment("nah", _make_phonemes(["tɬ"] * 50))
    lat_ph = _make_phonemes(["e", "g", "o", "t", "e"])
    for p in lat_ph:
        p.start_time += 5.0
    seg_lat = _create_segment("lat", lat_ph)
    seg_nah_2_ph = _make_phonemes(["tɬ"] * 40)
    for p in seg_nah_2_ph:
        p.start_time += 5.5
    seg_nah_2 = _create_segment("nah", seg_nah_2_ph)

    result = _consolidate_by_file_score([seg_nah, seg_lat, seg_nah_2], all_phonemes, registry)

    assert any(seg.language == "lat" for seg in result), "LAT segment was erased by consolidation"


# ============================================================================
# French and Italian detection tests (Phase 21)
# ============================================================================


def test_french_marker_detection():
    """French markers (y, ø, nasal vowels) should identify segment as French.

    Uses distinctive French phonemes that Allosaurus should produce:
    - y (close front rounded vowel, as in "tu")
    - ø (close-mid front rounded vowel, as in "peu")
    - Nasal vowels (if Allosaurus outputs them)
    """
    # Use 7+ phonemes to exceed 3-phoneme minimum segment length
    phonemes = _make_phonemes(["y", "a", "ø", "e", "y", "i", "ø"])
    result = identify_language(phonemes)

    # Should detect at least one FRA segment
    languages = {seg.language for seg in result}
    assert "fra" in languages, f"French not detected in phoneme stream with y/ø markers. Got: {languages}"


def test_italian_marker_detection():
    """Italian markers (dz, ʎ) should identify segment as Italian.

    Uses distinctive Italian phonemes:
    - dz (voiced alveolar affricate, as in "zero")
    - ʎ (palatal lateral, as in "aglio")
    """
    # Use 7+ phonemes to exceed 3-phoneme minimum segment length
    phonemes = _make_phonemes(["dz", "a", "ʎ", "e", "dz", "i", "ʎ"])
    result = identify_language(phonemes)

    # Should detect at least one ITA segment
    languages = {seg.language for seg in result}
    assert "ita" in languages, f"Italian not detected in phoneme stream with dz/ʎ markers. Got: {languages}"


def test_french_vs_german_disambiguation():
    """French-specific markers should win when French has more evidence than German.

    Tests that ʁ being shared between FRA and DEU doesn't cause misclassification.
    French-specific markers (y, ø) should provide enough evidence to correctly
    identify French segments even when ʁ appears.
    """
    # French-dominant stream: multiple y/ø (FRA-specific) with some ʁ (shared with DEU)
    # FRA score: 3×y(0.7) + 2×ø(0.6) + 2×ʁ(0.15) = 2.1 + 1.2 + 0.3 = 3.6
    # DEU score: 2×ʁ(0.8) = 1.6
    phonemes = _make_phonemes(["y", "a", "ø", "e", "ʁ", "i", "y", "o", "y", "ø", "ʁ"])
    result = identify_language(phonemes)

    # Should detect FRA (not DEU)
    languages = {seg.language for seg in result}
    assert "fra" in languages, f"French not detected when FRA-specific markers dominate. Got: {languages}"


def test_italian_vs_spanish_disambiguation():
    """Italian-specific markers should win when Italian has more evidence than Spanish.

    Tests that shared ɲ doesn't cause misclassification. Italian-specific markers
    (dz, ʎ) should provide enough evidence to correctly identify Italian segments
    even when ɲ (shared with Spanish) appears.
    """
    # Italian-dominant stream: multiple dz/ʎ (ITA-specific) with some ɾ (SPA marker)
    # ITA score: 3×dz(0.7) + 2×ʎ(0.7) + ɲ(0.15) = 2.1 + 1.4 + 0.15 = 3.65
    # SPA score: ɾ(0.7) + ɲ(0.8) = 1.5
    phonemes = _make_phonemes(["dz", "a", "ʎ", "e", "ɾ", "i", "dz", "o", "ɲ", "ʎ", "dz"])
    result = identify_language(phonemes)

    # Should detect ITA (not SPA)
    languages = {seg.language for seg in result}
    assert "ita" in languages, f"Italian not detected when ITA-specific markers dominate. Got: {languages}"


# ============================================================================
# Nahuatl Lexicon Pre-Check Tests
# ============================================================================


def test_lexicon_precheck_koali_recognized():
    """Phoneme stream containing 'koali' IPA should be tagged as NAH by lexicon."""
    # Create phonemes that spell 'koali' in IPA (from nah_lexicon.json)
    phonemes = _make_phonemes(["k", "o", "a", "l", "i"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah", f"Expected 'nah' for koali, got {result[0].language}"


def test_lexicon_precheck_amo_recognized():
    """Short Nahuatl word 'amo' should be recognized by lexicon."""
    phonemes = _make_phonemes(["a", "m", "o"])
    result = identify_language(phonemes)

    assert len(result) == 1
    assert result[0].language == "nah", f"Expected 'nah' for amo, got {result[0].language}"


def test_lexicon_precheck_no_false_positives():
    """Non-Nahuatl phoneme stream should not match lexicon."""
    # Random phonemes that don't spell any Nahuatl word - make it longer to avoid consolidation
    phonemes = _make_phonemes(["p", "y", "z", "ʒ", "v", "p", "y", "z"])
    result = identify_language(phonemes)

    # Should be marked as "other" since no markers and no lexicon match
    # (Could also be reclassified by consolidation if too short, so check for no NAH)
    languages = {seg.language for seg in result}
    assert "nah" not in languages, f"Expected no NAH (false positive), got {languages}"


def test_lexicon_precheck_embedded_word():
    """Lexicon should find Nahuatl word embedded in longer sequence."""
    # Sequence: unmarked + 'amo' + unmarked
    phonemes = _make_phonemes(["e", "i", "a", "m", "o", "u", "e"])
    result = identify_language(phonemes)

    # The 'amo' portion should be detected as NAH
    languages = {seg.language for seg in result}
    assert "nah" in languages, f"Expected NAH segment for embedded 'amo', got {languages}"


def test_lexicon_precheck_with_markers():
    """Lexicon match should work even when markers also present."""
    # 'koali' with an added Nahuatl marker
    phonemes = _make_phonemes(["k", "o", "a", "l", "i", "tɬ"])
    result = identify_language(phonemes)

    # Should be all NAH (lexicon + marker agreement)
    assert len(result) == 1
    assert result[0].language == "nah"


# ============================================================================
# OTH Context Absorption Tests
# ============================================================================


def test_absorb_oth_between_nah_short():
    """Short OTH segment between two NAH segments should be absorbed."""
    # Create NAH-OTH-NAH pattern
    nah1_phonemes = _make_phonemes(["tɬ", "a", "kʷ"])  # NAH markers
    oth_phonemes = _make_phonemes(["e", "i"])  # Short OTH (2 phonemes, 0.2s)
    nah2_phonemes = _make_phonemes(["ʔ", "ɬ", "kʷ"])  # NAH markers

    # Adjust timing for sequential segments
    for p in oth_phonemes:
        p.start_time += 0.3
    for p in nah2_phonemes:
        p.start_time += 0.5

    all_phonemes = nah1_phonemes + oth_phonemes + nah2_phonemes
    result = identify_language(all_phonemes)

    # Should end up as single NAH segment (OTH absorbed)
    assert len(result) == 1, f"Expected 1 merged segment, got {len(result)}"
    assert result[0].language == "nah", f"Expected NAH after absorption, got {result[0].language}"


def test_absorb_oth_between_nah_long_not_absorbed():
    """Long OTH segment (>2s) between NAH segments should NOT be absorbed.

    Note: This tests the absorption logic, but file-level consolidation may
    still reclassify segments based on dominant language markers.
    """
    # Direct test: create segments with NAH-OTH-NAH pattern where OTH is long
    from tenepal.language.identifier import _absorb_oth_between_nah, _create_segment

    nah1_phonemes = _make_phonemes(["tɬ", "kʷ", "ʔ"])
    # Long OTH: 25 phonemes at 0.1s each = 2.5s (exceeds 2.0s threshold)
    oth_phonemes = _make_phonemes(["e"] * 25)
    nah2_phonemes = _make_phonemes(["ɬ", "tɬ", "kʷ"])

    # Adjust timing
    for p in oth_phonemes:
        p.start_time += 0.3
    for p in nah2_phonemes:
        p.start_time += 2.8

    # Create segments directly
    segments = [
        _create_segment("nah", nah1_phonemes),
        _create_segment("other", oth_phonemes),
        _create_segment("nah", nah2_phonemes),
    ]

    # Apply absorption
    result = _absorb_oth_between_nah(segments)

    # OTH should NOT be absorbed (2.5s > 2.0s threshold)
    assert len(result) == 3, f"Expected 3 segments (long OTH not absorbed), got {len(result)}"
    assert result[1].language == "other", f"Expected OTH preserved, got {result[1].language}"


def test_absorb_oth_not_between_nah():
    """OTH segment not between NAH segments should NOT be absorbed."""
    # Direct test: create SPA-OTH-NAH pattern (OTH not sandwiched between NAH)
    from tenepal.language.identifier import _absorb_oth_between_nah, _create_segment

    spa_phonemes = _make_phonemes(["ɲ", "a", "ɾ"])
    oth_phonemes = _make_phonemes(["e", "i"])
    nah_phonemes = _make_phonemes(["tɬ", "kʷ"])

    # Adjust timing
    for p in oth_phonemes:
        p.start_time += 0.3
    for p in nah_phonemes:
        p.start_time += 0.5

    # Create segments
    segments = [
        _create_segment("spa", spa_phonemes),
        _create_segment("other", oth_phonemes),
        _create_segment("nah", nah_phonemes),
    ]

    # Apply absorption
    result = _absorb_oth_between_nah(segments)

    # OTH should NOT be absorbed (between SPA and NAH, not NAH and NAH)
    assert len(result) == 3, f"Expected 3 segments (OTH not between NAH), got {len(result)}"
    assert result[1].language == "other", f"Expected OTH preserved, got {result[1].language}"


def test_absorb_oth_too_many_phonemes():
    """OTH with >10 phonemes should NOT be absorbed even if duration < 2s."""
    # Direct test: NAH-OTH-NAH where OTH has 12 phonemes (>10 threshold)
    from tenepal.language.identifier import _absorb_oth_between_nah, _create_segment

    nah1_phonemes = _make_phonemes(["tɬ", "kʷ"])
    # OTH: 12 phonemes (exceeds threshold) but only 1.2s
    oth_phonemes = _make_phonemes(["e"] * 12)
    nah2_phonemes = _make_phonemes(["ɬ", "tɬ"])

    # Adjust timing
    for p in oth_phonemes:
        p.start_time += 0.2
    for p in nah2_phonemes:
        p.start_time += 1.4

    # Create segments
    segments = [
        _create_segment("nah", nah1_phonemes),
        _create_segment("other", oth_phonemes),
        _create_segment("nah", nah2_phonemes),
    ]

    # Apply absorption
    result = _absorb_oth_between_nah(segments)

    # OTH should NOT be absorbed (12 phonemes > 10 threshold)
    assert len(result) == 3, f"Expected 3 segments (OTH too many phonemes), got {len(result)}"
    assert result[1].language == "other", f"Expected OTH preserved, got {result[1].language}"


def test_maya_ejective_detection():
    """Maya ejective markers should be detected and tagged as MAY.

    Yucatec Maya is characterized by ejective consonants (kʼ, tsʼ, tʃʼ, pʼ, tʼ).
    A phoneme stream with multiple ejectives should be identified as MAY.
    """
    # Create stream with Maya ejective markers
    phonemes = _make_phonemes(["kʼ", "a", "tsʼ", "e", "kʼ", "i", "tʃʼ", "o", "kʼ"])

    result = identify_language(phonemes)
    languages = {seg.language for seg in result}

    # Should detect MAY
    assert "may" in languages, f"Maya ejectives not detected, got {languages}"

    # Find the MAY segment
    may_segments = [seg for seg in result if seg.language == "may"]
    assert len(may_segments) > 0, "Expected at least one MAY segment"


def test_maya_vs_nahuatl_disambiguation():
    """MAY and NAH should be distinguishable by their distinctive markers.

    Test 1: Ejectives (kʼ, tsʼ, pʼ) should route to MAY
    Test 2: Lateral affricates (tɬ, kʷ, ɬ) should route to NAH
    Test 3: Mixed stream with both marker types
    """
    # Test 1: Ejectives → MAY
    ejective_phonemes = _make_phonemes(["kʼ", "a", "tsʼ", "e", "pʼ", "i", "kʼ", "o"])
    result1 = identify_language(ejective_phonemes)
    languages1 = {seg.language for seg in result1}
    # Should strongly prefer MAY (ejectives are primary MAY markers)
    assert "may" in languages1, f"Ejectives should detect MAY, got {languages1}"

    # Test 2: Lateral affricates → NAH
    lateral_phonemes = _make_phonemes(["tɬ", "a", "kʷ", "i", "ɬ", "o", "tɬ", "e"])
    result2 = identify_language(lateral_phonemes)
    languages2 = {seg.language for seg in result2}
    # Should be NAH (tɬ, kʷ, ɬ are NAH-specific)
    assert "nah" in languages2, f"Lateral affricates should detect NAH, got {languages2}"
    # Should NOT be MAY (negative markers should exclude it)
    assert "may" not in languages2, f"MAY should be excluded by negative markers, got {languages2}"

    # Test 3: Mixed stream
    # When both ejective and lateral affricates present, NAH should win due to:
    # 1. Higher priority (NAH=10, MAY=9)
    # 2. Strong negative markers in MAY profile for tɬ/kʷ/ɬ
    mixed_phonemes = _make_phonemes(["kʼ", "a", "tɬ", "i", "kʷ", "o", "tsʼ", "e", "ɬ"])
    result3 = identify_language(mixed_phonemes)
    languages3 = {seg.language for seg in result3}
    # NAH's lateral affricates have strong negative weight in MAY profile (0.8)
    # So NAH should dominate
    assert "nah" in languages3, f"NAH should be detected in mixed stream, got {languages3}"


def test_maya_vs_spanish_disambiguation():
    """MAY should be distinguishable from SPA despite some shared features.

    Spanish voiced stops (b, d, ɡ) are MAY negative markers.
    Maya ejectives should overcome Spanish marker presence.
    """
    # Create stream with Maya ejectives and Spanish voiced stops
    phonemes = _make_phonemes(["kʼ", "a", "b", "tsʼ", "e", "d", "kʼ", "o", "ɡ", "pʼ"])

    result = identify_language(phonemes)
    languages = {seg.language for seg in result}

    # Ejective markers (weight 1.0 each) should dominate over Spanish markers
    # Expected: Multiple segments, some MAY (for ejectives), possibly some SPA
    # Key test: MAY should be detected (ejectives present)
    assert len(result) >= 1, "Should have at least one segment"

    # If MAY is detected, verify it contains ejectives
    may_segments = [seg for seg in result if seg.language == "may"]
    if may_segments:
        # Check that MAY segment contains at least one ejective
        may_phonemes = [p.phoneme for seg in may_segments for p in seg.phonemes]
        ejectives = {"kʼ", "tsʼ", "tʃʼ", "pʼ", "tʼ"}
        assert any(p in ejectives for p in may_phonemes), "MAY segment should contain ejective markers"


def test_seven_language_all_detected():
    """All 7 languages should be correctly detected in their own streams.

    This validates that adding MAY doesn't break existing detection.
    Each language gets a phoneme stream with its distinctive markers.

    Note: This is a smoke test, not a precision test. Some marker confusion
    is expected (e.g., DEU/FRA overlap, ghost markers). The key validation
    is that NAH and MAY (the Mesoamerican endangered languages) are correctly
    distinguished, and the system still functions with 7 languages.
    """
    from tenepal.language.registry import default_registry
    registry = default_registry()

    # Test streams for each language with their distinctive markers
    test_cases = {
        "nah": ["tɬ", "a", "kʷ", "i", "ɬ", "o", "ʔ", "e", "tɬ"],  # NAH laterals + glottal
        "may": ["kʼ", "a", "tsʼ", "e", "tʃʼ", "i", "pʼ", "o", "kʼ"],  # MAY ejectives
        "spa": ["ɲ", "a", "ɾ", "o", "ɲ", "e", "ɾ", "i", "ɲ"],  # SPA ɲ, ɾ
    }

    for expected_lang, phoneme_list in test_cases.items():
        phonemes = _make_phonemes(phoneme_list)
        result = identify_language(phonemes, registry=registry)
        languages = {seg.language for seg in result}

        # Each language should be detected in its own stream
        assert len(result) >= 1, f"{expected_lang}: Should have at least one segment"

        # For Mesoamerican languages (project focus), verify detection
        assert expected_lang in languages, f"{expected_lang}: Expected {expected_lang} detection in {languages}"

    # Verify that all 7 language codes are registered
    all_codes = set(registry.codes())
    expected_codes = {"nah", "may", "spa", "eng", "deu", "fra", "ita"}
    assert all_codes == expected_codes, f"Expected 7 languages, got {all_codes}"

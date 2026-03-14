"""Tests for phoneme frequency analysis and confusion matrix."""

import pytest

from tenepal.language.analyzer import (
    analyze_phonemes,
    build_confusion_matrix,
    format_analysis,
    PhonemeAnalysis,
    ProfileHits,
    ConfusionMatrix,
)
from tenepal.language.identifier import _tag_phonemes
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


def test_analyze_empty_input():
    """Empty phoneme list returns PhonemeAnalysis with zero counts."""
    result = analyze_phonemes([])

    assert result.total_phonemes == 0
    assert result.unique_phonemes == 0
    assert result.frequencies == {}
    assert result.profile_hits == {}


def test_analyze_frequency_counting():
    """Phoneme frequency counting works correctly."""
    phonemes = _make_phonemes(["a", "b", "a", "c", "a", "b"])
    result = analyze_phonemes(phonemes)

    assert result.total_phonemes == 6
    assert result.unique_phonemes == 3
    assert result.frequencies["a"] == 3
    assert result.frequencies["b"] == 2
    assert result.frequencies["c"] == 1

    # Check sorted by frequency descending
    freq_list = list(result.frequencies.items())
    assert freq_list[0] == ("a", 3)
    assert freq_list[1] == ("b", 2)
    assert freq_list[2] == ("c", 1)


def test_analyze_nahuatl_markers_detected():
    """Nahuatl marker phonemes are correctly identified in profile_hits."""
    # Use multiple Nahuatl markers
    phonemes = _make_phonemes(["tɬ", "a", "kʷ", "i", "ʔ", "e", "tɬ"])
    result = analyze_phonemes(phonemes)

    assert "nah" in result.profile_hits
    nah_hits = result.profile_hits["nah"]

    assert nah_hits.language_code == "nah"
    assert nah_hits.language_name == "Nahuatl"
    assert "tɬ" in nah_hits.markers_found
    assert nah_hits.markers_found["tɬ"] == 2  # appears twice
    assert "kʷ" in nah_hits.markers_found
    assert nah_hits.markers_found["kʷ"] == 1
    assert "ʔ" in nah_hits.markers_found
    assert nah_hits.markers_found["ʔ"] == 1


def test_analyze_ghost_markers_identified():
    """Markers that never appear are listed in markers_missing."""
    # Use phonemes with NO English markers (ghost markers have been removed from profiles)
    phonemes = _make_phonemes(["a", "b", "d", "ɡ", "o"])
    result = analyze_phonemes(phonemes)

    assert "eng" in result.profile_hits
    eng_hits = result.profile_hits["eng"]

    # English markers (æ, ʊ, ɪ) should all be missing since we used different phonemes
    # Note: ɝ and ɚ were removed as ghost markers
    assert "æ" in eng_hits.markers_missing
    assert "ʊ" in eng_hits.markers_missing
    assert "ɪ" in eng_hits.markers_missing
    assert len(eng_hits.markers_found) == 0


def test_confusion_matrix_shared_markers():
    """Shared markers are correctly identified in confusion matrix."""
    # Create custom registry where two profiles share a marker
    custom_registry = LanguageRegistry()

    lang_a = LanguageProfile(
        code="la",
        name="Language A",
        family="Test",
        marker_phonemes={"x", "y", "z"},
        absent_phonemes=set()
    )
    lang_b = LanguageProfile(
        code="lb",
        name="Language B",
        family="Test",
        marker_phonemes={"y", "z", "w"},  # y and z are shared with lang_a
        absent_phonemes=set()
    )

    custom_registry.register(lang_a)
    custom_registry.register(lang_b)

    # Use phonemes containing shared markers
    phonemes = _make_phonemes(["a", "y", "z", "i"])
    analysis = analyze_phonemes(phonemes, registry=custom_registry)
    confusion = build_confusion_matrix(analysis)

    # y and z should be in shared_markers
    assert "y" in confusion.shared_markers
    assert "z" in confusion.shared_markers
    assert set(confusion.shared_markers["y"]) == {"la", "lb"}
    assert set(confusion.shared_markers["z"]) == {"la", "lb"}

    # x and w are not shared
    assert "x" not in confusion.shared_markers
    assert "w" not in confusion.shared_markers


def test_confusion_matrix_ghost_markers():
    """Ghost markers dict correctly lists markers absent from input."""
    phonemes = _make_phonemes(["a", "e", "i", "o", "u"])  # Only vowels, no markers
    analysis = analyze_phonemes(phonemes)
    confusion = build_confusion_matrix(analysis)

    # All languages should have ghost markers
    assert "nah" in confusion.ghost_markers
    assert "spa" in confusion.ghost_markers
    assert "eng" in confusion.ghost_markers
    assert "deu" in confusion.ghost_markers

    # Nahuatl should have tɬ, kʷ, ʔ, etc. as ghost markers
    assert "tɬ" in confusion.ghost_markers["nah"]
    assert "kʷ" in confusion.ghost_markers["nah"]
    assert "ʔ" in confusion.ghost_markers["nah"]


def test_format_analysis_contains_sections():
    """format_analysis output contains all required section headers."""
    phonemes = _make_phonemes(["tɬ", "a", "kʷ", "b", "d"])
    analysis = analyze_phonemes(phonemes)
    confusion = build_confusion_matrix(analysis)

    report = format_analysis(analysis, confusion)

    assert "PHONEME FREQUENCY ANALYSIS" in report
    assert "PHONEME FREQUENCIES" in report
    assert "LANGUAGE PROFILE HITS" in report
    assert "CONFUSION MATRIX" in report
    assert "GHOST MARKERS" in report


def test_analyze_detection_count_matches_identifier():
    """Detection count in ProfileHits matches actual _tag_phonemes behavior."""
    from tenepal.language.registry import default_registry

    # Create known phoneme sequence with Nahuatl markers
    phonemes = _make_phonemes(["tɬ", "a", "kʷ", "i", "ʔ", "e", "ts"])

    # Get analysis detection_count
    analysis = analyze_phonemes(phonemes)
    nah_detection_count = analysis.profile_hits["nah"].detection_count

    # Get actual tagged count from _tag_phonemes
    registry = default_registry()
    tagged = _tag_phonemes(phonemes, registry)
    actual_nah_count = sum(1 for _, lang in tagged if lang == "nah")

    # They should match
    assert nah_detection_count == actual_nah_count


def test_analyze_sequences_found():
    """Bigram sequences are correctly identified in sequences_found."""
    # Create phonemes with k + w sequence (Nahuatl marker)
    phonemes = _make_phonemes(["a", "k", "w", "a", "i"])
    result = analyze_phonemes(phonemes)

    nah_hits = result.profile_hits["nah"]

    # k + w sequence should be found
    assert ("k", "w") in nah_hits.sequences_found


def test_analyze_sequences_missing():
    """Bigram sequences that never appear are in sequences_missing."""
    # Create phonemes with NO Nahuatl sequences
    phonemes = _make_phonemes(["a", "b", "d", "ɡ", "o"])
    result = analyze_phonemes(phonemes)

    nah_hits = result.profile_hits["nah"]

    # Nahuatl sequences should all be missing
    assert ("k", "w") in nah_hits.sequences_missing
    assert ("t", "l") in nah_hits.sequences_missing
    assert ("t", "ɬ") in nah_hits.sequences_missing


def test_analyze_with_modifiers_in_sequences():
    """Sequences with IPA modifiers are correctly matched (modifiers stripped)."""
    # t + lː should match ("t", "l") sequence
    phonemes = _make_phonemes(["a", "t", "lː", "a"])
    result = analyze_phonemes(phonemes)

    nah_hits = result.profile_hits["nah"]

    # ("t", "l") sequence should be found (ː stripped)
    assert ("t", "l") in nah_hits.sequences_found


def test_confusion_matrix_false_positive_candidates():
    """False positive candidates are identified when markers are shared."""
    # Create custom registry where lang_b shares markers with lang_a
    custom_registry = LanguageRegistry()

    lang_a = LanguageProfile(
        code="la",
        name="Language A",
        family="Test",
        marker_phonemes={"x", "y"},
        absent_phonemes=set(),
        priority=10  # Higher priority
    )
    lang_b = LanguageProfile(
        code="lb",
        name="Language B",
        family="Test",
        marker_phonemes={"y", "z"},  # y is shared with lang_a
        absent_phonemes=set(),
        priority=1  # Lower priority
    )

    custom_registry.register(lang_a)
    custom_registry.register(lang_b)

    # Use phonemes containing shared marker y
    phonemes = _make_phonemes(["a", "y", "i"])
    analysis = analyze_phonemes(phonemes, registry=custom_registry)
    confusion = build_confusion_matrix(analysis)

    # Both languages should have y as false positive candidate
    assert "la" in confusion.false_positive_candidates
    assert "lb" in confusion.false_positive_candidates
    assert "y" in confusion.false_positive_candidates["la"]
    assert "y" in confusion.false_positive_candidates["lb"]


def test_analyze_multiple_profiles():
    """Analysis correctly processes all profiles in registry."""
    # Use real markers (ghost markers have been removed from profiles)
    phonemes = _make_phonemes(["tɬ", "æ", "ʁ", "b"])
    result = analyze_phonemes(phonemes)

    # Should have hits for all default profiles
    assert "nah" in result.profile_hits
    assert "spa" in result.profile_hits
    assert "eng" in result.profile_hits
    assert "deu" in result.profile_hits

    # Each should detect their respective markers
    assert result.profile_hits["nah"].detection_count > 0  # tɬ
    assert result.profile_hits["eng"].detection_count > 0  # æ
    assert result.profile_hits["deu"].detection_count > 0  # ʁ
    assert result.profile_hits["spa"].detection_count > 0  # b


def test_format_analysis_percentage_calculation():
    """format_analysis correctly calculates and formats percentages."""
    phonemes = _make_phonemes(["a", "a", "b"])  # 2/3 a, 1/3 b
    analysis = analyze_phonemes(phonemes)
    confusion = build_confusion_matrix(analysis)

    report = format_analysis(analysis, confusion)

    # Check that percentages are present (should be ~66.67% and ~33.33%)
    assert "66.67%" in report or "66.66%" in report  # a
    assert "33.33%" in report or "33.34%" in report  # b


def test_analyze_zero_detection_count():
    """Profiles with no matching markers have detection_count of 0."""
    # Use only vowels - no markers for any language
    phonemes = _make_phonemes(["a", "e", "i", "o", "u"])
    result = analyze_phonemes(phonemes)

    # All profiles should have detection_count of 0
    for hits in result.profile_hits.values():
        assert hits.detection_count == 0


def test_analyze_trigrams_found():
    """Trigram sequences are correctly identified in trigrams_found."""
    # DEU has trigram (ʃ, t, ʁ)
    phonemes = _make_phonemes(["a", "ʃ", "t", "ʁ", "a"])
    result = analyze_phonemes(phonemes)

    deu_hits = result.profile_hits["deu"]
    assert ("ʃ", "t", "ʁ") in deu_hits.trigrams_found


def test_analyze_trigrams_missing():
    """Trigram sequences that never appear are in trigrams_missing."""
    # No DEU trigrams in input
    phonemes = _make_phonemes(["a", "b", "d", "ɡ", "o"])
    result = analyze_phonemes(phonemes)

    deu_hits = result.profile_hits["deu"]
    # All DEU trigrams should be missing
    assert len(deu_hits.trigrams_missing) == 5
    assert ("ʃ", "t", "ʁ") in deu_hits.trigrams_missing


def test_profile_hits_trigram_fields_exist():
    """ProfileHits should have trigrams_found and trigrams_missing fields."""
    phonemes = _make_phonemes(["a", "e", "i"])
    result = analyze_phonemes(phonemes)

    for hits in result.profile_hits.values():
        assert hasattr(hits, "trigrams_found")
        assert hasattr(hits, "trigrams_missing")
        assert isinstance(hits.trigrams_found, list)
        assert isinstance(hits.trigrams_missing, list)

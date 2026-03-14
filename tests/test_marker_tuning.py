"""Tests for tuned language profiles after marker weight optimization."""

import pytest

from tenepal.language.identifier import identify_language
from tenepal.language.registry import default_registry
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
        segments.append(
            PhonemeSegment(phoneme=phoneme, start_time=i * 0.1, duration=0.1)
        )
    return segments


def test_nah_audio_zero_false_positives():
    """Tuned profiles produce zero false ENG/DEU/SPA on NAH-dominant phoneme streams.

    This test simulates a Nahuatl-dominant phoneme pattern mixed with phonemes
    that caused false positives in Phase 19 before tuning. The tuned profiles
    with Phase G consolidation should correctly classify the entire stream as NAH.

    Phase 19 false positive counts from moctezuma_test.wav (224 phonemes):
    - ENG false positives: 22 (ɪ:20, ʊ:2)
    - DEU false positives: 7 (x:6, ʁ:1)
    - SPA false positives: 6 (ɾ:3, d:1, ɲ:1, ɡ:1)

    The test uses a 128-phoneme sequence (exceeds Phase G's 100-phoneme minimum)
    with strong NAH markers (tɬ, kʷ, ʔ, ʃ) mixed with the false positive phonemes.
    Phase G whole-file consolidation should recognize NAH dominance and consolidate.
    """
    # Create a NAH-dominant pattern with false positive phonemes mixed in
    # Repeated to exceed Phase G's 100-phoneme consolidation threshold
    base_pattern = [
        "tɬ",
        "a",
        "kʷ",
        "i",
        "ʔ",
        "ɪ",
        "ɪ",
        "ɪ",  # NAH markers + ENG false positives
        "ʃ",
        "o",
        "x",
        "x",
        "ɾ",
        "a",
        "ɲ",
        "d",  # NAH marker + DEU/SPA false positives
    ]
    # Repeat pattern 8 times = 128 phonemes (exceeds Phase G threshold)
    phonemes = _make_phonemes(base_pattern * 8)

    result = identify_language(phonemes)
    languages = {seg.language for seg in result}
    false_positives = languages - {"nah", "other"}

    # Phase G should consolidate to NAH only (no ENG/DEU/SPA segments)
    assert not false_positives, (
        f"False positives detected: {false_positives}. Phase G should consolidate to NAH only."
    )


def test_eng_markers_include_explicit_english_cues():
    """Verify English profile keeps explicit ENG cues used for NAH/ENG separation.

    New tuning intentionally enables strong English cues to catch long
    English outro segments that were previously tagged as NAH.
    """
    registry = default_registry()
    eng = registry.get("eng")
    required_markers = ["θ", "ð", "ɹ", "ʌ", "dʒ"]

    for marker in required_markers:
        assert marker in eng.marker_phonemes, (
            f"ENG marker {marker} missing from profile"
        )

    # Keep at least one previously problematic marker excluded.
    assert "ŋ" not in eng.marker_phonemes


def test_deu_ghost_markers_not_in_profile():
    """Verify ghost markers are absent from German profile.

    Known DEU ghost markers from Phase 19 analysis:
    - ç (voiceless palatal fricative)
    - ø (close-mid front rounded vowel)
    - ʏ (near-close near-front rounded vowel)
    """
    registry = default_registry()
    deu = registry.get("deu")
    ghost_markers = ["ç", "ø", "ʏ"]

    for ghost in ghost_markers:
        assert ghost not in deu.marker_phonemes, (
            f"Ghost marker {ghost} still in DEU profile (should be removed)"
        )


def test_spa_ghost_markers_not_in_profile():
    """Verify ghost markers are absent from Spanish profile.

    Known SPA ghost markers from Phase 19 analysis:
    - β (voiced bilabial fricative)
    - r (alveolar trill)
    - ɣ (voiced velar fricative)
    """
    registry = default_registry()
    spa = registry.get("spa")
    ghost_markers = ["β", "r", "ɣ"]

    for ghost in ghost_markers:
        assert ghost not in spa.marker_phonemes, (
            f"Ghost marker {ghost} still in SPA profile (should be removed)"
        )


def test_spa_voiced_plosives_low_weight():
    """Verify b, d, ɡ have low weights in Spanish profile.

    Per user decision in Phase 20: Spanish voiced plosives (b, d, ɡ) should
    have very low weights because they appear frequently in NAH audio despite
    not being native Nahuatl phonemes.

    Expected: weight <= 0.5 for all voiced plosives
    """
    registry = default_registry()
    spa = registry.get("spa")

    for plosive in ["b", "d", "ɡ"]:
        if plosive in spa.marker_phonemes:
            weight = spa.marker_weights.get(plosive, 1.0)
            assert weight <= 0.5, (
                f"SPA {plosive} weight {weight} too high (expected <= 0.5 for low-confidence marker)"
            )


def test_profile_thresholds():
    """Verify current per-language thresholds used by tuned profiles."""
    registry = default_registry()
    spa = registry.get("spa")
    eng = registry.get("eng")
    deu = registry.get("deu")

    # Per-segment thresholds calibrated for better language separation
    # ENG lowered to 1.0 to detect English segments with few markers
    # DEU lowered to 3.0 to allow detection with 3-4 'x' markers
    assert spa.threshold == 2.25, f"SPA threshold should be 2.25, got {spa.threshold}"
    assert eng.threshold == 1.0, f"ENG threshold should be 1.0, got {eng.threshold}"
    assert deu.threshold == 3.0, f"DEU threshold should be 3.0, got {deu.threshold}"


def test_nah_tl_safety_net_present():
    """Verify NAH tɬ marker is present despite being a ghost marker.

    The tɬ phoneme never appeared in real Allosaurus output from
    moctezuma_test.wav, making it technically a ghost marker. However,
    it was intentionally kept as a "safety net" marker because:

    1. It's a distinctive Nahuatl phoneme in linguistic theory
    2. It may appear with better Allosaurus models or other NAH audio
    3. Its presence doesn't cause false positives in other languages

    This test documents the deliberate exception to ghost marker elimination.
    """
    registry = default_registry()
    nah = registry.get("nah")

    assert "tɬ" in nah.marker_phonemes, "NAH tɬ safety net marker missing"


def test_profiles_have_marker_weights():
    """Verify all profiles have tuned marker weights.

    After Plan 20-02 tuning, all ENG/DEU/SPA profiles should have explicit
    marker_weights dictionaries with computed weights based on real audio
    frequency analysis.

    This test verifies that tuning was actually applied to the profiles.
    """
    registry = default_registry()

    for code in ["eng", "deu", "spa", "nah"]:
        profile = registry.get(code)
        assert profile.marker_weights, (
            f"{code} has no marker weights (tuning not applied?)"
        )

        # Verify weights are in reasonable range [0.0, 2.0]
        for phoneme, weight in profile.marker_weights.items():
            assert 0.0 <= weight <= 2.0, (
                f"{code} marker {phoneme} has unrealistic weight {weight}"
            )


def test_profiles_have_expected_thresholds():
    """Verify tuned profile thresholds used by current language-ID config."""
    registry = default_registry()

    expected = {"eng": 1.0, "deu": 3.0, "spa": 2.25, "nah": 0.0}
    for code, expected_threshold in expected.items():
        profile = registry.get(code)
        assert profile.threshold == expected_threshold, (
            f"{code} threshold {profile.threshold} should be {expected_threshold}"
        )


def test_nah_negative_markers_present():
    """Verify NAH profile has negative markers for voiced plosives/fricatives.

    The NAH profile includes negative_markers to penalize detection when
    phonemes that don't exist in Nahuatl phonology appear in the audio.

    Expected negative markers: b, d, ɡ, f, v, ʒ (all with penalty weight ~0.5)
    """
    registry = default_registry()
    nah = registry.get("nah")

    assert hasattr(nah, "negative_markers"), (
        "NAH profile missing negative_markers attribute"
    )

    expected_negatives = ["b", "d", "ɡ", "f", "v", "ʒ"]
    for phoneme in expected_negatives:
        assert phoneme in nah.negative_markers, f"NAH negative marker {phoneme} missing"
        penalty = nah.negative_markers[phoneme]
        assert 0.0 < penalty <= 1.0, (
            f"NAH negative marker {phoneme} penalty {penalty} out of range (0.0, 1.0]"
        )


def test_deu_has_bigram_and_trigram_markers():
    """Verify DEU profile has both bigram and trigram markers.

    German was the primary target for n-gram marker discovery in Plan 20-02.
    The tuned DEU profile should have:
    - marker_sequences (bigrams)
    - marker_trigrams (trigrams)
    - sequence_weights and trigram_weights

    This test verifies that n-gram discovery was successfully applied.
    """
    registry = default_registry()
    deu = registry.get("deu")

    assert deu.marker_sequences, "DEU profile missing bigram marker_sequences"
    assert hasattr(deu, "marker_trigrams"), (
        "DEU profile missing marker_trigrams attribute"
    )
    assert deu.marker_trigrams, "DEU profile has empty marker_trigrams"

    assert deu.sequence_weights, "DEU profile missing sequence_weights"
    assert hasattr(deu, "trigram_weights"), (
        "DEU profile missing trigram_weights attribute"
    )
    assert deu.trigram_weights, "DEU profile has empty trigram_weights"


def test_tuned_weights_reflect_specificity():
    """Verify marker weights reflect language specificity.

    High-specificity markers (appear frequently in target language, rarely
    in NAH) should have higher weights than low-specificity markers.

    Examples:
    - ENG ɪ: low weight (appears 20x in NAH audio)
    - ENG æ: higher weight (doesn't appear in NAH audio)
    - DEU x: high weight (distinctive German marker)
    - SPA d: very low weight (appears in NAH audio)
    """
    registry = default_registry()
    eng = registry.get("eng")
    deu = registry.get("deu")
    spa = registry.get("spa")

    # ENG: ɪ should have lower weight than æ
    if "ɪ" in eng.marker_weights and "æ" in eng.marker_weights:
        assert eng.marker_weights["ɪ"] < eng.marker_weights["æ"], (
            "ENG ɪ (frequent NAH false positive) should have lower weight than æ"
        )

    # DEU: x should have high weight (distinctive marker)
    if "x" in deu.marker_weights:
        assert deu.marker_weights["x"] >= 0.8, (
            "DEU x should have high weight (distinctive marker)"
        )

    # SPA: d should have very low weight (appears in NAH audio)
    if "d" in spa.marker_weights:
        assert spa.marker_weights["d"] <= 0.2, (
            "SPA d should have very low weight (frequent NAH false positive)"
        )


def test_fra_profile_loaded_with_markers():
    """Verify FRA profile exists in default_registry with proper structure.

    French profile should be loaded from fra.json with marker phonemes,
    weights, threshold, and metadata properly configured.
    """
    registry = default_registry()
    fra = registry.get("fra")

    assert fra is not None, "FRA profile not found in default registry"
    assert fra.code == "fra", f"Expected code 'fra', got '{fra.code}'"
    assert fra.name == "French", f"Expected name 'French', got '{fra.name}'"
    assert fra.family == "Indo-European", (
        f"Expected family 'Indo-European', got '{fra.family}'"
    )
    assert len(fra.marker_phonemes) >= 3, "FRA should have at least 3 marker phonemes"
    assert fra.marker_weights, "FRA should have marker weights configured"
    assert fra.threshold == 1.40, f"FRA threshold should be 1.40, got {fra.threshold}"


def test_ita_profile_loaded_with_markers():
    """Verify ITA profile exists in default_registry with proper structure.

    Italian profile should be loaded from ita.json with marker phonemes,
    weights, threshold, and metadata properly configured.
    """
    registry = default_registry()
    ita = registry.get("ita")

    assert ita is not None, "ITA profile not found in default registry"
    assert ita.code == "ita", f"Expected code 'ita', got '{ita.code}'"
    assert ita.name == "Italian", f"Expected name 'Italian', got '{ita.name}'"
    assert ita.family == "Indo-European", (
        f"Expected family 'Indo-European', got '{ita.family}'"
    )
    assert len(ita.marker_phonemes) >= 3, "ITA should have at least 3 marker phonemes"
    assert ita.marker_weights, "ITA should have marker weights configured"
    assert ita.threshold == 0.25, f"ITA threshold should be 0.25, got {ita.threshold}"


def test_six_language_priority_ordering():
    """Verify 6-language priority ordering is correct.

    NOTE: This test was written when only 6 languages existed (Phase 21).
    Phase 32 added MAY (Yucatec Maya) as a 7th language. This test now
    validates the original 6-language ordering still holds.

    NAH should have highest priority (endangered language preservation).
    DEU > ENG > SPA, FRA, ITA based on marker distinctiveness.
    All 6 original languages should be registered and have valid priority values.
    """
    registry = default_registry()

    # Verify original 6 codes present (MAY added in Phase 32)
    codes = set(registry.codes())
    original_codes = {"nah", "spa", "eng", "deu", "fra", "ita"}
    assert original_codes.issubset(codes), (
        f"Expected original 6 languages {original_codes} to be present in {codes}"
    )

    # Get profiles
    nah = registry.get("nah")
    deu = registry.get("deu")
    eng = registry.get("eng")
    spa = registry.get("spa")
    fra = registry.get("fra")
    ita = registry.get("ita")

    # NAH has highest priority (endangered language)
    assert nah.priority == 10, f"NAH priority should be 10, got {nah.priority}"

    # DEU > ENG priority
    assert deu.priority > eng.priority, (
        f"DEU priority ({deu.priority}) should be > ENG ({eng.priority})"
    )

    # ENG > SPA priority
    assert eng.priority > spa.priority, (
        f"ENG priority ({eng.priority}) should be > SPA ({spa.priority})"
    )

    # FRA and ITA have valid priorities (> 0)
    assert fra.priority > 0, f"FRA priority should be > 0, got {fra.priority}"
    assert ita.priority > 0, f"ITA priority should be > 0, got {ita.priority}"


def test_fra_markers_need_validation():
    """Document which FRA markers need audio validation.

    Known DEU ghost markers (ç, ø, ʏ) should NOT be in FRA profile.
    However, ø IS kept for FRA because it's a real French phoneme
    (as in "peu"), even though it was a ghost for German.

    This test documents that ø may still need validation with real
    French audio to confirm Allosaurus actually produces it for French.
    """
    registry = default_registry()
    fra = registry.get("fra")

    # DEU ghost markers that should NOT be in FRA
    # (except ø which IS a French phoneme, though possibly also a ghost)
    assert "ç" not in fra.marker_phonemes, "ç (DEU ghost) should not be in FRA profile"
    assert "ʏ" not in fra.marker_phonemes, "ʏ (DEU ghost) should not be in FRA profile"

    # Note: ø is kept in FRA profile because it's theoretically a French phoneme.
    # However, since it was a DEU ghost marker (never appeared in German audio),
    # it may also be a FRA ghost. This will be discovered during tune_profiles.py
    # validation with real French audio.


def test_no_regression_nah_with_six_languages():
    """Adding FRA/ITA must not create false positives on NAH audio.

    This is the critical regression test: the 6-language registry should
    produce zero false FRA/ITA segments on Nahuatl-dominant phoneme streams.
    Phase G consolidation should still work correctly with 6 languages.
    """
    # Create NAH-dominant pattern (same as test_nah_audio_zero_false_positives)
    base_pattern = [
        "tɬ",
        "a",
        "kʷ",
        "i",
        "ʔ",
        "ɪ",
        "ɪ",
        "ɪ",  # NAH markers + ENG false positives
        "ʃ",
        "o",
        "x",
        "x",
        "ɾ",
        "a",
        "ɲ",
        "d",  # NAH marker + DEU/SPA false positives
    ]
    # Repeat pattern 8 times = 128 phonemes (exceeds Phase G threshold)
    phonemes = _make_phonemes(base_pattern * 8)

    result = identify_language(phonemes)
    languages = {seg.language for seg in result}
    false_positives = languages - {"nah", "other"}

    # Should have zero false positives (no ENG/DEU/SPA/FRA/ITA segments)
    assert not false_positives, (
        f"False positives detected with 6 languages: {false_positives}. Phase G should consolidate to NAH only."
    )


def test_may_profile_loaded_with_markers():
    """Verify MAY profile exists in default_registry with proper structure.

    Yucatec Maya profile should be loaded from may.json with ejective markers,
    negative markers for NAH phonemes, and priority 9 (just below NAH's 10).
    """
    registry = default_registry()
    may = registry.get("may")

    assert may is not None, "MAY profile not found in default registry"
    assert may.code == "may", f"Expected code 'may', got '{may.code}'"
    assert may.name == "Yucatec Maya", f"Expected name 'Yucatec Maya', got '{may.name}'"
    assert may.family == "Mayan", f"Expected family 'Mayan', got '{may.family}'"
    assert may.priority == 9, f"MAY priority should be 9, got {may.priority}"

    # Verify ejective markers are present
    ejectives = {"kʼ", "tsʼ", "tʃʼ", "pʼ", "tʼ"}
    present_ejectives = ejectives & set(may.marker_phonemes)
    assert len(present_ejectives) >= 3, (
        f"MAY should have at least 3 ejective markers, found {present_ejectives}"
    )

    # Verify shared markers ʃ and ʔ are NOT in MAY markers (too generic)
    assert "ʃ" not in may.marker_phonemes, (
        "ʃ should be removed from MAY markers (too generic, causes false positives)"
    )
    assert "ʔ" not in may.marker_phonemes, (
        "ʔ should be removed from MAY markers (too generic, causes false positives)"
    )

    # Verify negative markers include NAH discriminators with strong penalties
    assert hasattr(may, "negative_markers"), (
        "MAY profile missing negative_markers attribute"
    )
    nah_discriminators = {"tɬ", "kʷ", "ɬ"}
    for phoneme in nah_discriminators:
        assert phoneme in may.negative_markers, (
            f"MAY should have negative marker for NAH phoneme {phoneme}"
        )
        penalty = may.negative_markers[phoneme]
        assert penalty >= 1.5, (
            f"MAY negative marker {phoneme} penalty {penalty} should be >= 1.5 for strong NAH discrimination"
        )

    assert may.threshold == 2.0, f"MAY threshold should be 2.0, got {may.threshold}"


def test_seven_language_priority_ordering():
    """Verify 7-language priority ordering is correct.

    NAH should have highest priority (endangered language preservation).
    MAY should be second highest (endangered Mesoamerican language).
    The priority chain: NAH(10) > MAY(9) > DEU(3) > ENG(2) > SPA(1) = FRA = ITA
    """
    registry = default_registry()

    # Verify all 7 codes present
    codes = set(registry.codes())
    expected_codes = {"nah", "may", "spa", "eng", "deu", "fra", "ita"}
    assert codes == expected_codes, f"Expected {expected_codes}, got {codes}"

    # Get profiles
    nah = registry.get("nah")
    may = registry.get("may")
    deu = registry.get("deu")
    eng = registry.get("eng")
    spa = registry.get("spa")
    fra = registry.get("fra")
    ita = registry.get("ita")

    # NAH has highest priority (endangered language)
    assert nah.priority == 10, f"NAH priority should be 10, got {nah.priority}"

    # MAY is second highest (endangered Mesoamerican language)
    assert may.priority == 9, f"MAY priority should be 9, got {may.priority}"

    # MAY > DEU priority
    assert may.priority > deu.priority, (
        f"MAY priority ({may.priority}) should be > DEU ({deu.priority})"
    )

    # DEU > ENG priority
    assert deu.priority > eng.priority, (
        f"DEU priority ({deu.priority}) should be > ENG ({eng.priority})"
    )

    # ENG > SPA priority
    assert eng.priority > spa.priority, (
        f"ENG priority ({eng.priority}) should be > SPA ({spa.priority})"
    )

    # FRA and ITA have valid priorities (> 0)
    assert fra.priority > 0, f"FRA priority should be > 0, got {fra.priority}"
    assert ita.priority > 0, f"ITA priority should be > 0, got {ita.priority}"


def test_no_regression_nah_with_seven_languages():
    """Adding MAY must not create false positives on NAH audio.

    This is the critical regression test: the 7-language registry should
    produce zero false MAY segments on Nahuatl-dominant phoneme streams.
    The MAY profile's negative markers for tɬ, kʷ, ɬ should prevent
    stealing NAH segments.
    """
    # Create NAH-dominant pattern with NAH-specific markers
    base_pattern = [
        "tɬ",
        "a",
        "kʷ",
        "i",
        "ʔ",
        "ɬ",
        "o",
        "tɬ",  # Strong NAH markers
        "ʃ",
        "a",
        "kʷ",
        "i",
        "ɬ",
        "o",
        "tɬ",
        "a",  # More NAH markers
    ]
    # Repeat pattern 8 times = 128 phonemes (exceeds Phase G threshold)
    phonemes = _make_phonemes(base_pattern * 8)

    result = identify_language(phonemes)
    languages = {seg.language for seg in result}

    # Should be NAH only, not MAY
    assert "nah" in languages or "other" in languages, "Should detect NAH or other"
    assert "may" not in languages, (
        "MAY false positive detected on NAH-dominant stream (tɬ, kʷ, ɬ should exclude MAY)"
    )


def test_may_negative_markers_exclude_nah_phonemes():
    """MAY negative markers should penalize NAH-only phonemes.

    When phoneme stream contains NAH-specific markers (tɬ, ɬ, kʷ),
    MAY should NOT be detected due to negative markers.
    NAH should be detected instead.
    """
    # Create stream with NAH-only markers mixed with neutral phonemes
    phonemes = _make_phonemes(
        ["tɬ", "a", "ɬ", "i", "kʷ", "o", "tɬ", "e", "ɬ", "u", "kʷ", "a"]
    )

    result = identify_language(phonemes)
    languages = {seg.language for seg in result}

    # NAH should be detected (has these markers)
    assert "nah" in languages, (
        f"NAH should be detected for tɬ/ɬ/kʷ markers, got {languages}"
    )

    # MAY should NOT be detected (negative markers should exclude it)
    assert "may" not in languages, (
        f"MAY should be excluded by negative markers for tɬ/ɬ/kʷ, got {languages}"
    )


def test_nah_with_glottal_stop_not_may():
    """NAH phonemes with glottal stop but no ejectives must NOT be tagged MAY.

    This is the core La Otra Conquista false positive scenario: Nahuatl audio
    produces ʔ (shared marker) but no ejective consonants. Before the fix,
    ʔ alone (weight 0.5) could reach the MAY threshold (0.5).

    After fix: ʔ removed from MAY markers, threshold raised to 2.0.
    """
    # NAH-like stream with glottal stops but NO ejectives
    phonemes = _make_phonemes(
        ["ʔ", "a", "ʃ", "i", "ʔ", "o", "a", "ʔ", "e", "ʃ", "i", "o"] * 11
    )

    result = identify_language(phonemes)
    languages = {seg.language for seg in result}

    assert "may" not in languages, (
        f"MAY detected without ejective evidence (ʔ and ʃ alone should not trigger MAY), got {languages}"
    )


def test_spanish_dialogue_not_may():
    """Spanish-like phoneme stream must NOT be tagged MAY.

    Simulates the common LOC false positive: Spanish dialogue with
    scattered ʔ artifacts from Allosaurus.
    """
    # Spanish-like phonemes with scattered ʔ (Allosaurus artifact)
    base = [
        "p", "o", "ɾ", "k", "e", "n", "o", "s", "t", "a", "d", "j", "a",
        "ʔ", "k", "e", "ɾ", "d", "o", "k", "o", "m", "b", "w", "e", "s",
        "t", "ɾ", "a", "s", "f", "o", "ɾ", "m", "a", "s",
    ]
    phonemes = _make_phonemes(base * 4)  # 140 phonemes

    result = identify_language(phonemes)
    languages = {seg.language for seg in result}

    assert "may" not in languages, (
        f"MAY detected in Spanish-like phoneme stream, got {languages}"
    )


def test_may_requires_ejective_for_detection():
    """MAY detection requires at least one ejective consonant.

    The ejective guard (Phase F.5) demotes MAY segments without ejective
    evidence (kʼ, tʼ, pʼ, tsʼ, tʃʼ) to 'other'. Only segments with
    actual ejective markers should survive as MAY.
    """
    # Stream WITH ejective markers — should be MAY
    with_ejectives = _make_phonemes(
        ["kʼ", "a", "tʼ", "i", "kʼ", "o", "pʼ", "a", "kʼ", "i", "tʼ", "o"] * 11
    )
    result = identify_language(with_ejectives)
    languages = {seg.language for seg in result}
    assert "may" in languages, (
        f"MAY should be detected when ejective markers are present, got {languages}"
    )

    # Stream WITHOUT ejective markers — should NOT be MAY
    without_ejectives = _make_phonemes(
        ["ʔ", "a", "ɓ", "i", "ʔ", "o", "a", "i", "ʔ", "o", "ɓ", "a"] * 11
    )
    result = identify_language(without_ejectives)
    languages = {seg.language for seg in result}
    assert "may" not in languages, (
        f"MAY should NOT be detected without ejective markers (ʔ/ɓ alone insufficient), got {languages}"
    )


def test_may_not_protected_from_consolidation():
    """MAY segments should NOT be protected from file-level consolidation.

    In v7.0, MAY was removed from _PROTECTED_FILE_CONSOLIDATION_LANGUAGES
    to allow file-level scoring to correct MAY false positives when another
    language clearly dominates the file (e.g., SPA in La Otra Conquista).
    """
    from tenepal.language.identifier import _PROTECTED_FILE_CONSOLIDATION_LANGUAGES

    assert "may" not in _PROTECTED_FILE_CONSOLIDATION_LANGUAGES, (
        "MAY should not be in _PROTECTED_FILE_CONSOLIDATION_LANGUAGES "
        "(causes 70 false positives in La Otra Conquista)"
    )

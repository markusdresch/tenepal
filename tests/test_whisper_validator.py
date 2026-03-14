"""Tests for WhisperValidator hallucination detection."""

import pytest
from pathlib import Path
from tenepal.validation import WhisperValidator, ValidationResult


class TestWhisperValidator:
    """Test suite for WhisperValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return WhisperValidator()

    def test_wordlist_loads(self, validator):
        """Test that validator initializes and loads word list."""
        assert validator.spanish_words is not None
        assert len(validator.spanish_words) > 0
        # Check for some common words
        assert "el" in validator.spanish_words
        assert "la" in validator.spanish_words
        assert "de" in validator.spanish_words

    def test_custom_wordlist_path(self, tmp_path):
        """Test that custom wordlist path can be provided."""
        # Create a temporary word list
        wordlist = tmp_path / "custom_words.txt"
        wordlist.write_text("hola\nadios\nmundo", encoding="utf-8")

        validator = WhisperValidator(wordlist_path=wordlist)
        assert "hola" in validator.spanish_words
        assert "adios" in validator.spanish_words
        assert "mundo" in validator.spanish_words

    def test_custom_wordlist_not_found(self):
        """Test that FileNotFoundError is raised for missing custom wordlist."""
        with pytest.raises(FileNotFoundError):
            WhisperValidator(wordlist_path=Path("/nonexistent/wordlist.txt"))

    def test_normalize_strips_accents(self, validator):
        """Test that _normalize_word strips accents correctly."""
        assert validator._normalize_word("señor") == validator._normalize_word("senor")
        assert validator._normalize_word("está") == validator._normalize_word("esta")
        assert validator._normalize_word("Moctezuma") == validator._normalize_word("moctezuma")
        assert validator._normalize_word("México") == "mexico"
        assert validator._normalize_word("José") == "jose"

    def test_normalize_strips_punctuation(self, validator):
        """Test that _normalize_word removes punctuation."""
        assert validator._normalize_word("hola!") == "hola"
        assert validator._normalize_word("¿Cómo?") == "como"
        assert validator._normalize_word("está.") == "esta"

    def test_normalize_lowercase(self, validator):
        """Test that _normalize_word converts to lowercase."""
        assert validator._normalize_word("HOLA") == "hola"
        assert validator._normalize_word("Soldados") == "soldados"

    def test_real_spanish_passes(self, validator):
        """Test that real Spanish text validates as VALID."""
        result = validator.validate("Soldados, estan listos. Vayan con nosotros.")
        assert result.is_valid, f"Real Spanish should pass: {result.reason}"
        assert result.confidence > 0.7
        assert "lexicon" in result.checks
        assert result.checks["lexicon"] > 0.6

    def test_hallucinated_maya_fails(self, validator):
        """Test that hallucinated Mayan text validates as INVALID."""
        result = validator.validate("Uchach, ayik, alilti, le'l l'uma. Ak, ayik, ali, le ba'alo, ba'.")
        assert not result.is_valid, f"Hallucination should fail: {result.reason}"
        assert result.confidence < 0.6  # Updated for new 5-check confidence formula
        assert result.checks["lexicon"] < 0.4 or result.checks["apostrophe"] < 0.5

    def test_mixed_with_proper_nouns_passes(self, validator):
        """Test that Spanish with proper nouns passes validation."""
        result = validator.validate("Moctezuma es el senor de los mexicas.")
        assert result.is_valid, f"Mixed text should pass: {result.reason}"
        assert result.checks["lexicon"] > 0.6

    def test_apostrophe_density_fails(self, validator):
        """Test that high apostrophe density flags hallucination."""
        result = validator.validate("Ak, k'an k'antakin. Uk'ahilo, chup.")
        assert not result.is_valid, f"High apostrophe density should fail: {result.reason}"
        assert "apostrophe" in result.reason.lower() or result.checks["lexicon"] < 0.4

    def test_pure_spanish_sentence(self, validator):
        """Test that pure Spanish sentence validates as VALID."""
        result = validator.validate("Buenos dias, como esta usted hoy?")
        assert result.is_valid, f"Pure Spanish should pass: {result.reason}"
        assert result.confidence > 0.7

    def test_empty_text(self, validator):
        """Test that empty text returns VALID (nothing to flag)."""
        result = validator.validate("")
        assert result.is_valid
        assert result.confidence >= 0.9  # Neutral scores should be high

    def test_single_word_spanish(self, validator):
        """Test that single Spanish word returns VALID."""
        result = validator.validate("hola")
        assert result.is_valid

    def test_single_word_unknown(self, validator):
        """Test that single unknown word returns INVALID."""
        result = validator.validate("uchach")
        assert not result.is_valid
        assert result.checks["lexicon"] < 0.4

    def test_lexicon_score_calculation(self, validator):
        """Test lexicon score calculation with mixed words."""
        # "el hombre uchach" = 2 Spanish words, 1 unknown
        result = validator.validate("el hombre uchach")
        # Should be approximately 2/3 = 0.67
        assert 0.6 <= result.checks["lexicon"] <= 0.7

    def test_apostrophe_density_score(self, validator):
        """Test apostrophe density score calculation."""
        # "k'an k'al" has 2 apostrophes in 10 characters = 0.2 density
        # Density / 0.04 = 5.0, capped at 1.0
        # Score = 1.0 - 1.0 = 0.0
        result = validator.validate("k'an k'al")
        # High density should give low score
        assert result.checks["apostrophe"] < 0.5

    def test_character_pattern_score(self, validator):
        """Test character pattern score for pure Spanish."""
        result = validator.validate("Soldados listos")
        # All characters are Spanish
        assert result.checks["character"] >= 0.95

    def test_validation_result_has_checks_dict(self, validator):
        """Test that ValidationResult contains all check scores."""
        result = validator.validate("Hola mundo")
        assert "lexicon" in result.checks
        assert "apostrophe" in result.checks
        assert "character" in result.checks
        assert isinstance(result.checks["lexicon"], float)
        assert isinstance(result.checks["apostrophe"], float)
        assert isinstance(result.checks["character"], float)

    def test_lexicon_check_filters_short_tokens(self, validator):
        """Test that _check_lexicon skips very short tokens."""
        # "el y o la" - "y" and "o" are 1 char, should be skipped
        score = validator._check_lexicon("el y o la")
        # Should only count "el" and "la" (both in word list)
        assert score == 1.0

    def test_apostrophe_density_empty_text(self, validator):
        """Test apostrophe density check with empty text."""
        score = validator._check_apostrophe_density("")
        assert score == 1.0  # Empty text is neutral

    def test_character_pattern_no_alpha(self, validator):
        """Test character pattern check with no alphabetic characters."""
        score = validator._check_character_pattern("123 456")
        assert score == 1.0  # No alpha chars is neutral

    def test_combined_logic_low_lexicon(self, validator):
        """Test combined decision logic: low lexicon triggers INVALID."""
        # Create text with mostly unknown words
        result = validator.validate("xochitl tlaloc quetzalcoatl")
        assert not result.is_valid
        assert "lexicon" in result.reason.lower()

    def test_combined_logic_apostrophe_and_moderate_lexicon(self, validator):
        """Test combined decision logic: apostrophes + moderate lexicon."""
        # Test text with apostrophe density + moderate lexicon
        # "el esta xochitl tlaloc" = 2 Spanish (el, esta), 2 unknown (xochitl, tlaloc)
        # Add many apostrophes: "el' esta' xochitl' tlaloc'" → 4 apostrophes in ~27 chars = 0.15 density
        # Apostrophe score: 1 - min(0.15/0.04, 1.0) = 1 - 1.0 = 0.0
        # Lexicon: 2/4 = 0.5
        # Rule: apostrophe < 0.5 AND lexicon < 0.6 → INVALID
        result = validator.validate("el' esta' xochitl' tlaloc'")
        assert not result.is_valid, f"Should fail with high apostrophes + moderate lexicon: {result}"

    def test_avg_log_prob_parameter_accepted(self, validator):
        """Test that avg_log_prob parameter is accepted (for future use)."""
        # Should not raise error even though not used yet
        result = validator.validate("Hola mundo", avg_log_prob=-0.5)
        assert result.is_valid

    def test_confidence_calculation(self, validator):
        """Test that confidence is weighted average of checks."""
        result = validator.validate("Soldados estan listos")
        # Confidence = 0.4 * lexicon + 0.15 * apostrophe + 0.15 * character + 0.15 * logprob + 0.15 * repetition
        expected = (
            0.4 * result.checks["lexicon"]
            + 0.15 * result.checks["apostrophe"]
            + 0.15 * result.checks["character"]
            + 0.15 * result.checks["avg_log_prob"]
            + 0.15 * result.checks["repetition"]
        )
        assert abs(result.confidence - expected) < 0.01

    def test_hernan_specific_vocabulary(self, validator):
        """Test that Hernan-specific proper nouns are recognized."""
        result = validator.validate("Cortes llego a Tenochtitlan con Marina.")
        assert result.is_valid, f"Hernan vocabulary should pass: {result.reason}"
        # Check that proper nouns are in word list
        assert "cortes" in validator.spanish_words
        assert "tenochtitlan" in validator.spanish_words
        assert "marina" in validator.spanish_words

    def test_accent_insensitive_matching(self, validator):
        """Test that accent-insensitive matching works."""
        # "esta" (without accent) should match "está" in text
        result = validator.validate("Esta es una prueba")
        assert result.is_valid
        # Both should be recognized
        assert result.checks["lexicon"] > 0.8

    # ========== VALID-04: avg_log_prob tests ==========

    def test_low_logprob_with_low_lexicon_fails(self, validator):
        """Test that low avg_log_prob with moderate lexicon fails."""
        # Text with moderate lexicon (~0.6) but very low avg_log_prob
        result = validator.validate("Hola soldados uchach ayik", avg_log_prob=-1.2)
        assert not result.is_valid, f"Low logprob + moderate lexicon should fail: {result.reason}"
        assert result.checks["lexicon"] < 0.7
        assert result.checks["avg_log_prob"] < 0.4

    def test_low_logprob_with_high_lexicon_passes(self, validator):
        """Test that high lexicon score passes even with low avg_log_prob."""
        # High lexicon should save the segment
        result = validator.validate("Hola amigos buenos dias", avg_log_prob=-1.2)
        assert result.is_valid, f"High lexicon should pass despite low logprob: {result.reason}"
        assert result.checks["lexicon"] > 0.7

    def test_normal_logprob_no_effect(self, validator):
        """Test that normal avg_log_prob doesn't affect decision."""
        # High avg_log_prob should not affect decision (based on lexicon only)
        result = validator.validate("hola uchach ayik", avg_log_prob=-0.3)
        # Should fail due to low lexicon, not logprob
        assert not result.is_valid
        assert result.checks["lexicon"] < 0.5

    def test_logprob_score_calculation(self, validator):
        """Test avg_log_prob score mapping."""
        # High logprob → high score
        score_high = validator._check_avg_log_prob(-0.3)
        assert score_high >= 0.79  # (-0.3 + 1.5) / 1.5 = 1.2 / 1.5 = 0.8

        # Low logprob → low score
        score_low = validator._check_avg_log_prob(-1.5)
        assert score_low == 0.0  # At threshold, exactly 0

        # Very low logprob → still 0
        score_very_low = validator._check_avg_log_prob(-2.0)
        assert score_very_low == 0.0

        # Perfect logprob → 1.0
        score_perfect = validator._check_avg_log_prob(0.0)
        assert score_perfect == 1.0

    # ========== VALID-05: Repetition detection tests ==========

    def test_repeated_phrase_detected(self, validator):
        """Test that repeated phrases are detected."""
        # "soldados vamos ahora" repeats twice → score 0.5
        result = validator.validate("soldados vamos ahora soldados vamos ahora")
        assert not result.is_valid, f"Repetition should be detected: {result.reason}"
        assert result.checks["repetition"] == 0.5  # 1.0 / 2 repetitions
        assert "repetition" in result.reason.lower() or "loop" in result.reason.lower()

    def test_repeated_3word_ngram_detected(self, validator):
        """Test that repeated 3-word n-grams are detected."""
        # "el señor de" appears twice → score 0.5
        result = validator.validate("el señor de el señor de los mexicas")
        assert not result.is_valid, f"3-word repetition should be detected: {result.reason}"
        assert result.checks["repetition"] == 0.5  # 1.0 / 2 repetitions
        assert "repetition" in result.reason.lower() or "loop" in result.reason.lower()

    def test_no_repetition_passes(self, validator):
        """Test that text without repetition passes."""
        result = validator.validate("Los soldados marcharon por el camino largo hacia el pueblo")
        # Should not flag for repetition (but lexicon might fail if words unknown)
        assert result.checks["repetition"] > 0.9

    def test_repetition_score_calculation(self, validator):
        """Test repetition detection scoring."""
        # "a b c a b c d e f" - "a b c" repeats twice
        score_repeat = validator._check_repetition("a b c a b c d e f")
        assert score_repeat <= 0.5  # 1.0 / 2 repetitions

        # No repetition
        score_no_repeat = validator._check_repetition("a b c d e f g")
        assert score_no_repeat == 1.0

    def test_short_text_no_repetition_check(self, validator):
        """Test that short text skips repetition check."""
        # Fewer than 6 words should return 1.0
        score = validator._check_repetition("a b c d e")
        assert score == 1.0

    # ========== Combined decision tests ==========

    def test_combined_all_checks_in_result(self, validator):
        """Test that ValidationResult contains all 5 checks."""
        result = validator.validate("Hola mundo", avg_log_prob=-0.5)
        assert len(result.checks) == 5, f"Expected 5 checks, got {len(result.checks)}"
        assert "lexicon" in result.checks
        assert "apostrophe" in result.checks
        assert "character" in result.checks
        assert "avg_log_prob" in result.checks
        assert "repetition" in result.checks

    def test_combined_logprob_and_lexicon_interaction(self, validator):
        """Test interaction between avg_log_prob and lexicon."""
        # Low logprob AND low lexicon → INVALID
        result = validator.validate("Hola soldados uchach ayik koali", avg_log_prob=-1.3)
        assert not result.is_valid, f"Low logprob + low lexicon should fail: {result.reason}"

    def test_repetition_overrides_lexicon(self, validator):
        """Test that repetition fails even with OK lexicon."""
        # All Spanish words but repeated 3 times: "vamos ahora todos" appears 3 times
        result = validator.validate("vamos ahora todos vamos ahora todos vamos ahora todos soldados")
        assert not result.is_valid, f"Repetition should fail: {result.reason}"
        assert result.checks["repetition"] <= 0.5  # Should trigger <= 0.5 threshold

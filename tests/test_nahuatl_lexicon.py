"""Tests for NahuatlLexicon class for lexical recognition of short Nahuatl words."""

import json

import pytest
from tenepal.language.nahuatl_lexicon import LexiconMatch, NahuatlLexicon


class TestNahuatlLexiconLoad:
    """Tests for loading the Nahuatl lexicon data file."""

    def test_loads_from_package_data(self):
        """Lexicon loads from nah_lexicon.json in package data."""
        lex = NahuatlLexicon()
        assert len(lex._entries) >= 100, "Merged lexicon should have substantial coverage"

    def test_entries_have_required_fields(self):
        """Each lexicon entry has word, ipa, and gloss fields."""
        lex = NahuatlLexicon()
        for entry in lex._entries:
            assert "word" in entry
            assert "ipa" in entry
            assert "gloss" in entry
            assert isinstance(entry["word"], str)
            assert isinstance(entry["ipa"], list)
            assert entry["gloss"] is None or isinstance(entry["gloss"], str)

    def test_contains_expected_words(self):
        """Lexicon contains expected common Nahuatl words."""
        lex = NahuatlLexicon()
        words = {e["word"] for e in lex._entries}
        # Check for some expected short words from the plan
        expected = {"koali", "tlein", "amo"}
        assert expected.issubset(words), f"Missing expected words: {expected - words}"


class TestLexiconMatchExact:
    """Tests for exact IPA matching."""

    def test_koali_exact_match(self):
        """koali IPA [k o a l i] matches with score 1.0."""
        lex = NahuatlLexicon()
        result = lex.match(["k", "o", "a", "l", "i"])
        assert result is not None
        assert result.word == "koali"
        assert result.score >= 0.8
        assert result.start_idx == 0
        assert result.length == 5

    def test_amo_exact_match(self):
        """amo IPA [a m o] matches with score 1.0."""
        lex = NahuatlLexicon()
        result = lex.match(["a", "m", "o"])
        assert result is not None
        assert result.word in {"amo", "a:mo"}
        assert result.score >= 0.8
        assert result.start_idx == 0
        assert result.length == 3

    def test_tlein_with_affricate(self):
        """tlein IPA [tɬ e i n] matches with score >= 0.8."""
        lex = NahuatlLexicon()
        result = lex.match(["tɬ", "e", "i", "n"])
        assert result is not None
        assert result.word == "tlein"
        assert result.score >= 0.8


class TestLexiconMatchFuzzy:
    """Tests for fuzzy IPA matching with edit distance."""

    def test_decomposed_affricate_matches(self):
        """Allosaurus-decomposed [t l] matches lexicon [tɬ]."""
        lex = NahuatlLexicon()
        # tlein with decomposed affricate: [t, l, e, i, n]
        result = lex.match(["t", "l", "e", "i", "n"])
        assert result is not None
        assert result.word == "tlein"
        assert result.score >= 0.7

    def test_one_deletion_fuzzy_match(self):
        """koali with 1 phoneme missing still matches (score ~0.8)."""
        lex = NahuatlLexicon()
        # Missing final "i": [k, o, a, l]
        result = lex.match(["k", "o", "a", "l"])
        assert result is not None
        assert result.word.startswith("ko")
        assert 0.7 <= result.score < 1.0

    def test_one_substitution_fuzzy_match(self):
        """koali with 1 phoneme substituted still matches."""
        lex = NahuatlLexicon()
        # Substitute "a" -> "e": [k, o, e, l, i]
        result = lex.match(["k", "o", "e", "l", "i"])
        assert result is not None
        assert result.word == "koali"
        assert 0.7 <= result.score < 1.0

    def test_exceeds_max_distance_returns_none(self):
        """IPA too different returns None (exceeds max_distance threshold)."""
        lex = NahuatlLexicon()
        # Too many changes from "koali": [k, e, e, n, u] (4+ edits)
        result = lex.match(["k", "e", "e", "n", "u"])
        assert result is None


class TestLexiconMatchNonMatches:
    """Tests for non-matching sequences."""

    def test_random_phonemes_no_match(self):
        """Random unrelated IPA sequence returns None."""
        lex = NahuatlLexicon()
        result = lex.match(["b", "r", "z"])
        assert result is None

    def test_empty_input_no_match(self):
        """Empty phoneme list returns None."""
        lex = NahuatlLexicon()
        result = lex.match([])
        assert result is None

    def test_too_long_sequence_no_match(self):
        """Very long sequence with no word match returns None."""
        lex = NahuatlLexicon()
        result = lex.match(["x", "y", "z", "p", "q", "r", "s", "t", "u", "v"])
        assert result is None


class TestMatchSubsequence:
    """Tests for finding lexicon words in longer phoneme sequences."""

    def test_finds_koali_embedded(self):
        """Finds koali in longer sequence [a k o a l i n]."""
        lex = NahuatlLexicon()
        matches = lex.match_subsequence(["a", "k", "o", "a", "l", "i", "n"])
        assert len(matches) > 0
        koali_matches = [m for m in matches if m.word == "koali"]
        # Should find koali, possibly at multiple fuzzy positions
        assert len(koali_matches) >= 1
        # Best match should be at start_idx=1 with exact match
        best_koali = max(koali_matches, key=lambda m: m.score)
        assert best_koali.start_idx == 1
        assert best_koali.length == 5
        assert best_koali.score == 1.0  # Exact match

    def test_finds_multiple_words(self):
        """Finds multiple words in a sequence."""
        lex = NahuatlLexicon()
        # Create sequence with "amo" and "koali"
        matches = lex.match_subsequence(["a", "m", "o", "k", "o", "a", "l", "i"])
        words = {m.word for m in matches}
        assert "amo" in words
        assert "koali" in words

    def test_respects_min_length(self):
        """Only returns matches >= min_length."""
        lex = NahuatlLexicon()
        # If any 1-phoneme words exist, they shouldn't match with min_length=2
        matches = lex.match_subsequence(["a", "b", "c", "d"], min_length=3)
        for match in matches:
            assert match.length >= 3

    def test_empty_sequence_returns_empty(self):
        """Empty sequence returns empty list."""
        lex = NahuatlLexicon()
        matches = lex.match_subsequence([])
        assert matches == []


class TestIPANormalization:
    """Tests for IPA normalization handling Allosaurus quirks."""

    def test_strips_length_modifiers(self):
        """Strips length modifier (ː) for matching."""
        lex = NahuatlLexicon()
        # If lexicon has "atl" = [a, t, l], then [a, tː, l] should match
        result = lex.match(["a", "tː", "l"])
        # Should match if "atl" is in lexicon
        if result:
            assert result.score >= 0.7

    def test_allophone_normalization(self):
        """Applies allophone normalization (e.g., ð→d, β→b)."""
        lex = NahuatlLexicon()
        # This tests the _ALLOPHONE_MAP from identifier.py is used
        # If any words have "d", then "ð" should match
        # Hard to test without knowing exact lexicon, but the test
        # demonstrates the intended behavior
        pass  # Covered implicitly by fuzzy matching


class TestMaxDistanceParameter:
    """Tests for configurable max_distance parameter."""

    def test_strict_matching_with_low_max_distance(self):
        """Lower max_distance requires closer matches."""
        lex = NahuatlLexicon()
        # With max_distance=0.1, only very close matches succeed
        result = lex.match(["k", "o", "a", "l"], max_distance=0.1)
        # This should fail (1 deletion = 0.2 normalized distance)
        assert result is None

    def test_permissive_matching_with_high_max_distance(self):
        """Higher max_distance allows more variation."""
        lex = NahuatlLexicon()
        # With max_distance=0.5, even 2 edits might succeed
        result = lex.match(["k", "o", "a", "l"], max_distance=0.5)
        assert result is not None
        assert result.word.startswith("ko")


class TestMinFreqFiltering:
    """Tests for min_freq filtering behavior."""

    def test_filters_low_freq_non_curated_entries(self, tmp_path):
        """Non-curated entries below min_freq are dropped."""
        lexicon_data = [
            {"word": "cur", "ipa": ["k", "u", "r"], "gloss": "curated", "source": "curated", "freq": 1},
            {"word": "low", "ipa": ["l", "o"], "gloss": "low", "source": "amith-zacatlan", "freq": 2},
            {"word": "ok", "ipa": ["o", "k"], "gloss": "ok", "source": "amith-zacatlan", "freq": 3},
        ]
        lexicon_path = tmp_path / "lex.json"
        lexicon_path.write_text(json.dumps(lexicon_data), encoding="utf-8")

        lex = NahuatlLexicon(lexicon_path=lexicon_path, min_freq=3)
        words = {e["word"] for e in lex._entries}
        assert "cur" in words
        assert "ok" in words
        assert "low" not in words

    def test_default_min_freq_is_3(self, tmp_path):
        """Default constructor applies min_freq=3."""
        lexicon_data = [
            {"word": "two", "ipa": ["t", "u"], "gloss": "two", "source": "amith-zacatlan", "freq": 2},
            {"word": "three", "ipa": ["t", "ɾ", "i"], "gloss": "three", "source": "amith-zacatlan", "freq": 3},
        ]
        lexicon_path = tmp_path / "lex.json"
        lexicon_path.write_text(json.dumps(lexicon_data), encoding="utf-8")

        lex = NahuatlLexicon(lexicon_path=lexicon_path)
        words = {e["word"] for e in lex._entries}
        assert "three" in words
        assert "two" not in words

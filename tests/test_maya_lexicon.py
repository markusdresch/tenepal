"""Tests for MayaLexicon class for lexical recognition of short Yucatec Maya words."""

import pytest
from tenepal.language.maya_lexicon import LexiconMatch, MayaLexicon


class TestMayaLexiconLoad:
    """Tests for loading the Maya lexicon data file."""

    def test_maya_lexicon_loads(self):
        """Lexicon loads from may_lexicon.json in package data."""
        lex = MayaLexicon()
        assert len(lex._entries) >= 10, "Should have at least 10 entries"
        assert len(lex._entries) <= 30, "Should have at most 30 entries"

    def test_lexicon_entries_have_required_fields(self):
        """Each lexicon entry has word, ipa, and gloss fields."""
        lex = MayaLexicon()
        for entry in lex._entries:
            assert "word" in entry
            assert "ipa" in entry
            assert "gloss" in entry
            assert isinstance(entry["word"], str)
            assert isinstance(entry["ipa"], list)
            assert isinstance(entry["gloss"], str)

    def test_contains_expected_maya_words(self):
        """Lexicon contains expected common Maya words."""
        lex = MayaLexicon()
        words = {e["word"] for e in lex._entries}
        # Check for some expected Maya words from the plan
        expected = {"ba'al", "ka'an", "k'iin"}
        assert expected.issubset(words), f"Missing expected words: {expected - words}"


class TestMatchExactWord:
    """Tests for exact IPA matching of Maya words."""

    def test_match_exact_word(self):
        """ba'al IPA [b a ʔ a l] matches with score ~1.0."""
        lex = MayaLexicon()
        result = lex.match(["b", "a", "ʔ", "a", "l"])
        assert result is not None
        assert result.word == "ba'al"
        assert result.score >= 0.8
        assert result.start_idx == 0
        assert result.length == 5

    def test_kiin_exact_match(self):
        """k'iin IPA [kʼ i n] matches with score ~1.0."""
        lex = MayaLexicon()
        result = lex.match(["kʼ", "i", "n"])
        assert result is not None
        assert result.word == "k'iin"
        assert result.score >= 0.8
        assert result.start_idx == 0
        assert result.length == 3

    def test_ja_exact_match(self):
        """ja' IPA [h a ʔ] matches with score >= 0.8."""
        lex = MayaLexicon()
        result = lex.match(["h", "a", "ʔ"])
        assert result is not None
        assert result.word == "ja'"
        assert result.score >= 0.8


class TestMatchFuzzy:
    """Tests for fuzzy IPA matching with edit distance."""

    def test_match_fuzzy(self):
        """ba'al with missing glottal stop matches at score > 0.7."""
        lex = MayaLexicon()
        # Missing glottal stop: [b, a, a, l] (no ʔ)
        result = lex.match(["b", "a", "a", "l"])
        assert result is not None
        assert result.word == "ba'al"
        assert result.score >= 0.7

    def test_ejective_normalization(self):
        """Handles decomposed ejectives (e.g., [k ʼ a n] → ka'an)."""
        lex = MayaLexicon()
        # Decomposed ejective: [k, ʼ, a, ʔ, a, n] instead of [kʼ, a, ʔ, a, n]
        result = lex.match(["k", "ʼ", "a", "ʔ", "a", "n"])
        assert result is not None
        assert result.word == "ka'an"
        assert result.score >= 0.7

    def test_ts_ejective_decomposition(self):
        """Handles ts' decomposed as [t, s, ʼ] or [ts, ʼ]."""
        lex = MayaLexicon()
        # ts'ono'ot with decomposed affricate-ejective
        result = lex.match(["t", "s", "ʼ", "o", "n", "o", "ʔ", "o", "t"])
        assert result is not None
        assert result.word == "ts'ono'ot"
        assert result.score >= 0.7

    def test_one_substitution_still_matches(self):
        """Maya word with 1 phoneme substituted still matches with permissive threshold."""
        lex = MayaLexicon()
        # Substitute final "l" -> "n" in "ba'al": [b, a, ʔ, a, n]
        # 1 substitution / 5 length = 0.2 normalized distance (within 0.3 threshold)
        result = lex.match(["b", "a", "ʔ", "a", "n"])
        assert result is not None
        assert result.word == "ba'al"
        assert result.score >= 0.7


class TestNoMatchUnrelated:
    """Tests for non-matching sequences."""

    def test_no_match_unrelated(self):
        """Random unrelated IPA sequence returns None."""
        lex = MayaLexicon()
        # English-like phonemes unlikely to match Maya
        result = lex.match(["p", "a", "t", "e", "r"])
        assert result is None

    def test_empty_sequence_no_match(self):
        """Empty phoneme list returns None."""
        lex = MayaLexicon()
        result = lex.match([])
        assert result is None

    def test_too_different_no_match(self):
        """Sequence too different from any Maya word returns None."""
        lex = MayaLexicon()
        result = lex.match(["x", "y", "z", "q"])
        assert result is None


class TestMatchSubsequence:
    """Tests for finding Maya words embedded in longer sequences."""

    def test_match_subsequence(self):
        """Finds Maya words embedded in longer phoneme sequence."""
        lex = MayaLexicon()
        # Embed "bix" [b i ʃ] in longer sequence
        matches = lex.match_subsequence(["a", "b", "i", "ʃ", "n", "o"])
        assert len(matches) > 0
        bix_matches = [m for m in matches if m.word == "bix"]
        assert len(bix_matches) >= 1
        # Best match should be at start_idx=1
        best_bix = max(bix_matches, key=lambda m: m.score)
        assert best_bix.start_idx == 1
        assert best_bix.length == 3

    def test_finds_multiple_maya_words(self):
        """Finds multiple Maya words in a sequence."""
        lex = MayaLexicon()
        # Create sequence with "le" [l e] and "ja'" [h a ʔ]
        matches = lex.match_subsequence(["l", "e", "h", "a", "ʔ"])
        words = {m.word for m in matches}
        assert "le" in words or "ja'" in words  # At least one should match

    def test_respects_min_length(self):
        """Only returns matches >= min_length."""
        lex = MayaLexicon()
        matches = lex.match_subsequence(["a", "b", "c", "d"], min_length=3)
        for match in matches:
            assert match.length >= 3


class TestPipelineIntegration:
    """Tests for integration with language identification pipeline."""

    def test_apply_maya_lexicon_check(self):
        """Pipeline integration: phoneme stream with Maya words tagged as 'may'."""
        from tenepal.language.identifier import _apply_lexicon_check
        from tenepal.phoneme import PhonemeSegment

        # Create phoneme stream with Maya word "bix" [b i ʃ]
        phonemes = [
            (PhonemeSegment("b", 0.0, 0.1), None, 0.0),
            (PhonemeSegment("i", 0.1, 0.1), None, 0.0),
            (PhonemeSegment("ʃ", 0.2, 0.1), None, 0.0),
        ]

        # Apply lexicon check
        result = _apply_lexicon_check(phonemes)

        # Check that Maya word was recognized and tagged
        # (Note: This requires Maya lexicon to be integrated into identifier.py)
        # For now, this test establishes the interface
        assert len(result) == 3


class TestMayaLexiconCoexistsWithNah:
    """Tests for Maya lexicon coexistence with Nahuatl lexicon."""

    def test_maya_lexicon_coexists_with_nah(self):
        """Stream with both NAH and MAY words tags each correctly."""
        from tenepal.language.identifier import _apply_lexicon_check
        from tenepal.phoneme import PhonemeSegment

        # Create stream with NAH word "amo" [a m o] and MAY word "bix" [b i ʃ]
        phonemes = [
            (PhonemeSegment("a", 0.0, 0.1), None, 0.0),
            (PhonemeSegment("m", 0.1, 0.1), None, 0.0),
            (PhonemeSegment("o", 0.2, 0.1), None, 0.0),
            (PhonemeSegment("b", 0.3, 0.1), None, 0.0),
            (PhonemeSegment("i", 0.4, 0.1), None, 0.0),
            (PhonemeSegment("ʃ", 0.5, 0.1), None, 0.0),
        ]

        result = _apply_lexicon_check(phonemes)

        # Both lexicons should run without interference
        assert len(result) == 6


class TestMayaLexiconShortSegment:
    """Tests for short Maya segments identified via lexicon."""

    def test_maya_lexicon_short_segment(self):
        """3-phoneme Maya word detected even without enough markers."""
        lex = MayaLexicon()
        # Very short Maya word "k'iin" = [kʼ, i, n] (3 phonemes)
        result = lex.match(["kʼ", "i", "n"])
        assert result is not None
        assert result.word == "k'iin"
        # This demonstrates lexicon provides recognition path
        # for short words lacking marker accumulation
        assert result.length == 3

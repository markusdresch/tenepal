"""Tests for Latin liturgical phrase recognition."""

import pytest

from tenepal.language.latin_lexicon import LatinLexicon


class TestLatinLexicon:
    """Tests for LatinLexicon keyword matching."""

    @pytest.fixture
    def lexicon(self):
        """Create LatinLexicon instance."""
        return LatinLexicon()

    def test_check_text_high_match(self, lexicon):
        """Test text with 5+ Latin keywords returns True."""
        text = "Ego te baptizo in nomine Patris"
        is_latin, count = lexicon.check_text(text)
        assert is_latin is True
        assert count >= 3

    def test_check_text_threshold_match(self, lexicon):
        """Test text with exactly 3 Latin keywords returns True."""
        text = "Sanctus Dominus Deus"
        is_latin, count = lexicon.check_text(text)
        assert is_latin is True
        assert count == 3

    def test_check_text_below_threshold(self, lexicon):
        """Test text with <3 Latin keywords returns False."""
        text = "Amen"
        is_latin, count = lexicon.check_text(text)
        assert is_latin is False
        assert count < 3

    def test_check_text_no_match(self, lexicon):
        """Test non-Latin text returns False."""
        text = "Hola mundo"
        is_latin, count = lexicon.check_text(text)
        assert is_latin is False
        assert count == 0

    def test_check_text_pater_noster(self, lexicon):
        """Test Pater Noster prayer matches."""
        text = "Pater Noster qui es in caelis"
        is_latin, count = lexicon.check_text(text)
        # Should match: pater, noster, qui (if included)
        assert is_latin is True
        assert count >= 3

    def test_check_text_case_insensitive(self, lexicon):
        """Test matching is case-insensitive."""
        text = "EGO TE BAPTIZO"
        is_latin, count = lexicon.check_text(text)
        assert is_latin is True
        assert count == 3

    def test_check_text_accents_normalized(self, lexicon):
        """Test accent normalization works."""
        text = "Pátér Nóstér quí"
        is_latin, count = lexicon.check_text(text)
        # Should match despite accents
        assert count >= 2

    def test_check_text_custom_threshold(self, lexicon):
        """Test custom min_matches threshold."""
        text = "Amen Amen"  # 2 matches (same word repeated)
        is_latin, count = lexicon.check_text(text, min_matches=2)
        # Note: "Amen Amen" has only 1 unique keyword
        assert is_latin is False  # Still need 2 unique matches

    def test_check_text_empty_string(self, lexicon):
        """Test empty string returns False."""
        is_latin, count = lexicon.check_text("")
        assert is_latin is False
        assert count == 0

    def test_check_text_ave_maria(self, lexicon):
        """Test Ave Maria prayer matches."""
        text = "Ave Maria gratia plena"
        is_latin, count = lexicon.check_text(text)
        assert is_latin is True
        assert count >= 3  # ave, maria, gratia, plena

    def test_check_text_mixed_language(self, lexicon):
        """Test mixed Latin-Spanish text."""
        text = "El padre dijo ego te baptizo in nomine"
        is_latin, count = lexicon.check_text(text)
        # Should find: ego, te, baptizo, nomine = 4 matches
        assert is_latin is True
        assert count >= 3

"""Tests for pronunciation rendering."""

import pytest

from tenepal.pronunciation.renderer import render_pronunciation, LOCALES


class TestPronunciationRenderer:
    def test_render_german_locale(self):
        phonemes = ["ʃ", "x", "tʃ", "ŋ", "θ", "ɣ", "ʔ", "tɬ", "kʷ", "j", "ts", "w", "a", "e", "i", "o", "u"]
        rendered = render_pronunciation(phonemes, "de")
        assert rendered == "sch ch tsch ng th gh ' tl ku j z w a e i o u"

    def test_render_spanish_locale(self):
        phonemes = ["ʃ", "x", "tʃ", "ŋ", "θ", "ʔ", "tɬ", "kʷ", "j", "ts", "w"]
        rendered = render_pronunciation(phonemes, "es")
        assert rendered == "sh j ch ng z ' tl cu y tz hu"

    def test_render_english_locale(self):
        phonemes = ["ʃ", "x", "tʃ", "ŋ", "θ", "ʔ", "tɬ", "kʷ", "j", "ts", "w"]
        rendered = render_pronunciation(phonemes, "en")
        assert rendered == "sh kh ch ng th ' tl kw y ts w"

    def test_missing_mapping_passes_through(self):
        rendered = render_pronunciation(["ɮ"], "en")
        assert rendered == "ɮ"

    def test_missing_mapping_logs_warning(self, caplog):
        with caplog.at_level("WARNING"):
            render_pronunciation(["ɮ"], "en")
        assert any("No pronunciation mapping" in record.message for record in caplog.records)

    def test_empty_input(self):
        assert render_pronunciation([], "en") == ""

    def test_render_full_segment(self):
        rendered = render_pronunciation(["t", "ʃ", "a"], "de")
        assert rendered == "t sch a"

    def test_available_locales(self):
        assert {"de", "es", "en"}.issubset(set(LOCALES.keys()))

    def test_invalid_locale_raises(self):
        with pytest.raises(ValueError):
            render_pronunciation(["a"], "xx")

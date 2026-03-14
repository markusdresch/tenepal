"""Tests for text-to-IPA conversion utilities."""

import math

import pytest

from tenepal.phoneme.backend import PhonemeSegment
from tenepal.phoneme.text_to_ipa import (
    G2PConverter,
    NahuatlMapG2P,
    NahuatlG2P,
    get_g2p_converter,
    text_to_phonemes,
    words_to_phonemes,
)


class TestNahuatlG2P:
    def test_niltze_greeting(self):
        result = NahuatlG2P().convert("Niltze", "nah")
        assert result == ["n", "i", "l", "ts", "e"]

    def test_tl_affricate(self):
        result = NahuatlG2P().convert("Nahuatl", "nah")
        assert "tɬ" in result

    def test_kw_labialized(self):
        result = NahuatlG2P().convert("cualli", "nah")
        assert "kʷ" in result

    def test_sh_fricative(self):
        result = NahuatlG2P().convert("xochitl", "nah")
        assert "ʃ" in result
        assert "tɬ" in result

    def test_ch_affricate(self):
        result = NahuatlG2P().convert("chilli", "nah")
        assert "tʃ" in result

    def test_empty_string(self):
        result = NahuatlG2P().convert("", "nah")
        assert result == []

    def test_unknown_characters_skipped(self):
        result = NahuatlG2P().convert("abc123", "nah")
        assert "a" in result
        assert len(result) > 0

    def test_case_insensitive(self):
        result_lower = NahuatlG2P().convert("Niltze", "nah")
        result_upper = NahuatlG2P().convert("NILTZE", "nah")
        assert result_lower == result_upper


class TestTimestampDistribution:
    def test_single_word_even_distribution(self):
        result = text_to_phonemes("Niltze", "nah", 0.0, 1.0)
        durations = [seg.duration for seg in result]
        assert len(set(durations)) == 1
        assert math.isclose(sum(durations), 1.0, abs_tol=1e-6)
        for idx in range(1, len(result)):
            assert math.isclose(
                result[idx].start_time,
                result[idx - 1].start_time + result[idx - 1].duration,
                abs_tol=1e-6,
            )

    def test_start_time_offset(self):
        result = text_to_phonemes("Niltze", "nah", 5.0, 1.0)
        assert math.isclose(result[0].start_time, 5.0, abs_tol=1e-6)
        last = result[-1]
        assert math.isclose(last.start_time + last.duration, 6.0, abs_tol=1e-6)

    def test_multi_word_distribution(self):
        result = text_to_phonemes("in Niltze", "nah", 0.0, 2.0)
        assert math.isclose(sum(seg.duration for seg in result), 2.0, abs_tol=1e-6)

    def test_zero_duration(self):
        result = text_to_phonemes("Niltze", "nah", 0.0, 0.0)
        assert all(seg.duration == 0.0 for seg in result)

    def test_empty_text_returns_empty(self):
        assert text_to_phonemes("", "nah", 0.0, 1.0) == []

    def test_returns_phoneme_segments(self):
        result = text_to_phonemes("Niltze", "nah", 0.0, 1.0)
        assert all(isinstance(seg, PhonemeSegment) for seg in result)


class TestWordsToPhonemes:
    def test_single_word_tuple(self):
        result = words_to_phonemes([("Niltze", 0.0, 0.5)], "nah")
        assert math.isclose(sum(seg.duration for seg in result), 0.5, abs_tol=1e-6)

    def test_multiple_word_tuples(self):
        result = words_to_phonemes(
            [("in", 0.0, 0.3), ("Niltze", 0.3, 0.7)],
            "nah",
        )
        assert result
        first_word_end = max(seg.start_time + seg.duration for seg in result[:2])
        last_word_end = max(seg.start_time + seg.duration for seg in result)
        assert math.isclose(first_word_end, 0.3, abs_tol=1e-6)
        assert math.isclose(last_word_end, 1.0, abs_tol=1e-6)

    def test_empty_words_list(self):
        assert words_to_phonemes([], "nah") == []

    def test_word_with_no_phonemes(self):
        result = words_to_phonemes([("123", 0.0, 0.5)], "nah")
        assert result == []


class TestG2PConverterFactory:
    def test_nah_returns_nahuatl_g2p(self):
        converter = get_g2p_converter("nah")
        assert isinstance(converter, G2PConverter)
        assert converter.convert("Nahuatl", "nah")

    def test_nahuatl_variants_return_nahuatl_g2p(self):
        for code in ["ncj", "ncl", "nch", "ncx", "nco", "ncu"]:
            converter = get_g2p_converter(code)
            assert isinstance(converter, G2PConverter)

    def test_unknown_language_returns_converter(self):
        converter = get_g2p_converter("xyz")
        assert converter is not None

    def test_nah_classical_alias_uses_map_converter(self):
        converter = get_g2p_converter("nah-classical")
        assert isinstance(converter, NahuatlMapG2P)

    def test_env_can_force_classical_variant(self, monkeypatch):
        monkeypatch.setenv("TENEPAL_NAH_G2P_VARIANT", "classical")
        converter = get_g2p_converter("nah")
        assert isinstance(converter, NahuatlMapG2P)
        assert converter.variant == "classical"


class TestGracefulFailure:
    def test_g2p_failure_returns_empty(self, monkeypatch):
        def boom(_language):
            raise RuntimeError("boom")

        monkeypatch.setattr("tenepal.phoneme.text_to_ipa.get_g2p_converter", boom)
        result = text_to_phonemes("test", "nah", 0.0, 1.0)
        assert result == []

    def test_g2p_failure_logs_warning(self, monkeypatch, caplog):
        def boom(_language):
            raise RuntimeError("boom")

        monkeypatch.setattr("tenepal.phoneme.text_to_ipa.get_g2p_converter", boom)
        with caplog.at_level("WARNING"):
            result = text_to_phonemes("test", "nah", 0.0, 1.0)
        assert result == []
        assert any("G2P failed" in record.message for record in caplog.records)

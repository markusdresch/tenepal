"""Tests for transcription language routing constants."""

import pytest

from tenepal.transcription.languages import (
    WHISPER_SUPPORTED,
    ISO_639_MAP,
    WHISPER_MODEL_SIZES,
)


def test_whisper_supported_contains_known_languages():
    """Verify WHISPER_SUPPORTED contains the 5 supported languages."""
    expected = {"spa", "eng", "deu", "fra", "ita"}
    assert WHISPER_SUPPORTED == expected


def test_whisper_supported_excludes_nahuatl():
    """Verify Nahuatl (nah) is not in WHISPER_SUPPORTED.

    Whisper supports 99 languages but Nahuatl is not among them.
    """
    assert "nah" not in WHISPER_SUPPORTED


def test_whisper_supported_excludes_other():
    """Verify 'other' fallback language is not in WHISPER_SUPPORTED."""
    assert "other" not in WHISPER_SUPPORTED


def test_iso_639_map_covers_all_supported():
    """Verify ISO_639_MAP has entries for every language in WHISPER_SUPPORTED."""
    for lang in WHISPER_SUPPORTED:
        assert lang in ISO_639_MAP, f"Missing ISO_639_MAP entry for {lang}"


def test_iso_639_map_values_are_two_letter():
    """Verify all ISO_639_MAP values are 2-character ISO 639-1 codes."""
    for lang, code in ISO_639_MAP.items():
        assert isinstance(code, str), f"{lang} maps to non-string: {code}"
        assert len(code) == 2, f"{lang} maps to invalid code: {code}"


def test_whisper_model_sizes():
    """Verify WHISPER_MODEL_SIZES contains all valid model sizes."""
    expected = {"tiny", "base", "small", "medium", "large", "turbo"}
    assert WHISPER_MODEL_SIZES == expected

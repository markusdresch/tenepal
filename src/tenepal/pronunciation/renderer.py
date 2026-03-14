"""Pronunciation rendering: IPA to user-friendly spelling conventions."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_GERMAN_MAP = {
    "tʃ": "tsch",
    "tɬ": "tl",
    "kʷ": "ku",
    "ts": "z",
    "ʃ": "sch",
    "x": "ch",
    "ŋ": "ng",
    "θ": "th",
    "ɣ": "gh",
    "ʔ": "'",
    "j": "j",
    "w": "w",
    "ð": "d",
    "ɲ": "nj",
    "ʒ": "sch",
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
}

_SPANISH_MAP = {
    "tʃ": "ch",
    "tɬ": "tl",
    "kʷ": "cu",
    "ts": "tz",
    "ʃ": "sh",
    "x": "j",
    "ŋ": "ng",
    "θ": "z",
    "ɣ": "g",
    "ʔ": "'",
    "j": "y",
    "w": "hu",
    "ð": "d",
    "ɲ": "ñ",
    "ʒ": "sh",
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
}

_ENGLISH_MAP = {
    "tʃ": "ch",
    "tɬ": "tl",
    "kʷ": "kw",
    "ts": "ts",
    "ʃ": "sh",
    "x": "kh",
    "ŋ": "ng",
    "θ": "th",
    "ɣ": "gh",
    "ʔ": "'",
    "j": "y",
    "w": "w",
    "ð": "th",
    "ɲ": "ny",
    "ʒ": "zh",
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
}

LOCALES = {"de": _GERMAN_MAP, "es": _SPANISH_MAP, "en": _ENGLISH_MAP}


def render_pronunciation(phonemes: list[str], locale: str) -> str:
    """Render IPA phoneme list into locale-specific spelling approximation."""
    if locale not in LOCALES:
        raise ValueError(f"Unsupported locale: {locale}")

    if not phonemes:
        return ""

    mapping = LOCALES[locale]
    rendered = []
    for phoneme in phonemes:
        if phoneme in mapping:
            rendered.append(mapping[phoneme])
        else:
            logger.warning(
                "No pronunciation mapping for '%s' in locale '%s'; rendering IPA unchanged",
                phoneme,
                locale,
            )
            rendered.append(phoneme)

    return " ".join(rendered)

"""Text-to-IPA conversion utilities for text-based ASR outputs."""

from __future__ import annotations

import logging
import os
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .backend import PhonemeSegment

logger = logging.getLogger(__name__)


class G2PConverter(ABC):
    """Abstract base class for grapheme-to-phoneme converters."""

    @abstractmethod
    def convert(self, word: str, language: str) -> list[str]:
        """Convert a single word to list of IPA phoneme strings."""


class NahuatlG2P(G2PConverter):
    """Rule-based Nahuatl G2P converter (no external dependencies)."""

    _vowels = {"a", "e", "i", "o", "u"}

    def convert(self, word: str, language: str) -> list[str]:
        if not word:
            return []

        text = word.lower()
        phonemes: list[str] = []
        i = 0
        length = len(text)

        while i < length:
            ch = text[i]
            nxt = text[i + 1] if i + 1 < length else ""
            nxt2 = text[i + 2] if i + 2 < length else ""
            prev = text[i - 1] if i - 1 >= 0 else ""

            # Longest-match-first rules
            if ch == "t" and nxt == "l":
                phonemes.append("tɬ")
                i += 2
                continue
            if ch == "t" and nxt == "z":
                phonemes.append("ts")
                i += 2
                continue
            if ch == "c" and nxt == "h":
                phonemes.append("tʃ")
                i += 2
                continue
            if ch == "s" and nxt == "h":
                phonemes.append("ʃ")
                i += 2
                continue
            if ch == "l" and nxt == "l":
                phonemes.append("l")
                i += 2
                continue
            if ch == "h" and nxt == "u" and nxt2 in self._vowels:
                phonemes.append("w")
                i += 2
                continue
            if ch == "u" and nxt == "h" and prev in self._vowels:
                phonemes.append("w")
                i += 2
                continue
            if ch == "c" and nxt == "u" and nxt2 in self._vowels:
                phonemes.append("kʷ")
                i += 2
                continue
            if ch == "q" and nxt == "u" and nxt2 in self._vowels:
                phonemes.append("kʷ")
                i += 2
                continue

            # Single-character rules
            if ch == "x":
                phonemes.append("ʃ")
            elif ch == "h":
                if i != length - 1:
                    phonemes.append("ʔ")
            elif ch == "c":
                if nxt in {"e", "i"}:
                    phonemes.append("s")
                else:
                    phonemes.append("k")
            elif ch == "z":
                phonemes.append("s")
            elif ch in self._vowels:
                phonemes.append(ch)
            elif ch == "p":
                phonemes.append("p")
            elif ch == "t":
                phonemes.append("t")
            elif ch == "k":
                phonemes.append("k")
            elif ch == "m":
                phonemes.append("m")
            elif ch == "n":
                phonemes.append("n")
            elif ch == "l":
                phonemes.append("l")
            elif ch == "s":
                phonemes.append("s")
            elif ch == "y":
                phonemes.append("j")
            elif ch == "w":
                phonemes.append("w")
            # unknown characters are skipped

            i += 1

        return phonemes


class NahuatlMapG2P(G2PConverter):
    """Map-based Nahuatl G2P using editable CSV rule tables."""

    _cache: dict[str, list[tuple[str, list[str]]]] = {}

    def __init__(self, variant: str = "modern") -> None:
        self.variant = variant if variant in {"modern", "classical"} else "modern"
        self.rules = self._load_rules(self.variant)

    @staticmethod
    def _map_path(variant: str) -> Path:
        filename = "nah-modern.csv" if variant == "modern" else "nah-classical.csv"
        return Path(__file__).resolve().parent.parent / "data" / "epitran_maps" / filename

    @classmethod
    def _load_rules(cls, variant: str) -> list[tuple[str, list[str]]]:
        if variant in cls._cache:
            return cls._cache[variant]
        path = cls._map_path(variant)
        if not path.exists():
            raise FileNotFoundError(f"Nahuatl map not found: {path}")

        rules: list[tuple[str, list[str]]] = []
        for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if idx == 0 or not raw.strip():
                continue
            parts = raw.split(",")
            if len(parts) < 2:
                continue
            g = parts[0].strip().lower()
            ipa = parts[1].strip()
            if not g or not ipa:
                continue
            rules.append((g, ipa.split()))

        rules.sort(key=lambda x: len(x[0]), reverse=True)
        cls._cache[variant] = rules
        return rules

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().replace("’", "'")
        return unicodedata.normalize("NFC", text)

    def convert(self, word: str, language: str) -> list[str]:
        if not word:
            return []
        text = self._normalize(word)
        phonemes: list[str] = []
        i = 0
        while i < len(text):
            matched = False
            for grapheme, ipa_parts in self.rules:
                if text.startswith(grapheme, i):
                    phonemes.extend(ipa_parts)
                    i += len(grapheme)
                    matched = True
                    break
            if matched:
                continue
            i += 1
        return phonemes


class EpitranG2P(G2PConverter):
    """Optional Epitran-based G2P converter."""

    _language_map = {
        "nah": "nah-Latn",
        "spa": "spa-Latn",
        "eng": "eng-Latn",
        "deu": "deu-Latn",
    }

    def __init__(self) -> None:
        try:
            import epitran  # noqa: F401

            self._epitran = epitran
            self._available = True
        except Exception:
            self._epitran = None
            self._available = False

    def convert(self, word: str, language: str) -> list[str]:
        if not self._available:
            return []
        lang_code = self._language_map.get(language)
        if not lang_code:
            return []

        transliterator = self._epitran.Epitran(lang_code)
        ipa = transliterator.transliterate(word)
        return [ch for ch in ipa if ch.strip()]


_NAHUATL_VARIANTS = {"nah", "ncj", "ncl", "nch", "ncx", "nco", "ncu"}


def _resolve_nahuatl_variant(language: str) -> str:
    env = os.getenv("TENEPAL_NAH_G2P_VARIANT", "").strip().lower()
    if env in {"modern", "classical"}:
        return env
    lang = (language or "").lower()
    if any(tag in lang for tag in ("classical", "cls", "florentine")):
        return "classical"
    return "modern"


def get_g2p_converter(language: str) -> G2PConverter:
    """Return a G2P converter appropriate for the requested language."""
    lang_norm = (language or "").lower()
    if lang_norm in _NAHUATL_VARIANTS or lang_norm.startswith("nah-"):
        try:
            return NahuatlMapG2P(_resolve_nahuatl_variant(lang_norm))
        except Exception as exc:
            logger.warning("Map-based Nahuatl G2P unavailable (%s), falling back to rules.", exc)
            return NahuatlG2P()

    epitran_converter = EpitranG2P()
    if epitran_converter._available:
        return epitran_converter

    logger.warning(
        "Epitran not available for language '%s'; falling back to Nahuatl rules.",
        language,
    )
    return NahuatlG2P()


def _flatten(items: Iterable[Iterable[str]]) -> list[str]:
    return [item for sub in items for item in sub]


def text_to_phonemes(
    text: str,
    language: str,
    start_time: float,
    duration: float,
) -> list[PhonemeSegment]:
    """Convert orthographic text to phoneme segments with distributed timing."""
    try:
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        converter = get_g2p_converter(language)
        word_phonemes: list[list[str]] = []
        for word in words:
            try:
                word_phonemes.append(converter.convert(word, language))
            except Exception as exc:
                logger.warning("G2P failed for word '%s': %s", word, exc)
                word_phonemes.append([])

        phonemes = _flatten(word_phonemes)
        total = len(phonemes)
        if total == 0:
            return []

        per_phoneme = duration / total if total else 0.0
        segments: list[PhonemeSegment] = []
        for idx, phoneme in enumerate(phonemes):
            seg_start = start_time + idx * per_phoneme
            segments.append(
                PhonemeSegment(
                    phoneme=phoneme,
                    start_time=seg_start,
                    duration=per_phoneme,
                )
            )
        return segments

    except Exception as exc:
        logger.warning("G2P failed for text '%s': %s", text, exc)
        return []


def words_to_phonemes(
    words: list[tuple[str, float, float]],
    language: str,
) -> list[PhonemeSegment]:
    """Convert word-level timestamps to phoneme segments."""
    if not words:
        return []

    converter = get_g2p_converter(language)
    segments: list[PhonemeSegment] = []

    for word, start_time, duration in words:
        try:
            phonemes = converter.convert(word, language)
        except Exception as exc:
            logger.warning("G2P failed for word '%s': %s", word, exc)
            continue

        total = len(phonemes)
        if total == 0:
            continue
        per_phoneme = duration / total if total else 0.0
        for idx, phoneme in enumerate(phonemes):
            seg_start = start_time + idx * per_phoneme
            segments.append(
                PhonemeSegment(
                    phoneme=phoneme,
                    start_time=seg_start,
                    duration=per_phoneme,
                )
            )

    return segments

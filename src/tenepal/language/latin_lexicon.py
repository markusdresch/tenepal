"""Latin liturgical phrase recognition for Whisper text segments.

This module provides keyword-based detection of Latin liturgical text
(e.g., from Catholic mass, sacraments) that Whisper transcribes correctly
but might be misidentified as Nahuatl or Spanish due to unfamiliar vocabulary.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LatinLexicon:
    """Keyword-based recognizer for Latin liturgical text.

    Uses a curated list of common Latin liturgical keywords (e.g., "ego",
    "baptizo", "patris") to detect Latin phrases in Whisper transcriptions.
    When 3+ keywords match, the text is classified as Latin.

    Attributes:
        _keywords: Set of Latin liturgical keywords (normalized lowercase)
    """

    def __init__(self, lexicon_path: Optional[Path] = None) -> None:
        """Initialize LatinLexicon with keywords from lat_lexicon.json.

        Args:
            lexicon_path: Optional custom path to lexicon JSON file.
                         If None, loads from package data.
        """
        if lexicon_path:
            if not lexicon_path.exists():
                raise FileNotFoundError(f"Lexicon not found: {lexicon_path}")
            with open(lexicon_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Load from package data using importlib.resources
            try:
                # Python 3.9+ approach
                from importlib.resources import files

                data_dir = files("tenepal.data")
                lexicon_file = data_dir / "lat_lexicon.json"
                lexicon_text = lexicon_file.read_text(encoding="utf-8")
                data = json.loads(lexicon_text)
            except (ImportError, AttributeError):
                # Fallback for older Python
                import pkg_resources

                lexicon_text = pkg_resources.resource_string(
                    "tenepal.data", "lat_lexicon.json"
                ).decode("utf-8")
                data = json.loads(lexicon_text)

        self._keywords = set(data["keywords"])
        logger.debug(f"Loaded {len(self._keywords)} Latin keywords")

    def check_text(
        self, text: str, min_matches: int = 3
    ) -> tuple[bool, int]:
        """Check if text contains Latin liturgical keywords.

        Normalizes text (lowercase, strip accents), tokenizes into words,
        and counts matches against Latin keyword set. Returns True if
        count >= min_matches.

        Args:
            text: Text to check (typically from Whisper transcription)
            min_matches: Minimum keyword matches to classify as Latin (default: 3)

        Returns:
            Tuple of (is_latin, match_count)
        """
        if not text:
            return (False, 0)

        # Normalize and tokenize
        normalized_text = self._normalize_text(text)
        words = self._tokenize(normalized_text)

        # Count unique keyword matches
        matches = set(words) & self._keywords
        count = len(matches)

        is_latin = count >= min_matches
        return (is_latin, count)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching.

        Converts to lowercase and strips accents/diacritics.

        Args:
            text: Raw text string

        Returns:
            Normalized text string
        """
        # Lowercase
        text = text.lower()

        # Strip accents using Unicode decomposition
        # NFD decomposes characters like "á" into "a" + combining acute accent
        nfd = unicodedata.normalize("NFD", text)
        # Filter out combining characters (category Mn)
        stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")

        return stripped

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize normalized text into words.

        Splits on whitespace and punctuation, keeping only alphabetic tokens.

        Args:
            text: Normalized text string

        Returns:
            List of word tokens
        """
        # Split on non-alphabetic characters
        words = re.findall(r"[a-z]+", text)
        return words

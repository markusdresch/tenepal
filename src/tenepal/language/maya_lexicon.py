"""Yucatec Maya lexicon for recognizing short known words from IPA sequences.

This module provides lexical recognition for short Maya words that might not
accumulate enough phoneme markers for language ID. Uses fuzzy IPA matching to
handle Allosaurus transcription variations, especially ejective decomposition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LexiconMatch:
    """Result of matching a phoneme sequence to a lexicon entry.

    Attributes:
        word: Orthographic Maya form (e.g., "ba'al")
        ipa: Canonical IPA phoneme list from lexicon
        score: Match confidence (0.0 to 1.0, where 1.0 = exact match)
        start_idx: Index in input where match begins (for subsequence matching)
        length: Number of phonemes matched
    """

    word: str
    ipa: list[str]
    score: float
    start_idx: int
    length: int


class MayaLexicon:
    """Lexicon-based recognizer for short Yucatec Maya words.

    Loads a curated list of common short Maya words with their canonical IPA
    transcriptions. Uses fuzzy edit-distance matching to handle Allosaurus
    transcription variations (especially ejective and affricate decomposition).

    Attributes:
        _entries: List of lexicon entries (dicts with word, ipa, gloss)
    """

    # Ejective normalization: Allosaurus often decomposes ejectives
    # e.g., kʼ → [k, ʼ], tsʼ → [t, s, ʼ], tʃʼ → [t, ʃ, ʼ]
    _EJECTIVE_DECOMPOSITIONS = {
        "kʼ": [("k", "ʼ")],
        "pʼ": [("p", "ʼ")],
        "tʼ": [("t", "ʼ")],
        "tsʼ": [("t", "s", "ʼ"), ("ts", "ʼ")],
        "tʃʼ": [("t", "ʃ", "ʼ"), ("tʃ", "ʼ")],
    }

    # Affricate normalization: Allosaurus often decomposes affricates
    _AFFRICATE_DECOMPOSITIONS = {
        "ts": ("t", "s"),
        "tʃ": ("t", "ʃ"),
    }

    # Import allophone normalization from identifier
    _ALLOPHONE_MAP = {
        "ð": "d",
        "β": "b",
        "ɣ": "ɡ",
        "ɸ": "f",
        "ɻ": "ɾ",
        "r": "ɾ",
    }

    # IPA spacing modifiers to strip
    _SPACING_MODIFIERS = frozenset("ːʲʰˤʼ")

    def __init__(self, lexicon_path: Optional[Path] = None) -> None:
        """Initialize MayaLexicon with data from may_lexicon.json.

        Args:
            lexicon_path: Optional custom path to lexicon JSON file.
                         If None, loads from package data.
        """
        if lexicon_path:
            if not lexicon_path.exists():
                raise FileNotFoundError(f"Lexicon not found: {lexicon_path}")
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self._entries = json.load(f)
        else:
            # Load from package data using importlib.resources
            try:
                # Python 3.9+ approach
                from importlib.resources import files

                data_dir = files("tenepal.data")
                lexicon_file = data_dir / "may_lexicon.json"
                lexicon_text = lexicon_file.read_text(encoding="utf-8")
                self._entries = json.loads(lexicon_text)
            except (ImportError, AttributeError):
                # Fallback for older Python
                import pkg_resources

                lexicon_text = pkg_resources.resource_string(
                    "tenepal.data", "may_lexicon.json"
                ).decode("utf-8")
                self._entries = json.loads(lexicon_text)

        logger.debug(f"Loaded {len(self._entries)} Maya lexicon entries")

    def match(
        self, phonemes: list[str], max_distance: float = 0.3
    ) -> LexiconMatch | None:
        """Find best matching lexicon entry for given phoneme sequence.

        Uses normalized edit distance for fuzzy matching. Returns best match
        if normalized distance <= max_distance, otherwise None.

        Args:
            phonemes: IPA phoneme sequence to match
            max_distance: Maximum normalized edit distance (0.0-1.0)

        Returns:
            LexiconMatch with best match and score, or None if no match
        """
        if not phonemes:
            return None

        # Normalize input phonemes
        normalized_input = self._normalize_sequence(phonemes)

        best_match: LexiconMatch | None = None
        best_score = 0.0

        for entry in self._entries:
            lexicon_ipa = entry["ipa"]
            normalized_lexicon = self._normalize_sequence(lexicon_ipa)

            # Try multiple variant combinations (ejectives and affricates)
            expanded_input = self._expand_segments(normalized_input)
            expanded_lexicon = self._expand_segments(normalized_lexicon)

            variants = [
                (normalized_input, normalized_lexicon),  # Direct comparison
                (expanded_input, normalized_lexicon),  # Expanded input vs normal lexicon
                (normalized_input, expanded_lexicon),  # Normal input vs expanded lexicon
            ]

            for input_seq, lexicon_seq in variants:
                if input_seq == variants[0][0] and lexicon_seq == variants[0][1]:
                    # Skip if it's the same as direct comparison
                    pass
                elif input_seq == normalized_input and lexicon_seq == normalized_lexicon:
                    # Always try direct comparison
                    pass
                elif input_seq == expanded_input and input_seq == normalized_input:
                    # Skip if expansion didn't change anything
                    continue
                elif lexicon_seq == expanded_lexicon and lexicon_seq == normalized_lexicon:
                    # Skip if expansion didn't change anything
                    continue

                distance = self._edit_distance(input_seq, lexicon_seq)
                max_len = max(len(input_seq), len(lexicon_seq))
                normalized_dist = distance / max_len if max_len > 0 else 0.0
                score = 1.0 - normalized_dist

                if normalized_dist <= max_distance and score > best_score:
                    best_score = score
                    best_match = LexiconMatch(
                        word=entry["word"],
                        ipa=lexicon_ipa,
                        score=score,
                        start_idx=0,
                        length=len(phonemes),
                    )

        return best_match

    def match_subsequence(
        self, phonemes: list[str], min_length: int = 2
    ) -> list[LexiconMatch]:
        """Find all lexicon words embedded in a longer phoneme sequence.

        Uses sliding window approach to scan for matches. Returns all matches
        with length >= min_length, sorted by score (best first).

        Args:
            phonemes: Long IPA phoneme sequence to scan
            min_length: Minimum word length to consider

        Returns:
            List of LexiconMatch objects, sorted by score descending
        """
        if not phonemes:
            return []

        matches: list[LexiconMatch] = []

        # For each entry, try sliding windows around its length
        for entry in self._entries:
            lexicon_ipa = entry["ipa"]
            if len(lexicon_ipa) < min_length:
                continue

            word_len = len(lexicon_ipa)
            # Try windows from word_len-1 to word_len+1 to allow fuzzy matching
            for window_size in range(max(min_length, word_len - 1), word_len + 2):
                for start_idx in range(len(phonemes) - window_size + 1):
                    window = phonemes[start_idx : start_idx + window_size]
                    match = self.match(window, max_distance=0.3)
                    if match and match.word == entry["word"]:
                        # Adjust start_idx and length to reflect position in full sequence
                        match.start_idx = start_idx
                        match.length = window_size
                        matches.append(match)

        # Remove duplicates: keep best match for each (word, start_idx) pair
        best_matches = {}
        for match in matches:
            key = (match.word, match.start_idx)
            if key not in best_matches or match.score > best_matches[key].score:
                best_matches[key] = match

        # Convert to list and sort by score descending
        unique_matches = list(best_matches.values())
        unique_matches.sort(key=lambda m: m.score, reverse=True)

        return unique_matches

    def _normalize_sequence(self, phonemes: list[str]) -> list[str]:
        """Normalize IPA sequence for matching.

        Strips modifiers, applies allophone normalization.

        Args:
            phonemes: Raw IPA phoneme list

        Returns:
            Normalized phoneme list
        """
        normalized = []
        for p in phonemes:
            # Strip spacing modifiers
            stripped = "".join(c for c in p if c not in self._SPACING_MODIFIERS)
            # Strip combining diacritics (Unicode category Mn)
            import unicodedata

            stripped = "".join(
                c for c in stripped if unicodedata.category(c) != "Mn"
            )
            # Apply allophone map
            normalized_phoneme = self._ALLOPHONE_MAP.get(stripped, stripped)
            normalized.append(normalized_phoneme)

        return normalized

    def _expand_segments(self, phonemes: list[str]) -> list[str]:
        """Expand ejectives and affricates into decomposed form for matching.

        Converts single ejectives/affricates into multi-phoneme sequences
        to handle Allosaurus decomposition. Also handles reverse: if sequence
        has decomposed form, contracts to ejective/affricate.

        Args:
            phonemes: Phoneme sequence

        Returns:
            Sequence with ejectives/affricates expanded
        """
        expanded = []
        i = 0
        while i < len(phonemes):
            phoneme = phonemes[i]

            # Check if this phoneme is an ejective (try all decomposition variants)
            if phoneme in self._EJECTIVE_DECOMPOSITIONS:
                # Use first decomposition variant
                decomp = self._EJECTIVE_DECOMPOSITIONS[phoneme][0]
                expanded.extend(decomp)
                i += 1
            # Check if this is a decomposed ejective (multiple phonemes)
            elif i + 1 < len(phonemes):
                # Try to contract decomposed ejectives back
                pair = (phoneme, phonemes[i + 1])
                triple = None
                if i + 2 < len(phonemes):
                    triple = (phoneme, phonemes[i + 1], phonemes[i + 2])

                ejective_contracted = None

                # Check for 3-phoneme ejective decompositions (tsʼ, tʃʼ)
                if triple:
                    for ej, decomps in self._EJECTIVE_DECOMPOSITIONS.items():
                        if triple in decomps:
                            ejective_contracted = ej
                            expanded.append(ejective_contracted)
                            i += 3
                            break

                # Check for 2-phoneme ejective decompositions (kʼ, pʼ, tʼ)
                if not ejective_contracted:
                    for ej, decomps in self._EJECTIVE_DECOMPOSITIONS.items():
                        if pair in decomps:
                            ejective_contracted = ej
                            expanded.append(ejective_contracted)
                            i += 2
                            break

                # Check for affricate decomposition
                if not ejective_contracted:
                    for aff, decomp in self._AFFRICATE_DECOMPOSITIONS.items():
                        if pair == decomp:
                            expanded.append(aff)
                            i += 2
                            ejective_contracted = True  # Use as flag
                            break

                if not ejective_contracted:
                    # Check if phoneme is affricate to expand
                    if phoneme in self._AFFRICATE_DECOMPOSITIONS:
                        p1, p2 = self._AFFRICATE_DECOMPOSITIONS[phoneme]
                        expanded.extend([p1, p2])
                        i += 1
                    else:
                        expanded.append(phoneme)
                        i += 1
            else:
                # Check if phoneme is affricate to expand
                if phoneme in self._AFFRICATE_DECOMPOSITIONS:
                    p1, p2 = self._AFFRICATE_DECOMPOSITIONS[phoneme]
                    expanded.extend([p1, p2])
                    i += 1
                else:
                    expanded.append(phoneme)
                    i += 1

        return expanded

    def _edit_distance(self, seq1: list[str], seq2: list[str]) -> int:
        """Calculate Levenshtein edit distance between two phoneme sequences.

        Uses dynamic programming approach.

        Args:
            seq1: First phoneme sequence
            seq2: Second phoneme sequence

        Returns:
            Edit distance (number of insertions, deletions, substitutions)
        """
        m, n = len(seq1), len(seq2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # deletion
                        dp[i][j - 1],  # insertion
                        dp[i - 1][j - 1],  # substitution
                    )

        return dp[m][n]

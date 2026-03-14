"""Nahuatl lexicon for recognizing short known words from IPA sequences.

This module provides lexical recognition for short Nahuatl words that might not
accumulate enough phoneme markers for language ID. Uses fuzzy IPA matching to
handle Allosaurus transcription variations.

Optimized: pre-normalized entries, direct entry comparison in match_subsequence,
early-exit edit distance with max_distance cutoff.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LexiconMatch:
    """Result of matching a phoneme sequence to a lexicon entry.

    Attributes:
        word: Orthographic Nahuatl form (e.g., "koali")
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


class NahuatlLexicon:
    """Lexicon-based recognizer for short Nahuatl words.

    Loads a curated list of common short Nahuatl words with their canonical IPA
    transcriptions. Uses fuzzy edit-distance matching to handle Allosaurus
    transcription variations (especially affricate decomposition).

    Attributes:
        _entries: List of lexicon entries (dicts with word, ipa, gloss)
    """

    # Affricate normalization: Allosaurus often decomposes affricates
    _AFFRICATE_DECOMPOSITIONS = {
        "tɬ": ("t", "l"),
        "ts": ("t", "s"),
        "tʃ": ("t", "ʃ"),
        "kʷ": ("k", "w"),
    }

    # Reverse map: decomposed pair -> affricate
    _AFFRICATE_CONTRACTIONS = {
        ("t", "l"): "tɬ",
        ("t", "s"): "ts",
        ("t", "ʃ"): "tʃ",
        ("k", "w"): "kʷ",
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

    def __init__(
        self,
        lexicon_path: Optional[Path] = None,
        min_freq: int = 3,
    ) -> None:
        """Initialize NahuatlLexicon with data from nah_lexicon_merged.json.

        Args:
            lexicon_path: Optional custom path to lexicon JSON file.
                         If None, loads from package data.
            min_freq: Minimum corpus frequency for an entry to be loaded.
                     Curated entries are always kept regardless of this
                     threshold.
        """
        if lexicon_path:
            if not lexicon_path.exists():
                raise FileNotFoundError(f"Lexicon not found: {lexicon_path}")
            with open(lexicon_path, "r", encoding="utf-8") as f:
                all_entries = json.load(f)
        else:
            try:
                from importlib.resources import files

                data_dir = files("tenepal.data")
                lexicon_file = data_dir / "nah_lexicon_merged.json"
                lexicon_text = lexicon_file.read_text(encoding="utf-8")
                all_entries = json.loads(lexicon_text)
            except (ImportError, AttributeError):
                import pkg_resources

                lexicon_text = pkg_resources.resource_string(
                    "tenepal.data", "nah_lexicon_merged.json"
                ).decode("utf-8")
                all_entries = json.loads(lexicon_text)

        # Keep curated entries unconditionally and filter rare corpus entries.
        self._entries = [
            e for e in all_entries
            if e.get("source") == "curated"
            or e.get("freq") is None
            or e.get("freq", 0) >= min_freq
        ]

        # ── Pre-compute normalized forms at init time ──
        # Avoids re-normalizing on every match call.
        self._prepared: list[dict] = []
        for e in self._entries:
            norm = self._normalize_sequence(e["ipa"])
            exp = self._expand_affricates(norm)
            self._prepared.append({
                "word": e["word"],
                "ipa": e["ipa"],
                "norm": norm,
                "exp": exp,
                "length": len(norm),
                # First-phoneme set for quick pre-filter
                "first": norm[0] if norm else "",
                "first_exp": exp[0] if exp else "",
            })

        # ── Build phoneme index for candidate pre-filtering ──
        # Maps first phoneme -> list of entry indices
        self._first_phoneme_index: dict[str, list[int]] = {}
        for idx, p in enumerate(self._prepared):
            for first in {p["first"], p["first_exp"]}:
                if first:
                    self._first_phoneme_index.setdefault(first, []).append(idx)

        logger.debug(
            f"Loaded {len(self._entries)} Nahuatl lexicon entries "
            f"(from {len(all_entries)} total, min_freq={min_freq})"
        )

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

        normalized_input = self._normalize_sequence(phonemes)
        expanded_input = self._expand_affricates(normalized_input)

        best_match: LexiconMatch | None = None
        best_score = 0.0

        for prep in self._prepared:
            score = self._score_entry(
                normalized_input, expanded_input,
                prep, max_distance, best_score,
            )
            if score is not None and score > best_score:
                best_score = score
                best_match = LexiconMatch(
                    word=prep["word"],
                    ipa=prep["ipa"],
                    score=score,
                    start_idx=0,
                    length=len(phonemes),
                )

        return best_match

    def match_subsequence(
        self, phonemes: list[str], min_length: int = 2, max_distance: float = 0.3,
    ) -> list[LexiconMatch]:
        """Find all lexicon words embedded in a longer phoneme sequence.

        Optimized: compares each window directly against the current entry
        instead of calling match() (which would rescan all entries).

        Args:
            phonemes: Long IPA phoneme sequence to scan
            min_length: Minimum word length to consider
            max_distance: Maximum normalized edit distance (0.0-1.0)

        Returns:
            List of LexiconMatch objects, sorted by score descending
        """
        if not phonemes:
            return []

        # Pre-normalize the full input sequence once
        full_norm = self._normalize_sequence(phonemes)
        full_exp = self._expand_affricates(full_norm)
        input_len = len(full_norm)

        # best_matches keyed by (word, start_idx)
        best_matches: dict[tuple[str, int], LexiconMatch] = {}

        for prep in self._prepared:
            word_len = prep["length"]
            if word_len < min_length:
                continue

            # Window sizes: word_len-1 to word_len+1
            min_win = max(min_length, word_len - 1)
            max_win = min(word_len + 2, input_len + 1)  # cap at input length

            for window_size in range(min_win, max_win):
                max_positions = input_len - window_size + 1
                if max_positions <= 0:
                    continue

                for start_idx in range(max_positions):
                    window_norm = full_norm[start_idx: start_idx + window_size]
                    window_exp = full_exp[start_idx: start_idx + window_size]

                    # Direct comparison against THIS entry only (not all entries)
                    score = self._score_entry(
                        window_norm, window_exp,
                        prep, max_distance, 0.0,
                    )
                    if score is None:
                        continue

                    key = (prep["word"], start_idx)
                    if key not in best_matches or score > best_matches[key].score:
                        best_matches[key] = LexiconMatch(
                            word=prep["word"],
                            ipa=prep["ipa"],
                            score=score,
                            start_idx=start_idx,
                            length=window_size,
                        )

        result = list(best_matches.values())
        result.sort(key=lambda m: m.score, reverse=True)
        return result

    def _score_entry(
        self,
        input_norm: list[str],
        input_exp: list[str],
        prep: dict,
        max_distance: float,
        min_score: float,
    ) -> float | None:
        """Score a single input against a single prepared entry.

        Tries normalized and expanded variants. Returns score if above
        min_score and within max_distance, else None.
        """
        best = None

        # Pairs to try: (input_variant, lexicon_variant)
        # Only try expanded variants if they differ from normalized
        pairs = [(input_norm, prep["norm"])]
        if input_exp != input_norm:
            pairs.append((input_exp, prep["norm"]))
        if prep["exp"] != prep["norm"]:
            pairs.append((input_norm, prep["exp"]))

        for inp, lex in pairs:
            max_len = max(len(inp), len(lex))
            if max_len == 0:
                continue

            # Early length check: if length difference alone exceeds threshold, skip
            len_diff = abs(len(inp) - len(lex))
            if len_diff / max_len > max_distance:
                continue

            # Compute max allowed absolute distance for early exit
            max_abs_dist = int(max_distance * max_len)

            distance = self._edit_distance_bounded(inp, lex, max_abs_dist)
            if distance is None:
                continue  # exceeded bound

            score = 1.0 - (distance / max_len)
            if score > (best or min_score):
                best = score

        return best

    def _normalize_sequence(self, phonemes: list[str]) -> list[str]:
        """Normalize IPA sequence for matching.

        Strips modifiers, applies allophone normalization.
        """
        normalized = []
        for p in phonemes:
            stripped = "".join(c for c in p if c not in self._SPACING_MODIFIERS)
            stripped = "".join(
                c for c in stripped if unicodedata.category(c) != "Mn"
            )
            normalized_phoneme = self._ALLOPHONE_MAP.get(stripped, stripped)
            normalized.append(normalized_phoneme)
        return normalized

    def _expand_affricates(self, phonemes: list[str]) -> list[str]:
        """Expand affricates into decomposed form for matching.

        Converts single affricates (tɬ, ts, tʃ, kʷ) into two-phoneme sequences
        (t+l, t+s, t+ʃ, k+w) to handle Allosaurus decomposition.
        Also handles reverse: if sequence has [t, l], replaces with [tɬ].
        """
        expanded = []
        i = 0
        while i < len(phonemes):
            phoneme = phonemes[i]

            if phoneme in self._AFFRICATE_DECOMPOSITIONS:
                p1, p2 = self._AFFRICATE_DECOMPOSITIONS[phoneme]
                expanded.extend([p1, p2])
                i += 1
            elif i + 1 < len(phonemes):
                pair = (phoneme, phonemes[i + 1])
                affricate = self._AFFRICATE_CONTRACTIONS.get(pair)
                if affricate:
                    expanded.append(affricate)
                    i += 2
                else:
                    expanded.append(phoneme)
                    i += 1
            else:
                expanded.append(phoneme)
                i += 1

        return expanded

    @staticmethod
    def _edit_distance_bounded(
        seq1: list[str], seq2: list[str], max_dist: int
    ) -> int | None:
        """Levenshtein edit distance with early exit.

        Returns distance if <= max_dist, else None (saves computation).
        Uses single-row DP for memory efficiency.
        """
        m, n = len(seq1), len(seq2)

        # Quick checks
        if abs(m - n) > max_dist:
            return None
        if m == 0:
            return n if n <= max_dist else None
        if n == 0:
            return m if m <= max_dist else None

        # Ensure m <= n for the single-row optimization
        if m > n:
            seq1, seq2 = seq2, seq1
            m, n = n, m

        # Single-row DP (O(min(m,n)) space)
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            row_min = curr[0]

            for j in range(1, n + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost # substitution
                )
                if curr[j] < row_min:
                    row_min = curr[j]

            # Early exit: if minimum in this row exceeds bound, no solution
            if row_min > max_dist:
                return None

            prev, curr = curr, prev

        return prev[n] if prev[n] <= max_dist else None

    # Keep old method for backward compat (tests etc.)
    def _edit_distance(self, seq1: list[str], seq2: list[str]) -> int:
        """Calculate Levenshtein edit distance (unbounded, for compat)."""
        result = self._edit_distance_bounded(seq1, seq2, max(len(seq1), len(seq2)))
        return result if result is not None else max(len(seq1), len(seq2))

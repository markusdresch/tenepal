"""Post-process Whisper segments to detect hallucinations.

Whisper hallucinates confident-looking but fake text for languages it doesn't
know (e.g., Nahuatl, Maya). This validator catches those hallucinations using
multiple heuristic checks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ValidationResult:
    """Result of Whisper segment validation.

    Attributes:
        is_valid: Overall verdict (True = real Spanish, False = hallucination)
        confidence: 0.0 (definitely hallucination) to 1.0 (definitely real)
        reason: Human-readable explanation of the decision
        checks: Individual check scores for debugging
    """

    is_valid: bool
    confidence: float
    reason: str
    checks: dict[str, float]


class WhisperValidator:
    """Validator for detecting Whisper hallucinations on indigenous language audio.

    Uses multiple heuristic checks:
    - Spanish lexicon lookup (VALID-01)
    - Apostrophe density detection (VALID-02)
    - Character pattern analysis (VALID-03)
    - Average log probability confidence (VALID-04)
    - Repetition loop detection (VALID-05)

    Attributes:
        spanish_words: Frozen set of normalized Spanish words for O(1) lookup
    """

    def __init__(self, wordlist_path: Optional[Path] = None) -> None:
        """Initialize WhisperValidator with Spanish word list.

        Args:
            wordlist_path: Optional custom path to Spanish word list.
                          If None, loads from package data (spa_wordlist.txt)

        Raises:
            FileNotFoundError: If wordlist_path provided but doesn't exist
        """
        if wordlist_path is not None:
            if not wordlist_path.exists():
                raise FileNotFoundError(f"Word list not found: {wordlist_path}")
            with open(wordlist_path, "r", encoding="utf-8") as f:
                words = f.read().split()
        else:
            # Load from package data
            try:
                # Try Python 3.9+ approach
                from importlib.resources import files

                data_dir = files("tenepal.data")
                wordlist_file = data_dir / "spa_wordlist.txt"
                words = wordlist_file.read_text(encoding="utf-8").split()
            except (ImportError, AttributeError):
                # Fallback for older Python or if files() not available
                import pkg_resources

                wordlist_text = pkg_resources.resource_string(
                    "tenepal.data", "spa_wordlist.txt"
                ).decode("utf-8")
                words = wordlist_text.split()

        # Normalize and store as frozenset for O(1) lookup
        self.spanish_words = frozenset(self._normalize_word(w) for w in words)

    def _normalize_word(self, word: str) -> str:
        """Normalize a word for matching: lowercase, strip accents, remove punctuation.

        Args:
            word: Raw word from text

        Returns:
            Normalized word (lowercase, no accents, alphanumeric only)

        Example:
            >>> v = WhisperValidator()
            >>> v._normalize_word("Señor")
            'senor'
            >>> v._normalize_word("está!")
            'esta'
        """
        # Lowercase
        word = word.lower()

        # Strip accents using Unicode normalization
        # NFD = decompose characters into base + combining marks
        # Filter out combining marks (category Mn)
        word = "".join(
            c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn"
        )

        # Keep only alphanumeric characters
        word = re.sub(r"[^a-z0-9]", "", word)

        return word

    def _check_lexicon(self, text: str) -> float:
        """Check what proportion of words are in Spanish lexicon (VALID-01).

        Args:
            text: Text to validate

        Returns:
            Score from 0.0 (no words recognized) to 1.0 (all words recognized)
        """
        if not text.strip():
            return 1.0  # Empty text is neutral

        # Tokenize by splitting on whitespace and punctuation
        tokens = re.findall(r"\w+", text)

        if not tokens:
            return 1.0  # No tokens = neutral

        # Normalize and filter tokens (skip very short ones)
        valid_tokens = [self._normalize_word(t) for t in tokens if len(t) >= 2]

        if not valid_tokens:
            return 1.0  # No valid tokens = neutral

        # Count recognized words
        recognized = sum(1 for token in valid_tokens if token in self.spanish_words)

        # Return recognition ratio
        return recognized / len(valid_tokens)

    def _check_apostrophe_density(self, text: str) -> float:
        """Check apostrophe density (VALID-02).

        Spanish uses apostrophes extremely rarely. High apostrophe density
        indicates Mayan/Nahuatl romanization (e.g., k'an, ba'alo).

        Args:
            text: Text to validate

        Returns:
            Score from 0.0 (high density = suspicious) to 1.0 (low/no apostrophes)
        """
        if not text:
            return 1.0  # Empty text is neutral

        # Count both ASCII apostrophe (') and right single quotation mark (')
        apostrophe_count = text.count("'") + text.count("'")

        # Calculate density
        density = apostrophe_count / len(text)

        # Threshold: density >= 0.04 is very suspicious
        # Return inverse score: high density → low score
        # min(density / 0.04, 1.0) caps at 1.0, then subtract from 1.0
        return 1.0 - min(density / 0.04, 1.0)

    def _check_character_pattern(self, text: str) -> float:
        """Check for non-Spanish character patterns (VALID-03).

        Spanish uses a known character set: a-z, accented vowels (áéíóú, ü),
        ñ, and standard punctuation. Characters outside this set indicate
        other languages.

        Args:
            text: Text to validate

        Returns:
            Score from 0.0 (many foreign chars) to 1.0 (all Spanish chars)
        """
        if not text:
            return 1.0  # Empty text is neutral

        # Define Spanish character set (after NFD normalization)
        # Base letters, accented vowels, ñ
        spanish_chars = set("abcdefghijklmnopqrstuvwxyzáéíóúüñ")

        # Count alphabetic characters
        alpha_chars = [c.lower() for c in text if c.isalpha()]

        if not alpha_chars:
            return 1.0  # No alphabetic characters = neutral

        # Count characters in Spanish set
        spanish_count = sum(1 for c in alpha_chars if c in spanish_chars)

        # Return ratio
        return spanish_count / len(alpha_chars)

    def _check_avg_log_prob(self, avg_log_prob: float) -> float:
        """Check Whisper's average log probability as confidence signal (VALID-04).

        Whisper provides avg_log_prob per segment. Very low values indicate
        the model is uncertain about its transcription, which combined with
        low lexicon scores suggests hallucination.

        Mapping:
            - avg_log_prob <= -1.5 → score 0.0
            - avg_log_prob = 0.0 → score 1.0
            - Linear interpolation between

        Args:
            avg_log_prob: Average log probability from Whisper segment

        Returns:
            Score from 0.0 (very low confidence) to 1.0 (high confidence)
        """
        # Linear mapping from [-1.5, 0.0] to [0.0, 1.0]
        # Score = (avg_log_prob + 1.5) / 1.5
        # Clamp to [0.0, 1.0] range
        score = (avg_log_prob + 1.5) / 1.5
        return max(0.0, min(1.0, score))

    def _check_repetition(self, text: str) -> float:
        """Check for repeated phrase loops (VALID-05).

        Whisper sometimes gets stuck in loops, repeating the same phrase
        multiple times. This is a strong hallucination signal.

        Detects repeated 3-word n-grams. If any 3-word sequence appears
        2+ times, the score decreases proportionally.

        Args:
            text: Text to validate

        Returns:
            Score from 0.0 (severe repetition) to 1.0 (no repetition)
        """
        if not text.strip():
            return 1.0  # Empty text is neutral

        # Tokenize into words (lowercase, basic cleaning)
        words = re.findall(r"\w+", text.lower())

        # Too short to check for 3-word repetitions
        if len(words) < 6:
            return 1.0

        # Generate all 3-word n-grams
        ngrams = []
        for i in range(len(words) - 2):
            ngram = " ".join(words[i : i + 3])
            ngrams.append(ngram)

        if not ngrams:
            return 1.0

        # Count occurrences of each n-gram
        from collections import Counter

        ngram_counts = Counter(ngrams)

        # Find maximum repetition count
        max_count = max(ngram_counts.values())

        if max_count == 1:
            # No repetition
            return 1.0
        else:
            # Score inversely proportional to repetition count
            # 2 repetitions → 0.5, 3 repetitions → 0.33, etc.
            return 1.0 / max_count

    def validate(self, text: str, avg_log_prob: float = 0.0) -> ValidationResult:
        """Validate a Whisper segment to detect hallucinations.

        Args:
            text: Transcribed text from Whisper
            avg_log_prob: Average log probability from Whisper segment

        Returns:
            ValidationResult with verdict, confidence, reason, and check scores

        Decision logic (order matters):
            1. If lexicon_score < 0.4 → INVALID (most words not Spanish)
            2. If repetition_score < 0.3 → INVALID (severe repetition loop)
            3. If apostrophe_score < 0.5 AND lexicon_score < 0.6 → INVALID
            4. If character_score < 0.7 AND lexicon_score < 0.6 → INVALID
            5. If logprob_score < 0.3 AND lexicon_score < 0.7 → INVALID
            6. Otherwise → VALID

        Confidence:
            Weighted average: 0.4 * lexicon + 0.15 * apostrophe + 0.15 * character
                            + 0.15 * logprob + 0.15 * repetition
        """
        # Run all checks
        lexicon_score = self._check_lexicon(text)
        apostrophe_score = self._check_apostrophe_density(text)
        character_score = self._check_character_pattern(text)
        logprob_score = self._check_avg_log_prob(avg_log_prob)
        repetition_score = self._check_repetition(text)

        checks = {
            "lexicon": lexicon_score,
            "apostrophe": apostrophe_score,
            "character": character_score,
            "avg_log_prob": logprob_score,
            "repetition": repetition_score,
        }

        # Helper to calculate confidence
        def calc_confidence():
            return (
                0.4 * lexicon_score
                + 0.15 * apostrophe_score
                + 0.15 * character_score
                + 0.15 * logprob_score
                + 0.15 * repetition_score
            )

        # Combined decision logic (order matters)
        if lexicon_score < 0.4:
            return ValidationResult(
                is_valid=False,
                confidence=calc_confidence(),
                reason=f"Low Spanish lexicon match ({lexicon_score:.2f} < 0.4) - likely hallucination",
                checks=checks,
            )

        if repetition_score <= 0.5:
            return ValidationResult(
                is_valid=False,
                confidence=calc_confidence(),
                reason=f"Repetition detected (score {repetition_score:.2f} <= 0.5) - likely hallucination loop",
                checks=checks,
            )

        if apostrophe_score < 0.5 and lexicon_score < 0.6:
            return ValidationResult(
                is_valid=False,
                confidence=calc_confidence(),
                reason=f"High apostrophe density (score {apostrophe_score:.2f}) with moderate lexicon ({lexicon_score:.2f}) - likely Mayan/Nahuatl romanization",
                checks=checks,
            )

        if character_score < 0.7 and lexicon_score < 0.6:
            return ValidationResult(
                is_valid=False,
                confidence=calc_confidence(),
                reason=f"Non-Spanish characters (score {character_score:.2f}) with moderate lexicon ({lexicon_score:.2f}) - likely foreign language",
                checks=checks,
            )

        if logprob_score < 0.3 and lexicon_score < 0.7:
            return ValidationResult(
                is_valid=False,
                confidence=calc_confidence(),
                reason=f"Low confidence (logprob score {logprob_score:.2f}) with moderate lexicon ({lexicon_score:.2f}) - likely hallucination",
                checks=checks,
            )

        # VALID
        return ValidationResult(
            is_valid=True,
            confidence=calc_confidence(),
            reason=f"Valid Spanish text (lexicon: {lexicon_score:.2f}, apostrophe: {apostrophe_score:.2f}, character: {character_score:.2f}, logprob: {logprob_score:.2f}, repetition: {repetition_score:.2f})",
            checks=checks,
        )

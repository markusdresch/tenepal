"""Nahuatl morpheme segmenter for agglutinative word decomposition.

Decomposes Nahuatl words into prefix-root-suffix chains using a slot-based
morphological model. Works on both orthographic text and IPA sequences.

Dual-purpose module:
1. LID signal: Successful parse = strong NAH evidence
2. Translation: Morpheme glosses → LLM composition for natural target language

Architecture:
    Input (text or IPA) → G2P normalization → Greedy prefix stripping
    → Root lookup (fuzzy) → Suffix stripping → MorphemeAnalysis

The parser doesn't require complete coverage. A partial parse
(recognized prefix + unknown root + recognized suffix) still provides
strong language identification signal and structural glossing.

References:
    Sullivan (1988) Compendium of Nahuatl Grammar
    Karttunen (1983) Analytical Dictionary of Nahuatl
    Lockhart (2001) Nahuatl as Written
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MorphemeType(Enum):
    """Morpheme category in the Nahuatl word structure."""
    SUBJECT_PREFIX = "subj"
    OBJECT_PREFIX = "obj"
    REFLEXIVE_PREFIX = "refl"
    DIRECTIONAL_PREFIX = "dir"
    ROOT = "root"
    DERIVATIONAL_SUFFIX = "deriv"
    TENSE_SUFFIX = "tense"
    NOUN_SUFFIX = "noun"
    PLURAL_SUFFIX = "pl"
    PARTICLE = "particle"
    UNKNOWN = "unknown"


@dataclass
class Morpheme:
    """A single identified morpheme within a word.

    Attributes:
        form: Orthographic form (e.g., "ni", "mitz", "notza")
        ipa: IPA representation as phone list
        gloss: Grammatical/semantic gloss (e.g., "1SG", "call")
        type: Morpheme category
        slot: Position slot in the word template (lower = more left)
        confidence: Match confidence (1.0 = exact, <1.0 = fuzzy)
    """
    form: str
    ipa: list[str]
    gloss: str
    type: MorphemeType
    slot: int = 0
    confidence: float = 1.0


@dataclass
class MorphemeAnalysis:
    """Complete morphological analysis of a Nahuatl word.

    Attributes:
        input_form: Original input (text or IPA string)
        morphemes: Ordered list of identified morphemes
        remainder: Unanalyzed portion (empty if fully parsed)
        coverage: Fraction of input explained by morphemes (0.0–1.0)
        is_nahuatl: Whether the analysis suggests this is Nahuatl
        interlinear: Morpheme-by-morpheme gloss string
        translation_hint: Rough compositional translation
    """
    input_form: str
    morphemes: list[Morpheme] = field(default_factory=list)
    remainder: str = ""
    coverage: float = 0.0
    is_nahuatl: bool = False
    interlinear: str = ""
    translation_hint: str = ""

    @property
    def lid_score(self) -> float:
        """Language ID confidence score derived from morphological parse.

        Returns a score between 0.0 and 1.0 indicating how likely this
        word is Nahuatl based on morphological evidence:
        - Full parse with known root: 0.9–1.0
        - Partial parse (prefix+suffix, unknown root): 0.5–0.8
        - Only prefix or suffix recognized: 0.2–0.4
        - Nothing recognized: 0.0
        """
        if not self.morphemes:
            return 0.0

        has_prefix = any(m.type in (
            MorphemeType.SUBJECT_PREFIX,
            MorphemeType.OBJECT_PREFIX,
            MorphemeType.REFLEXIVE_PREFIX,
            MorphemeType.DIRECTIONAL_PREFIX,
        ) for m in self.morphemes)

        has_root = any(m.type == MorphemeType.ROOT for m in self.morphemes)

        has_suffix = any(m.type in (
            MorphemeType.TENSE_SUFFIX,
            MorphemeType.NOUN_SUFFIX,
            MorphemeType.DERIVATIONAL_SUFFIX,
            MorphemeType.PLURAL_SUFFIX,
        ) for m in self.morphemes)

        has_particle = any(m.type == MorphemeType.PARTICLE for m in self.morphemes)

        # Scoring: morphological structure = strong NAH evidence
        if has_root and (has_prefix or has_suffix):
            return min(0.95, 0.7 + self.coverage * 0.3)
        if has_root:
            return min(0.85, 0.5 + self.coverage * 0.35)
        if has_prefix and has_suffix:
            # Recognized frame around unknown root — still very NAH
            return min(0.80, 0.5 + self.coverage * 0.3)
        if has_particle:
            return min(0.70, 0.4 + self.coverage * 0.3)
        if has_prefix or has_suffix:
            # Single affix — weaker but nonzero signal
            return min(0.40, 0.2 + self.coverage * 0.2)
        return 0.0


class NahuatlMorphemeSegmenter:
    """Agglutinative morpheme segmenter for Nahuatl.

    Loads morpheme data from nah_morphemes.json and applies a greedy
    left-to-right decomposition strategy:
    1. Strip prefixes (subject → object → reflexive → directional)
    2. Match root from dictionary (fuzzy)
    3. Strip suffixes (derivational → tense → noun → plural)

    The parser is intentionally greedy and forgiving — partial parses
    are useful for both LID and translation scaffolding.
    """

    def __init__(self, morpheme_path: Optional[Path] = None) -> None:
        if morpheme_path:
            with open(morpheme_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            try:
                from importlib.resources import files
                data_dir = files("tenepal.data")
                morph_file = data_dir / "nah_morphemes.json"
                morph_text = morph_file.read_text(encoding="utf-8")
                self._data = json.loads(morph_text)
            except (ImportError, AttributeError):
                import pkg_resources
                morph_text = pkg_resources.resource_string(
                    "tenepal.data", "nah_morphemes.json"
                ).decode("utf-8")
                self._data = json.loads(morph_text)

        # Build lookup structures
        self._prefixes = self._build_prefix_table()
        self._suffixes = self._build_suffix_table()
        self._roots = self._build_root_table()
        self._particles = self._build_particle_table()

        logger.debug(
            "Loaded morpheme DB: %d prefixes, %d suffixes, %d roots, %d particles",
            len(self._prefixes), len(self._suffixes),
            len(self._roots), len(self._particles),
        )

    def _build_prefix_table(self) -> list[dict]:
        """Build prefix list sorted by length (longest first) for greedy match."""
        prefixes = []
        for category, mtype in [
            ("subject_prefixes", MorphemeType.SUBJECT_PREFIX),
            ("object_prefixes", MorphemeType.OBJECT_PREFIX),
            ("reflexive_prefixes", MorphemeType.REFLEXIVE_PREFIX),
            ("directional_prefixes", MorphemeType.DIRECTIONAL_PREFIX),
        ]:
            for entry in self._data.get(category, []):
                prefixes.append({
                    "form": entry["form"],
                    "ipa": entry["ipa"],
                    "gloss": entry["gloss"],
                    "type": mtype,
                    "slot": entry.get("slot", 0),
                })
        # Sort by IPA length descending (longest match first)
        prefixes.sort(key=lambda x: len(x["ipa"]), reverse=True)
        return prefixes

    def _build_suffix_table(self) -> list[dict]:
        """Build suffix list sorted by length (longest first)."""
        suffixes = []
        for category, mtype in [
            ("verbal_suffixes", MorphemeType.DERIVATIONAL_SUFFIX),
            ("tense_aspect_suffixes", MorphemeType.TENSE_SUFFIX),
            ("noun_suffixes", MorphemeType.NOUN_SUFFIX),
            ("relational_suffixes", MorphemeType.DERIVATIONAL_SUFFIX),
        ]:
            for entry in self._data.get(category, []):
                suffixes.append({
                    "form": entry["form"],
                    "ipa": entry["ipa"],
                    "gloss": entry["gloss"],
                    "type": mtype,
                    "slot": entry.get("slot", 10),
                })
        suffixes.sort(key=lambda x: len(x["ipa"]), reverse=True)
        return suffixes

    def _build_root_table(self) -> list[dict]:
        """Build root dictionary sorted by IPA length (longest first)."""
        roots = []
        for entry in self._data.get("root_dictionary", []):
            roots.append({
                "form": entry["form"],
                "ipa": entry["ipa"],
                "gloss": entry["gloss"],
                "class": entry.get("class", "noun"),
            })
        roots.sort(key=lambda x: len(x["ipa"]), reverse=True)
        return roots

    def _build_particle_table(self) -> list[dict]:
        """Build particle lookup sorted by length."""
        particles = []
        for entry in self._data.get("common_particles", []):
            particles.append({
                "form": entry["form"],
                "ipa": entry["ipa"],
                "gloss": entry["gloss"],
            })
        particles.sort(key=lambda x: len(x["ipa"]), reverse=True)
        return particles

    def analyze(self, word: str, as_ipa: bool = False) -> MorphemeAnalysis:
        """Analyze a single word/IPA sequence into morphemes.

        Args:
            word: Input word (orthographic text or space-separated IPA)
            as_ipa: If True, treat input as IPA phone sequence

        Returns:
            MorphemeAnalysis with decomposition results
        """
        if not word or not word.strip():
            return MorphemeAnalysis(input_form=word)

        # Convert to IPA phone list
        if as_ipa:
            phones = word.strip().split()
        else:
            phones = self._text_to_ipa(word)

        if not phones:
            return MorphemeAnalysis(input_form=word)

        original_len = len(phones)
        morphemes: list[Morpheme] = []
        remaining = list(phones)

        # Check if it's a known particle first (whole-word match)
        particle = self._match_particle(remaining)
        if particle and len(particle["ipa"]) >= len(remaining) - 1:
            morphemes.append(Morpheme(
                form=particle["form"],
                ipa=particle["ipa"],
                gloss=particle["gloss"],
                type=MorphemeType.PARTICLE,
                slot=0,
                confidence=1.0,
            ))
            consumed = len(particle["ipa"])
            remaining = remaining[consumed:]
            return self._build_result(word, morphemes, remaining, original_len)

        # Phase 1: Strip prefixes (greedy, left-to-right)
        seen_slots: set[int] = set()
        for _ in range(4):  # max 4 prefix layers
            prefix = self._match_prefix(remaining, seen_slots)
            if not prefix:
                break
            morphemes.append(Morpheme(
                form=prefix["form"],
                ipa=prefix["ipa"],
                gloss=prefix["gloss"],
                type=prefix["type"],
                slot=prefix["slot"],
                confidence=1.0,
            ))
            consumed = len(prefix["ipa"])
            remaining = remaining[consumed:]
            seen_slots.add(prefix["slot"])

        # Phase 2: Try to match root
        root, suffix_remainder = self._match_root(remaining)
        if root:
            morphemes.append(Morpheme(
                form=root["form"],
                ipa=root["ipa"],
                gloss=root["gloss"],
                type=MorphemeType.ROOT,
                slot=5,
                confidence=root.get("confidence", 1.0),
            ))
            remaining = suffix_remainder
        else:
            # No root found — try suffix stripping from the end
            # to see if we can isolate an unknown root
            suffix_morphemes, core = self._strip_suffixes_reverse(remaining)
            if suffix_morphemes:
                # We found suffixes — the core is an unknown root
                if core:
                    morphemes.append(Morpheme(
                        form="".join(core),
                        ipa=core,
                        gloss="?",
                        type=MorphemeType.UNKNOWN,
                        slot=5,
                        confidence=0.0,
                    ))
                morphemes.extend(suffix_morphemes)
                remaining = []

        # Phase 3: Strip suffixes from what's left after root
        if remaining and root:
            seen_suffix_slots: set[int] = set()
            for _ in range(4):  # max 4 suffix layers
                suffix = self._match_suffix(remaining, seen_suffix_slots)
                if not suffix:
                    break
                morphemes.append(Morpheme(
                    form=suffix["form"],
                    ipa=suffix["ipa"],
                    gloss=suffix["gloss"],
                    type=suffix["type"],
                    slot=suffix["slot"],
                    confidence=1.0,
                ))
                consumed = len(suffix["ipa"])
                remaining = remaining[consumed:]
                seen_suffix_slots.add(suffix["slot"])

        # If we still have remainder and no morphemes at all, mark as unknown
        if remaining and not morphemes:
            morphemes.append(Morpheme(
                form="".join(remaining),
                ipa=remaining,
                gloss="?",
                type=MorphemeType.UNKNOWN,
                slot=5,
                confidence=0.0,
            ))
            remaining = []

        return self._build_result(word, morphemes, remaining, original_len)

    def analyze_text(self, text: str, as_ipa: bool = False) -> list[MorphemeAnalysis]:
        """Analyze a multi-word text string.

        Args:
            text: Input text (words separated by spaces)
            as_ipa: If True, split on double-space for word boundaries

        Returns:
            List of MorphemeAnalysis, one per word
        """
        if not text:
            return []

        if as_ipa:
            # IPA mode: words separated by "  " (double space), phones by single space
            words = text.strip().split("  ")
        else:
            words = text.strip().split()

        return [self.analyze(w, as_ipa=as_ipa) for w in words if w.strip()]

    def lid_score_text(self, text: str, as_ipa: bool = False) -> float:
        """Compute aggregate NAH language ID score for a text.

        Averages per-word lid_score, weighted by word length.
        Useful as an additional LID signal in the pipeline.

        Args:
            text: Input text or IPA sequence
            as_ipa: Whether input is IPA

        Returns:
            Aggregate score between 0.0 and 1.0
        """
        analyses = self.analyze_text(text, as_ipa=as_ipa)
        if not analyses:
            return 0.0

        total_weight = 0.0
        weighted_score = 0.0
        for analysis in analyses:
            # Weight by number of phones in original input
            n_phones = len(self._text_to_ipa(analysis.input_form)) if not as_ipa else len(analysis.input_form.split())
            weight = max(1, n_phones)
            weighted_score += analysis.lid_score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _text_to_ipa(self, word: str) -> list[str]:
        """Convert orthographic Nahuatl to IPA using the existing G2P."""
        try:
            from tenepal.phoneme.text_to_ipa import get_g2p_converter
            converter = get_g2p_converter("nah")
            return converter.convert(word, "nah")
        except Exception:
            # Minimal fallback if G2P not available
            return list(word.lower())

    def _match_prefix(self, phones: list[str], seen_slots: set[int]) -> Optional[dict]:
        """Find the longest matching prefix in phone sequence."""
        for prefix in self._prefixes:
            plen = len(prefix["ipa"])
            if plen > len(phones):
                continue
            if prefix["slot"] in seen_slots:
                continue
            if self._phones_match(phones[:plen], prefix["ipa"]):
                return prefix
        return None

    def _match_suffix(self, phones: list[str], seen_slots: set[int]) -> Optional[dict]:
        """Find the longest matching suffix at the START of remaining phones."""
        for suffix in self._suffixes:
            slen = len(suffix["ipa"])
            if slen > len(phones):
                continue
            if suffix["slot"] in seen_slots:
                continue
            if self._phones_match(phones[:slen], suffix["ipa"]):
                return suffix
        return None

    def _strip_suffixes_reverse(self, phones: list[str]) -> tuple[list[Morpheme], list[str]]:
        """Strip suffixes from the END of phone sequence, working backwards.

        Returns (suffix_morphemes, remaining_core).
        """
        suffixes_found: list[Morpheme] = []
        remaining = list(phones)
        seen_slots: set[int] = set()

        for _ in range(4):
            matched = False
            for suffix in self._suffixes:
                slen = len(suffix["ipa"])
                if slen > len(remaining):
                    continue
                if suffix["slot"] in seen_slots:
                    continue
                if self._phones_match(remaining[-slen:], suffix["ipa"]):
                    suffixes_found.insert(0, Morpheme(
                        form=suffix["form"],
                        ipa=suffix["ipa"],
                        gloss=suffix["gloss"],
                        type=suffix["type"],
                        slot=suffix["slot"],
                        confidence=1.0,
                    ))
                    remaining = remaining[:-slen]
                    seen_slots.add(suffix["slot"])
                    matched = True
                    break
            if not matched:
                break

        return suffixes_found, remaining

    def _match_root(self, phones: list[str]) -> tuple[Optional[dict], list[str]]:
        """Find the longest matching root at the start of phone sequence.

        Returns (root_entry, remaining_phones_after_root) or (None, phones).
        Uses fuzzy matching: allows 1 substitution for roots >= 4 phones.
        """
        best_root = None
        best_remainder: list[str] = phones
        best_confidence = 0.0

        for root in self._roots:
            rlen = len(root["ipa"])
            if rlen > len(phones):
                continue

            # Exact match
            if self._phones_match(phones[:rlen], root["ipa"]):
                conf = 1.0
                if conf > best_confidence or (conf == best_confidence and rlen > len(best_root.get("ipa", []) if best_root else [])):
                    best_root = {**root, "confidence": conf}
                    best_remainder = phones[rlen:]
                    best_confidence = conf
                continue

            # Fuzzy match: allow 1 substitution for roots >= 4 phones
            if rlen >= 4:
                mismatches = sum(
                    1 for a, b in zip(phones[:rlen], root["ipa"])
                    if not self._phone_match(a, b)
                )
                if mismatches <= 1:
                    conf = 1.0 - (mismatches * 0.2)
                    if conf > best_confidence:
                        best_root = {**root, "confidence": conf}
                        best_remainder = phones[rlen:]
                        best_confidence = conf

        return best_root, best_remainder

    def _match_particle(self, phones: list[str]) -> Optional[dict]:
        """Check if the entire phone sequence matches a known particle."""
        for particle in self._particles:
            plen = len(particle["ipa"])
            if abs(plen - len(phones)) > 1:  # allow ±1 length difference
                continue
            if self._phones_match(phones[:plen], particle["ipa"]):
                return particle
        return None

    def _phones_match(self, seq1: list[str], seq2: list[str]) -> bool:
        """Check if two phone sequences match (with allophone normalization)."""
        if len(seq1) != len(seq2):
            return False
        return all(self._phone_match(a, b) for a, b in zip(seq1, seq2))

    @staticmethod
    def _phone_match(a: str, b: str) -> bool:
        """Check if two phones match, considering common allophones."""
        if a == b:
            return True
        # Normalize common confusions
        _EQUIV = {
            "ð": "d", "β": "b", "ɣ": "ɡ", "ɸ": "f",
            "r": "ɾ", "ɻ": "ɾ",
            "tɬ": "tl", "tl": "tɬ",
            "ts": "tz", "tz": "ts",
        }
        na = _EQUIV.get(a, a)
        nb = _EQUIV.get(b, b)
        return na == nb

    def _build_result(
        self,
        input_form: str,
        morphemes: list[Morpheme],
        remainder: list[str],
        original_len: int,
    ) -> MorphemeAnalysis:
        """Build final MorphemeAnalysis from parse results."""
        # Calculate coverage
        consumed = sum(len(m.ipa) for m in morphemes)
        coverage = consumed / original_len if original_len > 0 else 0.0

        # Build interlinear gloss
        interlinear = "-".join(m.gloss for m in morphemes)
        if remainder:
            interlinear += f"-[{''.join(remainder)}]"

        # Build translation hint
        translation_hint = self._compose_translation(morphemes)

        # Determine if this looks like Nahuatl
        analysis = MorphemeAnalysis(
            input_form=input_form,
            morphemes=morphemes,
            remainder="".join(remainder),
            coverage=coverage,
            interlinear=interlinear,
            translation_hint=translation_hint,
        )
        analysis.is_nahuatl = analysis.lid_score >= 0.3
        return analysis

    @staticmethod
    def _compose_translation(morphemes: list[Morpheme]) -> str:
        """Compose a rough translation hint from morpheme glosses.

        This is intentionally simple — real translation should use
        LLM composition with the interlinear as input.
        """
        parts = []
        subject = ""
        obj = ""
        verb = ""
        noun = ""

        for m in morphemes:
            if m.type == MorphemeType.SUBJECT_PREFIX:
                subject = m.gloss.split("(")[1].rstrip(")") if "(" in m.gloss else m.gloss
            elif m.type == MorphemeType.OBJECT_PREFIX:
                obj = m.gloss.split("(")[1].rstrip(")") if "(" in m.gloss else m.gloss
            elif m.type == MorphemeType.ROOT:
                if m.gloss != "?":
                    parts.append(m.gloss)
            elif m.type == MorphemeType.PARTICLE:
                parts.append(m.gloss.split("(")[1].rstrip(")") if "(" in m.gloss else m.gloss)
            elif m.type == MorphemeType.NOUN_SUFFIX:
                pass  # grammatical, not lexical
            elif m.type == MorphemeType.TENSE_SUFFIX:
                parts.append(f"[{m.gloss}]")

        # Assemble
        result_parts = []
        if subject:
            result_parts.append(subject)
        if obj:
            result_parts.append(obj)
        result_parts.extend(parts)

        return " ".join(result_parts) if result_parts else ""


# Module-level singleton for convenience
_segmenter: Optional[NahuatlMorphemeSegmenter] = None


def get_segmenter() -> NahuatlMorphemeSegmenter:
    """Get or lazily initialize the module-level segmenter singleton."""
    global _segmenter
    if _segmenter is None:
        _segmenter = NahuatlMorphemeSegmenter()
    return _segmenter


def analyze_word(word: str, as_ipa: bool = False) -> MorphemeAnalysis:
    """Convenience function: analyze a single word."""
    return get_segmenter().analyze(word, as_ipa=as_ipa)


def analyze_text(text: str, as_ipa: bool = False) -> list[MorphemeAnalysis]:
    """Convenience function: analyze multi-word text."""
    return get_segmenter().analyze_text(text, as_ipa=as_ipa)


def lid_score(text: str, as_ipa: bool = False) -> float:
    """Convenience function: get aggregate NAH LID score for text."""
    return get_segmenter().lid_score_text(text, as_ipa=as_ipa)

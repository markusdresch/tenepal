"""Language identification from phoneme streams.

This module implements language identification by analyzing phoneme sequences
and matching them against language profiles in a registry.
"""

import logging
import unicodedata
from dataclasses import dataclass

import numpy as np

from tenepal.language.registry import LanguageRegistry, default_registry
from tenepal.phoneme import PhonemeSegment
from tenepal.prosody import (
    extract_prosody,
    score_all_profiles as score_prosody_profiles,
)
from tenepal.fusion import fuse_scores

logger = logging.getLogger(__name__)

# IPA spacing modifier letters to strip for sequence matching
_SPACING_MODIFIERS = frozenset("ːʲʰˤʼ")

# Lazy-loaded Nahuatl lexicon singleton
_nahuatl_lexicon = None


def _get_lexicon():
    """Get or lazily initialize the Nahuatl lexicon singleton."""
    global _nahuatl_lexicon
    if _nahuatl_lexicon is None:
        from tenepal.language.nahuatl_lexicon import NahuatlLexicon

        _nahuatl_lexicon = NahuatlLexicon()
    return _nahuatl_lexicon


# Lazy-loaded Maya lexicon singleton
_maya_lexicon = None

# Preserve explicit high-value language assignments during file-level
# consolidation so dominant-language fallback does not erase them.
_PROTECTED_FILE_CONSOLIDATION_LANGUAGES = frozenset({"lat", "eng"})


def _get_maya_lexicon():
    """Get or lazily initialize the Maya lexicon singleton."""
    global _maya_lexicon
    if _maya_lexicon is None:
        from tenepal.language.maya_lexicon import MayaLexicon

        _maya_lexicon = MayaLexicon()
    return _maya_lexicon


# Allophone normalization map: base-character variants → canonical phoneme
# These are allophones that Allosaurus outputs as separate IPA symbols
# rather than base + diacritic. Covers Spanish lenition and common variants.
_ALLOPHONE_MAP: dict[str, str] = {
    "ð": "d",  # Spanish intervocalic /d/ → [ð]
    "β": "b",  # Spanish intervocalic /b/ → [β]
    "ɣ": "ɡ",  # Spanish intervocalic /ɡ/ → [ɣ]
    "ɸ": "f",  # Bilabial fricative → labiodental
    "ɻ": "ɾ",  # Retroflex approximant → tap (common Allosaurus confusion)
    "r": "ɾ",  # Trill → tap (for non-trilling varieties)
}


def _strip_modifiers(phoneme: str) -> str:
    """Strip IPA diacritics and modifier letters to get the base phoneme.

    Used for sequence matching so that e.g. t + lː matches the
    sequence ("t", "l"). Strips combining diacritics (◌̪, ◌̟, ◌̃)
    and spacing modifiers (ː, ʲ, ʰ).
    """
    return "".join(
        c
        for c in phoneme
        if c not in _SPACING_MODIFIERS and unicodedata.category(c) != "Mn"
    )


@dataclass
class LanguageSegment:
    """Container for a language-labeled segment with timing information.

    Attributes:
        language: Language code (ISO 639) or "other" for unidentified
        phonemes: List of PhonemeSegments in this language segment
        start_time: Start time in seconds (from first phoneme)
        end_time: End time in seconds (last phoneme start + duration)
        speaker: Speaker label (e.g., "Speaker A"), None = no diarization
        confidence: Weighted confidence score for this segment (sum of marker weights)
    """

    language: str
    phonemes: list[PhonemeSegment]
    start_time: float
    end_time: float
    speaker: str | None = None
    confidence: float = 0.0


def identify_language(
    segments: list[PhonemeSegment],
    registry: LanguageRegistry | None = None,
    audio_data: tuple[np.ndarray, int] | None = None,
) -> list[LanguageSegment]:
    """Identify language from phoneme stream and segment by language.

    Takes a flat list of phoneme segments and produces language-labeled
    segments by:
    1. Tagging each phoneme with a language based on weighted marker scoring
    2. Grouping consecutive same-language phonemes into segments
    3. Merging short segments (< 3 phonemes) into neighbors
    4. Optionally fusing with prosodic evidence if audio_data is provided

    Args:
        segments: List of PhonemeSegment objects with phoneme and timing
        registry: LanguageRegistry to use (default: default_registry())
        audio_data: Optional tuple of (samples_array, sample_rate) for prosody extraction.
                    When provided, prosodic features are extracted and fused with phoneme scores.
                    When None or extraction fails, falls back to phoneme-only scoring.

    Returns:
        List of LanguageSegment objects with language labels, timing, and confidence scores
    """
    if not segments:
        return []

    if registry is None:
        registry = default_registry()

    # Extract prosodic features if audio data is available
    prosody_features = None
    if audio_data is not None:
        try:
            samples, sample_rate = audio_data
            prosody_features = extract_prosody(samples, sample_rate)
        except Exception as exc:
            logger.warning(
                "Prosody extraction failed, falling back to phoneme-only: %s", exc
            )
            prosody_features = None

    # Phase A: Score each phoneme with weighted language detection
    scored_phonemes = _score_phonemes(segments, registry)

    # Phase A.5: Lexicon pre-check for known Nahuatl words
    scored_phonemes = _apply_lexicon_check(scored_phonemes)

    # Phase B: Group consecutive same-language phonemes
    raw_segments = _group_by_language_scored(scored_phonemes)

    # Phase C: Merge short segments into neighbors
    merged_segments = _merge_short_segments(raw_segments)

    # Phase D: Reclassify mixed segments by language priority
    reclassified = _reclassify_by_priority(merged_segments, registry)

    # Phase E: Absorb isolated low-priority islands into neighbors
    absorbed = _absorb_islands(reclassified, registry)

    # Phase F: Apply segment-level confidence thresholds
    thresholded = _apply_segment_thresholds(absorbed, registry)

    # Phase F.5: Guard MAY segments — require ejective evidence
    guarded = _guard_may_ejectives(thresholded)

    # Phase G: Consolidate using whole-file scoring context
    consolidated = _consolidate_by_file_score(
        guarded, segments, registry, prosody_features
    )

    # Phase G.5: Absorb short OTH segments between NAH segments
    consolidated = _absorb_oth_between_nah(consolidated)

    return consolidated


def _score_phonemes(
    phonemes: list[PhonemeSegment], registry: LanguageRegistry
) -> list[tuple[PhonemeSegment, str | None, float]]:
    """Tag each phoneme with its most likely language using weighted scoring.

    Computes weighted scores for each language at each phoneme position,
    considering both bigram sequences and single markers. The language with
    the highest score wins, provided the score exceeds the language's threshold.

    Args:
        phonemes: List of PhonemeSegment objects
        registry: LanguageRegistry with language profiles

    Returns:
        List of (PhonemeSegment, language_code or None, score) tuples.
        None means no language exceeded its threshold (unmarked phoneme).
        Score represents the per-phoneme contribution to segment confidence.
    """
    profiles = registry.all_profiles()

    # Initialize scores: scores[i][lang_code] = accumulated weight at position i
    scores: list[dict[str, float]] = [{} for _ in phonemes]
    # Track which positions were tagged by trigrams or bigrams per language
    trigram_tagged: list[set[str]] = [set() for _ in phonemes]
    bigram_tagged: list[set[str]] = [set() for _ in phonemes]

    # Pass 1: Check trigram sequences (highest specificity)
    for profile in profiles:
        trigrams = getattr(profile, "marker_trigrams", set())
        if not trigrams:
            continue
        trigram_weights = getattr(profile, "trigram_weights", {})
        for i in range(len(phonemes) - 2):
            triple = (
                _strip_modifiers(phonemes[i].phoneme),
                _strip_modifiers(phonemes[i + 1].phoneme),
                _strip_modifiers(phonemes[i + 2].phoneme),
            )
            if triple in trigrams:
                weight = trigram_weights.get(triple, 1.0)
                third_weight = weight / 3.0
                for j in (i, i + 1, i + 2):
                    scores[j][profile.code] = (
                        scores[j].get(profile.code, 0.0) + third_weight
                    )
                    trigram_tagged[j].add(profile.code)

    # Pass 2: Check bigram sequences (skip trigram-tagged positions for same language)
    for profile in profiles:
        if not profile.marker_sequences:
            continue
        for i in range(len(phonemes) - 1):
            pair = (
                _strip_modifiers(phonemes[i].phoneme),
                _strip_modifiers(phonemes[i + 1].phoneme),
            )
            if pair in profile.marker_sequences:
                weight = profile.sequence_weights.get(pair, 1.0)
                half_weight = weight / 2.0
                for j in (i, i + 1):
                    if profile.code not in trigram_tagged[j]:
                        scores[j][profile.code] = (
                            scores[j].get(profile.code, 0.0) + half_weight
                        )
                        bigram_tagged[j].add(profile.code)

    # Pass 3: Single phoneme markers (skip trigram/bigram-tagged positions for same language)
    # Try both raw and stripped forms so that allophonic variants (b̞→b, d̪→d, ɡ̤→ɡ)
    # match profiles while preserving exact matches for markers with modifiers (lʲ, t͡ʃʲ)
    for i, phoneme in enumerate(phonemes):
        raw = phoneme.phoneme
        stripped = _strip_modifiers(raw)
        for profile in profiles:
            # Try raw first (exact match preserves markers like lʲ, k̟ʲ)
            matched = None
            if raw in profile.marker_phonemes:
                matched = raw
            elif stripped != raw and stripped in profile.marker_phonemes:
                matched = stripped
            # Also check allophone map for base-character variants (ð→d, β→b, ɣ→ɡ)
            elif not matched:
                normalized = _ALLOPHONE_MAP.get(stripped)
                if normalized and normalized in profile.marker_phonemes:
                    matched = normalized
            if matched:
                if (
                    profile.code not in trigram_tagged[i]
                    and profile.code not in bigram_tagged[i]
                ):
                    weight = profile.marker_weights.get(matched, 1.0)
                    scores[i][profile.code] = weight

    # Determine winning language at each position
    tagged: list[tuple[PhonemeSegment, str | None, float]] = []
    for i, phoneme in enumerate(phonemes):
        if not scores[i]:
            # No language scored at this position
            tagged.append((phoneme, None, 0.0))
            continue

        # Find language with highest score
        best_language = None
        best_score = 0.0
        best_priority = -1

        for lang_code, score in scores[i].items():
            profile = registry.get(lang_code)
            if not profile:
                continue

            # Only exclude zero-scoring languages (segment-level thresholds applied later)
            if score <= 0:
                continue

            # Check if this is the best candidate (highest score, or highest priority on tie)
            if score > best_score or (
                score == best_score and profile.priority > best_priority
            ):
                best_language = lang_code
                best_score = score
                best_priority = profile.priority

        tagged.append((phoneme, best_language, best_score))

    return tagged


def _apply_lexicon_check(
    scored_phonemes: list[tuple[PhonemeSegment, str | None, float]],
) -> list[tuple[PhonemeSegment, str | None, float]]:
    """Apply Nahuatl and Maya lexicon pre-check to override marker-based scoring.

    Scans the phoneme stream for known Nahuatl and Maya words using fuzzy IPA matching.
    When a lexicon match is found, overrides the language tag to "nah" or "may" and
    sets the score to the match confidence for all phonemes in the matched range.

    This provides a direct recognition path for short words that might not
    accumulate enough marker evidence to exceed detection thresholds.

    Nahuatl lexicon runs first for backward compatibility. If a position was tagged
    by both NAH and MAY lexicons, the first match wins (NAH check runs first).

    Args:
        scored_phonemes: List of (PhonemeSegment, language_code, score) tuples
                        from _score_phonemes output

    Returns:
        Modified scored_phonemes list with lexicon matches overriding original tags
    """
    if not scored_phonemes:
        return scored_phonemes

    # Create a mutable copy of scored_phonemes
    result = list(scored_phonemes)

    # Extract just the phoneme strings for lexicon matching
    phoneme_strings = [seg.phoneme for seg, _, _ in scored_phonemes]

    # Apply Nahuatl lexicon first (backward compatibility)
    nah_lexicon = _get_lexicon()
    nah_matches = nah_lexicon.match_subsequence(phoneme_strings, min_length=2)

    # Apply each NAH match (starting with best matches first)
    for match in nah_matches:
        # Override language and score for all phonemes in the matched range
        for i in range(match.start_idx, match.start_idx + match.length):
            if i < len(result):
                phoneme_seg, _, _ = result[i]
                result[i] = (phoneme_seg, "nah", match.score)

    # Apply Maya lexicon second
    maya_lexicon = _get_maya_lexicon()
    maya_matches = maya_lexicon.match_subsequence(phoneme_strings, min_length=4)

    # Apply each MAY match (only if position not already tagged by NAH lexicon)
    for match in maya_matches:
        # Override language and score for all phonemes in the matched range
        for i in range(match.start_idx, match.start_idx + match.length):
            if i < len(result):
                phoneme_seg, current_lang, current_score = result[i]
                # Only override if not already tagged by NAH lexicon
                if current_lang != "nah" or current_score == 0.0:
                    result[i] = (phoneme_seg, "may", match.score)

    return result


def _tag_phonemes(
    phonemes: list[PhonemeSegment], registry: LanguageRegistry
) -> list[tuple[PhonemeSegment, str | None]]:
    """Tag each phoneme with its most likely language.

    Legacy function for backward compatibility. Calls _score_phonemes
    and discards score information.

    Args:
        phonemes: List of PhonemeSegment objects
        registry: LanguageRegistry with language profiles

    Returns:
        List of (PhonemeSegment, language_code or None) tuples.
        None means no marker match (unmarked phoneme).
    """
    scored = _score_phonemes(phonemes, registry)
    return [(phoneme, language) for phoneme, language, _ in scored]


def _group_by_language_scored(
    scored_phonemes: list[tuple[PhonemeSegment, str | None, float]],
) -> list[LanguageSegment]:
    """Group consecutive phonemes with the same language tag, propagating confidence scores.

    Unmarked phonemes (tagged as None) are absorbed into the preceding
    language segment if one exists, otherwise they start a new "other" segment.

    Args:
        scored_phonemes: List of (PhonemeSegment, language_code or None, score) tuples

    Returns:
        List of LanguageSegment objects with confidence scores (before merging)
    """
    if not scored_phonemes:
        return []

    segments = []
    current_language = scored_phonemes[0][1] or "other"
    current_phonemes = [scored_phonemes[0][0]]
    current_scores = [scored_phonemes[0][2]]

    for phoneme, language, score in scored_phonemes[1:]:
        # If this phoneme has a language marker
        if language is not None:
            if language == current_language:
                # Same language: continue segment
                current_phonemes.append(phoneme)
                current_scores.append(score)
            else:
                # Different language: finish current segment, start new one
                confidence = sum(current_scores)
                segments.append(
                    _create_segment(current_language, current_phonemes, confidence)
                )
                current_language = language
                current_phonemes = [phoneme]
                current_scores = [score]
        else:
            # Unmarked phoneme: absorb into current segment with score 0.0
            current_phonemes.append(phoneme)
            current_scores.append(0.0)

    # Add final segment
    confidence = sum(current_scores)
    segments.append(_create_segment(current_language, current_phonemes, confidence))

    return segments


def _group_by_language(
    tagged_phonemes: list[tuple[PhonemeSegment, str | None]],
) -> list[LanguageSegment]:
    """Group consecutive phonemes with the same language tag.

    Legacy function for backward compatibility. Converts to scored format
    and calls _group_by_language_scored.

    Args:
        tagged_phonemes: List of (PhonemeSegment, language_code or None) tuples

    Returns:
        List of LanguageSegment objects (before merging)
    """
    scored = [(phoneme, language, 0.0) for phoneme, language in tagged_phonemes]
    return _group_by_language_scored(scored)


def _create_segment(
    language: str, phonemes: list[PhonemeSegment], confidence: float = 0.0
) -> LanguageSegment:
    """Create a LanguageSegment from language code and phoneme list.

    Args:
        language: Language code or "other"
        phonemes: List of PhonemeSegment objects
        confidence: Weighted confidence score for this segment

    Returns:
        LanguageSegment with calculated timing and confidence
    """
    start_time = phonemes[0].start_time
    end_time = phonemes[-1].start_time + phonemes[-1].duration

    return LanguageSegment(
        language=language,
        phonemes=phonemes,
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
    )


def _merge_short_segments(
    segments: list[LanguageSegment], min_length: int = 3
) -> list[LanguageSegment]:
    """Merge short segments into their neighbors, combining confidence scores.

    Short segments (< min_length phonemes) are absorbed into adjacent segments.
    Merging continues iteratively until no more short segments exist.
    Confidence scores are summed when segments merge.

    Args:
        segments: List of LanguageSegment objects
        min_length: Minimum number of phonemes for a segment to stand alone

    Returns:
        List of LanguageSegment objects after merging
    """
    if not segments:
        return []

    # Keep merging until no changes occur
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0

        while i < len(segments):
            segment = segments[i]

            # Check if this segment is too short
            if len(segment.phonemes) < min_length:
                # Determine where to merge
                if i == 0 and len(segments) > 1:
                    # First segment: merge into next
                    next_seg = segments[i + 1]
                    merged_phonemes = segment.phonemes + next_seg.phonemes
                    merged_confidence = segment.confidence + next_seg.confidence
                    merged.append(
                        _create_segment(
                            next_seg.language, merged_phonemes, merged_confidence
                        )
                    )
                    i += 2  # Skip next segment (already merged)
                    changed = True

                elif i == len(segments) - 1 and len(merged) > 0:
                    # Last segment: merge into previous
                    prev_seg = merged.pop()
                    merged_phonemes = prev_seg.phonemes + segment.phonemes
                    merged_confidence = prev_seg.confidence + segment.confidence
                    merged.append(
                        _create_segment(
                            prev_seg.language, merged_phonemes, merged_confidence
                        )
                    )
                    i += 1
                    changed = True

                elif 0 < i < len(segments) - 1:
                    # Middle segment: check neighbors
                    prev_seg = merged[-1] if merged else None
                    next_seg = segments[i + 1]

                    if prev_seg and prev_seg.language == next_seg.language:
                        # Neighbors have same language: merge all three
                        merged.pop()  # Remove previous segment
                        merged_phonemes = (
                            prev_seg.phonemes + segment.phonemes + next_seg.phonemes
                        )
                        merged_confidence = (
                            prev_seg.confidence
                            + segment.confidence
                            + next_seg.confidence
                        )
                        merged.append(
                            _create_segment(
                                prev_seg.language, merged_phonemes, merged_confidence
                            )
                        )
                        i += 2  # Skip next segment
                        changed = True

                    elif prev_seg:
                        # Neighbors differ: merge into larger neighbor
                        if len(prev_seg.phonemes) >= len(next_seg.phonemes):
                            # Merge into previous
                            merged.pop()
                            merged_phonemes = prev_seg.phonemes + segment.phonemes
                            merged_confidence = prev_seg.confidence + segment.confidence
                            merged.append(
                                _create_segment(
                                    prev_seg.language,
                                    merged_phonemes,
                                    merged_confidence,
                                )
                            )
                        else:
                            # Merge into next
                            merged_phonemes = segment.phonemes + next_seg.phonemes
                            merged_confidence = segment.confidence + next_seg.confidence
                            merged.append(
                                _create_segment(
                                    next_seg.language,
                                    merged_phonemes,
                                    merged_confidence,
                                )
                            )
                            i += 1  # Skip next segment

                        i += 1
                        changed = True

                    else:
                        # No previous segment (shouldn't happen in middle)
                        merged.append(segment)
                        i += 1

                else:
                    # Single short segment: keep it
                    merged.append(segment)
                    i += 1

            else:
                # Segment is long enough: keep it
                merged.append(segment)
                i += 1

        segments = merged

    return segments


def _compute_segment_score(
    phonemes: list[PhonemeSegment],
    profile,
) -> float:
    """Compute total marker score for a list of phonemes against a profile.

    Sums unigram + bigram + trigram contributions. Used by confidence-aware
    reclassification to compare competing languages on the same segment.

    Args:
        phonemes: List of PhonemeSegment objects
        profile: LanguageProfile to score against

    Returns:
        Total weighted score for this profile on these phonemes
    """
    score = 0.0
    tagged = [False] * len(phonemes)

    # Trigrams (highest specificity)
    trigrams = getattr(profile, "marker_trigrams", set())
    trigram_weights = getattr(profile, "trigram_weights", {})
    if trigrams:
        for i in range(len(phonemes) - 2):
            triple = (
                _strip_modifiers(phonemes[i].phoneme),
                _strip_modifiers(phonemes[i + 1].phoneme),
                _strip_modifiers(phonemes[i + 2].phoneme),
            )
            if triple in trigrams:
                weight = trigram_weights.get(triple, 1.0)
                third_weight = weight / 3.0
                for j in (i, i + 1, i + 2):
                    if not tagged[j]:
                        score += third_weight
                        tagged[j] = True

    # Bigrams
    if profile.marker_sequences:
        for i in range(len(phonemes) - 1):
            pair = (
                _strip_modifiers(phonemes[i].phoneme),
                _strip_modifiers(phonemes[i + 1].phoneme),
            )
            if pair in profile.marker_sequences:
                weight = profile.sequence_weights.get(pair, 1.0)
                half_weight = weight / 2.0
                for j in (i, i + 1):
                    if not tagged[j]:
                        score += half_weight
                        tagged[j] = True

    # Single markers (with allophone normalization)
    for i, phoneme in enumerate(phonemes):
        if tagged[i]:
            continue
        raw = phoneme.phoneme
        stripped = _strip_modifiers(raw)
        matched = None
        if raw in profile.marker_phonemes:
            matched = raw
        elif stripped != raw and stripped in profile.marker_phonemes:
            matched = stripped
        else:
            normalized = _ALLOPHONE_MAP.get(stripped)
            if normalized and normalized in profile.marker_phonemes:
                matched = normalized
        if matched:
            weight = profile.marker_weights.get(matched, 1.0)
            score += weight
            tagged[i] = True

    # Absent phoneme penalties (negative markers, also normalized)
    negative_markers = getattr(profile, "negative_markers", {})
    if negative_markers:
        for phoneme in phonemes:
            raw = phoneme.phoneme
            stripped = _strip_modifiers(raw)
            normalized = _ALLOPHONE_MAP.get(stripped, stripped)
            penalty = (
                negative_markers.get(raw, 0.0)
                or negative_markers.get(stripped, 0.0)
                or negative_markers.get(normalized, 0.0)
            )
            if penalty > 0.0:
                score -= penalty

    return score


def _reclassify_by_priority(
    segments: list[LanguageSegment], registry: LanguageRegistry
) -> list[LanguageSegment]:
    """Reclassify segments using bidirectional confidence-aware scoring.

    Re-scores each segment's phonemes against ALL candidate profiles
    (including absent phoneme penalties). Picks the language with the
    highest score. Also processes "other" segments to recover language
    labels lost during short-segment merging.

    Args:
        segments: List of LanguageSegment objects after merging
        registry: LanguageRegistry with priority-aware profiles

    Returns:
        List of LanguageSegment objects after reclassification and re-merging
    """
    if not segments:
        return []

    profiles = registry.all_profiles()

    for segment in segments:
        current_profile = registry.get(segment.language)
        current_score = (
            _compute_segment_score(segment.phonemes, current_profile)
            if current_profile
            else 0.0
        )
        current_priority = current_profile.priority if current_profile else 0

        best_lang = segment.language
        best_score = current_score
        best_priority = current_priority

        for profile in profiles:
            if profile.code == segment.language:
                continue
            candidate_score = _compute_segment_score(segment.phonemes, profile)
            if candidate_score > best_score or (
                candidate_score == best_score and profile.priority > best_priority
            ):
                best_lang = profile.code
                best_score = candidate_score
                best_priority = profile.priority

        if best_score > 0 and best_lang != segment.language:
            segment.language = best_lang
            segment.confidence = best_score

    # Re-merge adjacent segments that now share the same language
    return _merge_adjacent(segments)


def _merge_adjacent(segments: list[LanguageSegment]) -> list[LanguageSegment]:
    """Merge adjacent segments that share the same language, combining confidence scores."""
    if not segments:
        return []
    result = [segments[0]]
    for seg in segments[1:]:
        if seg.language == result[-1].language:
            merged_phonemes = result[-1].phonemes + seg.phonemes
            merged_confidence = result[-1].confidence + seg.confidence
            result[-1] = _create_segment(
                result[-1].language, merged_phonemes, merged_confidence
            )
        else:
            result.append(seg)
    return result


def _absorb_islands(
    segments: list[LanguageSegment], registry: LanguageRegistry
) -> list[LanguageSegment]:
    """Absorb isolated low-priority segments into higher-priority neighbors.

    A segment is an "island" when both neighbors are the same language and
    that language has higher priority. This handles cases where a phoneme
    recognizer produces false markers (e.g., Allosaurus outputs /ɡ/ from
    Nahuatl audio, creating a spurious Spanish segment inside Nahuatl speech).

    Confidence scores are combined when segments are absorbed.

    Args:
        segments: List of LanguageSegment objects after reclassification
        registry: LanguageRegistry with priority-aware profiles

    Returns:
        List of LanguageSegment objects after island absorption
    """
    if len(segments) < 3:
        return segments

    changed = True
    while changed:
        changed = False
        result = []
        i = 0

        while i < len(segments):
            seg = segments[i]

            # Check if this segment is an island (both neighbors same language)
            if 0 < i < len(segments) - 1:
                prev_lang = result[-1].language if result else None
                next_lang = segments[i + 1].language

                if prev_lang and prev_lang == next_lang and prev_lang != seg.language:
                    seg_priority = _get_priority(seg.language, registry)
                    neighbor_priority = _get_priority(prev_lang, registry)

                    if neighbor_priority > seg_priority:
                        # Confidence check: only absorb if the neighbor language
                        # actually scores higher on the island's phonemes
                        neighbor_profile = registry.get(prev_lang)
                        island_profile = registry.get(seg.language)
                        if neighbor_profile and island_profile:
                            neighbor_score = _compute_segment_score(
                                seg.phonemes, neighbor_profile
                            )
                            island_score = _compute_segment_score(
                                seg.phonemes, island_profile
                            )
                            if island_score > neighbor_score:
                                # Island's own language scores higher — keep it
                                result.append(seg)
                                i += 1
                                continue

                        # Absorb: merge into previous neighbor
                        prev_seg = result.pop()
                        merged_phonemes = prev_seg.phonemes + seg.phonemes
                        merged_confidence = prev_seg.confidence + seg.confidence
                        result.append(
                            _create_segment(
                                prev_lang, merged_phonemes, merged_confidence
                            )
                        )
                        changed = True
                        i += 1
                        continue

            result.append(seg)
            i += 1

        segments = _merge_adjacent(result)

    return segments


def _apply_segment_thresholds(
    segments: list[LanguageSegment], registry: LanguageRegistry
) -> list[LanguageSegment]:
    """Reclassify segments whose cumulative confidence doesn't meet the language threshold.

    Each language profile defines a threshold representing the minimum cumulative
    marker evidence needed to declare that language. Segments below their language's
    threshold are reclassified as "other". Adjacent same-language segments are then
    merged.

    Args:
        segments: List of LanguageSegment objects after absorption
        registry: LanguageRegistry with threshold-aware profiles

    Returns:
        List of LanguageSegment objects after threshold filtering
    """
    result = []
    for seg in segments:
        if seg.language == "other":
            result.append(seg)
            continue
        profile = registry.get(seg.language)
        if profile and profile.threshold > 0 and seg.confidence <= profile.threshold:
            result.append(_create_segment("other", seg.phonemes, 0.0))
        else:
            result.append(seg)
    return _merge_adjacent(result)


# IPA ejective modifier (U+02BC) used by Maya ejective consonants
_EJECTIVE_MARKER = "ʼ"


def _guard_may_ejectives(
    segments: list[LanguageSegment],
) -> list[LanguageSegment]:
    """Demote MAY segments that lack ejective evidence to 'other'.

    Yucatec Maya is distinguished from Nahuatl primarily by ejective
    consonants (kʼ, tʼ, pʼ, tsʼ, tʃʼ). Without at least one ejective
    in the phoneme stream, a MAY classification is almost certainly a
    false positive caused by shared markers (ʔ, ʃ) or lexicon noise.

    Args:
        segments: List of LanguageSegment objects after threshold filtering

    Returns:
        List of LanguageSegment objects with ejective-less MAY demoted
    """
    changed = False
    for seg in segments:
        if seg.language != "may":
            continue
        # Check if any phoneme contains an ejective marker
        has_ejective = any(_EJECTIVE_MARKER in p.phoneme for p in seg.phonemes)
        if not has_ejective:
            seg.language = "other"
            seg.confidence = 0.0
            changed = True

    if changed:
        return _merge_adjacent(segments)
    return segments


def _consolidate_by_file_score(
    segments: list[LanguageSegment],
    all_phonemes: list[PhonemeSegment],
    registry: LanguageRegistry,
    prosody_features=None,
) -> list[LanguageSegment]:
    """Consolidate segments using whole-file scoring as tiebreaker.

    When one language clearly dominates the whole-file score, reclassify
    minority-language segments to the dominant language if the dominant
    language scores at least as well on those segment phonemes.

    This corrects fragmentation where per-segment scoring assigns segments
    to other languages due to shared phoneme artifacts, even though the
    overall evidence strongly favors one language.

    Args:
        segments: List of LanguageSegment objects after threshold filtering
        all_phonemes: Original flat phoneme list for whole-file scoring
        registry: LanguageRegistry with profiles
        prosody_features: Optional ProsodyFeatures object for score fusion

    Returns:
        List of LanguageSegment objects after file-level consolidation
    """
    if not segments or not all_phonemes:
        return segments

    # Only consolidate files with enough phonemes for reliable scoring
    if len(all_phonemes) < 100:
        return segments

    profiles = registry.all_profiles()
    if not profiles:
        return segments

    # Compute whole-file scores for each language
    file_scores = {}
    for profile in profiles:
        file_scores[profile.code] = _compute_segment_score(all_phonemes, profile)

    # Fuse with prosody scores if available
    if prosody_features is not None:
        try:
            prosody_scores = score_prosody_profiles(prosody_features)
            file_scores = fuse_scores(file_scores, prosody_scores)
        except Exception as exc:
            logger.warning("Prosody fusion failed, using phoneme-only: %s", exc)

    # Find the top two languages by whole-file score
    sorted_langs = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_langs) < 2:
        return segments

    dominant_lang, dominant_score = sorted_langs[0]
    second_lang, second_score = sorted_langs[1]

    # Only consolidate if dominant language clearly leads:
    # - dominant score must be positive
    # - dominant must beat second by at least 20% relative margin
    if dominant_score <= 0:
        return segments
    if second_score > 0 and dominant_score < second_score * 1.25:
        return segments

    # Check marker density: dominant language must have sufficient evidence per
    # phoneme. Low density means sparse markers that could appear in any language
    # (e.g., ɾ in non-Spanish audio). Without this check, long files with sparse
    # universal markers would be incorrectly consolidated to one language.
    density = dominant_score / len(all_phonemes)
    high_density = density >= 0.015

    # When one language clearly dominates with sufficient density, reclassify
    # all segments (including "other") to the dominant language.
    # With low density, only consolidate minority-language segments — preserve
    # "other" segments as genuine low-confidence regions.
    # Additionally, only consolidate if the dominant language scores higher on
    # that segment than the segment's current language.
    changed = False
    dominant_profile = registry.get(dominant_lang)
    for segment in segments:
        if segment.language != dominant_lang:
            if segment.language in _PROTECTED_FILE_CONSOLIDATION_LANGUAGES:
                continue  # preserve explicit MAY/LAT/ENG segments
            if segment.language == "other" and not high_density:
                continue  # preserve "other" when evidence is sparse
            if segment.language != "other" and dominant_profile:
                current_profile = registry.get(segment.language)
                if current_profile:
                    current_seg_score = _compute_segment_score(
                        segment.phonemes, current_profile
                    )
                    dominant_seg_score = _compute_segment_score(
                        segment.phonemes, dominant_profile
                    )
                    if current_seg_score > dominant_seg_score:
                        continue  # segment scores higher for its current language
            segment.language = dominant_lang
            changed = True

    if changed:
        return _merge_adjacent(segments)
    return segments


def _absorb_oth_between_nah(segments: list[LanguageSegment]) -> list[LanguageSegment]:
    """Absorb short OTH segments that are sandwiched between NAH segments.

    This handles the common pattern in film audio where a brief unidentified
    segment sits between two recognized Nahuatl speech segments. If the OTH
    segment(s) between NAH segments are short (total duration <= 2.0 seconds
    AND total phonemes <= 10), reclassify them as NAH.

    Args:
        segments: List of LanguageSegment objects after consolidation

    Returns:
        List of LanguageSegment objects after OTH absorption and merging
    """
    if len(segments) < 3:
        return segments

    changed = True
    while changed:
        changed = False
        result = []
        i = 0

        while i < len(segments):
            seg = segments[i]

            # Check if this is an OTH segment between two NAH segments
            if seg.language == "other" and i > 0 and i < len(segments) - 1:
                # Look at the actual previous segment (already in result list)
                prev_lang = result[-1].language if result else None
                next_seg = segments[i + 1]

                if prev_lang == "nah" and next_seg.language == "nah":
                    # Check if OTH is short enough to absorb
                    duration = seg.end_time - seg.start_time
                    num_phonemes = len(seg.phonemes)

                    if duration <= 2.0 and num_phonemes <= 10:
                        # Absorb: reclassify OTH as NAH
                        seg.language = "nah"
                        changed = True

            result.append(seg)
            i += 1

        segments = result

        # Merge adjacent NAH segments after absorption
        if changed:
            segments = _merge_adjacent(segments)

    return segments


def _get_priority(language: str, registry: LanguageRegistry) -> int:
    """Get the priority of a language code, returning 0 for unknown/other."""
    if language == "other":
        return 0
    profile = registry.get(language)
    return profile.priority if profile else 0

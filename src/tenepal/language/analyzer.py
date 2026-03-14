"""Phoneme frequency analysis and confusion matrix tools.

This module provides diagnostic tools for measuring language detection accuracy
by analyzing phoneme frequencies and building confusion matrices from real
Allosaurus output. Used for marker validation and tuning.
"""

from dataclasses import dataclass, field

from tenepal.language.identifier import _strip_modifiers, _tag_phonemes
from tenepal.language.registry import LanguageRegistry, default_registry
from tenepal.phoneme import PhonemeSegment


@dataclass
class ProfileHits:
    """Analysis of how a language profile's markers appear in phoneme input.

    Attributes:
        language_code: ISO 639 code
        language_name: Common name
        markers_found: Marker phoneme -> count of occurrences in input
        markers_missing: Marker phonemes that never appeared (ghost markers)
        sequences_found: Bigram sequences that matched
        sequences_missing: Bigram sequences that never matched
        trigrams_found: Trigram sequences that matched
        trigrams_missing: Trigram sequences that never matched
        detection_count: How many phonemes would be tagged to this language
    """
    language_code: str
    language_name: str
    markers_found: dict[str, int]
    markers_missing: set[str]
    sequences_found: list[tuple[str, str]]
    sequences_missing: list[tuple[str, str]]
    trigrams_found: list[tuple[str, str, str]] = field(default_factory=list)
    trigrams_missing: list[tuple[str, str, str]] = field(default_factory=list)
    detection_count: int = 0


@dataclass
class PhonemeAnalysis:
    """Results of phoneme frequency analysis.

    Attributes:
        total_phonemes: Total count of phonemes
        unique_phonemes: Count of distinct phoneme strings
        frequencies: Phoneme string -> count, sorted by frequency descending
        profile_hits: Language code -> ProfileHits for each profile
    """
    total_phonemes: int
    unique_phonemes: int
    frequencies: dict[str, int]
    profile_hits: dict[str, ProfileHits]


@dataclass
class ConfusionMatrix:
    """Confusion matrix identifying shared markers and false positives.

    Attributes:
        shared_markers: Phoneme -> list of language codes claiming it as marker
        false_positive_candidates: Language code -> list of markers that appeared
            but may be false positives (marker also claimed by higher-priority language)
        ghost_markers: Language code -> set of markers never appearing in input
    """
    shared_markers: dict[str, list[str]]
    false_positive_candidates: dict[str, list[str]]
    ghost_markers: dict[str, set[str]]


def analyze_phonemes(
    phonemes: list[PhonemeSegment],
    registry: LanguageRegistry | None = None
) -> PhonemeAnalysis:
    """Analyze phoneme frequencies and profile marker hits.

    Args:
        phonemes: List of PhonemeSegment objects from recognition
        registry: LanguageRegistry to analyze against (default: default_registry())

    Returns:
        PhonemeAnalysis with frequency counts and profile hits
    """
    if registry is None:
        registry = default_registry()

    if not phonemes:
        return PhonemeAnalysis(
            total_phonemes=0,
            unique_phonemes=0,
            frequencies={},
            profile_hits={}
        )

    # Count phoneme frequencies
    phoneme_strings = [p.phoneme for p in phonemes]
    freq_dict: dict[str, int] = {}
    for phoneme in phoneme_strings:
        freq_dict[phoneme] = freq_dict.get(phoneme, 0) + 1

    # Sort by frequency descending
    sorted_frequencies = dict(
        sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    )

    total_count = len(phoneme_strings)
    unique_count = len(freq_dict)

    # Build phoneme set for fast lookup
    phoneme_set = set(phoneme_strings)

    # Analyze each profile
    profiles = registry.all_profiles()
    profile_hits_dict: dict[str, ProfileHits] = {}

    for profile in profiles:
        # Find which markers appeared and which didn't
        markers_found: dict[str, int] = {}
        markers_missing: set[str] = set()

        for marker in profile.marker_phonemes:
            if marker in phoneme_set:
                markers_found[marker] = freq_dict[marker]
            else:
                markers_missing.add(marker)

        # Check sequences
        sequences_found: list[tuple[str, str]] = []
        sequences_missing: list[tuple[str, str]] = []

        for seq in profile.marker_sequences:
            # Check if sequence appears anywhere in phoneme stream
            found = False
            for i in range(len(phonemes) - 1):
                pair = (
                    _strip_modifiers(phonemes[i].phoneme),
                    _strip_modifiers(phonemes[i + 1].phoneme),
                )
                if pair == seq:
                    found = True
                    break

            if found:
                sequences_found.append(seq)
            else:
                sequences_missing.append(seq)

        # Check trigrams
        trigrams_found: list[tuple[str, str, str]] = []
        trigrams_missing: list[tuple[str, str, str]] = []
        profile_trigrams = getattr(profile, "marker_trigrams", set())

        for tri in profile_trigrams:
            found = False
            for i in range(len(phonemes) - 2):
                triple = (
                    _strip_modifiers(phonemes[i].phoneme),
                    _strip_modifiers(phonemes[i + 1].phoneme),
                    _strip_modifiers(phonemes[i + 2].phoneme),
                )
                if triple == tri:
                    found = True
                    break
            if found:
                trigrams_found.append(tri)
            else:
                trigrams_missing.append(tri)

        # Calculate detection_count by replicating _tag_phonemes logic
        detection_count = _count_detections_for_profile(phonemes, profile)

        profile_hits_dict[profile.code] = ProfileHits(
            language_code=profile.code,
            language_name=profile.name,
            markers_found=markers_found,
            markers_missing=markers_missing,
            sequences_found=sequences_found,
            sequences_missing=sequences_missing,
            trigrams_found=trigrams_found,
            trigrams_missing=trigrams_missing,
            detection_count=detection_count,
        )

    return PhonemeAnalysis(
        total_phonemes=total_count,
        unique_phonemes=unique_count,
        frequencies=sorted_frequencies,
        profile_hits=profile_hits_dict
    )


def _count_detections_for_profile(
    phonemes: list[PhonemeSegment],
    profile
) -> int:
    """Count how many phonemes would be tagged to a specific profile.

    Replicates the _score_phonemes logic but counts only for one profile.
    """
    count = 0
    tagged = [False] * len(phonemes)

    # Pass 1: Check trigram sequences
    trigrams = getattr(profile, "marker_trigrams", set())
    if trigrams:
        for i in range(len(phonemes) - 2):
            triple = (
                _strip_modifiers(phonemes[i].phoneme),
                _strip_modifiers(phonemes[i + 1].phoneme),
                _strip_modifiers(phonemes[i + 2].phoneme),
            )
            if triple in trigrams:
                for j in (i, i + 1, i + 2):
                    if not tagged[j]:
                        count += 1
                        tagged[j] = True

    # Pass 2: Check bigram sequences
    if profile.marker_sequences:
        for i in range(len(phonemes) - 1):
            pair = (
                _strip_modifiers(phonemes[i].phoneme),
                _strip_modifiers(phonemes[i + 1].phoneme),
            )
            if pair in profile.marker_sequences:
                for j in (i, i + 1):
                    if not tagged[j]:
                        count += 1
                        tagged[j] = True

    # Pass 3: Single phoneme markers
    for i, phoneme in enumerate(phonemes):
        if not tagged[i] and phoneme.phoneme in profile.marker_phonemes:
            count += 1
            tagged[i] = True

    return count


def build_confusion_matrix(analysis: PhonemeAnalysis) -> ConfusionMatrix:
    """Build confusion matrix from phoneme analysis.

    Identifies which markers are shared between languages and which may be
    causing false positives.

    Args:
        analysis: PhonemeAnalysis from analyze_phonemes()

    Returns:
        ConfusionMatrix with shared markers and false positive candidates
    """
    # Build reverse mapping: phoneme -> languages that claim it
    marker_to_languages: dict[str, list[str]] = {}

    for lang_code, hits in analysis.profile_hits.items():
        # Only include markers that actually appeared
        for marker in hits.markers_found.keys():
            if marker not in marker_to_languages:
                marker_to_languages[marker] = []
            marker_to_languages[marker].append(lang_code)

    # Find shared markers (claimed by multiple languages)
    shared_markers = {
        phoneme: langs
        for phoneme, langs in marker_to_languages.items()
        if len(langs) > 1
    }

    # Find false positive candidates
    # A marker is a false positive candidate if it appeared in the input
    # but is also claimed by a higher-priority language
    false_positive_candidates: dict[str, list[str]] = {}

    # For each language, check if its markers are also claimed by higher-priority languages
    # This requires access to priorities, but we only have ProfileHits
    # We can infer from the shared_markers which languages share markers

    for lang_code, hits in analysis.profile_hits.items():
        fp_candidates = []
        for marker in hits.markers_found.keys():
            if marker in shared_markers:
                # This marker is shared with other languages
                fp_candidates.append(marker)
        if fp_candidates:
            false_positive_candidates[lang_code] = fp_candidates

    # Ghost markers: markers that never appeared
    ghost_markers = {
        lang_code: hits.markers_missing
        for lang_code, hits in analysis.profile_hits.items()
        if hits.markers_missing
    }

    return ConfusionMatrix(
        shared_markers=shared_markers,
        false_positive_candidates=false_positive_candidates,
        ghost_markers=ghost_markers
    )


def format_analysis(analysis: PhonemeAnalysis, confusion: ConfusionMatrix) -> str:
    """Format analysis results as human-readable text report.

    Args:
        analysis: PhonemeAnalysis from analyze_phonemes()
        confusion: ConfusionMatrix from build_confusion_matrix()

    Returns:
        Multi-section text report with frequencies, profile hits, confusion matrix,
        and ghost markers
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("PHONEME FREQUENCY ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Total phonemes: {analysis.total_phonemes}")
    lines.append(f"Unique phonemes: {analysis.unique_phonemes}")
    lines.append("")

    # Section 1: Phoneme Frequencies
    lines.append("-" * 80)
    lines.append("PHONEME FREQUENCIES")
    lines.append("-" * 80)
    lines.append(f"{'Phoneme':<15} {'Count':<10} {'Percentage':<10}")
    lines.append("-" * 80)

    for phoneme, count in analysis.frequencies.items():
        percentage = (count / analysis.total_phonemes) * 100
        lines.append(f"{phoneme:<15} {count:<10} {percentage:>6.2f}%")

    lines.append("")

    # Section 2: Language Profile Hits
    lines.append("-" * 80)
    lines.append("LANGUAGE PROFILE HITS")
    lines.append("-" * 80)

    for lang_code, hits in analysis.profile_hits.items():
        lines.append(f"\n{hits.language_name} ({hits.language_code.upper()}):")
        lines.append(f"  Detection count: {hits.detection_count} phonemes")

        if hits.markers_found:
            lines.append(f"  Markers found ({len(hits.markers_found)}):")
            for marker, count in sorted(hits.markers_found.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {marker}: {count}")
        else:
            lines.append("  Markers found: None")

        if hits.sequences_found:
            lines.append(f"  Sequences found ({len(hits.sequences_found)}):")
            for seq in hits.sequences_found:
                lines.append(f"    {seq[0]} + {seq[1]}")

        if hits.markers_missing:
            lines.append(f"  Markers missing ({len(hits.markers_missing)}):")
            missing_sorted = sorted(hits.markers_missing)
            lines.append(f"    {', '.join(missing_sorted)}")

        if hits.sequences_missing:
            lines.append(f"  Sequences missing ({len(hits.sequences_missing)}):")
            for seq in hits.sequences_missing:
                lines.append(f"    {seq[0]} + {seq[1]}")

        if hits.trigrams_found:
            lines.append(f"  Trigrams found ({len(hits.trigrams_found)}):")
            for tri in hits.trigrams_found:
                lines.append(f"    {tri[0]} + {tri[1]} + {tri[2]}")

        if hits.trigrams_missing:
            lines.append(f"  Trigrams missing ({len(hits.trigrams_missing)}):")
            for tri in hits.trigrams_missing:
                lines.append(f"    {tri[0]} + {tri[1]} + {tri[2]}")

    lines.append("")

    # Section 3: Confusion Matrix
    lines.append("-" * 80)
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 80)

    if confusion.shared_markers:
        lines.append("Shared markers (claimed by multiple languages):")
        for phoneme, langs in sorted(confusion.shared_markers.items()):
            langs_str = ", ".join(lang.upper() for lang in langs)
            lines.append(f"  {phoneme}: {langs_str}")
    else:
        lines.append("Shared markers: None (all markers are language-specific)")

    lines.append("")

    if confusion.false_positive_candidates:
        lines.append("False positive candidates:")
        for lang_code, markers in sorted(confusion.false_positive_candidates.items()):
            markers_str = ", ".join(markers)
            lines.append(f"  {lang_code.upper()}: {markers_str}")
    else:
        lines.append("False positive candidates: None")

    lines.append("")

    # Section 4: Ghost Markers
    lines.append("-" * 80)
    lines.append("GHOST MARKERS")
    lines.append("-" * 80)
    lines.append("(Markers that never appear in Allosaurus output)")
    lines.append("")

    if confusion.ghost_markers:
        for lang_code, markers in sorted(confusion.ghost_markers.items()):
            if markers:
                lines.append(f"{lang_code.upper()}: {', '.join(sorted(markers))}")
    else:
        lines.append("None - all markers appeared in input")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)

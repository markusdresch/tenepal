"""Speaker language profile building and inheritance logic.

This module builds per-speaker language profiles from high-confidence Whisper
segments and applies language inheritance to uncertain segments when speakers
have established language patterns.
"""

from dataclasses import dataclass

from tenepal.transcription.languages import WHISPER_LANG_REVERSE
from tenepal.language.identifier import LanguageSegment

PROTECTED_INHERITANCE_LANGUAGES = frozenset({"may", "lat"})


@dataclass
class SpeakerProfile:
    """Language profile for a speaker built from high-confidence segments.

    Attributes:
        speaker: Speaker label (e.g., "Speaker A")
        primary_language: Most common language (ISO 639-3 code, e.g., "spa")
        segment_count: Number of high-confidence segments used
        language_distribution: Mapping of language code → count
    """

    speaker: str
    primary_language: str
    segment_count: int
    language_distribution: dict[str, int]

    def meets_inheritance_threshold(
        self, min_segments: int = 5, min_ratio: float = 0.8
    ) -> bool:
        """Check if speaker has enough evidence for language inheritance.

        Args:
            min_segments: Minimum number of segments required (default: 5)
            min_ratio: Minimum ratio of primary language (default: 0.8 = 80%)

        Returns:
            True if speaker has sufficient evidence for inheritance
        """
        if self.segment_count < min_segments:
            return False
        primary_count = self.language_distribution.get(self.primary_language, 0)
        return (primary_count / self.segment_count) >= min_ratio


def build_speaker_profiles(
    assigned_pairs: list[tuple],
    confidence_threshold: float = -0.3,
) -> dict[str, SpeakerProfile]:
    """Build speaker language profiles from high-confidence Whisper segments.

    Only segments with avg_log_prob > confidence_threshold are used for
    profile building. Whisper language codes (ISO 639-1) are mapped to
    Tenepal codes (ISO 639-3) using WHISPER_LANG_REVERSE.

    Args:
        assigned_pairs: List of (WhisperAutoSegment, SpeakerSegment) tuples
                       from deduplication module
        confidence_threshold: Minimum avg_log_prob to include (default: -0.3)

    Returns:
        Dictionary mapping speaker label → SpeakerProfile
    """
    if not assigned_pairs:
        return {}

    # Group segments by speaker
    speaker_languages: dict[str, list[str]] = {}

    for whisper_seg, speaker_seg in assigned_pairs:
        # Skip low-confidence segments
        if whisper_seg.avg_log_prob < confidence_threshold:
            continue

        speaker = speaker_seg.speaker

        # Map Whisper language (ISO 639-1) to Tenepal (ISO 639-3)
        whisper_lang = whisper_seg.language
        tenepal_lang = WHISPER_LANG_REVERSE.get(whisper_lang, whisper_lang)

        if speaker not in speaker_languages:
            speaker_languages[speaker] = []
        speaker_languages[speaker].append(tenepal_lang)

    # Build profiles for each speaker
    profiles: dict[str, SpeakerProfile] = {}

    for speaker, languages in speaker_languages.items():
        # Count language occurrences
        distribution: dict[str, int] = {}
        for lang in languages:
            distribution[lang] = distribution.get(lang, 0) + 1

        # Find primary language (most frequent)
        primary_language = max(distribution, key=distribution.get)

        profiles[speaker] = SpeakerProfile(
            speaker=speaker,
            primary_language=primary_language,
            segment_count=len(languages),
            language_distribution=distribution,
        )

    return profiles


def apply_speaker_inheritance(
    segments: list[LanguageSegment],
    profiles: dict[str, SpeakerProfile],
    min_segments: int = 5,
    min_ratio: float = 0.8,
    max_inherit_duration: float = 1.0,
) -> list[LanguageSegment]:
    """Apply speaker language inheritance to uncertain segments.

    Language inheritance applies when:
    - SPKR-02: Allosaurus-only segment (transcription_backend != "whisper")
               with language="other" inherits from speaker profile
    - SPKR-03: Segment under max_inherit_duration seconds without Whisper
               transcription inherits from speaker profile

    Segments are only modified if:
    - Speaker has a profile that meets inheritance threshold
    - Segment meets one of the inheritance criteria above

    Args:
        segments: List of LanguageSegment objects to process
        profiles: Dictionary mapping speaker label → SpeakerProfile
        min_segments: Minimum segments for inheritance (default: 5)
        min_ratio: Minimum primary language ratio (default: 0.8)
        max_inherit_duration: Maximum duration for SPKR-03 (default: 1.0s)

    Returns:
        Modified list of LanguageSegment objects with inheritance applied
    """
    for segment in segments:
        # Skip if no speaker label
        if not segment.speaker:
            continue

        # Skip if speaker has no profile
        if segment.speaker not in profiles:
            continue

        profile = profiles[segment.speaker]

        # Skip if speaker doesn't meet inheritance threshold
        if not profile.meets_inheritance_threshold(min_segments, min_ratio):
            continue

        # Check if this segment has Whisper transcription
        has_whisper = getattr(segment, "transcription_backend", None) == "whisper"

        # NEVER change Whisper-labeled segments
        if has_whisper:
            continue

        # Preserve explicitly detected Maya/Latin segments from blanket inheritance.
        if segment.language in PROTECTED_INHERITANCE_LANGUAGES:
            continue

        # Calculate segment duration
        duration = segment.end_time - segment.start_time

        # Apply inheritance rules
        should_inherit = False

        # SPKR-02: Allosaurus-only segment with "other" language
        if segment.language == "other":
            should_inherit = True

        # SPKR-03: Short segment without Whisper match
        if duration < max_inherit_duration:
            should_inherit = True

        if should_inherit:
            segment.language = profile.primary_language

    return segments

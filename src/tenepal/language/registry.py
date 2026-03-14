"""Language profile registry for phonetic language identification.

This module provides the data model for language profiles, which define
languages by their distinctive phoneme inventories rather than hardcoded logic.
This enables extensible language identification.
"""

from dataclasses import dataclass, field


@dataclass
class LanguageProfile:
    """Profile defining a language by its phonetic characteristics.

    Attributes:
        code: ISO 639 language code (e.g., "nah" for Nahuatl, "spa" for Spanish)
        name: Common name of the language
        family: Language family classification
        marker_phonemes: Set of distinctive IPA phonemes that identify this language
        absent_phonemes: Set of phonemes absent in this language but common in contact languages
        marker_sequences: Set of phoneme pairs (bigrams) that identify this language
            when they appear consecutively (e.g., ("k", "w") for Nahuatl /kʷ/)
        priority: Higher priority languages win when a segment contains markers
            from multiple languages (e.g., Nahuatl > Spanish for mixed segments)
        marker_weights: Weights for marker phonemes (0.0-1.0). Phonemes not in this dict
            default to weight 1.0. Higher weights indicate more language-specific markers.
        sequence_weights: Weights for marker sequences (0.0-1.0). Sequences not in this dict
            default to weight 1.0. Higher weights indicate more language-specific bigrams.
        threshold: Minimum weighted score required to tag a segment as this language (0.0-1.0).
            0.0 means any marker match suffices (backward compatible default).
        marker_trigrams: Set of phoneme triples (trigrams) that identify this language
            when they appear consecutively. Higher specificity than bigrams.
        trigram_weights: Weights for marker trigrams. Trigrams not in this dict
            default to weight 1.0.
        negative_markers: Phonemes that count against this language (reserved for future use).
    """
    code: str
    name: str
    family: str
    marker_phonemes: set[str]
    absent_phonemes: set[str]
    marker_sequences: set[tuple[str, str]] = field(default_factory=set)
    priority: int = 0
    marker_weights: dict[str, float] = field(default_factory=dict)
    sequence_weights: dict[tuple[str, str], float] = field(default_factory=dict)
    threshold: float = 0.0
    marker_trigrams: set[tuple[str, str, str]] = field(default_factory=set)
    trigram_weights: dict[tuple[str, str, str], float] = field(default_factory=dict)
    negative_markers: dict[str, float] = field(default_factory=dict)


LANGUAGE_CODE_ALIASES = {
    "esp": "spa",
    "ger": "deu",
}


def normalize_language_code(code: str) -> str:
    """Normalize language codes to ISO 639-3 defaults."""
    normalized = code.lower()
    return LANGUAGE_CODE_ALIASES.get(normalized, normalized)


class LanguageRegistry:
    """Registry for managing language profiles.

    Provides methods to register, retrieve, and query language profiles.
    """

    def __init__(self):
        """Initialize an empty language registry."""
        self._profiles: dict[str, LanguageProfile] = {}

    def register(self, profile: LanguageProfile) -> None:
        """Register a language profile.

        Args:
            profile: LanguageProfile to register

        Raises:
            ValueError: If a profile with the same code is already registered
        """
        if profile.code in self._profiles:
            raise ValueError(f"Language code '{profile.code}' already registered")
        self._profiles[profile.code] = profile

    def get(self, code: str) -> LanguageProfile | None:
        """Retrieve a language profile by code.

        Args:
            code: ISO 639 language code

        Returns:
            LanguageProfile if found, None otherwise
        """
        normalized = normalize_language_code(code)
        return self._profiles.get(normalized)

    def all_profiles(self) -> list[LanguageProfile]:
        """Get all registered language profiles.

        Returns:
            List of all registered LanguageProfile objects
        """
        return list(self._profiles.values())

    def codes(self) -> list[str]:
        """Get all registered language codes.

        Returns:
            List of all registered language codes
        """
        return list(self._profiles.keys())


def default_registry() -> LanguageRegistry:
    """Create a language registry pre-loaded with default profiles.

    Loads profiles from external JSON configuration files in the profiles/ directory.
    This enables semi-automated tuning of marker weights and thresholds.

    Currently includes:
    - Nahuatl (nah): Uto-Aztecan language with distinctive phonemes
    - Spanish (spa): Indo-European language with voiced plosives and fricatives
    - English (eng): Indo-European language with distinctive vowels and consonants
    - German (deu): Indo-European language with distinctive fricatives and vowels

    Returns:
        LanguageRegistry with all default language profiles loaded from JSON configs
    """
    from tenepal.language.profile_loader import load_default_profiles

    registry = LanguageRegistry()
    for profile in load_default_profiles():
        registry.register(profile)
    return registry

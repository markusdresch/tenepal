"""Language identification module.

Provides language profile registry and identification capabilities
for phonetic language classification.
"""

from tenepal.language.analyzer import (
    PhonemeAnalysis,
    ProfileHits,
    ConfusionMatrix,
    analyze_phonemes,
    build_confusion_matrix,
    format_analysis,
)
from tenepal.language.formatter import (
    format_language_segments,
    print_language_segments,
)
from tenepal.language.identifier import (
    LanguageSegment,
    identify_language,
)
from tenepal.language.profile_loader import (
    load_profile,
    load_profiles_from_directory,
    load_default_profiles,
)
from tenepal.language.registry import (
    LanguageProfile,
    LanguageRegistry,
    default_registry,
    normalize_language_code,
)
from tenepal.language.smoother import (
    smooth_by_speaker,
    SpeakerLanguageStats,
)
from tenepal.language.speaker_profile import (
    SpeakerProfile,
    build_speaker_profiles,
    apply_speaker_inheritance,
)
from tenepal.language.maya_lexicon import (
    MayaLexicon,
)

__all__ = [
    "LanguageProfile",
    "LanguageRegistry",
    "default_registry",
    "normalize_language_code",
    "load_profile",
    "load_profiles_from_directory",
    "load_default_profiles",
    "LanguageSegment",
    "identify_language",
    "format_language_segments",
    "print_language_segments",
    "PhonemeAnalysis",
    "ProfileHits",
    "ConfusionMatrix",
    "analyze_phonemes",
    "build_confusion_matrix",
    "format_analysis",
    "smooth_by_speaker",
    "SpeakerLanguageStats",
    "SpeakerProfile",
    "build_speaker_profiles",
    "apply_speaker_inheritance",
    "MayaLexicon",
]

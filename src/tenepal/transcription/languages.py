"""Language routing constants for Whisper transcription integration.

This module defines which languages Tenepal routes to faster-whisper for
transcription versus phoneme-based recognition via Allosaurus.

Whisper supports 99 languages but Nahuatl (nah) is NOT among them. Therefore:
- Whisper-supported languages (spa, eng, deu, fra, ita) → faster-whisper
- Other languages (nah, etc.) → Allosaurus for IPA phoneme transcription

The ISO_639_MAP translates Tenepal's internal ISO 639-3 codes to the
ISO 639-1 codes required by Whisper's API.
"""

from typing import Final

# Languages Tenepal routes to Whisper for transcription
# These are the languages present in both Tenepal's profiles and Whisper's 99-language set
WHISPER_SUPPORTED: Final[frozenset[str]] = frozenset({"spa", "eng", "deu", "fra", "ita"})

# Maps ISO 639-3 (Tenepal internal) to ISO 639-1 (Whisper API requirement)
ISO_639_MAP: Final[dict[str, str]] = {
    "spa": "es",  # Spanish
    "eng": "en",  # English
    "deu": "de",  # German
    "fra": "fr",  # French
    "ita": "it",  # Italian
}

# Reverse map: ISO 639-1 (Whisper output) → ISO 639-3 (Tenepal internal)
WHISPER_LANG_REVERSE: Final[dict[str, str]] = {
    **{v: k for k, v in ISO_639_MAP.items()},
    # Common Whisper confusions for short clips (validator checks actual text):
    "pt": "spa",  # Portuguese → treat as Spanish
    "ca": "spa",  # Catalan → treat as Spanish
    "gl": "spa",  # Galician → treat as Spanish
}

# Valid faster-whisper model sizes
WHISPER_MODEL_SIZES: Final[frozenset[str]] = frozenset({
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "turbo",
})

"""Terminal display formatting for language-tagged segments."""

import sys

from tenepal.language.identifier import LanguageSegment

# ANSI color codes for terminal output
LANGUAGE_COLORS = {
    "nah": "\033[32m",  # Green for Nahuatl (indigenous language, primary focus)
    "spa": "\033[33m",  # Yellow for Spanish
    "eng": "\033[34m",  # Blue for English
    "deu": "\033[35m",  # Magenta for German
    "may": "\033[36m",  # Cyan for Yucatec Maya
    "lat": "\033[31m",  # Red for Latin liturgical segments
    "other": "\033[2m",  # Dim/gray for unidentified
}
RESET = "\033[0m"


def _language_label(code: str) -> str:
    """Convert language code to display label.

    Args:
        code: Language code (ISO 639) or "other"

    Returns:
        Three-character uppercase label for display
    """
    label_map = {
        "nah": "NAH",
        "spa": "SPA",
        "eng": "ENG",
        "deu": "DEU",
        "fra": "FRA",
        "ita": "ITA",
        "may": "MAY",
        "lat": "LAT",
    }
    return label_map.get(code, "OTH")


def format_language_segments(
    segments: list[LanguageSegment],
    use_color: bool = True,
    pronounce: str | None = None,
) -> str:
    """Format language-tagged segments for display.

    Args:
        segments: List of LanguageSegment objects
        use_color: If True and stdout is a tty, use ANSI color codes

    Returns:
        Formatted string representation with language labels and phonemes
    """
    if not segments:
        return ""

    # Disable colors if output is not a terminal (piped/redirected)
    use_color = use_color and sys.stdout.isatty()

    lines = []
    lines.append("Language-Tagged Phoneme Segments")
    lines.append("-" * 60)

    # Count segments by language for summary
    language_counts = {}

    for segment in segments:
        # Extract language label and color
        label = _language_label(segment.language)
        color = LANGUAGE_COLORS.get(segment.language, LANGUAGE_COLORS["other"])

        # Count this language
        language_counts[segment.language] = language_counts.get(segment.language, 0) + 1

        # Format time range
        time_range = f"{segment.start_time:.2f}s - {segment.end_time:.2f}s"

        # Format phonemes (space-separated IPA or locale rendering)
        if pronounce:
            from tenepal.pronunciation import render_pronunciation

            phonemes_str = render_pronunciation([p.phoneme for p in segment.phonemes], pronounce)
        else:
            phonemes_str = " ".join(p.phoneme for p in segment.phonemes)

        # Build line: time | [Speaker | label] or [label] | phonemes
        if segment.speaker:
            tag = f"[{segment.speaker} | {label}]"
        else:
            tag = f"[{label}]"
        line = f"{time_range:<20} {tag} {phonemes_str}"

        # Apply color if enabled
        if use_color:
            line = f"{color}{line}{RESET}"

        lines.append(line)

    # Add footer with summary
    lines.append("-" * 60)

    # Build summary string
    summary_parts = []
    # Show in stable priority order
    for lang_code in ["nah", "may", "spa", "eng", "deu", "fra", "ita", "lat", "other"]:
        if lang_code in language_counts:
            lang_name = {
                "nah": "Nahuatl",
                "may": "Maya",
                "spa": "Spanish",
                "eng": "English",
                "deu": "German",
                "fra": "French",
                "ita": "Italian",
                "lat": "Latin",
                "other": "Other"
            }[lang_code]
            summary_parts.append(f"{lang_name}: {language_counts[lang_code]}")

    summary = f"Total segments: {len(segments)} ({', '.join(summary_parts)})"
    lines.append(summary)

    return "\n".join(lines)


def print_language_segments(
    segments: list[LanguageSegment],
    use_color: bool = True,
    pronounce: str | None = None,
) -> None:
    """Print language-tagged segments to stdout with header/footer decoration.

    Args:
        segments: List of LanguageSegment objects
        use_color: If True and stdout is a tty, use ANSI color codes
    """
    print("\nLanguage Identification Results:")
    print("=" * 60)

    output = format_language_segments(segments, use_color, pronounce=pronounce)
    print(output)

    print("=" * 60)

"""Terminal display formatting for phoneme streams."""

from .recognizer import PhonemeSegment


def format_phonemes(segments: list[PhonemeSegment], show_timestamps: bool = True) -> str:
    """Format phoneme segments for display.

    Args:
        segments: List of PhonemeSegment objects
        show_timestamps: If True, show timestamps; if False, just phonemes

    Returns:
        Formatted string representation
    """
    if not segments:
        return ""

    if show_timestamps:
        # Format as table with aligned columns
        lines = ["Time      Duration  Phoneme"]
        lines.append("-" * 30)

        for seg in segments:
            time_str = f"{seg.start_time:.3f}s"
            duration_str = f"{seg.duration:.3f}s"
            lines.append(f"{time_str:<10}{duration_str:<10}{seg.phoneme}")

        return "\n".join(lines)
    else:
        # Just join phonemes with spaces
        return " ".join(seg.phoneme for seg in segments)


def print_phonemes(segments: list[PhonemeSegment], show_timestamps: bool = True) -> None:
    """Print phoneme stream to stdout.

    Args:
        segments: List of PhonemeSegment objects
        show_timestamps: If True, show timestamps; if False, just phonemes
    """
    if show_timestamps:
        print("\nPhoneme Stream (IPA with timestamps):")
        print("=" * 40)

    output = format_phonemes(segments, show_timestamps)
    print(output)

    if show_timestamps:
        print("=" * 40)
        print(f"Total segments: {len(segments)}")

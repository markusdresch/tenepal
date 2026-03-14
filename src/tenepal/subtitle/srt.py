"""SRT subtitle format generation from language-tagged phoneme segments.

SRT (SubRip) format:
- Sequential cue numbers starting at 1
- Timestamps: HH:MM:SS,mmm --> HH:MM:SS,mmm
- Cue text with language tags: [NAH] phonemes
- Blank line separator between cues
"""

from pathlib import Path

from tenepal.language.formatter import _language_label
from tenepal.language.identifier import LanguageSegment


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm.

    Args:
        seconds: Time in seconds (can be float with fractional seconds)

    Returns:
        Timestamp string in SRT format (e.g., "00:01:05,123")
    """
    # Calculate hours, minutes, seconds, milliseconds
    hours = int(seconds // 3600)
    remaining = seconds - (hours * 3600)
    minutes = int(remaining // 60)
    remaining = remaining - (minutes * 60)
    secs = int(remaining)
    millis = int(round((remaining - secs) * 1000))

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def merge_consecutive_segments(segments: list[LanguageSegment]) -> list[LanguageSegment]:
    """Merge consecutive segments with the same language code.

    Combines adjacent segments that have the same language, merging their
    phoneme lists and extending the time range.

    Args:
        segments: List of LanguageSegment objects

    Returns:
        List of LanguageSegment objects with consecutive same-language segments merged
    """
    if not segments:
        return []

    merged = []
    current_segment = segments[0]

    for next_segment in segments[1:]:
        if next_segment.language == current_segment.language and next_segment.speaker == current_segment.speaker:
            # Same language and speaker: merge by combining phonemes and extending time range
            merged_phonemes = current_segment.phonemes + next_segment.phonemes
            current_segment = LanguageSegment(
                language=current_segment.language,
                phonemes=merged_phonemes,
                start_time=current_segment.start_time,
                end_time=next_segment.end_time,
                speaker=current_segment.speaker
            )
        else:
            # Different language: save current segment and start new one
            merged.append(current_segment)
            current_segment = next_segment

    # Add the last segment
    merged.append(current_segment)

    return merged


def format_srt(segments: list[LanguageSegment], pronounce: str | None = None) -> str:
    """Generate SRT subtitle content from language-tagged segments.

    Processes segments through merging and formatting pipeline:
    1. Merge consecutive same-language segments
    2. Format each segment as numbered SRT cue with timestamp and tagged phonemes

    Args:
        segments: List of LanguageSegment objects

    Returns:
        Complete SRT subtitle content as string (empty string if no segments)
    """
    if not segments:
        return ""

    # Merge consecutive same-language segments
    merged_segments = merge_consecutive_segments(segments)

    # Format each segment as SRT cue
    cues = []
    for i, segment in enumerate(merged_segments, start=1):
        # Cue number
        cue_num = str(i)

        # Timestamp line
        start_timestamp = format_timestamp(segment.start_time)
        end_timestamp = format_timestamp(segment.end_time)
        timestamp_line = f"{start_timestamp} --> {end_timestamp}"

        # Cue text: [Speaker | LANG] or [LANG] + transcription or phonemes
        language_tag = _language_label(segment.language)
        if segment.speaker:
            cue_tag = f"[{segment.speaker} | {language_tag}]"
        else:
            cue_tag = f"[{language_tag}]"

        # Check for transcription text (from Whisper via TranscriptionRouter)
        transcription = getattr(segment, 'transcription', None)
        if transcription:
            # Use readable text from Whisper transcription
            cue_text = f"{cue_tag} {transcription}"
        else:
            # Use IPA phonemes (existing behavior for Allosaurus/no-Whisper mode)
            if pronounce:
                from tenepal.pronunciation import render_pronunciation

                phonemes_str = render_pronunciation([p.phoneme for p in segment.phonemes], pronounce)
            else:
                phonemes_str = " ".join(p.phoneme for p in segment.phonemes)
            cue_text = f"{cue_tag} {phonemes_str}"

        # Combine into cue (number, timestamp, text, blank line)
        cue = f"{cue_num}\n{timestamp_line}\n{cue_text}\n"
        cues.append(cue)

    # Join cues with blank line separator
    return "\n".join(cues) + "\n"


def write_srt(
    segments: list[LanguageSegment],
    output_path: str | Path,
    pronounce: str | None = None,
) -> Path:
    """Write SRT subtitle content to a file.

    Generates SRT content from segments and writes to the specified path with UTF-8 encoding.
    Creates parent directories if they don't exist.

    Args:
        segments: List of LanguageSegment objects to format as SRT
        output_path: Path where SRT file should be written (str or Path)

    Returns:
        Path object pointing to the written file

    Example:
        >>> segments = [LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1)]
        >>> output = write_srt(segments, "output.srt")
        >>> output.exists()
        True
    """
    # Convert to Path object
    path = Path(output_path)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Generate SRT content
    srt_content = format_srt(segments, pronounce=pronounce)

    # Write to file with UTF-8 encoding (for IPA characters)
    path.write_text(srt_content, encoding="utf-8")

    return path

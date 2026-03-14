"""Speaker statistics formatting."""

from typing import List


def format_speaker_stats(segments: List) -> str:
    """Format speaker statistics from language segments.

    Calculates speaking time per speaker and formats as percentages.

    Args:
        segments: List of LanguageSegment objects with speaker labels

    Returns:
        Formatted string like "Detected 3 speakers: Speaker A (45%), Speaker B (35%), Speaker C (20%)"
        or "No speakers detected" if no speakers found
    """
    # Collect speaker speaking times
    speaker_times = {}

    for seg in segments:
        if seg.speaker is None:
            continue

        duration = seg.end_time - seg.start_time
        if seg.speaker not in speaker_times:
            speaker_times[seg.speaker] = 0.0
        speaker_times[seg.speaker] += duration

    if not speaker_times:
        return "No speakers detected"

    # Calculate total time
    total_time = sum(speaker_times.values())

    # Sort by speaking time descending
    sorted_speakers = sorted(
        speaker_times.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Format percentages
    speaker_parts = []
    for speaker, time in sorted_speakers:
        percentage = int(round(100 * time / total_time))
        speaker_parts.append(f"{speaker} ({percentage}%)")

    # Format output
    count = len(speaker_times)
    plural = "speaker" if count == 1 else "speakers"
    speakers_str = ", ".join(speaker_parts)

    return f"Detected {count} {plural}: {speakers_str}"

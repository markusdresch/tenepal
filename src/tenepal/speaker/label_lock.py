"""Speaker label locking for stable labels across re-diarization passes."""

from typing import Dict, List
from tenepal.speaker.diarizer import SpeakerSegment


class LabelLocker:
    """Stabilize speaker labels across re-diarization passes.

    Ensures speaker labels (Speaker A, Speaker B, etc.) remain consistent
    even when pyannote.audio reassigns internal labels across diarization passes.
    Uses time overlap to match speakers from new passes to previous labels.
    """

    def __init__(self):
        """Initialize the label locker."""
        self._previous_segments: List[SpeakerSegment] = []
        self._label_map: Dict[str, str] = {}
        self._next_index: int = 0

    def stabilize(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Stabilize speaker labels across diarization passes.

        First call: Maps speakers to alphabetical labels (A, B, C...) by order.
        Subsequent calls: Matches new speakers to previous speakers by time overlap,
        assigns same label. New speakers get next available letter.

        Args:
            segments: List of SpeakerSegment from pyannote diarization

        Returns:
            New list of SpeakerSegment with stabilized speaker labels
        """
        if not segments:
            return []

        # First pass: identity mapping
        if not self._previous_segments:
            return self._first_pass(segments)

        # Subsequent passes: match by time overlap
        return self._match_by_overlap(segments)

    def _first_pass(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Process first diarization pass - assign labels by appearance order."""
        result = []
        unique_speakers = {}  # Original label -> stable label

        for seg in segments:
            if seg.speaker not in unique_speakers:
                unique_speakers[seg.speaker] = self._letter_label(self._next_index)
                self._next_index += 1

            stable_label = unique_speakers[seg.speaker]
            result.append(SpeakerSegment(
                speaker=stable_label,
                start_time=seg.start_time,
                end_time=seg.end_time
            ))

        # Store for next pass
        self._previous_segments = result
        self._label_map = unique_speakers

        return result

    def _match_by_overlap(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Match new speakers to previous by time overlap."""
        result = []
        new_to_stable: Dict[str, str] = {}

        # Build reverse map: stable label -> list of time ranges
        stable_times: Dict[str, List[tuple]] = {}
        for seg in self._previous_segments:
            if seg.speaker not in stable_times:
                stable_times[seg.speaker] = []
            stable_times[seg.speaker].append((seg.start_time, seg.end_time))

        for seg in segments:
            # If we've already mapped this speaker, reuse it
            if seg.speaker in new_to_stable:
                stable_label = new_to_stable[seg.speaker]
            else:
                # Find best match by time overlap
                stable_label = self._find_best_match(seg, stable_times)

                if stable_label is None:
                    # New speaker not in previous pass
                    stable_label = self._letter_label(self._next_index)
                    self._next_index += 1

                new_to_stable[seg.speaker] = stable_label

            result.append(SpeakerSegment(
                speaker=stable_label,
                start_time=seg.start_time,
                end_time=seg.end_time
            ))

        # Update for next pass
        self._previous_segments = result

        return result

    def _find_best_match(
        self,
        segment: SpeakerSegment,
        stable_times: Dict[str, List[tuple]]
    ) -> str | None:
        """Find stable label with maximum time overlap."""
        best_label = None
        best_overlap = 0.0

        for stable_label, time_ranges in stable_times.items():
            total_overlap = 0.0
            for start, end in time_ranges:
                overlap = self._compute_overlap(
                    segment.start_time, segment.end_time,
                    start, end
                )
                total_overlap += overlap

            if total_overlap > best_overlap:
                best_overlap = total_overlap
                best_label = stable_label

        # Return label only if there was actual overlap
        return best_label if best_overlap > 0 else None

    def _compute_overlap(self, a_start: float, a_end: float,
                        b_start: float, b_end: float) -> float:
        """Compute time overlap between two segments."""
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        return max(0.0, overlap_end - overlap_start)

    def _letter_label(self, index: int) -> str:
        """Convert index to Speaker A, Speaker B, etc."""
        letter = chr(ord('A') + index)
        return f"Speaker {letter}"

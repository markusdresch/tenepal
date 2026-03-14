"""Tests for speaker diarization module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tenepal.audio.loader import AudioData
from tenepal.speaker.diarizer import SpeakerSegment, _letter_label, diarize, slice_audio_by_speaker


# --- Mock helpers for pyannote ---

class MockSegment:
    """Mock pyannote Segment with .start and .end attributes."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class MockAnnotation:
    """Mock pyannote Annotation that implements itertracks(yield_label=True)."""

    def __init__(self, tracks: list[tuple[MockSegment, str, str]]):
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        if yield_label:
            return iter(self._tracks)
        return iter([(seg, track) for seg, track, _ in self._tracks])


# --- Tests ---

class TestSpeakerSegment:
    def test_speaker_segment_creation(self):
        """SpeakerSegment dataclass works correctly."""
        seg = SpeakerSegment(speaker="Speaker A", start_time=0.0, end_time=5.0)
        assert seg.speaker == "Speaker A"
        assert seg.start_time == 0.0
        assert seg.end_time == 5.0

    def test_speaker_segment_equality(self):
        """Two SpeakerSegments with same values are equal."""
        seg1 = SpeakerSegment(speaker="Speaker A", start_time=0.0, end_time=5.0)
        seg2 = SpeakerSegment(speaker="Speaker A", start_time=0.0, end_time=5.0)
        assert seg1 == seg2


class TestLetterLabel:
    def test_letter_label_first(self):
        """_letter_label(0) returns 'Speaker A'."""
        assert _letter_label(0) == "Speaker A"

    def test_letter_label_last_single(self):
        """_letter_label(25) returns 'Speaker Z'."""
        assert _letter_label(25) == "Speaker Z"

    def test_letter_label_middle(self):
        """_letter_label(1) returns 'Speaker B'."""
        assert _letter_label(1) == "Speaker B"

    def test_letter_label_double_first(self):
        """_letter_label(26) returns 'Speaker AA' for >26 speakers."""
        assert _letter_label(26) == "Speaker AA"

    def test_letter_label_double_second(self):
        """_letter_label(27) returns 'Speaker AB'."""
        assert _letter_label(27) == "Speaker AB"


class TestDiarizeFallback:
    def test_diarize_fallback_no_token(self, monkeypatch, tmp_path):
        """When HUGGINGFACE_TOKEN not set, diarize() returns single fallback segment."""
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

        # Reset the cached pipeline so _load_pipeline runs fresh
        import tenepal.speaker.diarizer as mod
        mod._pipeline = None
        mod._pipeline_loaded = False

        # Create a short WAV file for duration detection
        sample_rate = 16000
        duration = 3.0
        samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
        wav_path = tmp_path / "test.wav"

        import soundfile as sf
        sf.write(str(wav_path), samples, sample_rate)

        result = diarize(wav_path)
        assert len(result) == 1
        assert result[0].speaker == "Speaker ?"
        assert result[0].start_time == 0.0
        assert abs(result[0].end_time - duration) < 0.1

    def test_diarize_fallback_no_pyannote(self, monkeypatch, tmp_path):
        """When pyannote import fails, returns fallback segment."""
        import tenepal.speaker.diarizer as mod
        mod._pipeline = None
        mod._pipeline_loaded = False

        # Even with a token, if pyannote is not importable, fallback
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")

        # Patch the import to fail
        with patch.object(mod, "_load_pipeline", return_value=None):
            sample_rate = 16000
            duration = 2.0
            samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
            wav_path = tmp_path / "test.wav"

            import soundfile as sf
            sf.write(str(wav_path), samples, sample_rate)

            result = diarize(wav_path)
            assert len(result) == 1
            assert result[0].speaker == "Speaker ?"


class TestDiarizeWithMock:
    def test_diarize_with_mock_pipeline(self, tmp_path):
        """Mock pyannote Pipeline returns 2 speakers with correct labels."""
        import tenepal.speaker.diarizer as mod

        # Build mock annotation: 2 speakers, 2 segments each
        tracks = [
            (MockSegment(0.0, 2.5), "track_0", "SPEAKER_00"),
            (MockSegment(2.5, 5.0), "track_1", "SPEAKER_01"),
            (MockSegment(5.0, 7.5), "track_2", "SPEAKER_00"),
            (MockSegment(7.5, 10.0), "track_3", "SPEAKER_01"),
        ]
        mock_annotation = MockAnnotation(tracks)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_annotation

        with patch.object(mod, "_load_pipeline", return_value=mock_pipeline):
            # Create a short WAV file
            sample_rate = 16000
            samples = np.zeros(int(sample_rate * 10), dtype=np.float32)
            wav_path = tmp_path / "test.wav"

            import soundfile as sf
            sf.write(str(wav_path), samples, sample_rate)

            result = diarize(wav_path)

        assert len(result) == 4
        # Check labels: SPEAKER_00 appears first -> "Speaker A"
        assert result[0].speaker == "Speaker A"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 2.5
        assert result[1].speaker == "Speaker B"
        assert result[1].start_time == 2.5
        assert result[1].end_time == 5.0
        assert result[2].speaker == "Speaker A"
        assert result[3].speaker == "Speaker B"

    def test_diarize_overlap(self, tmp_path):
        """Overlapping segments from two speakers are both present."""
        import tenepal.speaker.diarizer as mod

        # Two speakers talking at the same time (overlap from 2.0 to 3.0)
        tracks = [
            (MockSegment(0.0, 3.0), "track_0", "SPEAKER_00"),
            (MockSegment(2.0, 5.0), "track_1", "SPEAKER_01"),
        ]
        mock_annotation = MockAnnotation(tracks)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_annotation

        with patch.object(mod, "_load_pipeline", return_value=mock_pipeline):
            sample_rate = 16000
            samples = np.zeros(int(sample_rate * 5), dtype=np.float32)
            wav_path = tmp_path / "test.wav"

            import soundfile as sf
            sf.write(str(wav_path), samples, sample_rate)

            result = diarize(wav_path)

        assert len(result) == 2
        # Sorted by start_time; Speaker A at 0.0, Speaker B at 2.0
        assert result[0].speaker == "Speaker A"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 3.0
        assert result[1].speaker == "Speaker B"
        assert result[1].start_time == 2.0
        assert result[1].end_time == 5.0

    def test_speaker_labels_consistent(self, tmp_path):
        """Labels assigned by first appearance order, not pyannote internal names."""
        import tenepal.speaker.diarizer as mod

        # 3 speakers, appearing in non-sequential pyannote order
        tracks = [
            (MockSegment(0.0, 1.0), "track_0", "SPEAKER_02"),
            (MockSegment(1.0, 2.0), "track_1", "SPEAKER_00"),
            (MockSegment(2.0, 3.0), "track_2", "SPEAKER_01"),
            (MockSegment(3.0, 4.0), "track_3", "SPEAKER_02"),
        ]
        mock_annotation = MockAnnotation(tracks)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_annotation

        with patch.object(mod, "_load_pipeline", return_value=mock_pipeline):
            sample_rate = 16000
            samples = np.zeros(int(sample_rate * 4), dtype=np.float32)
            wav_path = tmp_path / "test.wav"

            import soundfile as sf
            sf.write(str(wav_path), samples, sample_rate)

            result = diarize(wav_path)

        assert len(result) == 4
        # SPEAKER_02 appears first -> "Speaker A"
        # SPEAKER_00 appears second -> "Speaker B"
        # SPEAKER_01 appears third -> "Speaker C"
        assert result[0].speaker == "Speaker A"  # SPEAKER_02
        assert result[1].speaker == "Speaker B"  # SPEAKER_00
        assert result[2].speaker == "Speaker C"  # SPEAKER_01
        assert result[3].speaker == "Speaker A"  # SPEAKER_02 again


class TestSliceAudioBySpeaker:
    def test_slice_audio_by_speaker(self):
        """Slicing 10s audio into two 5s segments produces correct sample counts."""
        sample_rate = 16000
        total_samples = sample_rate * 10  # 10 seconds
        samples = np.arange(total_samples, dtype=np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=10.0,
            source_format="wav",
        )

        segments = [
            SpeakerSegment(speaker="Speaker A", start_time=0.0, end_time=5.0),
            SpeakerSegment(speaker="Speaker B", start_time=5.0, end_time=10.0),
        ]

        result = slice_audio_by_speaker(audio, segments)
        assert len(result) == 2

        seg_a, audio_a = result[0]
        seg_b, audio_b = result[1]

        assert seg_a.speaker == "Speaker A"
        assert seg_b.speaker == "Speaker B"
        assert len(audio_a.samples) == sample_rate * 5
        assert len(audio_b.samples) == sample_rate * 5
        assert abs(audio_a.duration - 5.0) < 0.01
        assert abs(audio_b.duration - 5.0) < 0.01
        assert audio_a.source_format == "wav"

        # Verify actual sample values (sequential)
        assert audio_a.samples[0] == 0.0
        assert audio_b.samples[0] == float(sample_rate * 5)

    def test_slice_audio_overlap(self):
        """Overlapping segments produce overlapping audio slices."""
        sample_rate = 16000
        total_samples = sample_rate * 10
        samples = np.arange(total_samples, dtype=np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=10.0,
            source_format="wav",
        )

        # Overlap from 3.0 to 5.0
        segments = [
            SpeakerSegment(speaker="Speaker A", start_time=0.0, end_time=5.0),
            SpeakerSegment(speaker="Speaker B", start_time=3.0, end_time=10.0),
        ]

        result = slice_audio_by_speaker(audio, segments)
        assert len(result) == 2

        _, audio_a = result[0]
        _, audio_b = result[1]

        # Speaker A: 0-5s = 80000 samples
        assert len(audio_a.samples) == sample_rate * 5
        # Speaker B: 3-10s = 112000 samples
        assert len(audio_b.samples) == sample_rate * 7

        # Overlapping region shares same sample values
        overlap_start_sample = sample_rate * 3
        assert audio_a.samples[overlap_start_sample] == audio_b.samples[0]

    def test_slice_audio_clamps_indices(self):
        """Segments extending beyond audio duration are clamped."""
        sample_rate = 16000
        total_samples = sample_rate * 5  # 5 seconds
        samples = np.arange(total_samples, dtype=np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=5.0,
            source_format="wav",
        )

        # Segment extends beyond audio
        segments = [
            SpeakerSegment(speaker="Speaker A", start_time=3.0, end_time=10.0),
        ]

        result = slice_audio_by_speaker(audio, segments)
        assert len(result) == 1
        _, audio_a = result[0]

        # Should be clamped to 3.0-5.0 = 2 seconds
        assert len(audio_a.samples) == sample_rate * 2

    def test_slice_audio_skips_zero_length(self):
        """Zero-length segments are skipped."""
        sample_rate = 16000
        total_samples = sample_rate * 5
        samples = np.arange(total_samples, dtype=np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=5.0,
            source_format="wav",
        )

        segments = [
            SpeakerSegment(speaker="Speaker A", start_time=3.0, end_time=3.0),  # zero-length
            SpeakerSegment(speaker="Speaker B", start_time=1.0, end_time=4.0),
        ]

        result = slice_audio_by_speaker(audio, segments)
        assert len(result) == 1
        assert result[0][0].speaker == "Speaker B"

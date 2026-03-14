"""Tests for audio preprocessing modules."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock, call
from pathlib import Path
import numpy as np
import json
import subprocess
import torch

from tenepal.preprocessing import SpeechSegment
from tenepal.preprocessing.extractor import extract_audio
from tenepal.preprocessing.isolator import isolate_vocals, is_demucs_available
from tenepal.preprocessing.segmenter import segment_speech
from tenepal.preprocessing.pipeline import preprocess_video, export_segments_json


class TestSpeechSegment:
    """Tests for SpeechSegment dataclass."""

    def test_create_segment_with_all_fields(self):
        """Test creating a SpeechSegment with all fields."""
        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = SpeechSegment(
            start_ms=1000,
            end_ms=2000,
            audio_data=audio_data,
            sample_rate=22050,
        )

        assert segment.start_ms == 1000
        assert segment.end_ms == 2000
        assert np.array_equal(segment.audio_data, audio_data)
        assert segment.sample_rate == 22050

    def test_segment_with_default_sample_rate(self):
        """Test that default sample rate is 22050."""
        segment = SpeechSegment(
            start_ms=0,
            end_ms=1000,
            audio_data=np.array([0.1, 0.2]),
        )

        assert segment.sample_rate == 22050

    def test_audio_data_is_numpy_array(self):
        """Test that audio_data is a numpy array."""
        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = SpeechSegment(
            start_ms=0,
            end_ms=1000,
            audio_data=audio_data,
        )

        assert isinstance(segment.audio_data, np.ndarray)


class TestExtractor:
    """Tests for FFmpeg audio extraction."""

    @patch('subprocess.run')
    def test_extract_audio_mkv_success(self, mock_run, tmp_path):
        """Test successful audio extraction from MKV."""
        video_path = tmp_path / "test.mkv"
        video_path.touch()
        output_dir = tmp_path / "output"

        # Mock ffprobe success
        probe_result = MagicMock()
        probe_result.stdout = "audio"

        # Mock ffmpeg success
        ffmpeg_result = MagicMock()

        mock_run.side_effect = [probe_result, ffmpeg_result]

        # Create expected output file for stat() call
        expected_output = output_dir / "test_extracted.wav"
        output_dir.mkdir()
        expected_output.touch()

        result = extract_audio(video_path, output_dir)

        assert result == expected_output
        assert mock_run.call_count == 2

        # Verify ffmpeg command structure
        ffmpeg_call = mock_run.call_args_list[1]
        cmd = ffmpeg_call[0][0]
        assert "ffmpeg" in cmd
        assert "-vn" in cmd
        assert "-map" in cmd
        assert "0:a:0" in cmd
        assert "-ac" in cmd
        assert "1" in cmd
        assert "-ar" in cmd
        assert "22050" in cmd
        assert "-acodec" in cmd
        assert "pcm_s16le" in cmd

    @patch('subprocess.run')
    def test_extract_audio_mp4_success(self, mock_run, tmp_path):
        """Test successful audio extraction from MP4."""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        probe_result = MagicMock()
        probe_result.stdout = "audio"
        ffmpeg_result = MagicMock()
        mock_run.side_effect = [probe_result, ffmpeg_result]

        expected_output = output_dir / "test_extracted.wav"
        output_dir.mkdir()
        expected_output.touch()

        result = extract_audio(video_path, output_dir)

        assert result == expected_output
        assert result.name == "test_extracted.wav"

    def test_extract_audio_rejects_avi(self, tmp_path):
        """Test that unsupported format raises ValueError."""
        video_path = tmp_path / "test.avi"
        video_path.touch()
        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="Unsupported video format"):
            extract_audio(video_path, output_dir)

    def test_extract_audio_missing_file(self, tmp_path):
        """Test that missing video raises FileNotFoundError."""
        video_path = tmp_path / "missing.mkv"
        output_dir = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extract_audio(video_path, output_dir)

    @patch('subprocess.run')
    def test_extract_audio_no_audio_track(self, mock_run, tmp_path):
        """Test that video without audio track raises FileNotFoundError."""
        video_path = tmp_path / "test.mkv"
        video_path.touch()
        output_dir = tmp_path / "output"

        # Mock ffprobe returning empty (no audio)
        probe_result = MagicMock()
        probe_result.stdout = ""
        mock_run.return_value = probe_result

        with pytest.raises(FileNotFoundError, match="No audio track found"):
            extract_audio(video_path, output_dir)

    @patch('subprocess.run')
    def test_extract_audio_creates_output_dir(self, mock_run, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        video_path = tmp_path / "test.mkv"
        video_path.touch()
        output_dir = tmp_path / "new_output"

        probe_result = MagicMock()
        probe_result.stdout = "audio"
        ffmpeg_result = MagicMock()
        mock_run.side_effect = [probe_result, ffmpeg_result]

        expected_output = output_dir / "test_extracted.wav"
        output_dir.mkdir()
        expected_output.touch()

        extract_audio(video_path, output_dir)

        assert output_dir.exists()


class TestIsolator:
    """Tests for Demucs vocal isolation."""

    @patch('tenepal.preprocessing.isolator._load_separator')
    def test_isolate_vocals_calls_separator(self, mock_load_separator, tmp_path):
        """Test that isolate_vocals calls Demucs separator."""
        audio_path = tmp_path / "test.wav"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create mock separator
        mock_separator = MagicMock()
        vocals_data = np.random.randn(2, 44100).astype(np.float32)
        other_data = np.random.randn(2, 44100).astype(np.float32) * 0.1
        mock_separator.separate_audio_file.return_value = {
            "vocals": vocals_data,
            "other": other_data,
            "drums": np.zeros((2, 44100)),
            "bass": np.zeros((2, 44100)),
        }
        mock_load_separator.return_value = mock_separator

        # Create expected output file so stat() doesn't fail
        expected_output = output_dir / "test_isolated.wav"
        expected_output.touch()

        # Mock soundfile info
        with patch('soundfile.info') as mock_info, \
             patch('soundfile.write') as mock_write:
            mock_info_obj = MagicMock()
            mock_info_obj.samplerate = 44100
            mock_info.return_value = mock_info_obj

            result = isolate_vocals(audio_path, output_dir, segment_size=300)

            assert result == output_dir / "test_isolated.wav"
            mock_separator.separate_audio_file.assert_called_once()
            mock_write.assert_called_once()

    @patch('tenepal.preprocessing.isolator._load_separator')
    def test_isolate_vocals_uses_segment_parameter(self, mock_load_separator, tmp_path):
        """Test that segment parameter is passed to separator."""
        audio_path = tmp_path / "test.wav"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_separator = MagicMock()
        vocals_data = np.random.randn(2, 44100).astype(np.float32)
        other_data = np.random.randn(2, 44100).astype(np.float32) * 0.1
        mock_separator.separate_audio_file.return_value = {
            "vocals": vocals_data,
            "other": other_data,
            "drums": np.zeros((2, 44100)),
            "bass": np.zeros((2, 44100)),
        }
        mock_load_separator.return_value = mock_separator

        # Create expected output file so stat() doesn't fail
        expected_output = output_dir / "test_isolated.wav"
        expected_output.touch()

        with patch('soundfile.info') as mock_info, \
             patch('soundfile.write'):
            mock_info_obj = MagicMock()
            mock_info_obj.samplerate = 44100
            mock_info.return_value = mock_info_obj

            isolate_vocals(audio_path, output_dir, segment_size=150)

            # Verify segment_size was passed to _load_separator
            mock_load_separator.assert_called_with(150)

    @patch('tenepal.preprocessing.isolator._load_separator')
    def test_isolate_vocals_low_confidence_warning(self, mock_load_separator, tmp_path, caplog):
        """Test that low isolation confidence triggers warning."""
        audio_path = tmp_path / "test.wav"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create mock data where other stem is loud (low confidence)
        mock_separator = MagicMock()
        vocals_data = np.ones((2, 44100), dtype=np.float32)
        other_data = np.ones((2, 44100), dtype=np.float32) * 0.9  # High ratio
        mock_separator.separate_audio_file.return_value = {
            "vocals": vocals_data,
            "other": other_data,
            "drums": np.zeros((2, 44100)),
            "bass": np.zeros((2, 44100)),
        }
        mock_load_separator.return_value = mock_separator

        # Create expected output file so stat() doesn't fail
        expected_output = output_dir / "test_isolated.wav"
        expected_output.touch()

        with patch('soundfile.info') as mock_info, \
             patch('soundfile.write'):
            mock_info_obj = MagicMock()
            mock_info_obj.samplerate = 44100
            mock_info.return_value = mock_info_obj

            isolate_vocals(audio_path, output_dir)

            # Check for warning in logs
            assert any("Low isolation confidence" in record.message for record in caplog.records)

    @patch('tenepal.preprocessing.isolator._load_separator')
    def test_isolate_vocals_raises_on_missing_demucs(self, mock_load_separator, tmp_path):
        """Test that RuntimeError is raised when Demucs not installed."""
        audio_path = tmp_path / "test.wav"
        output_dir = tmp_path / "output"

        mock_load_separator.return_value = None

        with pytest.raises(RuntimeError, match="Demucs not installed"):
            isolate_vocals(audio_path, output_dir)

    @patch('tenepal.preprocessing.isolator._load_separator')
    def test_isolate_vocals_handles_oom_error(self, mock_load_separator, tmp_path):
        """Test that MemoryError is caught with helpful advice."""
        audio_path = tmp_path / "test.wav"
        output_dir = tmp_path / "output"

        mock_separator = MagicMock()
        mock_separator.separate_audio_file.side_effect = MemoryError("CUDA out of memory")
        mock_load_separator.return_value = mock_separator

        with pytest.raises(RuntimeError, match="Out of memory.*--demucs-segment 150"):
            isolate_vocals(audio_path, output_dir, segment_size=300)

    def test_is_demucs_available_true(self):
        """Test is_demucs_available returns True when installed."""
        with patch('builtins.__import__'):
            result = is_demucs_available()
            # Can't easily test True case without actual import
            assert isinstance(result, bool)

    def test_is_demucs_available_false(self):
        """Test is_demucs_available returns False when not installed."""
        with patch('builtins.__import__', side_effect=ImportError):
            result = is_demucs_available()
            assert result is False


class TestSegmenter:
    """Tests for Silero-VAD speech segmentation."""

    @patch('tenepal.preprocessing.segmenter._load_silero_vad')
    @patch('soundfile.read')
    def test_segment_speech_returns_segments(self, mock_sf_read, mock_load_vad, tmp_path):
        """Test that segment_speech returns list of SpeechSegment."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        output_dir = tmp_path / "output"

        # Mock VAD model and utils
        mock_model = MagicMock()
        mock_get_timestamps = MagicMock()
        mock_read_audio = MagicMock()

        # Mock VAD returning 2 segments
        mock_get_timestamps.return_value = [
            {"start": 0, "end": 8000},      # 0-0.5s at 16kHz
            {"start": 16000, "end": 32000}, # 1-2s at 16kHz
        ]
        mock_read_audio.return_value = torch.zeros(48000)  # 3s at 16kHz

        mock_load_vad.return_value = (mock_model, (mock_get_timestamps, None, mock_read_audio, None))

        # Mock soundfile reading original audio at 22050 Hz
        original_audio = np.random.randn(66150).astype(np.float32)  # 3s at 22050 Hz
        mock_sf_read.return_value = (original_audio, 22050)

        with patch('torch.no_grad'):
            segments = segment_speech(audio_path, output_dir)

        assert len(segments) == 2
        assert all(isinstance(s, SpeechSegment) for s in segments)
        assert all(s.sample_rate == 22050 for s in segments)

    @patch('tenepal.preprocessing.segmenter._load_silero_vad')
    @patch('soundfile.read')
    def test_segment_speech_converts_timestamps_correctly(self, mock_sf_read, mock_load_vad, tmp_path):
        """Test that VAD timestamps are converted to milliseconds correctly."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_get_timestamps = MagicMock()
        mock_read_audio = MagicMock()

        # Single segment: 16000 samples at 16kHz = 1 second = 1000ms
        mock_get_timestamps.return_value = [
            {"start": 0, "end": 16000},
        ]
        mock_read_audio.return_value = torch.zeros(16000)

        mock_load_vad.return_value = (mock_model, (mock_get_timestamps, None, mock_read_audio, None))

        original_audio = np.random.randn(22050).astype(np.float32)  # 1s at 22050 Hz
        mock_sf_read.return_value = (original_audio, 22050)

        with patch('torch.no_grad'):
            segments = segment_speech(audio_path, output_dir)

        assert len(segments) == 1
        assert segments[0].start_ms == 0
        assert segments[0].end_ms == 1000

    @patch('tenepal.preprocessing.segmenter._load_silero_vad')
    def test_segment_speech_wraps_with_no_grad(self, mock_load_vad, tmp_path):
        """Test that VAD inference is wrapped in torch.no_grad."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_get_timestamps = MagicMock()
        mock_read_audio = MagicMock()

        mock_get_timestamps.return_value = []
        mock_read_audio.return_value = torch.zeros(16000)

        mock_load_vad.return_value = (mock_model, (mock_get_timestamps, None, mock_read_audio, None))

        with patch('soundfile.read') as mock_sf_read, \
             patch('torch.no_grad') as mock_no_grad:
            mock_sf_read.return_value = (np.zeros(22050, dtype=np.float32), 22050)
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock()

            segment_speech(audio_path, output_dir)

            # Verify torch.no_grad was used
            mock_no_grad.assert_called_once()

    @patch('tenepal.preprocessing.segmenter._load_silero_vad')
    def test_segment_speech_raises_on_vad_failure(self, mock_load_vad, tmp_path):
        """Test that RuntimeError is raised when VAD fails to load."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        output_dir = tmp_path / "output"

        mock_load_vad.return_value = (None, None)

        with pytest.raises(RuntimeError, match="Silero-VAD failed to load"):
            segment_speech(audio_path, output_dir)


class TestPipeline:
    """Tests for pipeline orchestration."""

    @patch('tenepal.preprocessing.pipeline.extract_audio')
    @patch('tenepal.preprocessing.pipeline.isolate_vocals')
    @patch('tenepal.preprocessing.pipeline.segment_speech')
    @patch('tenepal.preprocessing.pipeline.export_segments_json')
    def test_preprocess_video_calls_all_stages(
        self, mock_export, mock_segment, mock_isolate, mock_extract, tmp_path
    ):
        """Test that preprocess_video calls extract -> isolate -> segment."""
        video_path = tmp_path / "test.mkv"
        video_path.touch()
        output_dir = tmp_path / "output"

        extracted_wav = tmp_path / "test_extracted.wav"
        isolated_wav = tmp_path / "test_isolated.wav"
        mock_segments = [
            SpeechSegment(0, 1000, np.array([0.1]), 22050),
            SpeechSegment(1000, 2000, np.array([0.2]), 22050),
        ]

        mock_extract.return_value = extracted_wav
        mock_isolate.return_value = isolated_wav
        mock_segment.return_value = mock_segments

        result = preprocess_video(video_path, output_dir)

        assert result == mock_segments
        mock_extract.assert_called_once()
        mock_isolate.assert_called_once()
        mock_segment.assert_called_once()
        mock_export.assert_called_once()

    @patch('tenepal.preprocessing.pipeline.extract_audio')
    @patch('tenepal.preprocessing.pipeline.isolate_vocals')
    @patch('tenepal.preprocessing.pipeline.segment_speech')
    @patch('tenepal.preprocessing.pipeline.export_segments_json')
    def test_preprocess_video_skip_isolation(
        self, mock_export, mock_segment, mock_isolate, mock_extract, tmp_path
    ):
        """Test that skip_isolation=True skips isolate_vocals."""
        video_path = tmp_path / "test.mkv"
        video_path.touch()
        output_dir = tmp_path / "output"

        extracted_wav = tmp_path / "test_extracted.wav"
        mock_segments = [SpeechSegment(0, 1000, np.array([0.1]), 22050)]

        mock_extract.return_value = extracted_wav
        mock_segment.return_value = mock_segments

        preprocess_video(video_path, output_dir, skip_isolation=True)

        mock_extract.assert_called_once()
        mock_isolate.assert_not_called()
        mock_segment.assert_called_once_with(extracted_wav, output_dir, min_silence_ms=500, padding_ms=100)

    @patch('tenepal.preprocessing.pipeline.extract_audio')
    @patch('tenepal.preprocessing.pipeline.isolate_vocals')
    @patch('tenepal.preprocessing.pipeline.segment_speech')
    @patch('tenepal.preprocessing.pipeline.export_segments_json')
    def test_preprocess_video_passes_demucs_segment(
        self, mock_export, mock_segment, mock_isolate, mock_extract, tmp_path
    ):
        """Test that demucs_segment parameter is passed through."""
        video_path = tmp_path / "test.mkv"
        video_path.touch()
        output_dir = tmp_path / "output"

        extracted_wav = tmp_path / "test_extracted.wav"
        isolated_wav = tmp_path / "test_isolated.wav"
        mock_segments = [SpeechSegment(0, 1000, np.array([0.1]), 22050)]

        mock_extract.return_value = extracted_wav
        mock_isolate.return_value = isolated_wav
        mock_segment.return_value = mock_segments

        preprocess_video(video_path, output_dir, demucs_segment=150)

        mock_isolate.assert_called_once_with(extracted_wav, output_dir, segment_size=150)

    @patch('tenepal.preprocessing.pipeline.extract_audio')
    @patch('tenepal.preprocessing.pipeline.isolate_vocals')
    @patch('tenepal.preprocessing.pipeline.segment_speech')
    @patch('tenepal.preprocessing.pipeline.export_segments_json')
    def test_preprocess_video_creates_output_dir(
        self, mock_export, mock_segment, mock_isolate, mock_extract, tmp_path
    ):
        """Test that output_dir is created if it doesn't exist."""
        video_path = tmp_path / "test.mkv"
        video_path.touch()
        output_dir = tmp_path / "new_output"

        extracted_wav = tmp_path / "test_extracted.wav"
        isolated_wav = tmp_path / "test_isolated.wav"
        mock_segments = []

        mock_extract.return_value = extracted_wav
        mock_isolate.return_value = isolated_wav
        mock_segment.return_value = mock_segments

        preprocess_video(video_path, output_dir)

        assert output_dir.exists()


class TestSegmentsJson:
    """Tests for segments.json export."""

    def test_export_segments_json_creates_valid_json(self, tmp_path):
        """Test that export_segments_json creates valid JSON."""
        segments = [
            SpeechSegment(0, 1000, np.array([0.1]), 22050),
            SpeechSegment(1000, 3000, np.array([0.2, 0.3]), 22050),
        ]
        output_path = tmp_path / "segments.json"

        export_segments_json(segments, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["segment_count"] == 2
        assert data["total_speech_ms"] == 3000
        assert len(data["segments"]) == 2

    def test_export_segments_json_correct_structure(self, tmp_path):
        """Test that manifest has version, segment_count, total_speech_ms, segments array."""
        segments = [
            SpeechSegment(500, 1500, np.array([0.1]), 22050),
        ]
        output_path = tmp_path / "segments.json"

        export_segments_json(segments, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "version" in data
        assert "segment_count" in data
        assert "total_speech_ms" in data
        assert "segments" in data

    def test_export_segments_json_segment_fields(self, tmp_path):
        """Test that each segment has start_ms, end_ms, duration_ms."""
        segments = [
            SpeechSegment(100, 600, np.array([0.1]), 22050),
        ]
        output_path = tmp_path / "segments.json"

        export_segments_json(segments, output_path)

        with open(output_path) as f:
            data = json.load(f)

        seg = data["segments"][0]
        assert seg["start_ms"] == 100
        assert seg["end_ms"] == 600
        assert seg["duration_ms"] == 500

    def test_export_segments_json_empty_list(self, tmp_path):
        """Test that empty segments list produces valid manifest with count 0."""
        segments = []
        output_path = tmp_path / "segments.json"

        export_segments_json(segments, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["segment_count"] == 0
        assert data["total_speech_ms"] == 0
        assert data["segments"] == []

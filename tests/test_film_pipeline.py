"""Tests for film pipeline orchestration module."""
import gc
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest


# Helper: Create a mock LanguageSegment
def _mock_segment(start_time, end_time, language, speaker=None):
    """Create a mock LanguageSegment for testing."""
    seg = MagicMock()
    seg.start_time = start_time
    seg.end_time = end_time
    seg.language = language
    seg.speaker = speaker
    seg.phonemes = []
    return seg


# Helper: Setup mock StageProgress context manager
def _setup_mock_progress(mock_progress_cls):
    """Setup mock for StageProgress context manager."""
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.advance = MagicMock()
    mock_progress_cls.return_value = mock_progress
    return mock_progress


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_calls_stages_in_order(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test that process_film calls all stages in correct order."""
    from tenepal.orchestration import process_film

    # Setup mock progress
    _setup_mock_progress(mock_progress_cls)

    # Create a fake video file
    video = tmp_path / "test.mkv"
    video.touch()

    # Create fake audio file for path resolution
    isolated_audio = tmp_path / "test_isolated.wav"
    isolated_audio.touch()

    # Mock returns
    mock_preprocess.return_value = []  # Empty segments list from preprocessing
    mock_segments = [
        _mock_segment(0.0, 1.0, "eng", "Speaker A"),
        _mock_segment(1.0, 2.0, "spa", "Speaker B"),
    ]
    mock_process.return_value = mock_segments

    # Call process_film
    srt_path = process_film(video, enable_diarization=True, whisper_model="base")

    # Assert stages called in order
    assert mock_preprocess.called
    assert mock_preprocess.call_args[0][0] == video
    assert mock_preprocess.call_args[1]["skip_isolation"] is False
    assert mock_preprocess.call_args[1]["demucs_segment"] == 300

    # Assert cleanup called after preprocessing
    cleanup_calls = mock_cleanup.call_args_list
    assert any("preprocessing" in str(call) for call in cleanup_calls)

    # Assert process_audio called
    assert mock_process.called
    assert mock_process.call_args[1]["enable_diarization"] is True
    assert mock_process.call_args[1]["whisper_model"] == "base"

    # Assert cleanup called after transcription
    assert any("transcription" in str(call) for call in cleanup_calls)

    # Assert write_srt called
    assert mock_srt.called
    assert mock_srt.call_args[0][0] == mock_segments

    # Assert return value is SRT path
    assert srt_path == video.with_suffix(".srt")


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_default_output_path(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test that process_film uses default output path when not specified."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "movie.mkv"
    video.touch()

    isolated_audio = tmp_path / "movie_isolated.wav"
    isolated_audio.touch()

    mock_preprocess.return_value = []
    mock_process.return_value = [_mock_segment(0.0, 1.0, "eng")]

    # Call without output_path
    srt_path = process_film(video)

    # Assert write_srt called with default path
    assert mock_srt.called
    expected_path = video.with_suffix(".srt")
    assert mock_srt.call_args[0][1] == expected_path
    assert srt_path == expected_path


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_custom_output_path(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test that process_film respects custom output path."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "movie.mkv"
    video.touch()

    isolated_audio = tmp_path / "movie_isolated.wav"
    isolated_audio.touch()

    custom_output = tmp_path / "output" / "custom.srt"

    mock_preprocess.return_value = []
    mock_process.return_value = [_mock_segment(0.0, 1.0, "eng")]

    # Call with custom output_path
    srt_path = process_film(video, output_path=custom_output)

    # Assert write_srt called with custom path
    assert mock_srt.called
    assert mock_srt.call_args[0][1] == custom_output
    assert srt_path == custom_output


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_skip_isolation_uses_extracted_audio(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test that skip_isolation uses extracted audio path."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "test.mkv"
    video.touch()

    extracted_audio = tmp_path / "test_extracted.wav"
    extracted_audio.touch()

    mock_preprocess.return_value = []
    mock_process.return_value = [_mock_segment(0.0, 1.0, "eng")]

    # Call with skip_isolation=True
    process_film(video, skip_isolation=True)

    # Assert process_audio called with extracted audio
    assert mock_process.called
    audio_path = mock_process.call_args[0][0]
    assert str(audio_path).endswith("_extracted.wav")


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_with_isolation_uses_isolated_audio(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test that skip_isolation=False uses isolated audio path."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "test.mkv"
    video.touch()

    isolated_audio = tmp_path / "test_isolated.wav"
    isolated_audio.touch()

    mock_preprocess.return_value = []
    mock_process.return_value = [_mock_segment(0.0, 1.0, "eng")]

    # Call with skip_isolation=False (default)
    process_film(video, skip_isolation=False)

    # Assert process_audio called with isolated audio
    assert mock_process.called
    audio_path = mock_process.call_args[0][0]
    assert str(audio_path).endswith("_isolated.wav")


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_file_not_found(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test that process_film raises FileNotFoundError for missing audio."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "test.mkv"
    video.touch()

    # Don't create audio files - audio path resolution will fail

    mock_preprocess.return_value = []

    # Should raise FileNotFoundError when audio file not found
    with pytest.raises(FileNotFoundError):
        process_film(video)


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_graceful_degradation_diarization(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test graceful degradation when diarization fails."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "test.mkv"
    video.touch()

    isolated_audio = tmp_path / "test_isolated.wav"
    isolated_audio.touch()

    mock_preprocess.return_value = []

    # First call fails with diarization error, second succeeds
    mock_process.side_effect = [
        RuntimeError("HUGGINGFACE_TOKEN not set"),
        [_mock_segment(0.0, 1.0, "eng")],
    ]

    # Should retry without diarization and succeed
    srt_path = process_film(video, enable_diarization=True)

    # Assert process_audio called twice
    assert mock_process.call_count == 2

    # First call with diarization
    assert mock_process.call_args_list[0][1]["enable_diarization"] is True

    # Second call without diarization
    assert mock_process.call_args_list[1][1]["enable_diarization"] is False

    # Pipeline completes successfully
    assert srt_path == video.with_suffix(".srt")


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_graceful_degradation_whisper(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test graceful degradation when Whisper fails after diarization fallback."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "test.mkv"
    video.touch()

    isolated_audio = tmp_path / "test_isolated.wav"
    isolated_audio.touch()

    mock_preprocess.return_value = []

    # Sequence: First call fails (diarization error) → retry without diarization fails (Whisper error) → retry without Whisper succeeds
    mock_process.side_effect = [
        RuntimeError("Diarization failed"),
        ImportError("faster-whisper not installed"),
        [_mock_segment(0.0, 1.0, "eng")],
    ]

    # Should retry without diarization, then without Whisper, and succeed
    srt_path = process_film(video, enable_diarization=True, whisper_model="base")

    # Assert process_audio called three times
    assert mock_process.call_count == 3

    # First call with diarization and Whisper
    assert mock_process.call_args_list[0][1]["enable_diarization"] is True
    assert mock_process.call_args_list[0][1]["whisper_model"] == "base"

    # Second call without diarization, with Whisper
    assert mock_process.call_args_list[1][1]["enable_diarization"] is False
    assert mock_process.call_args_list[1][1]["whisper_model"] == "base"

    # Third call without diarization or Whisper
    assert mock_process.call_args_list[2][1]["enable_diarization"] is False
    assert mock_process.call_args_list[2][1]["whisper_model"] is None

    # Pipeline completes successfully
    assert srt_path == video.with_suffix(".srt")


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_preprocessing_failure_is_fatal(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test that preprocessing failures are fatal (not degraded)."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "test.mkv"
    video.touch()

    # Preprocessing fails
    mock_preprocess.side_effect = RuntimeError("Demucs OOM")

    # Should propagate RuntimeError
    with pytest.raises(RuntimeError, match="Preprocessing stage failed"):
        process_film(video)


def test_cleanup_stage_calls_gc_and_torch():
    """Test cleanup_stage forces garbage collection and clears CUDA cache."""
    from tenepal.orchestration.lifecycle import cleanup_stage
    import sys

    # Create a mock torch module
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.empty_cache = MagicMock()

    with patch("tenepal.orchestration.lifecycle.gc.collect") as mock_gc:
        with patch.dict(sys.modules, {"torch": mock_torch}):
            cleanup_stage("test_stage")

            # Assert gc.collect called
            assert mock_gc.called

            # Assert CUDA cache cleared
            assert mock_torch.cuda.empty_cache.called


def test_cleanup_stage_with_backend_unload():
    """Test cleanup_stage calls backend unload() method."""
    from tenepal.orchestration.lifecycle import cleanup_stage

    # Create mock backend with unload method
    mock_backend = MagicMock()
    mock_backend.unload = MagicMock()
    # Ensure hasattr checks work correctly
    mock_backend.unload_backends = None  # No unload_backends attribute
    del mock_backend.unload_backends  # Remove it so hasattr returns False

    with patch("tenepal.orchestration.lifecycle.gc.collect"):
        with patch("tenepal.orchestration.lifecycle.unload_gpu_models"):
            cleanup_stage("test_stage", backend=mock_backend)

    # Assert backend.unload() called
    assert mock_backend.unload.called


def test_stage_progress_context_manager():
    """Test StageProgress context manager and advance method."""
    from tenepal.orchestration.progress import StageProgress

    with patch("tenepal.orchestration.progress.tqdm") as mock_tqdm_cls:
        mock_pbar = MagicMock()
        mock_tqdm_cls.return_value = mock_pbar

        # Use context manager with correct parameter name
        with StageProgress(total_stages=3) as progress:
            progress.advance("Stage 1")
            progress.advance("Stage 2")
            progress.advance("Stage 3")

        # Assert tqdm called with correct params
        assert mock_tqdm_cls.called
        assert mock_tqdm_cls.call_args[1]["total"] == 3

        # Assert advance called updates
        assert mock_pbar.update.call_count == 3
        assert mock_pbar.set_description.call_count == 3

        # Assert close called on exit
        assert mock_pbar.close.called


def test_unload_gpu_models_no_torch():
    """Test unload_gpu_models gracefully handles missing torch."""
    from tenepal.orchestration.lifecycle import unload_gpu_models

    with patch("tenepal.orchestration.lifecycle.gc.collect") as mock_gc:
        with patch.dict("sys.modules", {"torch": None}):
            # Should not raise exception
            unload_gpu_models()

            # gc.collect still called
            assert mock_gc.called


@patch("tenepal.orchestration.film_pipeline.StageProgress")
@patch("tenepal.orchestration.film_pipeline.cleanup_stage")
@patch("tenepal.orchestration.film_pipeline.write_srt")
@patch("tenepal.orchestration.film_pipeline.format_speaker_stats", return_value="Stats")
@patch("tenepal.orchestration.film_pipeline.print_language_segments")
@patch("tenepal.orchestration.film_pipeline.process_audio")
@patch("tenepal.orchestration.film_pipeline.preprocess_video")
def test_process_film_fallback_to_extracted_audio(
    mock_preprocess,
    mock_process,
    mock_print_lang,
    mock_stats,
    mock_srt,
    mock_cleanup,
    mock_progress_cls,
    tmp_path,
):
    """Test audio path fallback when isolated audio missing."""
    from tenepal.orchestration import process_film

    _setup_mock_progress(mock_progress_cls)

    video = tmp_path / "test.mkv"
    video.touch()

    # Create only extracted audio (isolated missing)
    extracted_audio = tmp_path / "test_extracted.wav"
    extracted_audio.touch()

    mock_preprocess.return_value = []
    mock_process.return_value = [_mock_segment(0.0, 1.0, "eng")]

    # Call without skip_isolation (expects isolated, should fall back)
    srt_path = process_film(video, skip_isolation=False)

    # Assert process_audio called with fallback extracted audio
    assert mock_process.called
    audio_path = mock_process.call_args[0][0]
    assert str(audio_path).endswith("_extracted.wav")

    # Pipeline completes successfully
    assert srt_path == video.with_suffix(".srt")

"""Tests for WhisperBackend ASR implementation."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from dataclasses import dataclass

# Mock faster_whisper module before importing WhisperBackend
mock_faster_whisper = MagicMock()
sys.modules["faster_whisper"] = mock_faster_whisper

from tenepal.phoneme.backend import ASRBackend, PhonemeSegment
from tenepal.phoneme.whisper_backend import WhisperBackend


def test_whisper_backend_inherits_asr_backend():
    """Verify WhisperBackend extends ASRBackend."""
    assert issubclass(WhisperBackend, ASRBackend)


def test_whisper_backend_name():
    """Verify backend name is 'whisper'."""
    assert WhisperBackend.name == "whisper"


def test_whisper_backend_model_size_validation():
    """Verify invalid model size raises ValueError."""
    with pytest.raises(ValueError, match="Invalid model_size"):
        WhisperBackend(model_size="invalid")


def test_whisper_backend_lazy_load():
    """Verify model is not loaded until _get_model() is called."""
    mock_whisper = MagicMock()
    mock_faster_whisper.WhisperModel = mock_whisper

    backend = WhisperBackend(model_size="base")
    # Model should not be loaded yet (just initialized, not calling _get_model)
    assert backend._model is None

    # Now trigger lazy load
    backend._get_model()
    mock_whisper.assert_called_once()


def test_whisper_backend_recognize_forces_language():
    """CRITICAL: Verify recognize() forces language parameter, never auto-detects."""
    # Mock WhisperModel and its transcribe method
    mock_model = MagicMock()
    mock_segment = Mock()
    mock_segment.start = 0.0
    mock_segment.end = 1.5
    mock_segment.text = "hola mundo"
    mock_model.transcribe.return_value = ([mock_segment], {})

    # Create a fake audio file for the test
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        mock_faster_whisper.WhisperModel.return_value = mock_model
        backend = WhisperBackend(model_size="base", device="cpu")

        # Call recognize with explicit language
        result = backend.recognize(tmp_path, lang="es")

        # Verify transcribe was called with language="es"
        mock_model.transcribe.assert_called_once()
        call_args = mock_model.transcribe.call_args
        assert "language" in call_args.kwargs
        assert call_args.kwargs["language"] == "es"

        # Verify vad_filter=False (Tenepal handles VAD)
        assert call_args.kwargs["vad_filter"] is False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_whisper_backend_recognize_returns_phoneme_segments():
    """Verify recognize() returns list of PhonemeSegment with transcribed text."""
    # Mock WhisperModel and segments
    mock_model = MagicMock()
    mock_segment1 = Mock()
    mock_segment1.start = 0.0
    mock_segment1.end = 1.5
    mock_segment1.text = "hello world"

    mock_segment2 = Mock()
    mock_segment2.start = 1.5
    mock_segment2.end = 3.2
    mock_segment2.text = "testing whisper"

    mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], {})

    # Create a fake audio file for the test
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        mock_faster_whisper.WhisperModel.return_value = mock_model
        backend = WhisperBackend(model_size="base", device="cpu")
        result = backend.recognize(tmp_path, lang="en")

        assert len(result) == 2
        assert all(isinstance(seg, PhonemeSegment) for seg in result)

        # Verify first segment
        assert result[0].phoneme == "hello world"
        assert result[0].start_time == 0.0
        assert result[0].duration == pytest.approx(1.5)

        # Verify second segment
        assert result[1].phoneme == "testing whisper"
        assert result[1].start_time == 1.5
        assert result[1].duration == pytest.approx(1.7)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_whisper_backend_is_available_true():
    """Verify is_available() returns True when faster_whisper is installed."""
    # Module is already mocked at top of file
    assert WhisperBackend.is_available() is True


def test_whisper_backend_is_available_false():
    """Verify is_available() returns False when faster_whisper is not installed."""
    # Need to patch the import itself
    import sys
    original_modules = sys.modules.copy()
    try:
        # Remove faster_whisper from sys.modules if it exists
        sys.modules.pop("faster_whisper", None)
        # Make import fail
        with patch.dict("sys.modules", {"faster_whisper": None}):
            assert WhisperBackend.is_available() is False
    finally:
        # Restore original state
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_whisper_backend_device_resolution_auto_cuda():
    """Verify device='auto' resolves to 'cuda' when available."""
    # torch is imported inside _resolve_device method
    import torch as real_torch
    with patch.object(real_torch.cuda, "is_available", return_value=True):
        backend = WhisperBackend(device="auto")
        assert backend.device == "cuda"


def test_whisper_backend_device_resolution_auto_cpu():
    """Verify device='auto' resolves to 'cpu' when cuda unavailable."""
    # torch is imported inside _resolve_device method
    import torch as real_torch
    with patch.object(real_torch.cuda, "is_available", return_value=False):
        backend = WhisperBackend(device="auto")
        assert backend.device == "cpu"


def test_whisper_backend_device_resolution_explicit():
    """Verify explicit device values are preserved."""
    backend = WhisperBackend(device="cpu")
    assert backend.device == "cpu"


def test_whisper_backend_compute_type_resolution_auto_cuda():
    """Verify compute_type='auto' resolves to 'float16' on cuda."""
    import torch as real_torch
    with patch.object(real_torch.cuda, "is_available", return_value=True):
        backend = WhisperBackend(device="cuda", compute_type="auto")
        assert backend.compute_type == "float16"


def test_whisper_backend_compute_type_resolution_auto_cpu():
    """Verify compute_type='auto' resolves to 'int8' on cpu."""
    backend = WhisperBackend(device="cpu", compute_type="auto")
    assert backend.compute_type == "int8"


def test_whisper_backend_registered():
    """Verify WhisperBackend is registered in backend registry."""
    from tenepal.phoneme.backend import list_backends

    # Import whisper_backend to trigger registration
    import tenepal.phoneme.whisper_backend

    backends = list_backends()
    assert "whisper" in backends


def test_whisper_backend_unload():
    """Verify unload() clears model and GPU memory."""
    mock_model = MagicMock()
    mock_faster_whisper.WhisperModel.return_value = mock_model

    import torch as real_torch
    with patch.object(real_torch.cuda, "empty_cache") as mock_empty_cache:
        backend = WhisperBackend(device="cuda")
        backend._get_model()  # Load model

        # Unload
        backend.unload()

        # Verify model is cleared
        assert backend._model is None

        # Verify GPU cache cleared
        mock_empty_cache.assert_called_once()


def test_transcribe_auto_file_level_language():
    """Verify transcribe_auto() uses file-level language for all segments.

    Per-segment detect_language was removed because it returns wrong codes
    for short clips. All segments now get info.language from file-level detection.
    """
    import tempfile
    import numpy as np
    import soundfile as sf

    # Create a temp audio file (3 seconds @ 16kHz)
    duration = 3.0
    sr = 16000
    samples = np.zeros(int(duration * sr), dtype=np.float32)
    tmp = tempfile.mkstemp(suffix=".wav")[1]
    sf.write(tmp, samples, sr)

    try:
        # Mock WhisperModel
        mock_model = MagicMock()

        # Mock 3 segments from transcribe()
        mock_segs = []
        for i in range(3):
            seg = Mock()
            seg.text = f"segment {i}"
            seg.start = float(i)
            seg.end = float(i + 0.9)
            seg.avg_logprob = -0.2
            mock_segs.append(seg)

        mock_info = Mock()
        mock_info.language = "es"  # File-level detection

        mock_model.transcribe.return_value = (mock_segs, mock_info)

        mock_faster_whisper.WhisperModel.return_value = mock_model
        backend = WhisperBackend(model_size="base", device="cpu")

        result = backend.transcribe_auto(Path(tmp))

        # All segments get file-level language
        assert len(result) == 3
        assert all(r.language == "es" for r in result)

        # No per-segment detect_language calls
        mock_model.detect_language.assert_not_called()

    finally:
        Path(tmp).unlink(missing_ok=True)


def test_transcribe_auto_no_detect_language_calls():
    """Verify detect_language() is NOT called (file-level detection only)."""
    import tempfile
    import numpy as np
    import soundfile as sf

    # Create a temp audio file
    tmp = tempfile.mkstemp(suffix=".wav")[1]
    sf.write(tmp, np.zeros(48000, dtype=np.float32), 16000)

    try:
        mock_model = MagicMock()

        # Mock 3 segments
        mock_segs = [Mock(text=f"s{i}", start=float(i), end=float(i+0.5), avg_logprob=-0.2)
                     for i in range(3)]
        mock_info = Mock(language="de")
        mock_model.transcribe.return_value = (mock_segs, mock_info)

        mock_faster_whisper.WhisperModel.return_value = mock_model
        backend = WhisperBackend(model_size="base", device="cpu")

        result = backend.transcribe_auto(Path(tmp))

        # All segments get file-level language (de)
        assert all(r.language == "de" for r in result)
        # No per-segment detection
        mock_model.detect_language.assert_not_called()

    finally:
        Path(tmp).unlink(missing_ok=True)

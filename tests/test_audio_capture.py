"""Tests for live audio capture from PulseAudio/PipeWire."""

import subprocess
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from tenepal.audio.capture import AudioCapture
from tenepal.audio.loader import AudioData


class TestAudioCaptureInit:
    """Tests for AudioCapture initialization and detection."""

    def test_detects_pulseaudio(self):
        """Test that AudioCapture detects PulseAudio when available."""
        with patch("shutil.which") as mock_which:
            # Simulate pactl and parec available
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                capture = AudioCapture()
                assert capture._audio_system == "pulseaudio"

    def test_detects_pipewire(self):
        """Test that AudioCapture detects PipeWire when PulseAudio not available."""
        with patch("shutil.which") as mock_which:
            # Simulate pw-cli and pw-record available, but not pactl/parec
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pw-cli", "pw-record"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="123"):
                capture = AudioCapture()
                assert capture._audio_system == "pipewire"

    def test_raises_when_no_audio_system(self):
        """Test that AudioCapture raises RuntimeError when neither system is available."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Neither PulseAudio nor PipeWire"):
                AudioCapture()

    def test_chunk_size_calculation(self):
        """Test that bytes_per_chunk is calculated correctly."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                # Test with default parameters
                capture = AudioCapture(chunk_duration=3.0, sample_rate=16000)
                # 16000 samples/sec * 3 sec * 2 bytes/sample = 96000 bytes
                assert capture.bytes_per_chunk == 96000

                # Test with custom parameters
                capture = AudioCapture(chunk_duration=5.0, sample_rate=44100)
                # 44100 samples/sec * 5 sec * 2 bytes/sample = 441000 bytes
                assert capture.bytes_per_chunk == 441000


class TestAudioCaptureMonitorDetection:
    """Tests for monitor source detection."""

    def test_finds_pulseaudio_monitor(self):
        """Test that PulseAudio monitor source is found correctly."""
        mock_output = """1	alsa_output.pci-0000_00_1f.3.analog-stereo.monitor	module-alsa-card.c	s16le 2ch 44100Hz	SUSPENDED
2	alsa_input.pci-0000_00_1f.3.analog-stereo	module-alsa-card.c	s16le 2ch 44100Hz	SUSPENDED"""

        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(stdout=mock_output, returncode=0)

                capture = AudioCapture()
                assert "monitor" in capture._monitor_source.lower()

    def test_raises_when_no_pulseaudio_monitor(self):
        """Test that RuntimeError is raised when no PulseAudio monitor found."""
        mock_output = """1	alsa_input.pci-0000_00_1f.3.analog-stereo	module-alsa-card.c	s16le 2ch 44100Hz	SUSPENDED"""

        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(stdout=mock_output, returncode=0)

                with pytest.raises(RuntimeError, match="No PulseAudio monitor source"):
                    AudioCapture()

    def test_finds_pipewire_monitor(self):
        """Test that PipeWire monitor node is found correctly."""
        mock_output = """	id 45, type PipeWire:Interface:Node/3
		object.serial = "45"
		node.name = "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
		node.description = "Monitor of Built-in Audio Analog Stereo"
	id 46, type PipeWire:Interface:Node/3
		object.serial = "46"
		node.name = "alsa_input.pci-0000_00_1f.3.analog-stereo"
		node.description = "Built-in Audio Analog Stereo"""

        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pw-cli", "pw-record"] else None
            mock_which.side_effect = which_side_effect

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(stdout=mock_output, returncode=0)

                capture = AudioCapture()
                # Should find node ID 45 (the monitor)
                assert capture._monitor_source == "45"


class TestAudioCaptureLifecycle:
    """Tests for start/stop lifecycle."""

    def test_start_creates_process(self):
        """Test that start() creates a subprocess."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                with patch("subprocess.Popen") as mock_popen:
                    mock_process = Mock()
                    mock_popen.return_value = mock_process

                    capture = AudioCapture()
                    capture.start()

                    assert capture._process == mock_process
                    mock_popen.assert_called_once()

    def test_raises_when_starting_twice(self):
        """Test that start() raises when already started."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                with patch("subprocess.Popen"):
                    capture = AudioCapture()
                    capture.start()

                    with pytest.raises(RuntimeError, match="already started"):
                        capture.start()

    def test_stop_terminates_process(self):
        """Test that stop() terminates the subprocess."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                with patch("subprocess.Popen") as mock_popen:
                    mock_process = Mock()
                    mock_popen.return_value = mock_process

                    capture = AudioCapture()
                    capture.start()
                    capture.stop()

                    mock_process.terminate.assert_called_once()
                    mock_process.wait.assert_called()
                    assert capture._process is None

    def test_context_manager_protocol(self):
        """Test that AudioCapture works as a context manager."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                with patch("subprocess.Popen") as mock_popen:
                    mock_process = Mock()
                    mock_popen.return_value = mock_process

                    with AudioCapture() as capture:
                        assert capture._process == mock_process

                    # Should have stopped after context exit
                    mock_process.terminate.assert_called_once()


class TestAudioCaptureChunks:
    """Tests for audio chunk generation."""

    def test_chunks_yields_audio_data(self):
        """Test that chunks() yields AudioData objects."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                with patch("subprocess.Popen") as mock_popen:
                    # Create mock audio data: 1 second of silence at 16kHz
                    # 16000 samples * 2 bytes = 32000 bytes
                    mock_audio_bytes = bytes(32000)

                    mock_process = Mock()
                    mock_process.stdout = Mock()
                    mock_process.stdout.read = Mock(side_effect=[mock_audio_bytes, b""])
                    mock_popen.return_value = mock_process

                    capture = AudioCapture(chunk_duration=1.0, sample_rate=16000)
                    capture.start()

                    chunks = list(capture.chunks())

                    assert len(chunks) == 1
                    assert isinstance(chunks[0], AudioData)
                    assert chunks[0].sample_rate == 16000
                    assert len(chunks[0].samples) == 16000

    def test_chunks_raises_when_not_started(self):
        """Test that chunks() raises when capture not started."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                capture = AudioCapture()

                with pytest.raises(RuntimeError, match="not started"):
                    next(capture.chunks())

    def test_chunks_preprocesses_audio(self):
        """Test that chunks() returns preprocessed (normalized) audio."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/" + cmd if cmd in ["pactl", "parec"] else None
            mock_which.side_effect = which_side_effect

            with patch.object(AudioCapture, "_find_monitor_source", return_value="test.monitor"):
                with patch("subprocess.Popen") as mock_popen:
                    # Create mock audio with max amplitude (will need normalization)
                    samples_int16 = np.array([32767, -32768, 16384, -16384], dtype=np.int16)
                    mock_audio_bytes = samples_int16.tobytes()

                    mock_process = Mock()
                    mock_process.stdout = Mock()
                    mock_process.stdout.read = Mock(side_effect=[mock_audio_bytes, b""])
                    mock_popen.return_value = mock_process

                    capture = AudioCapture(chunk_duration=0.001, sample_rate=16000)
                    capture.start()

                    chunks = list(capture.chunks())

                    # Should be normalized to peak 0.9
                    assert len(chunks) == 1
                    peak = np.abs(chunks[0].samples).max()
                    assert abs(peak - 0.9) < 0.01  # Allow small floating point error

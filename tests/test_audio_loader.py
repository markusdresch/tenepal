"""Tests for audio loading and preprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tenepal.audio import AudioData, load_audio, preprocess_audio, save_wav


def generate_sine_wave(frequency: float = 440.0, duration: float = 1.0,
                       sample_rate: int = 44100, num_channels: int = 1) -> np.ndarray:
    """Generate a sine wave for testing.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        num_channels: Number of channels (1=mono, 2=stereo)

    Returns:
        Audio samples as float32 array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * frequency * t)

    if num_channels == 2:
        # Create stereo by stacking mono signal
        samples = np.stack([samples, samples], axis=1)

    return samples


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_load_wav_mono(self, tmp_path):
        """Test loading a mono WAV file."""
        # Create test WAV file
        samples = generate_sine_wave(440.0, 1.0, 44100, num_channels=1)
        wav_path = tmp_path / "test_mono.wav"
        sf.write(str(wav_path), samples, 44100)

        # Load audio
        audio = load_audio(wav_path)

        # Verify
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 44100
        assert audio.source_format == "wav"
        assert abs(audio.duration - 1.0) < 0.01  # Allow small tolerance
        assert audio.samples.ndim == 1  # Mono
        assert audio.samples.dtype == np.float32

    def test_load_wav_stereo(self, tmp_path):
        """Test loading a stereo WAV file."""
        # Create test WAV file
        samples = generate_sine_wave(440.0, 1.0, 44100, num_channels=2)
        wav_path = tmp_path / "test_stereo.wav"
        sf.write(str(wav_path), samples, 44100)

        # Load audio
        audio = load_audio(wav_path)

        # Verify
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 44100
        assert audio.source_format == "wav"
        assert audio.samples.ndim == 2  # Stereo
        assert audio.samples.shape[1] == 2  # Two channels

    def test_load_missing_file(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            load_audio("nonexistent_file.wav")

    def test_load_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported formats."""
        # Create a file with unsupported extension
        unsupported_path = tmp_path / "test.ogg"
        unsupported_path.touch()

        with pytest.raises(ValueError, match="Unsupported audio format"):
            load_audio(unsupported_path)


class TestPreprocessAudio:
    """Tests for preprocess_audio function."""

    def test_stereo_to_mono_conversion(self):
        """Test that stereo audio is converted to mono."""
        # Create stereo audio
        stereo_samples = generate_sine_wave(440.0, 0.5, 44100, num_channels=2)
        audio = AudioData(
            samples=stereo_samples,
            sample_rate=44100,
            duration=0.5,
            source_format="wav"
        )

        # Preprocess
        processed = preprocess_audio(audio, target_sr=44100)

        # Verify mono
        assert processed.samples.ndim == 1
        assert processed.sample_rate == 44100

    def test_resampling(self):
        """Test that audio is resampled to target rate."""
        # Create audio at 44100 Hz
        samples = generate_sine_wave(440.0, 1.0, 44100, num_channels=1)
        audio = AudioData(
            samples=samples,
            sample_rate=44100,
            duration=1.0,
            source_format="wav"
        )

        # Resample to 16000 Hz
        processed = preprocess_audio(audio, target_sr=16000)

        # Verify
        assert processed.sample_rate == 16000
        # New number of samples should be approximately 16000
        assert abs(len(processed.samples) - 16000) < 100
        assert processed.duration < 1.01  # Duration preserved with small tolerance

    def test_normalization(self):
        """Test that audio is normalized to peak amplitude of 0.9."""
        # Create audio with known peak
        samples = np.array([0.1, 0.5, -0.5, 0.3], dtype=np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=16000,
            duration=0.00025,
            source_format="wav"
        )

        # Preprocess
        processed = preprocess_audio(audio, target_sr=16000)

        # Verify peak is 0.9
        peak = np.abs(processed.samples).max()
        assert abs(peak - 0.9) < 0.01

    def test_zero_audio_normalization(self):
        """Test that silent audio doesn't cause division by zero."""
        # Create silent audio
        samples = np.zeros(1000, dtype=np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=16000,
            duration=0.0625,
            source_format="wav"
        )

        # Preprocess - should not raise error
        processed = preprocess_audio(audio, target_sr=16000)

        # Verify still silent
        assert processed.samples.max() == 0.0
        assert processed.samples.min() == 0.0


class TestSaveWav:
    """Tests for save_wav function."""

    def test_save_and_reload(self, tmp_path):
        """Test that saved WAV can be reloaded successfully."""
        # Create audio
        samples = generate_sine_wave(440.0, 0.5, 16000, num_channels=1)
        audio = AudioData(
            samples=samples,
            sample_rate=16000,
            duration=0.5,
            source_format="wav"
        )

        # Save
        wav_path = tmp_path / "output.wav"
        result_path = save_wav(audio, wav_path)

        # Verify file was created
        assert result_path.exists()
        assert result_path == wav_path

        # Reload and verify
        reloaded = load_audio(wav_path)
        assert reloaded.sample_rate == 16000
        assert abs(reloaded.duration - 0.5) < 0.01
        # Samples should be very similar (WAV encoding introduces small errors)
        assert np.allclose(reloaded.samples, audio.samples, atol=1e-4)

    def test_save_creates_parent_directory(self, tmp_path):
        """Test that save_wav creates parent directories if needed."""
        # Create nested path that doesn't exist
        wav_path = tmp_path / "subdir1" / "subdir2" / "output.wav"

        # Create audio
        samples = generate_sine_wave(440.0, 0.1, 16000, num_channels=1)
        audio = AudioData(
            samples=samples,
            sample_rate=16000,
            duration=0.1,
            source_format="wav"
        )

        # Save - should create directories
        save_wav(audio, wav_path)

        # Verify file exists
        assert wav_path.exists()


class TestRoundTrip:
    """Integration tests for full pipeline."""

    def test_load_preprocess_save_reload(self, tmp_path):
        """Test complete round-trip: load → preprocess → save → reload."""
        # Create original WAV at 44100 Hz stereo
        original_samples = generate_sine_wave(440.0, 1.0, 44100, num_channels=2)
        original_path = tmp_path / "original.wav"
        sf.write(str(original_path), original_samples, 44100)

        # Load
        audio = load_audio(original_path)
        assert audio.sample_rate == 44100
        assert audio.samples.ndim == 2  # Stereo

        # Preprocess to 16kHz mono
        processed = preprocess_audio(audio, target_sr=16000)
        assert processed.sample_rate == 16000
        assert processed.samples.ndim == 1  # Mono

        # Save
        output_path = tmp_path / "processed.wav"
        save_wav(processed, output_path)

        # Reload
        reloaded = load_audio(output_path)
        assert reloaded.sample_rate == 16000
        assert reloaded.samples.ndim == 1  # Mono
        assert np.allclose(reloaded.samples, processed.samples, atol=1e-4)

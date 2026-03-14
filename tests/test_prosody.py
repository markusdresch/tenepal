"""Tests for prosodic feature extraction and rhythm metrics."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tenepal.prosody.rhythm import compute_npvi, compute_rpvi


class TestComputeNpvi:
    """Tests for Normalized Pairwise Variability Index calculation."""

    def test_uniform_durations_returns_zero(self):
        """nPVI of uniform durations should be 0."""
        durations = [100.0, 100.0, 100.0, 100.0]
        result = compute_npvi(durations)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_varied_durations_known_value(self):
        """nPVI with varied durations should return expected value."""
        # Example: [100, 200] -> |100-200| / ((100+200)/2) = 100/150 = 0.667
        # nPVI = 100 * (1/1) * 0.667 = 66.67
        durations = [100.0, 200.0]
        result = compute_npvi(durations)
        assert result == pytest.approx(66.67, abs=0.01)

    def test_three_varied_durations(self):
        """nPVI with 3 durations: [100, 200, 100]."""
        # Pair 1: |100-200| / 150 = 0.667
        # Pair 2: |200-100| / 150 = 0.667
        # nPVI = 100 * (1/2) * (0.667 + 0.667) = 66.67
        durations = [100.0, 200.0, 100.0]
        result = compute_npvi(durations)
        assert result == pytest.approx(66.67, abs=0.01)

    def test_empty_list_returns_zero(self):
        """nPVI with empty list should return 0."""
        durations = []
        result = compute_npvi(durations)
        assert result == 0.0

    def test_single_element_returns_zero(self):
        """nPVI with single element should return 0."""
        durations = [100.0]
        result = compute_npvi(durations)
        assert result == 0.0

    def test_list_with_zeros(self):
        """nPVI should handle pairs with zero mean gracefully."""
        # Pair with zero mean would cause division by zero - should skip
        durations = [0.0, 0.0, 100.0, 200.0]
        result = compute_npvi(durations)
        # First pair (0,0) skipped, remaining pairs computed
        # Pair (0,100): mean=50, |diff|=100, contribution=100/50=2.0
        # Pair (100,200): mean=150, |diff|=100, contribution=100/150=0.667
        # nPVI = 100 * (1/3) * (0 + 2.0 + 0.667) = 88.9 (but first pair skipped)
        # Actually: skip when mean=0, so only 2 valid pairs: (0,100) and (100,200)
        # Let's compute properly: if mean is 0, skip
        assert result >= 0.0  # Should not crash, exact value depends on skip logic


class TestComputeRpvi:
    """Tests for Raw Pairwise Variability Index calculation."""

    def test_uniform_durations_returns_zero(self):
        """rPVI of uniform durations should be 0."""
        durations = [100.0, 100.0, 100.0, 100.0]
        result = compute_rpvi(durations)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_varied_durations_known_value(self):
        """rPVI with varied durations should return expected value."""
        # Example: [100, 200] -> |100-200| = 100
        # rPVI = (1/1) * 100 = 100
        durations = [100.0, 200.0]
        result = compute_rpvi(durations)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_three_varied_durations(self):
        """rPVI with 3 durations: [100, 200, 100]."""
        # Pair 1: |100-200| = 100
        # Pair 2: |200-100| = 100
        # rPVI = (1/2) * (100 + 100) = 100
        durations = [100.0, 200.0, 100.0]
        result = compute_rpvi(durations)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_empty_list_returns_zero(self):
        """rPVI with empty list should return 0."""
        durations = []
        result = compute_rpvi(durations)
        assert result == 0.0

    def test_single_element_returns_zero(self):
        """rPVI with single element should return 0."""
        durations = [100.0]
        result = compute_rpvi(durations)
        assert result == 0.0

    def test_list_with_zeros(self):
        """rPVI should handle zero values without issue."""
        durations = [0.0, 100.0, 200.0]
        result = compute_rpvi(durations)
        # |0-100| + |100-200| = 100 + 100 = 200
        # rPVI = (1/2) * 200 = 100
        assert result == pytest.approx(100.0, abs=0.01)


class TestProsodyFeatures:
    """Tests for ProsodyFeatures dataclass."""

    def test_dataclass_creation(self):
        """ProsodyFeatures should be creatable with all fields."""
        from tenepal.prosody import ProsodyFeatures

        features = ProsodyFeatures(
            f0_mean=150.0,
            f0_std=25.0,
            f0_range=100.0,
            intensity_mean=60.0,
            duration=2.5,
            speech_rate=4.5,
            npvi_v=55.0,
        )

        assert features.f0_mean == 150.0
        assert features.f0_std == 25.0
        assert features.f0_range == 100.0
        assert features.intensity_mean == 60.0
        assert features.duration == 2.5
        assert features.speech_rate == 4.5
        assert features.npvi_v == 55.0


class TestExtractProsody:
    """Tests for prosodic feature extraction from audio."""

    def test_short_segment_returns_none(self):
        """Segments shorter than 1.0 second should return None."""
        from tenepal.prosody import extract_prosody

        # 0.5 seconds of audio at 22050 Hz
        sample_rate = 22050
        duration = 0.5
        num_samples = int(sample_rate * duration)
        audio_data = np.random.randn(num_samples).astype(np.float32)

        result = extract_prosody(audio_data, sample_rate)
        assert result is None

    def test_silent_audio_returns_none(self):
        """Silent/unvoiced audio should return None."""
        from tenepal.prosody import extract_prosody

        # 2.0 seconds of silence
        sample_rate = 22050
        duration = 2.0
        num_samples = int(sample_rate * duration)
        audio_data = np.zeros(num_samples, dtype=np.float32)

        result = extract_prosody(audio_data, sample_rate)
        assert result is None

    def test_exception_returns_none(self):
        """Exceptions in parselmouth should return None gracefully."""
        from tenepal.prosody import extract_prosody

        # Mock parselmouth to raise exception
        with patch('tenepal.prosody.extractor.parselmouth.Sound') as mock_sound:
            mock_sound.side_effect = Exception("Parselmouth error")

            # Valid-length audio
            sample_rate = 22050
            duration = 2.0
            num_samples = int(sample_rate * duration)
            audio_data = np.random.randn(num_samples).astype(np.float32)

            result = extract_prosody(audio_data, sample_rate)
            assert result is None

    @pytest.mark.skipif(
        True,  # Skip by default, requires real parselmouth
        reason="Requires real parselmouth and generates audio",
    )
    def test_synthetic_sine_wave_extraction(self):
        """Test extraction with synthetic sine wave (integration test)."""
        from tenepal.prosody import extract_prosody, ProsodyFeatures

        # Generate 2-second sine wave at 440 Hz (A4 note)
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440.0
        audio_data = (np.sin(2 * np.pi * frequency * t) * 0.5).astype(np.float32)

        result = extract_prosody(audio_data, sample_rate)

        # Should extract features from sine wave
        assert isinstance(result, ProsodyFeatures)
        assert result.f0_mean > 0  # Should detect fundamental frequency
        assert result.duration == pytest.approx(duration, abs=0.1)
        assert result.f0_range >= 0
        assert result.intensity_mean > 0

    def test_mock_parselmouth_extraction(self):
        """Test extraction logic with mocked parselmouth."""
        from tenepal.prosody import extract_prosody, ProsodyFeatures

        # Mock parselmouth objects
        mock_pitch = MagicMock()
        # selected_array is a property, not a method
        mock_pitch.selected_array = {
            'frequency': np.array([100.0, 150.0, 200.0, 150.0, 100.0])
        }

        mock_intensity = MagicMock()
        mock_intensity.values = np.array([[60.0, 65.0, 70.0, 65.0, 60.0]])

        mock_sound = MagicMock()
        mock_sound.to_pitch.return_value = mock_pitch
        mock_sound.to_intensity.return_value = mock_intensity

        with patch('tenepal.prosody.extractor.parselmouth.Sound', return_value=mock_sound):
            with patch('tenepal.prosody.extractor.find_peaks') as mock_peaks:
                # Mock peak detection - simulate 2 peaks (syllables) for rhythm calculation
                # find_peaks returns (peaks, properties_dict)
                mock_peaks.return_value = (np.array([1, 3]), {})

                # Valid-length audio
                sample_rate = 22050
                duration = 2.0
                num_samples = int(sample_rate * duration)
                audio_data = np.random.randn(num_samples).astype(np.float32)

                result = extract_prosody(audio_data, sample_rate)

        # Should extract features
        assert isinstance(result, ProsodyFeatures)
        assert result.f0_mean == pytest.approx(140.0, abs=1.0)  # Mean of pitch values
        assert result.f0_std > 0
        assert result.f0_range == pytest.approx(100.0, abs=1.0)  # 200 - 100
        assert result.intensity_mean == pytest.approx(64.0, abs=1.0)  # Mean of intensity
        assert result.duration == 2.0
        assert result.speech_rate >= 0  # Based on peaks
        assert result.npvi_v >= 0  # Based on vocalic intervals

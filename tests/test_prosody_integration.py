"""Integration tests for prosody + fusion in language identifier."""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from tenepal.language import identify_language, LanguageSegment
from tenepal.phoneme import PhonemeSegment
from tenepal.prosody import ProsodyFeatures


class TestProsodyIntegration(unittest.TestCase):
    """Test prosody integration into identify_language()."""

    def setUp(self):
        """Create phoneme fixtures for testing."""
        # Create a list of 10 phonemes for basic tests
        self.basic_phonemes = [
            PhonemeSegment(phoneme="t", start_time=i * 0.1, duration=0.1)
            for i in range(10)
        ]

        # Create a long list (100+ phonemes) for consolidation tests
        self.long_phonemes = [
            PhonemeSegment(phoneme="t", start_time=i * 0.1, duration=0.1)
            for i in range(150)
        ]

    def test_identify_language_without_audio_data_unchanged(self):
        """Call identify_language without audio_data, verify no crash."""
        # Should work exactly as before (backward compat)
        segments = identify_language(self.basic_phonemes)
        self.assertIsInstance(segments, list)
        for seg in segments:
            self.assertIsInstance(seg, LanguageSegment)

    def test_identify_language_with_audio_data_none_unchanged(self):
        """Explicitly pass audio_data=None, verify same result."""
        segments = identify_language(self.basic_phonemes, audio_data=None)
        self.assertIsInstance(segments, list)
        for seg in segments:
            self.assertIsInstance(seg, LanguageSegment)

    def test_identify_language_accepts_audio_data(self):
        """Pass valid audio_data tuple, verify no crash."""
        # 1 second of silence at 22050 Hz
        audio_data = (np.zeros(22050, dtype=np.float32), 22050)
        segments = identify_language(self.basic_phonemes, audio_data=audio_data)
        self.assertIsInstance(segments, list)
        for seg in segments:
            self.assertIsInstance(seg, LanguageSegment)

    def test_identify_language_prosody_fallback_on_short_audio(self):
        """Pass audio_data with < 1 second, verify fallback to phoneme-only."""
        # 100 samples = ~4.5ms at 22050 Hz (way too short)
        audio_data = (np.zeros(100, dtype=np.float32), 22050)

        # Mock extract_prosody to verify it returns None for short audio
        with patch("tenepal.language.identifier.extract_prosody") as mock_extract:
            mock_extract.return_value = None
            segments = identify_language(self.basic_phonemes, audio_data=audio_data)

        # Should succeed (fallback to phoneme-only)
        self.assertIsInstance(segments, list)
        mock_extract.assert_called_once()

    def test_identify_language_prosody_fallback_on_extraction_failure(self):
        """Mock extract_prosody to raise exception, verify graceful fallback."""
        audio_data = (np.zeros(22050, dtype=np.float32), 22050)

        with patch("tenepal.language.identifier.extract_prosody") as mock_extract:
            mock_extract.side_effect = RuntimeError("Parselmouth failed")
            segments = identify_language(self.basic_phonemes, audio_data=audio_data)

        # Should succeed (fallback to phoneme-only)
        self.assertIsInstance(segments, list)
        for seg in segments:
            self.assertIsInstance(seg, LanguageSegment)

    def test_consolidate_uses_fused_scores_when_prosody_available(self):
        """Mock prosody extraction and fusion, verify they are called during consolidation."""
        audio_data = (np.zeros(22050, dtype=np.float32), 22050)

        # Create mock prosody features
        mock_features = ProsodyFeatures(
            f0_mean=200.0,
            f0_std=30.0,
            f0_range=150.0,
            intensity_mean=60.0,
            duration=1.0,
            speech_rate=4.5,
            npvi_v=45.0,
        )

        # Mock prosody extraction to return features
        with patch("tenepal.language.identifier.extract_prosody") as mock_extract, \
             patch("tenepal.language.identifier.score_prosody_profiles") as mock_score, \
             patch("tenepal.language.identifier.fuse_scores") as mock_fuse:

            mock_extract.return_value = mock_features
            mock_score.return_value = {"spa": 0.8, "eng": 0.5, "deu": 0.3}
            # fuse_scores should return fused scores
            mock_fuse.return_value = {"spa": 0.7, "eng": 0.4, "deu": 0.2}

            segments = identify_language(self.long_phonemes, audio_data=audio_data)

        # Verify calls
        mock_extract.assert_called_once()
        # score_prosody_profiles and fuse_scores should be called during consolidation
        # (only if len(phonemes) >= 100)
        mock_score.assert_called_once_with(mock_features)
        mock_fuse.assert_called_once()

        self.assertIsInstance(segments, list)

    def test_fuse_scores_not_called_without_audio(self):
        """Verify fuse_scores is NOT called when audio_data is None."""
        with patch("tenepal.language.identifier.fuse_scores") as mock_fuse:
            segments = identify_language(self.long_phonemes, audio_data=None)

        # fuse_scores should NOT be called (no prosody available)
        mock_fuse.assert_not_called()
        self.assertIsInstance(segments, list)


if __name__ == "__main__":
    unittest.main()

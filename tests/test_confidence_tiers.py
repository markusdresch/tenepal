"""Test confidence tier classification for Whisper segments."""

from __future__ import annotations

import pytest

from tenepal.validation.confidence_tiers import (
    ConfidenceTier,
    classify_confidence,
    split_by_confidence_tier,
)


class TestClassifyConfidence:
    """Test classify_confidence function with various avg_log_prob values."""

    def test_classify_high_confidence(self):
        """HIGH tier: avg_log_prob = -0.1 (> -0.3)."""
        assert classify_confidence(-0.1) == ConfidenceTier.HIGH

    def test_classify_high_boundary(self):
        """HIGH tier: avg_log_prob = -0.29 (just above -0.3)."""
        assert classify_confidence(-0.29) == ConfidenceTier.HIGH

    def test_classify_medium_confidence(self):
        """MEDIUM tier: avg_log_prob = -0.5 (between -0.7 and -0.3)."""
        assert classify_confidence(-0.5) == ConfidenceTier.MEDIUM

    def test_classify_medium_boundaries(self):
        """MEDIUM tier: boundaries at -0.3 and -0.7 inclusive."""
        assert classify_confidence(-0.3) == ConfidenceTier.MEDIUM
        assert classify_confidence(-0.7) == ConfidenceTier.MEDIUM

    def test_classify_low_confidence(self):
        """LOW tier: avg_log_prob = -1.0 (< -0.7)."""
        assert classify_confidence(-1.0) == ConfidenceTier.LOW

    def test_classify_low_boundary(self):
        """LOW tier: avg_log_prob = -0.7001 (just below -0.7)."""
        assert classify_confidence(-0.7001) == ConfidenceTier.LOW

    def test_classify_zero(self):
        """HIGH tier: avg_log_prob = 0.0 (perfect confidence)."""
        assert classify_confidence(0.0) == ConfidenceTier.HIGH

    def test_classify_very_low(self):
        """LOW tier: avg_log_prob = -2.0 (very low confidence)."""
        assert classify_confidence(-2.0) == ConfidenceTier.LOW


class MockSegment:
    """Mock Whisper segment with avg_log_prob attribute."""

    def __init__(self, avg_log_prob: float, text: str = ""):
        self.avg_log_prob = avg_log_prob
        self.text = text

    def __repr__(self):
        return f"MockSegment(avg_log_prob={self.avg_log_prob}, text='{self.text}')"


class TestSplitByConfidenceTier:
    """Test split_by_confidence_tier function."""

    def test_split_by_tier_empty(self):
        """Empty list returns three empty lists."""
        high, medium, low = split_by_confidence_tier([])
        assert high == []
        assert medium == []
        assert low == []

    def test_split_by_tier_all_high(self):
        """All segments with avg_log_prob > -0.3 go to high list."""
        segments = [
            MockSegment(-0.1, "one"),
            MockSegment(-0.2, "two"),
            MockSegment(-0.29, "three"),
        ]
        high, medium, low = split_by_confidence_tier(segments)
        assert len(high) == 3
        assert len(medium) == 0
        assert len(low) == 0
        assert high == segments

    def test_split_by_tier_mixed(self):
        """5 segments across all tiers are correctly sorted."""
        segments = [
            MockSegment(-0.1, "high1"),  # HIGH
            MockSegment(-0.5, "med1"),  # MEDIUM
            MockSegment(-1.0, "low1"),  # LOW
            MockSegment(-0.3, "med2"),  # MEDIUM (boundary)
            MockSegment(-0.8, "low2"),  # LOW
        ]
        high, medium, low = split_by_confidence_tier(segments)

        assert len(high) == 1
        assert len(medium) == 2
        assert len(low) == 2

        assert high[0].text == "high1"
        assert medium[0].text == "med1"
        assert medium[1].text == "med2"
        assert low[0].text == "low1"
        assert low[1].text == "low2"

    def test_split_preserves_order(self):
        """Segments within each tier maintain their original order."""
        segments = [
            MockSegment(-0.1, "high1"),
            MockSegment(-0.2, "high2"),
            MockSegment(-0.15, "high3"),
        ]
        high, medium, low = split_by_confidence_tier(segments)

        # Order preserved
        assert high[0].text == "high1"
        assert high[1].text == "high2"
        assert high[2].text == "high3"

"""Whisper confidence tier classification.

Classifies Whisper segments into three confidence tiers that determine
how their language identification is handled:

- HIGH (avg_log_prob > -0.3): Use Whisper language unconditionally
- MEDIUM (-0.7 <= avg_log_prob <= -0.3): Keep Whisper text, defer to Allosaurus for language
- LOW (avg_log_prob < -0.7): Prefer Allosaurus phonemes and speaker context
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ConfidenceTier(Enum):
    """Confidence tier enumeration for Whisper segments."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Threshold constants (exposed for testing and pipeline configuration)
HIGH_CONFIDENCE_THRESHOLD = -0.3  # avg_log_prob > this → HIGH
LOW_CONFIDENCE_THRESHOLD = -0.7  # avg_log_prob < this → LOW
# MEDIUM is between LOW and HIGH inclusive on both boundaries


def classify_confidence(avg_log_prob: float) -> ConfidenceTier:
    """Classify a Whisper segment's confidence into a tier.

    Args:
        avg_log_prob: Average log probability from Whisper segment

    Returns:
        ConfidenceTier.HIGH, MEDIUM, or LOW based on thresholds

    Boundary rules:
        - HIGH: avg_log_prob > -0.3
        - MEDIUM: -0.7 <= avg_log_prob <= -0.3
        - LOW: avg_log_prob < -0.7
    """
    if avg_log_prob > HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceTier.HIGH
    elif avg_log_prob < LOW_CONFIDENCE_THRESHOLD:
        return ConfidenceTier.LOW
    else:
        return ConfidenceTier.MEDIUM


def split_by_confidence_tier(segments: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
    """Split Whisper segments into (high, medium, low) confidence tiers.

    Each segment must have an avg_log_prob attribute.
    Returns (high_confidence, medium_confidence, low_confidence) lists.
    Order within each tier is preserved from the input.

    Args:
        segments: List of objects with avg_log_prob attribute

    Returns:
        Tuple of (high, medium, low) lists, order preserved within each tier
    """
    high_confidence: list[Any] = []
    medium_confidence: list[Any] = []
    low_confidence: list[Any] = []

    for segment in segments:
        tier = classify_confidence(segment.avg_log_prob)
        if tier == ConfidenceTier.HIGH:
            high_confidence.append(segment)
        elif tier == ConfidenceTier.MEDIUM:
            medium_confidence.append(segment)
        else:  # LOW
            low_confidence.append(segment)

    return high_confidence, medium_confidence, low_confidence

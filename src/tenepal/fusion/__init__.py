"""Score normalization and fusion for multi-channel language identification.

This package combines phoneme-based and prosody-based language detection
scores using weighted fusion with adaptive per-language-pair weights.
"""

from .normalizer import normalize_phoneme_scores
from .scorer import FusionWeights, default_fusion_weights, fuse_scores

__all__ = [
    "normalize_phoneme_scores",
    "fuse_scores",
    "FusionWeights",
    "default_fusion_weights",
]

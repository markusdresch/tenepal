"""Prosodic feature extraction and rhythm metrics for language identification.

This package provides tools to extract prosodic features (pitch, intensity, rhythm)
from audio segments and compute PVI rhythm metrics for distinguishing stress-timed
from syllable-timed languages.
"""

from .classifier import (
    ProsodyProfile,
    load_prosody_profiles,
    score_all_profiles,
    score_prosody_profile,
)
from .extractor import ProsodyFeatures, extract_prosody
from .rhythm import compute_npvi, compute_rpvi

__all__ = [
    "ProsodyFeatures",
    "ProsodyProfile",
    "extract_prosody",
    "compute_npvi",
    "compute_rpvi",
    "load_prosody_profiles",
    "score_prosody_profile",
    "score_all_profiles",
]

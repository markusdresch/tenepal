"""Nahuatl morphological analysis for LID and translation.

Provides morpheme segmentation for polysynthetic Nahuatl words,
serving dual purposes:
1. Language identification boost (successful parse = strong NAH signal)
2. Translation scaffolding (morpheme glosses → LLM composition)
"""

from tenepal.morphology.segmenter import (
    MorphemeAnalysis,
    MorphemeType,
    Morpheme,
    NahuatlMorphemeSegmenter,
    analyze_word,
    analyze_text,
    lid_score,
    get_segmenter,
)

__all__ = [
    "MorphemeAnalysis",
    "MorphemeType",
    "Morpheme",
    "NahuatlMorphemeSegmenter",
    "analyze_word",
    "analyze_text",
    "lid_score",
    "get_segmenter",
]

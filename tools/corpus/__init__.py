"""Corpus indexing utilities for NAH ground-truth evaluation corpus.

Provides:
    CorpusSample             — dataclass representing one indexed audio+transcript pair
    build_corpus_index       — function to build manifest from TRS files
    compute_hallucination_stats — Whisper confusion matrix and CI computation
"""

from tools.corpus.index import CorpusSample, build_corpus_index
from tools.corpus.hallucination_stats import compute_hallucination_stats

__all__ = ["CorpusSample", "build_corpus_index", "compute_hallucination_stats"]

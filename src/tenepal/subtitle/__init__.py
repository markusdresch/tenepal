"""Subtitle export module for Tenepal.

Provides subtitle generation in various formats (SRT, VTT, etc.)
from language-tagged phoneme segments.
"""

from tenepal.subtitle.srt import format_srt, write_srt

__all__ = ["format_srt", "write_srt"]

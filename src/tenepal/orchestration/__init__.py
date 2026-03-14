"""End-to-end orchestration for full film processing pipeline.

This module provides the process_film() entry point that chains all v4.0
components into a unified pipeline with GPU memory management, progress
reporting, and graceful degradation.
"""

from .film_pipeline import process_film

__all__ = ["process_film"]

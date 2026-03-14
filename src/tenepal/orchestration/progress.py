"""Progress reporting for pipeline stages using tqdm.

Provides a context manager wrapper around tqdm for stage-by-stage progress
tracking during long-form audio processing (40+ minutes).
"""

import sys
from tqdm import tqdm


class StageProgress:
    """Context manager for multi-stage progress tracking.

    Wraps tqdm to provide stage-level progress updates during pipeline
    execution. Progress bar is automatically disabled when output is not
    a TTY (e.g., logging to file, CI environments).

    Example:
        >>> with StageProgress(total=5, description="Processing film") as progress:
        ...     progress.advance("Stage 1/5: Preprocessing")
        ...     # ... do work ...
        ...     progress.advance("Stage 2/5: Diarization")
        ...     # ... do work ...
    """

    def __init__(self, total_stages: int, description: str = "Processing film") -> None:
        """Initialize progress tracker.

        Args:
            total_stages: Number of stages in the pipeline
            description: Overall task description (shown on progress bar)
        """
        self.total_stages = total_stages
        self.description = description
        self.pbar = None

    def __enter__(self):
        """Create and return tqdm progress bar."""
        self.pbar = tqdm(
            total=self.total_stages,
            unit="stage",
            desc=self.description,
            disable=not sys.stdout.isatty()
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close tqdm progress bar."""
        if self.pbar:
            self.pbar.close()
        return False

    def advance(self, stage_name: str) -> None:
        """Advance progress bar to next stage.

        Args:
            stage_name: Name/description of the stage that just completed
        """
        if self.pbar:
            self.pbar.set_description(stage_name)
            self.pbar.update(1)

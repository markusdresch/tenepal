"""GPU memory lifecycle management for sequential model loading.

Provides utilities to free GPU memory between pipeline stages, enabling
processing of long-form audio without OOM errors from model accumulation.
"""

import gc
import logging

logger = logging.getLogger(__name__)


def unload_gpu_models() -> None:
    """Unload GPU models and clear CUDA cache.

    Forces Python garbage collection and clears the CUDA memory cache if
    torch is available and CUDA is active. Safe to call even if torch is
    not installed or CUDA is not available.

    This function is called between pipeline stages to free GPU memory from
    models that are no longer needed (Demucs, pyannote, Whisper, etc.).
    """
    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        # torch not installed, nothing to do
        pass


def cleanup_stage(stage_name: str, backend=None) -> None:
    """Clean up resources after a pipeline stage completes.

    Unloads backend models (if provided) and frees GPU memory. This ensures
    that each pipeline stage releases its GPU resources before the next stage
    begins, preventing memory accumulation across stages.

    Args:
        stage_name: Name of the completed stage (for logging)
        backend: Optional backend object with unload() or unload_backends() method

    Example:
        >>> router = TranscriptionRouter(whisper_model="base")
        >>> # ... use router ...
        >>> cleanup_stage("transcription", backend=router)
    """
    # Unload backend if provided
    if backend is not None:
        if hasattr(backend, "unload_backends"):
            # TranscriptionRouter has unload_backends()
            backend.unload_backends()
        elif hasattr(backend, "unload"):
            # Other backends have unload()
            backend.unload()

    # Free GPU memory
    unload_gpu_models()

    logger.info("Stage complete: %s, GPU memory released", stage_name)

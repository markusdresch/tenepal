"""WhisperBackend ASR implementation using faster-whisper.

This backend provides transcription (not phoneme recognition) for languages
supported by OpenAI's Whisper model. It integrates into Tenepal's backend
registry and follows the ASRBackend interface for consistency.

CRITICAL: This backend ALWAYS forces the language parameter to Whisper.
Never allows auto-detection - language must come from Tenepal's analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, Union

from .backend import ASRBackend, PhonemeSegment


@dataclass
class WhisperAutoSegment:
    """Segment from Whisper auto-detect transcription.

    Attributes:
        text: Transcribed text
        start: Start time in seconds
        end: End time in seconds
        language: ISO 639-1 code detected by Whisper (e.g., "es")
        avg_log_prob: Average log probability (confidence proxy)
    """

    text: str
    start: float
    end: float
    language: str
    avg_log_prob: float


class WhisperBackend(ASRBackend):
    """faster-whisper ASR backend for transcription.

    This backend uses faster-whisper (optimized Whisper via CTranslate2) for
    transcription of audio segments identified as Whisper-supported languages.

    Key differences from phoneme backends:
    - Returns transcribed text (not IPA phonemes)
    - Requires explicit language code (ISO 639-1)
    - Supports GPU acceleration
    - Larger models (hundreds of MB to GBs)

    GPU memory usage (approx):
    - tiny: ~400 MB
    - base: ~500 MB
    - small: ~1 GB
    - medium: ~2 GB
    - large: ~3 GB
    - turbo: ~3 GB

    Attributes:
        name: Backend identifier for registry
        MODEL_SIZES: Valid model size choices
    """

    name: ClassVar[str] = "whisper"
    MODEL_SIZES = {"tiny", "base", "small", "medium", "large", "turbo"}

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        """Initialize WhisperBackend.

        Args:
            model_size: Model size (tiny, base, small, medium, large, turbo)
            device: Device for inference ("auto", "cpu", "cuda")
            compute_type: Precision ("auto", "float16", "int8", "float32")

        Raises:
            ValueError: If model_size is invalid
        """
        if model_size not in self.MODEL_SIZES:
            valid = ", ".join(sorted(self.MODEL_SIZES))
            raise ValueError(f"Invalid model_size '{model_size}'. Valid: {valid}")

        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.compute_type = self._resolve_compute_type(compute_type, self.device)
        self._model: Optional[object] = None

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _resolve_compute_type(self, compute_type: str, device: str) -> str:
        """Resolve compute type based on device."""
        if compute_type == "auto":
            return "float16" if device == "cuda" else "int8"
        return compute_type

    def _get_model(self):
        """Get or initialize the cached WhisperModel.

        Lazy-loads the model on first use to avoid loading during initialization.
        """
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "faster-whisper is not available. Install with: pip install faster-whisper"
                ) from exc

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def recognize(
        self, audio_path: Union[str, Path], lang: str = "en"
    ) -> list[PhonemeSegment]:
        """Transcribe audio using Whisper with forced language.

        CRITICAL: Always passes explicit language parameter to Whisper.
        Never allows auto-detection - defeats Tenepal's purpose.

        Args:
            audio_path: Path to audio file
            lang: ISO 639-1 language code (e.g., "es", "en", "de")

        Returns:
            List of PhonemeSegment with transcribed text in phoneme field

        Note:
            - vad_filter=False: Tenepal handles VAD in preprocessing (Phase 22)
            - beam_size=5: Balances quality and speed
            - Text stored in phoneme field for pipeline compatibility
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._get_model()

        # CRITICAL: Force language parameter - never auto-detect
        segments, _ = model.transcribe(
            str(audio_path),
            language=lang,  # Explicit language from Tenepal analysis
            vad_filter=False,  # Tenepal handles VAD
            beam_size=5,
        )

        # Convert Whisper segments to PhonemeSegment
        # Store transcription in phoneme field for pipeline compatibility
        result: list[PhonemeSegment] = []
        for seg in segments:
            result.append(
                PhonemeSegment(
                    phoneme=seg.text.strip(),
                    start_time=seg.start,
                    duration=seg.end - seg.start,
                )
            )

        return result

    def transcribe_auto(self, audio_path: Union[str, Path]) -> list[WhisperAutoSegment]:
        """Transcribe audio with automatic language detection.

        Uses Whisper's auto-detect mode (language=None) to transcribe without
        forcing a language. Returns segments with text, timing, detected language,
        and confidence (avg_log_prob).

        Uses file-level language detection (info.language) for all segments.
        Per-segment detection was removed because it returns wrong codes for
        short clips, causing most segments to map to "other".

        Args:
            audio_path: Path to audio file

        Returns:
            List of WhisperAutoSegment with text, timing, language, confidence

        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._get_model()

        segments, info = model.transcribe(
            str(audio_path),
            language=None,  # AUTO-DETECT
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
        )

        # Convert segments generator to list
        segments_list = list(segments)

        # Use file-level language for all segments (WhisperValidator checks text)
        result: list[WhisperAutoSegment] = []
        for s in segments_list:
            result.append(
                WhisperAutoSegment(
                    text=s.text.strip(),
                    start=s.start,
                    end=s.end,
                    language=info.language,
                    avg_log_prob=s.avg_logprob,
                )
            )

        return result

    def transcribe_segment(
        self,
        samples: "np.ndarray",
        sample_rate: int,
        vad_filter: bool = False,
        initial_prompt: str | None = None,
        pad_seconds: float = 0.5,
    ) -> list[WhisperAutoSegment]:
        """Transcribe a single audio segment with configurable parameters.

        Designed for rescue pass: re-tries short segments that Whisper's VAD
        skipped in the main pass. Optionally pads with silence for better
        boundary detection.

        Args:
            samples: Audio samples as numpy array (mono, any sample rate)
            sample_rate: Sample rate of the samples
            vad_filter: Whether to use Whisper's VAD filter (default: False for rescue)
            initial_prompt: Vocabulary prompt to guide transcription
            pad_seconds: Silence padding before/after segment (default: 0.5s)

        Returns:
            List of WhisperAutoSegment (usually 0 or 1 for short segments)
        """
        import numpy as np
        import tempfile
        import os
        import soundfile as sf

        model = self._get_model()

        # Pad with silence if requested
        if pad_seconds > 0:
            pad_samples = int(pad_seconds * sample_rate)
            silence = np.zeros(pad_samples, dtype=samples.dtype)
            padded = np.concatenate([silence, samples, silence])
        else:
            padded = samples

        # Write to temp file (faster-whisper needs a file path)
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="tenepal_rescue_")
        os.close(fd)

        try:
            sf.write(temp_path, padded, sample_rate)

            kwargs = {
                "language": None,  # auto-detect
                "vad_filter": vad_filter,
                "beam_size": 5,
            }
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt

            segments, info = model.transcribe(temp_path, **kwargs)
            segments_list = list(segments)

            result: list[WhisperAutoSegment] = []
            for s in segments_list:
                # Adjust timestamps to remove padding offset
                adj_start = max(0.0, s.start - pad_seconds)
                adj_end = max(0.0, s.end - pad_seconds)
                if adj_end <= 0:
                    continue  # segment is entirely within leading padding
                result.append(
                    WhisperAutoSegment(
                        text=s.text.strip(),
                        start=adj_start,
                        end=adj_end,
                        language=info.language,
                        avg_log_prob=s.avg_logprob,
                    )
                )

            return result
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def unload(self) -> None:
        """Unload model and clear GPU memory.

        Call this when switching between models or cleaning up GPU resources.
        """
        self._model = None

        # Clear GPU cache if on CUDA
        if self.device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if faster-whisper is available.

        Returns:
            True if faster-whisper can be imported
        """
        try:
            from faster_whisper import WhisperModel  # noqa: F401

            return True
        except Exception:
            return False

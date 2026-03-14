"""TranscriptionRouter for directing language segments to optimal ASR backends.

The router is the core intelligence of Whisper integration -- it uses Tenepal's
language identification results to select the right transcription tool per segment:
- Known languages (spa/eng/deu/fra/ita) → Whisper for readable text
- Nahuatl and unknowns → Allosaurus for IPA phonemes

This enables Tenepal to produce readable text output alongside IPA for the first time.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import soundfile as sf

from tenepal.audio import load_audio
from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme.backend import get_backend
from tenepal.transcription.languages import ISO_639_MAP, WHISPER_SUPPORTED

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Unified transcription output from any backend.

    Attributes:
        text: Transcribed text (Whisper) or IPA phonemes (Allosaurus)
        start_time: Start time from LanguageSegment (preserved from Phase 22/23)
        end_time: End time from LanguageSegment (preserved from Phase 22/23)
        language: ISO 639-3 code from Tenepal language identification
        backend: "whisper" or "allosaurus" - which backend transcribed this
        is_text: True for readable text (Whisper), False for IPA phonemes (Allosaurus)
    """
    text: str
    start_time: float
    end_time: float
    language: str
    backend: str
    is_text: bool = True


class TranscriptionRouter:
    """Routes language-identified segments to optimal ASR backends.

    The router examines each LanguageSegment's language code and routes to:
    - WhisperBackend for spa/eng/deu/fra/ita (produces readable text)
    - AllosaurusBackend for nah and other (produces IPA phonemes)

    Segment timing from Phase 22/23 language identification is preserved in
    TranscriptionResults (Whisper's internal timing is discarded).
    """

    def __init__(self, whisper_model: str = "base", whisper_device: str = "auto") -> None:
        """Initialize the transcription router.

        Args:
            whisper_model: Whisper model size (tiny/base/small/medium/large/turbo)
            whisper_device: Device for Whisper ("auto", "cuda", "cpu")
        """
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self._whisper_backend = None
        self._allosaurus_backend = None

    def transcribe_segments(
        self,
        language_segments: list[LanguageSegment],
        audio_path: Union[str, Path]
    ) -> list[TranscriptionResult]:
        """Transcribe language-identified segments using appropriate backends.

        Routes each segment to Whisper (for readable text) or Allosaurus (for IPA)
        based on the segment's language code. Extracts segment audio, transcribes,
        and assembles TranscriptionResults with preserved timing.

        Args:
            language_segments: List of LanguageSegment from language identification
            audio_path: Path to full audio file for segment extraction

        Returns:
            List of TranscriptionResult objects with text, timing, language, backend info

        Raises:
            RuntimeError: If required backend is unavailable
            Exception: If transcription fails
        """
        if not language_segments:
            return []

        results = []

        for segment in language_segments:
            if segment.language in WHISPER_SUPPORTED:
                result = self._transcribe_with_whisper(segment, audio_path)
            else:
                result = self._transcribe_with_allosaurus(segment, audio_path)

            results.append(result)

        return results

    def _transcribe_with_whisper(
        self,
        segment: LanguageSegment,
        audio_path: Union[str, Path]
    ) -> TranscriptionResult:
        """Transcribe segment using WhisperBackend for readable text.

        Args:
            segment: LanguageSegment with language in WHISPER_SUPPORTED
            audio_path: Path to full audio file

        Returns:
            TranscriptionResult with backend="whisper", is_text=True
        """
        # Lazy-load Whisper backend
        if self._whisper_backend is None:
            self._whisper_backend = get_backend(
                "whisper",
                model_size=self.whisper_model,
                device=self.whisper_device
            )

        # Map ISO 639-3 to ISO 639-1 for Whisper API
        whisper_lang = ISO_639_MAP[segment.language]

        # Extract segment audio to temp file
        temp_path = self._extract_segment_audio(segment, audio_path)

        try:
            # Transcribe with forced language parameter (NEVER auto-detect)
            phoneme_segments = self._whisper_backend.recognize(
                temp_path,
                lang=whisper_lang
            )

            # Concatenate Whisper's sub-segment texts into single string
            # (Whisper may split one LanguageSegment into multiple words/phrases)
            text = " ".join(seg.phoneme for seg in phoneme_segments)

            # Return result with original LanguageSegment timing (NOT Whisper's internal timing)
            return TranscriptionResult(
                text=text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                language=segment.language,
                backend="whisper",
                is_text=True
            )

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def _transcribe_with_allosaurus(
        self,
        segment: LanguageSegment,
        audio_path: Union[str, Path]
    ) -> TranscriptionResult:
        """Transcribe segment using AllosaurusBackend for IPA phonemes.

        Args:
            segment: LanguageSegment with language not in WHISPER_SUPPORTED
            audio_path: Path to full audio file

        Returns:
            TranscriptionResult with backend="allosaurus", is_text=False
        """
        # Lazy-load Allosaurus backend
        if self._allosaurus_backend is None:
            self._allosaurus_backend = get_backend("allosaurus")

        # Extract segment audio to temp file
        temp_path = self._extract_segment_audio(segment, audio_path)

        try:
            # Transcribe with IPA mode
            phoneme_segments = self._allosaurus_backend.recognize(
                temp_path,
                lang="ipa"
            )

            # Join phonemes with spaces
            text = " ".join(seg.phoneme for seg in phoneme_segments)

            # Return result with original LanguageSegment timing
            return TranscriptionResult(
                text=text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                language=segment.language,
                backend="allosaurus",
                is_text=False
            )

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def _extract_segment_audio(
        self,
        segment: LanguageSegment,
        audio_path: Union[str, Path]
    ) -> Path:
        """Extract segment audio from full file and write to temp WAV.

        Args:
            segment: LanguageSegment with start_time and end_time
            audio_path: Path to full audio file

        Returns:
            Path to temporary WAV file containing segment audio
        """
        # Load full audio
        audio_data = load_audio(audio_path)
        samples = audio_data.samples
        sample_rate = audio_data.sample_rate

        # Calculate sample range
        start_sample = int(segment.start_time * sample_rate)
        end_sample = int(segment.end_time * sample_rate)

        # Extract segment samples
        segment_samples = samples[start_sample:end_sample]

        # Write to temp file
        fd, temp_path_str = tempfile.mkstemp(suffix=".wav", prefix="tenepal_segment_")
        os.close(fd)  # Close the file descriptor
        temp_path = Path(temp_path_str)

        sf.write(str(temp_path), segment_samples, sample_rate)

        return temp_path

    def unload_backends(self) -> None:
        """Unload cached backends for GPU memory management.

        Calls unload() on WhisperBackend if available and clears CUDA cache.
        Used in Phase 25 for sequential model lifecycle management.
        """
        if self._whisper_backend and hasattr(self._whisper_backend, "unload"):
            self._whisper_backend.unload()
            self._whisper_backend = None

        # Clear CUDA cache if torch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

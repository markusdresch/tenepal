"""Phoneme recognition dispatching to ASR backends."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .backend import PhonemeSegment, get_backend


def recognize_phonemes(
    audio_path: Union[str, Path],
    lang: str = "ipa",
    backend: str = "allosaurus",
    **backend_kwargs,
) -> list[PhonemeSegment]:
    """Recognize phonemes from audio file with timestamps."""
    recognizer = get_backend(backend, **backend_kwargs)
    return recognizer.recognize(audio_path, lang=lang)

"""Phoneme recognition and formatting."""

from .backend import ASRBackend, AllosaurusBackend, get_backend, list_backends
from .omnilingual_backend import OmnilingualBackend
from .dual_backend import DualBackend
from .formatter import format_phonemes, print_phonemes
from .recognizer import PhonemeSegment, recognize_phonemes
from .text_to_ipa import text_to_phonemes, words_to_phonemes

# Optional backend - may not be available
try:
    from .whisper_backend import WhisperBackend
except ImportError:
    WhisperBackend = None

__all__ = [
    "ASRBackend",
    "AllosaurusBackend",
    "OmnilingualBackend",
    "DualBackend",
    "WhisperBackend",
    "get_backend",
    "list_backends",
    "PhonemeSegment",
    "recognize_phonemes",
    "format_phonemes",
    "print_phonemes",
    "text_to_phonemes",
    "words_to_phonemes",
]

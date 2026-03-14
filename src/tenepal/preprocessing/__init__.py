"""Audio preprocessing pipeline for film/video files."""
from .extractor import SpeechSegment, extract_audio
from .isolator import isolate_vocals, is_demucs_available
from .segmenter import segment_speech
from .pipeline import preprocess_video, export_segments_json

__all__ = [
    "SpeechSegment",
    "extract_audio",
    "isolate_vocals",
    "is_demucs_available",
    "segment_speech",
    "preprocess_video",
    "export_segments_json",
]

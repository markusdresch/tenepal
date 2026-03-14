"""Dual backend that runs Allosaurus and Omnilingual in parallel."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, Union
from pathlib import Path

from .backend import ASRBackend, PhonemeSegment, get_backend
from .backend import AllosaurusBackend
from .omnilingual_backend import OmnilingualBackend


class DualBackend(ASRBackend):
    """Run Allosaurus and Omnilingual and merge by confidence."""

    name: ClassVar[str] = "dual"

    def __init__(self, model_size: str = "300M") -> None:
        self.model_size = model_size

    def recognize(self, audio_path: Union[str, Path], lang: str = "ipa") -> list[PhonemeSegment]:
        errors = []
        segments: list[PhonemeSegment] = []

        try:
            allosaurus = get_backend("allosaurus")
            segments.extend(self._with_default_confidence(allosaurus.recognize(audio_path, lang=lang), 0.6))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

        try:
            omnilingual = get_backend("omnilingual", model_size=self.model_size)
            segments.extend(self._with_default_confidence(omnilingual.recognize(audio_path, lang=lang), 0.7))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

        if not segments:
            raise RuntimeError("Dual backend failed: no backend produced results") from (errors[0] if errors else None)

        return self._merge_by_confidence(segments)

    @classmethod
    def is_available(cls) -> bool:
        return AllosaurusBackend.is_available() or OmnilingualBackend.is_available()

    @staticmethod
    def _with_default_confidence(segments: list[PhonemeSegment], default_conf: float) -> list[PhonemeSegment]:
        updated = []
        for seg in segments:
            if seg.confidence is None:
                updated.append(replace(seg, confidence=default_conf))
            else:
                updated.append(seg)
        return updated

    @staticmethod
    def _merge_by_confidence(segments: list[PhonemeSegment]) -> list[PhonemeSegment]:
        if not segments:
            return []
        ordered = sorted(segments, key=lambda s: (s.start_time, s.duration))
        merged: list[PhonemeSegment] = []

        for seg in ordered:
            if not merged:
                merged.append(seg)
                continue

            last = merged[-1]
            last_end = last.start_time + last.duration
            seg_end = seg.start_time + seg.duration
            overlaps = seg.start_time < last_end

            if not overlaps:
                merged.append(seg)
                continue

            last_conf = last.confidence if last.confidence is not None else 0.0
            seg_conf = seg.confidence if seg.confidence is not None else 0.0

            if seg_conf > last_conf:
                merged[-1] = seg
            else:
                # keep existing
                pass

        return merged

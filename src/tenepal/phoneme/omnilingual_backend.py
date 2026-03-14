"""Omnilingual ASR backend with subprocess isolation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import ClassVar, Union

from .backend import ASRBackend, PhonemeSegment
from .language_codes import resolve_language_code
from .text_to_ipa import text_to_phonemes
from ..audio import load_audio


class OmnilingualBackend(ASRBackend):
    """Omnilingual ASR backend running in a Python 3.12 subprocess."""

    name: ClassVar[str] = "omnilingual"
    VENV_PATH: ClassVar[Path] = Path.home() / ".tenepal" / "omnilingual-venv"
    MODEL_CARDS: ClassVar[dict[str, str]] = {
        "300M": "omniASR_LLM_300M_v2",
        "7B": "omniASR_LLM_7B_v2",
    }

    def __init__(self, model_size: str = "300M") -> None:
        if model_size not in self.MODEL_CARDS:
            raise ValueError(f"Unknown model size: {model_size}")
        self.model_size = model_size

    def recognize(self, audio_path: Union[str, Path], lang: str = "nah") -> list[PhonemeSegment]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = load_audio(audio_path)
        duration = audio.duration

        resolved = resolve_language_code(lang, backend=self.name)
        language = self._to_omnilingual_lang(resolved)
        model_card = self.MODEL_CARDS[self.model_size]
        python_path = self.VENV_PATH / "bin" / "python"
        worker_path = Path(__file__).parent / "omnilingual_worker.py"

        cmd = [
            str(python_path),
            str(worker_path),
            "--audio-path",
            str(audio_path),
            "--language",
            language,
            "--model-card",
            model_card,
            "--device",
            self._detect_device(),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Omnilingual ASR subprocess timed out") from exc

        try:
            payload = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Omnilingual ASR returned invalid JSON") from exc

        if "error" in payload:
            raise RuntimeError(payload["error"])

        text = payload.get("text", "")
        return text_to_phonemes(text, lang, start_time=0.0, duration=duration)

    def _detect_device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "auto"

    @classmethod
    def _to_omnilingual_lang(cls, code: str) -> str:
        if "_" in code:
            return code
        return f"{code}_Latn"

    @classmethod
    def is_available(cls) -> bool:
        python_path = cls.VENV_PATH / "bin" / "python"
        worker_path = Path(__file__).parent / "omnilingual_worker.py"
        return python_path.exists() and worker_path.exists()

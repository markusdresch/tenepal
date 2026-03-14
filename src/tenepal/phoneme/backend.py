"""Backend abstraction for phoneme recognition."""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Union

from ..audio import load_audio, preprocess_audio, save_wav
from .language_codes import resolve_language_code


@dataclass
class PhonemeSegment:
    """Container for a phoneme with timing information."""

    phoneme: str
    start_time: float
    duration: float
    confidence: Optional[float] = None


class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    name: ClassVar[str]

    @abstractmethod
    def recognize(self, audio_path: Union[str, Path], lang: str = "ipa") -> list[PhonemeSegment]:
        """Recognize phonemes from an audio file."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if backend dependencies are available."""


class AllosaurusBackend(ASRBackend):
    """Allosaurus-based phoneme recognizer with adaptive blank-bias support.

    The CTC decoder in Allosaurus uses greedy decoding where blank often wins
    over actual phonemes, especially for unknown languages like Nahuatl.
    The blank_bias parameter reduces blank's advantage:
    - bias=0: Default behavior (58% blank on average)
    - bias=2-3: Sweet spot for recovering missed phonemes
    - bias=5+: Noisy, may hallucinate

    Adaptive strategy: Use bias=0 for SPA, bias=3 for NAH/MAY/OTH.
    """

    name: ClassVar[str] = "allosaurus"

    def __init__(self) -> None:
        self._model = None
        self._blank_idx: Optional[int] = None
        self._phone_list: Optional[dict] = None

    def _get_model(self):
        """Get or initialize the cached Allosaurus model."""
        if self._model is None:
            try:
                from allosaurus.app import read_recognizer
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError("Allosaurus is not available") from exc
            self._model = read_recognizer()

            # Cache blank index and phone list for bias decoding
            self._phone_list = self._model.lm.inventory.unit.id_to_unit
            for idx, phone in self._phone_list.items():
                if phone == "<blk>":
                    self._blank_idx = idx
                    break

        return self._model

    def recognize(
        self,
        audio_path: Union[str, Path],
        lang: str = "ipa",
        blank_bias: float = 0.0,
    ) -> list[PhonemeSegment]:
        """Recognize phonemes from audio file with timestamps.

        Args:
            audio_path: Path to audio file
            lang: Language code for phoneme inventory (default: 'ipa')
            blank_bias: CTC blank bias reduction (0=default, 2-3=recommended for NAH)

        Returns:
            List of PhonemeSegment with timing information
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = load_audio(audio_path)
        audio = preprocess_audio(audio, target_sr=16000)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            save_wav(audio, temp_path)

            model = self._get_model()
            lang_id = resolve_language_code(lang, backend=self.name)

            # Use bias-adjusted decoding if blank_bias > 0
            if blank_bias > 0 and self._blank_idx is not None:
                return self._recognize_with_bias(
                    str(temp_path), lang_id, blank_bias, audio.duration
                )

            # Default: use standard Allosaurus recognize
            result = model.recognize(str(temp_path), lang_id=lang_id, timestamp=True)

            segments: list[PhonemeSegment] = []
            for line in result.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    duration = float(parts[1])
                    phoneme = parts[2]

                    segments.append(
                        PhonemeSegment(
                            phoneme=phoneme,
                            start_time=start_time,
                            duration=duration,
                        )
                    )

            return segments

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _recognize_with_bias(
        self,
        temp_path: str,
        lang_id: str,
        blank_bias: float,
        audio_duration: float,
    ) -> list[PhonemeSegment]:
        """Internal: CTC decoding with reduced blank bias.

        Manipulates logits before argmax to reduce blank's advantage.
        """
        try:
            import numpy as np
            import torch
            from allosaurus.audio import read_audio
            from allosaurus.am.utils import move_to_tensor
        except ImportError as exc:
            raise RuntimeError("Required dependencies not available") from exc

        model = self._get_model()

        # Get acoustic features
        audio_allo = read_audio(temp_path)
        feat = model.pm.compute(audio_allo)
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)

        tensor_feat, tensor_feat_len = move_to_tensor(
            [feats, feat_len], model.config.device_id
        )

        # Get log probabilities from acoustic model
        with torch.no_grad():
            tensor_lprobs = model.am(tensor_feat, tensor_feat_len)

        if model.config.device_id >= 0:
            lprobs = tensor_lprobs.cpu().numpy()[0]
        else:
            lprobs = tensor_lprobs.numpy()[0]

        # Apply blank bias: subtract from blank logit
        lprobs[:, self._blank_idx] -= blank_bias

        # Greedy CTC decode with collapse
        decoded_indices = np.argmax(lprobs, axis=1)

        # Convert to phones with CTC collapse
        phones = []
        prev_idx = -1
        for idx in decoded_indices:
            if idx != prev_idx:
                if idx != self._blank_idx:
                    phone = self._phone_list.get(int(idx), f"?{idx}")
                    phones.append(phone)
                prev_idx = idx

        # Create segments with estimated timing
        num_frames = len(decoded_indices)
        frame_duration = audio_duration / num_frames if num_frames > 0 else 0.01
        segments: list[PhonemeSegment] = []

        if phones:
            phone_duration = audio_duration / len(phones)
            for i, phone in enumerate(phones):
                segments.append(
                    PhonemeSegment(
                        phoneme=phone,
                        start_time=i * phone_duration,
                        duration=phone_duration,
                    )
                )

        return segments

    @classmethod
    def is_available(cls) -> bool:
        try:
            import allosaurus  # noqa: F401

            return True
        except Exception:
            return False


_BACKENDS: Dict[str, type[ASRBackend]] = {}
_BACKEND_INSTANCES: Dict[tuple, ASRBackend] = {}


def register_backend(name: str, backend_cls: type[ASRBackend]) -> None:
    """Register a backend class under a name."""
    _BACKENDS[name] = backend_cls


def list_backends() -> list[str]:
    """List available backend names."""
    return sorted(_BACKENDS.keys())


def get_backend(name: str = "allosaurus", **kwargs) -> ASRBackend:
    """Return a cached backend instance by name."""
    cache_key = (name, tuple(sorted(kwargs.items())))
    if cache_key in _BACKEND_INSTANCES:
        return _BACKEND_INSTANCES[cache_key]

    backend_cls = _BACKENDS.get(name)
    if backend_cls is None:
        available = ", ".join(list_backends())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    if not backend_cls.is_available():
        raise RuntimeError(
            f"Backend '{name}' is not available. Install dependencies or use another backend."
        )

    instance = backend_cls(**kwargs)
    _BACKEND_INSTANCES[cache_key] = instance
    return instance


register_backend(AllosaurusBackend.name, AllosaurusBackend)

try:  # Optional backend import
    from .omnilingual_backend import OmnilingualBackend

    register_backend(OmnilingualBackend.name, OmnilingualBackend)
except Exception:
    pass

try:  # Optional backend import
    from .dual_backend import DualBackend

    register_backend(DualBackend.name, DualBackend)
except Exception:
    pass

try:  # Optional backend import
    from .whisper_backend import WhisperBackend

    register_backend(WhisperBackend.name, WhisperBackend)
except Exception:
    pass

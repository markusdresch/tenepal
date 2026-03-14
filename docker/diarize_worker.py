"""pyannote diarization worker script for container execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path


def _letter_label(index: int) -> str:
    if index < 26:
        return f"Speaker {chr(65 + index)}"
    first = chr(65 + (index // 26) - 1)
    second = chr(65 + (index % 26))
    return f"Speaker {first}{second}"


def _error(message: str) -> int:
    print(json.dumps({"error": message}))
    return 1


def _load_wav(audio_path: Path) -> dict[str, object]:
    """Load a WAV file into a pyannote-compatible in-memory dict."""
    try:
        import numpy as np
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"NumPy/Torch required for WAV loading: {exc}") from exc

    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sampwidth = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if sampwidth == 1:
        data = np.frombuffer(frames, dtype=np.uint8)
        data = data.astype(np.int16) - 128
        scale = 128.0
    elif sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16)
        scale = 32768.0
    elif sampwidth == 3:
        raw = np.frombuffer(frames, dtype=np.uint8)
        raw = raw.reshape(-1, 3)
        data = raw[:, 0].astype(np.int32)
        data |= raw[:, 1].astype(np.int32) << 8
        data |= raw[:, 2].astype(np.int32) << 16
        sign_mask = 1 << 23
        data = (data ^ sign_mask) - sign_mask
        scale = float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32)
        scale = float(1 << 31)
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if data.size % channels != 0:
        raise RuntimeError("WAV data size does not align with channel count")

    data = data.reshape(-1, channels).T.astype(np.float32) / scale
    waveform = torch.from_numpy(data)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _patch_torchcodec_fallback() -> None:
    """Provide a minimal torchcodec fallback for WAV files."""
    try:
        import numpy as np
        import torch
        import pyannote.audio.core.io as io
    except Exception:
        return

    if hasattr(io, "AudioDecoder"):
        return

    class _FallbackAudioSamples:
        def __init__(self, data: "torch.Tensor", sample_rate: int) -> None:
            self.data = data
            self.sample_rate = sample_rate

    class _FallbackAudioStreamMetadata:
        def __init__(self, sample_rate: int, duration_seconds: float) -> None:
            self.sample_rate = sample_rate
            self.duration_seconds_from_header = duration_seconds

    class _FallbackAudioDecoder:
        def __init__(self, source) -> None:  # noqa: ANN001 - external API
            self._source = source

        def _read(self) -> tuple["torch.Tensor", int]:
            wav_file = wave.open(self._source, "rb")
            try:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                sampwidth = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                frames = wav_file.readframes(frame_count)
            finally:
                wav_file.close()

            if sampwidth == 1:
                data = np.frombuffer(frames, dtype=np.uint8)
                data = data.astype(np.int16) - 128
                scale = 128.0
            elif sampwidth == 2:
                data = np.frombuffer(frames, dtype=np.int16)
                scale = 32768.0
            elif sampwidth == 3:
                raw = np.frombuffer(frames, dtype=np.uint8)
                raw = raw.reshape(-1, 3)
                data = raw[:, 0].astype(np.int32)
                data |= raw[:, 1].astype(np.int32) << 8
                data |= raw[:, 2].astype(np.int32) << 16
                sign_mask = 1 << 23
                data = (data ^ sign_mask) - sign_mask
                scale = float(1 << 23)
            elif sampwidth == 4:
                data = np.frombuffer(frames, dtype=np.int32)
                scale = float(1 << 31)
            else:
                raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

            data = data.reshape(-1, channels).T.astype(np.float32) / scale
            waveform = torch.from_numpy(data)
            return waveform, sample_rate

        @property
        def metadata(self) -> "_FallbackAudioStreamMetadata":
            waveform, sample_rate = self._read()
            duration = waveform.shape[1] / sample_rate
            return _FallbackAudioStreamMetadata(sample_rate, duration)

        def get_all_samples(self) -> "_FallbackAudioSamples":
            waveform, sample_rate = self._read()
            return _FallbackAudioSamples(waveform, sample_rate)

        def get_samples_played_in_range(self, start: float, end: float) -> "_FallbackAudioSamples":
            waveform, sample_rate = self._read()
            start_idx = max(0, int(start * sample_rate))
            end_idx = max(start_idx, int(end * sample_rate))
            return _FallbackAudioSamples(waveform[:, start_idx:end_idx], sample_rate)

    io.AudioSamples = _FallbackAudioSamples
    io.AudioStreamMetadata = _FallbackAudioStreamMetadata
    io.AudioDecoder = _FallbackAudioDecoder


def main() -> int:
    parser = argparse.ArgumentParser(description="pyannote diarization worker")
    parser.add_argument("--audio-path", required=True)
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return _error("HUGGINGFACE_TOKEN not set")

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        return _error(f"Audio file not found: {audio_path}")

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # noqa: BLE001 - surface import errors in JSON
        return _error(f"pyannote.audio not available: {exc}")

    try:
        import torch

        try:
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
        except AttributeError:
            pass
    except Exception:
        pass

    try:
        import inspect
        import torch
        import pyannote.audio.core.task as task_module

        safe_classes = [
            obj
            for _, obj in inspect.getmembers(task_module, inspect.isclass)
            if obj.__module__ == task_module.__name__
        ]
        if safe_classes:
            try:
                torch.serialization.add_safe_globals(safe_classes)
            except AttributeError:
                pass
    except Exception:
        pass

    _patch_torchcodec_fallback()

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
    except TypeError:
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
        except Exception as exc:  # noqa: BLE001 - surface pipeline errors in JSON
            return _error(f"Failed to load pipeline: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface pipeline errors in JSON
        return _error(f"Failed to load pipeline: {exc}")

    device = None
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    except Exception:
        device = None

    try:
        if device is not None:
            pipeline.to(device)
            print(f"Pipeline device: {device}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline device set failed ({device}): {exc}", file=sys.stderr)

    try:
        if audio_path.suffix.lower() == ".wav":
            audio_input = _load_wav(audio_path)
        else:
            audio_input = str(audio_path)

        result = pipeline(audio_input)
        annotation = getattr(result, "speaker_diarization", result)

        label_map: dict[str, str] = {}
        label_counter = 0
        segments: list[dict[str, float | str]] = []

        for segment, _track, speaker_label in annotation.itertracks(yield_label=True):
            if speaker_label not in label_map:
                label_map[speaker_label] = _letter_label(label_counter)
                label_counter += 1

            segments.append(
                {
                    "speaker": label_map[speaker_label],
                    "start_time": float(segment.start),
                    "end_time": float(segment.end),
                }
            )

        segments.sort(key=lambda entry: (entry["start_time"], entry["speaker"]))
        print(json.dumps({"segments": segments}))
        return 0
    except Exception as exc:  # noqa: BLE001 - worker must be resilient
        return _error(str(exc))


if __name__ == "__main__":
    sys.exit(main())

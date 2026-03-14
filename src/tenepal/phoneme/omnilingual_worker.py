"""Omnilingual ASR worker script for Python 3.12 subprocess execution."""

from __future__ import annotations

import argparse
import json
import sys


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Omnilingual ASR worker")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto" or not device:
        device = _detect_device()

    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        pipeline = ASRInferencePipeline(model_card=args.model_card, device=device)
        result = pipeline.transcribe([args.audio_path], lang=[args.language])

        text = ""
        language = args.language
        if isinstance(result, list) and result:
            entry = result[0]
            if isinstance(entry, dict):
                text = entry.get("text", "")
                language = entry.get("language", language)
            else:
                text = str(entry)
        elif isinstance(result, dict):
            text = result.get("text", "")
            language = result.get("language", language)
        else:
            text = str(result)

        payload = {"text": text, "language": language}
        print(json.dumps(payload))
        return 0
    except Exception as exc:  # noqa: BLE001 - worker must be resilient
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

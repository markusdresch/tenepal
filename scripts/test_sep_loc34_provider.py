#!/usr/bin/env python3
"""Provider-agnostic LOC34 separation test.

Examples:
  python scripts/test_sep_loc34_provider.py --provider modal
  python scripts/test_sep_loc34_provider.py --provider runpod \
      --runpod-endpoint-id <id> --runpod-api-key <key>
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import soundfile as sf

from tenepal.runtime import InferenceRequest, create_provider


def _load_first_60s_wav(path: Path) -> bytes:
    data, sr = sf.read(str(path))
    chunk = data[: 60 * sr]
    buf = io.BytesIO()
    sf.write(buf, chunk, sr, format="WAV")
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["modal", "runpod"], default="modal")
    parser.add_argument("--runpod-endpoint-id", default=None)
    parser.add_argument("--runpod-api-key", default=None)
    args = parser.parse_args()

    input_vocals = Path("validation_video/La-Otra-Conquista_34m15-44m25.vocals.wav")
    input_raw = Path("validation_video/La-Otra-Conquista_34m15-44m25.wav")

    if not input_vocals.exists():
        raise SystemExit(f"Missing input file: {input_vocals}")

    provider_kwargs = {}
    if args.provider == "runpod":
        if args.runpod_endpoint_id:
            provider_kwargs["endpoint_id"] = args.runpod_endpoint_id
        if args.runpod_api_key:
            provider_kwargs["api_key"] = args.runpod_api_key
    provider = create_provider(args.provider, **provider_kwargs)

    out_dir = Path("eq_comparison_results/sep_test_loc34")
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_bytes = input_vocals.read_bytes()
    sep_req = InferenceRequest(
        operation="separate_voices_sepformer",
        payload={
            "audio_bytes": audio_bytes,
            "filename": "LOC_34-44.wav",
            "start_s": 0.0,
            "end_s": 60.0,
        },
    )
    sep_result = provider.run(sep_req).output

    print(f"SepFormer: {sep_result['num_sources']} sources")
    for src in sep_result["sources"]:
        spk = src.get("speaker_id", "SPEAKER_XX")
        wav_path = out_dir / f"sepformer_{spk}.wav"
        wav_path.write_bytes(src["wav_bytes"])
        print(f"  {spk}: {src.get('duration_s', 0.0):.1f}s -> {wav_path}")

    vv_bytes = _load_first_60s_wav(input_raw if input_raw.exists() else input_vocals)
    vv_req = InferenceRequest(
        operation="run_vibevoice",
        payload={
            "audio_bytes": vv_bytes,
            "filename": "LOC_34-44_60s.wav",
            "context_info": (
                "Scene from La Otra Conquista (1999), 16th-century Mexico. "
                "Spanish colonial dialogue with occasional Nahuatl phrases. "
                "Names: Topiltzin, Fray Diego, Tecuichpo. "
                "Nahuatl words: Tonantzin, tlatoani, teotl."
            ),
            "chunk_offset_s": 0.0,
        },
    )
    vv_result = provider.run(vv_req).output

    vv_json_path = out_dir / "vibevoice_loc34_60s.json"
    vv_save = {k: v for k, v in vv_result.items() if k != "raw_output"}
    vv_json_path.write_text(json.dumps(vv_save, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VibeVoice segments: {vv_result.get('num_segments', 0)} -> {vv_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


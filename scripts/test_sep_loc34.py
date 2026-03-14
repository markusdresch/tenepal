#!/usr/bin/env python3
"""Test SepFormer + VibeVoice on LOC 34-44 first 60s.

Run:  modal run scripts/test_sep_loc34.py
"""
import modal
import json
import os
import sys
from pathlib import Path

# Import the app from tenepal_modal
sys.path.insert(0, str(Path(__file__).parent.parent))
from tenepal_modal import (
    app, separate_voices_sepformer, run_vibevoice,
)


@app.local_entrypoint()
def test_sep_loc34():
    input_wav = Path("validation_video/La-Otra-Conquista_34m15-44m25.vocals.wav")
    if not input_wav.exists():
        print(f"ERROR: {input_wav} not found")
        return

    audio_bytes = input_wav.read_bytes()
    out_dir = Path("eq_comparison_results/sep_test_loc34")
    out_dir.mkdir(exist_ok=True)

    # --- SepFormer: first 60s ---
    print("=" * 60)
    print("SEPFORMER on first 60s...")
    print("=" * 60)
    sep_result = separate_voices_sepformer.remote(
        audio_bytes=audio_bytes,
        filename="LOC_34-44.wav",
        start_s=0.0,
        end_s=60.0,
    )

    print(f"SepFormer: {sep_result['num_sources']} sources, SR={sep_result['sample_rate']}Hz")
    for src in sep_result["sources"]:
        spk = src["speaker_id"]
        wav_path = out_dir / f"sepformer_{spk}.wav"
        wav_path.write_bytes(src["wav_bytes"])
        print(f"  {spk}: {src['duration_s']:.1f}s → {wav_path}")

    # --- VibeVoice: first 60s ---
    print()
    print("=" * 60)
    print("VIBEVOICE on first 60s...")
    print("=" * 60)

    # VibeVoice needs raw audio (not vocals) for diarization
    raw_wav = Path("validation_video/La-Otra-Conquista_34m15-44m25.wav")
    if raw_wav.exists():
        # Extract first 60s
        import soundfile as sf
        import numpy as np
        import io

        data, sr = sf.read(str(raw_wav))
        chunk = data[:60 * sr]
        buf = io.BytesIO()
        sf.write(buf, chunk, sr, format="WAV")
        vv_bytes = buf.getvalue()
    else:
        # Fallback to vocals
        import soundfile as sf
        import numpy as np
        import io

        data, sr = sf.read(str(input_wav))
        chunk = data[:60 * sr]
        buf = io.BytesIO()
        sf.write(buf, chunk, sr, format="WAV")
        vv_bytes = buf.getvalue()

    vv_result = run_vibevoice.remote(
        audio_bytes=vv_bytes,
        filename="LOC_34-44_60s.wav",
        context_info=(
            "Scene from La Otra Conquista (1999), 16th-century Mexico. "
            "Spanish colonial dialogue with occasional Nahuatl phrases. "
            "Names: Topiltzin, Fray Diego, Tecuichpo. "
            "Nahuatl words: Tonantzin, tlatoani, teotl."
        ),
        chunk_offset_s=0.0,
    )

    print(f"\nVibeVoice: {vv_result['num_segments']} segments")
    vv_json_path = out_dir / "vibevoice_loc34_60s.json"
    # Remove raw_output for smaller file
    vv_save = {k: v for k, v in vv_result.items() if k != "raw_output"}
    vv_json_path.write_text(json.dumps(vv_save, indent=2, ensure_ascii=False))
    print(f"  Saved: {vv_json_path}")

    for seg in vv_result["segments"]:
        spk = seg.get("speaker_id", "?")
        print(f"  [{seg['start_time']:5.1f}-{seg['end_time']:5.1f}s] "
              f"SPK={spk}: {seg['text'][:60]}")

    print("\n✅ Done. Check:", out_dir)

#!/usr/bin/env python3
"""Test SepFormer + VibeVoice on LOC 34-44 error region (190-350s).

8 of 13 SPA→NAH errors are in this range:
  cue 56 @ 191s, cue 65 @ 218s, cue 71 @ 235s, cue 74 @ 242s,
  cue 76 @ 276s, cue 81 @ 299s, cue 82 @ 303s, cue 83 @ 306s

Run:  modal run scripts/test_sep_loc34_errors.py::test_sep_errors
"""
import modal
import json
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tenepal_modal import (
    app, separate_voices_sepformer, run_vibevoice,
)


@app.local_entrypoint()
def test_sep_errors():
    import soundfile as sf
    import numpy as np

    vocals_path = Path("validation_video/La-Otra-Conquista_34m15-44m25.vocals.wav")
    raw_path = Path("validation_video/La-Otra-Conquista_34m15-44m25.wav")

    out_dir = Path("eq_comparison_results/sep_test_loc34_errors")
    out_dir.mkdir(exist_ok=True)

    # Error cues with timestamps (relative to clip start)
    error_cues = [
        (56, 191.0, 192.5),
        (65, 217.7, 218.9),
        (71, 234.7, 235.6),
        (74, 242.3, 244.4),
        (76, 275.8, 277.0),
        (81, 298.7, 299.3),
        (82, 302.5, 304.6),
        (83, 306.0, 306.8),
        (93, 336.6, 337.7),
        (95, 345.9, 348.5),
    ]

    # --- SepFormer in 30s chunks around error regions ---
    print("=" * 60)
    print("SEPFORMER on error regions (30s chunks)")
    print("=" * 60)
    vocals_bytes = vocals_path.read_bytes()

    # Group errors into 30s windows
    windows = []
    for cue, start, end in error_cues:
        # 30s window centered on error, with 5s context each side
        w_start = max(0, start - 10)
        w_end = w_start + 30
        # Check if fits in existing window
        merged = False
        for i, (ws, we, cues) in enumerate(windows):
            if w_start < we and w_end > ws:  # overlap
                windows[i] = (min(ws, w_start), max(we, w_end), cues + [(cue, start, end)])
                merged = True
                break
        if not merged:
            windows.append((w_start, w_end, [(cue, start, end)]))

    # Launch SepFormer calls in parallel
    sep_handles = []
    for w_start, w_end, cues in windows:
        cue_ids = [c[0] for c in cues]
        print(f"  Launching SepFormer {w_start:.0f}-{w_end:.0f}s (cues {cue_ids})")
        h = separate_voices_sepformer.spawn(
            audio_bytes=vocals_bytes,
            filename="LOC_34-44.wav",
            start_s=w_start,
            end_s=w_end,
        )
        sep_handles.append((w_start, w_end, cues, h))

    # Collect results
    for w_start, w_end, cues, h in sep_handles:
        try:
            sep_result = h.get()
            cue_ids = [c[0] for c in cues]
            print(f"\n  Window {w_start:.0f}-{w_end:.0f}s (cues {cue_ids}):")
            print(f"    {sep_result['num_sources']} sources, SR={sep_result['sample_rate']}Hz")
            for src in sep_result["sources"]:
                spk = src["speaker_id"]
                tag = f"cues{'_'.join(str(c) for c in cue_ids)}"
                wav_path = out_dir / f"sepformer_{w_start:.0f}s_{spk}.wav"
                wav_path.write_bytes(src["wav_bytes"])
                print(f"    {spk}: {src['duration_s']:.1f}s → {wav_path}")
        except Exception as e:
            print(f"  Window {w_start:.0f}-{w_end:.0f}s FAILED: {e}")

    # --- VibeVoice on raw audio (needs full acoustic context) ---
    # Cover 180-360s in one 3-minute chunk
    VV_START = 180.0
    VV_END = 360.0
    print()
    print("=" * 60)
    print(f"VIBEVOICE on {VV_START:.0f}-{VV_END:.0f}s (error region)")
    print("=" * 60)

    raw_data, raw_sr = sf.read(str(raw_path))
    chunk = raw_data[int(VV_START * raw_sr):int(VV_END * raw_sr)]
    buf = io.BytesIO()
    sf.write(buf, chunk, raw_sr, format="WAV")
    vv_bytes = buf.getvalue()

    vv_result = run_vibevoice.remote(
        audio_bytes=vv_bytes,
        filename="LOC_34-44_errors.wav",
        context_info=(
            "Scene from La Otra Conquista (1999), 16th-century Mexico. "
            "Multiple speakers, often overlapping. Spanish colonial dialogue "
            "with occasional Nahuatl phrases. "
            "Names: Topiltzin, Fray Diego, Tecuichpo, Capitán Cristóbal. "
            "Nahuatl words: Tonantzin, tlatoani, teotl, tlacameh."
        ),
        chunk_offset_s=VV_START,
    )

    vv_json_path = out_dir / "vibevoice_errors.json"
    vv_save = {k: v for k, v in vv_result.items() if k != "raw_output"}
    vv_json_path.write_text(json.dumps(vv_save, indent=2, ensure_ascii=False))

    print(f"\nVibeVoice: {vv_result['num_segments']} segments")
    for seg in vv_result["segments"]:
        spk = seg.get("speaker_id", "?")
        abs_start = seg["start_time"]
        abs_end = seg["end_time"]
        print(f"  [{abs_start:6.1f}-{abs_end:6.1f}s] SPK={spk}: {seg['text'][:70]}")

    print(f"\n✅ Done. Results in: {out_dir}")

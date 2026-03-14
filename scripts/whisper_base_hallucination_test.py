#!/usr/bin/env python3
"""Whisper-base hallucination test on NAH regression fixture.

Runs faster-whisper with 'base' model on the 25 NAH segments from
hernan_nah_25.json and compares language detection distribution against
the existing tiny-model results.

Usage:
    ./venv/bin/python scripts/whisper_base_hallucination_test.py
    ./venv/bin/python scripts/whisper_base_hallucination_test.py --model small
    ./venv/bin/python scripts/whisper_base_hallucination_test.py --model medium
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

FIXTURE = REPO_ROOT / "tests/regression/fixtures/hernan_nah_25.json"
TINY_RESULTS = REPO_ROOT / "tests/regression/reports/hernan_nah_25_baseline.json"


def wilson_ci(successes: int, total: int, z: float = 1.96):
    if total == 0:
        return (0.0, 1.0)
    n, p, z2 = total, successes / total, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def run_whisper_on_segment(audio_path: Path, start_s: float, end_s: float, model, pad_s: float = 0.1):
    """Slice segment from WAV and detect language with Whisper."""
    try:
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio[:, 0]

        start_sample = max(0, int((start_s - pad_s) * sr))
        end_sample = min(len(audio), int((end_s + pad_s) * sr))
        segment_audio = audio[start_sample:end_sample].astype(np.float32)

        if len(segment_audio) < sr * 0.3:
            return "too_short"

        # Write to temp WAV and run Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        sf.write(tmp_path, segment_audio, sr)
        _, info = model.transcribe(tmp_path, language=None, vad_filter=False)
        Path(tmp_path).unlink(missing_ok=True)
        return info.language

    except Exception as e:
        return f"error:{e}"


def main():
    parser = argparse.ArgumentParser(description="Whisper hallucination test — model comparison")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--output", default=None, help="JSON output path (default: auto)")
    args = parser.parse_args()

    print(f"Loading fixture: {FIXTURE}")
    fixture = json.loads(FIXTURE.read_text())
    segments = fixture["segments"]
    print(f"  {len(segments)} NAH segments")

    print(f"\nLoading faster-whisper ({args.model})...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
    except ImportError:
        print("ERROR: faster_whisper not installed. Run: pip install faster-whisper")
        sys.exit(1)

    results = []
    lang_counts: dict[str, int] = {}
    errors = 0

    for i, seg in enumerate(segments, 1):
        audio_path = REPO_ROOT / seg["source_audio"]
        seg_id = seg["id"]
        start_s, end_s = seg["start_s"], seg["end_s"]
        duration = end_s - start_s

        if not audio_path.exists():
            print(f"  [{i:2d}/{len(segments)}] SKIP — audio missing: {audio_path.name}")
            errors += 1
            continue

        detected = run_whisper_on_segment(audio_path, start_s, end_s, model)
        lang_counts[detected] = lang_counts.get(detected, 0) + 1

        marker = "✓" if detected not in ("nah",) else "NAH(!"
        print(f"  [{i:2d}/{len(segments)}] {seg_id} ({duration:.1f}s) → {detected} {marker}")
        results.append({"id": seg_id, "start_s": start_s, "end_s": end_s,
                         "duration_s": round(duration, 3), "whisper_lang": detected})

    # Stats
    n_total = len(results)
    n_nah_codes = sum(lang_counts.get(c, 0) for c in ("nah", "nah-naz"))
    n_hallucinated = n_total - n_nah_codes
    hall_rate = n_hallucinated / n_total if n_total > 0 else 0.0
    hall_ci = wilson_ci(n_hallucinated, n_total)

    print(f"\n{'='*50}")
    print(f"Model: {args.model}  |  N={n_total}  |  Errors={errors}")
    print(f"Hallucination rate: {hall_rate*100:.1f}%  (95% CI: {hall_ci[0]*100:.1f}%-{hall_ci[1]*100:.1f}%)")
    print()
    print(f"{'Language':<20} {'Count':>5}  {'%':>6}  {'95% CI'}")
    print("-" * 50)
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = count / n_total * 100 if n_total else 0
        ci = wilson_ci(count, n_total)
        print(f"  {lang:<18} {count:>5}  {pct:>5.1f}%  [{ci[0]*100:.1f}%-{ci[1]*100:.1f}%]")

    # Load tiny results for comparison
    if TINY_RESULTS.exists():
        print(f"\n{'='*50}")
        print("Comparison with tiny model (from existing baseline):")
        tiny = json.loads(TINY_RESULTS.read_text())
        tiny_langs = tiny.get("whisper_hallucination_langs", {})
        if tiny_langs:
            for lang, info in sorted(tiny_langs.items(), key=lambda x: -x[1].get("count", 0)):
                tiny_count = info.get("count", 0)
                base_count = lang_counts.get(lang, 0)
                print(f"  {lang:<18} tiny={tiny_count:>3}  {args.model}={base_count:>3}")

    # Save output
    output_path = args.output or str(REPO_ROOT / f"tests/regression/reports/hernan_nah_25_{args.model}.json")
    output = {
        "model": args.model,
        "n_segments": n_total,
        "n_errors": errors,
        "hallucination_rate": round(hall_rate, 4),
        "hallucination_ci_95": hall_ci,
        "lang_counts": lang_counts,
        "per_segment": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()

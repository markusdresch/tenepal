#!/usr/bin/env python3
"""Window sweep test for "Capitán" segment on Modal GPU.

Sweeps audio window offsets to find optimal framing for Allosaurus and wav2vec2.
Target IPA: k a p i t a n (7 phonemes)

Usage:
    modal run scripts/window_sweep_capitan.py
"""

import modal
import time

# Modal setup
app = modal.App("window-sweep-capitan")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install(
        "allosaurus",
        "soundfile",
        "numpy",
        "torch",
        "torchaudio",
        "transformers",
        "phonemizer",
    )
)

volume = modal.Volume.from_name("tenepal-models", create_if_missing=True)
CACHE_DIR = "/tenepal-models"

# Capitán segment from E01 (from last Modal run)
SEGMENT_START = 35.991
SEGMENT_END = 36.531
TARGET_IPA = ["k", "a", "p", "i", "t", "a", "n"]


def score_ipa(ipa_str: str, target: list[str]) -> tuple[float, int]:
    """Score IPA output against target sequence.

    Returns: (score 0-1, matching_phones)
    """
    if not ipa_str:
        return 0.0, 0

    phones = ipa_str.lower().split()

    # Normalize phones
    normalize = {
        "ɑ": "a", "ɒ": "a", "æ": "a", "ʌ": "a",
        "kʰ": "k", "pʰ": "p", "tʰ": "t",
        "ɪ": "i", "iː": "i",
        "ɾ": "r", "ɹ": "r",
    }
    phones = [normalize.get(p, p) for p in phones]

    # Count matching phones (order-aware subsequence)
    matches = 0
    target_idx = 0
    for phone in phones:
        if target_idx < len(target) and phone == target[target_idx]:
            matches += 1
            target_idx += 1

    score = matches / len(target)
    return score, matches


@app.function(
    image=image,
    gpu="T4",
    volumes={CACHE_DIR: volume},
    timeout=900,
)
def run_window_sweep(audio_bytes: bytes, sr: int):
    """Run window sweep on Modal GPU."""
    import numpy as np
    import soundfile as sf
    import torch
    import io
    from pathlib import Path

    print("Loading audio...")
    audio_data, audio_sr = sf.read(io.BytesIO(audio_bytes))
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    print(f"Audio: {len(audio_data)/audio_sr:.1f}s, sr={audio_sr}")

    # Initialize Allosaurus
    print("Initializing Allosaurus...")
    import allosaurus
    from allosaurus.app import read_recognizer
    from pathlib import Path as AlloPath

    allo_pkg_dir = AlloPath(allosaurus.__file__).parent / "pretrained"
    allo_cache = Path(CACHE_DIR) / "allosaurus" / "pretrained"
    if allo_cache.exists() and not allo_pkg_dir.exists():
        allo_pkg_dir.symlink_to(allo_cache)
    allo_model = read_recognizer()
    print("Allosaurus ready")

    # Initialize wav2vec2
    print("Initializing wav2vec2...")
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

    model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    w2v2_processor = Wav2Vec2Processor.from_pretrained(model_id)
    w2v2_model = Wav2Vec2ForCTC.from_pretrained(model_id).to("cuda")
    print("wav2vec2 ready")

    def run_allosaurus(start_s, end_s):
        """Run Allosaurus on a segment."""
        import tempfile
        start_idx = int(start_s * audio_sr)
        end_idx = int(end_s * audio_sr)
        chunk = audio_data[start_idx:end_idx]
        if len(chunk) < audio_sr * 0.08:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, audio_sr)
            try:
                result = allo_model.recognize(f.name)
                return result
            except Exception as e:
                return f"ERROR:{e}"

    def run_wav2vec2(start_s, end_s):
        """Run wav2vec2 on a segment."""
        start_idx = int(start_s * audio_sr)
        end_idx = int(end_s * audio_sr)
        chunk = audio_data[start_idx:end_idx]
        if len(chunk) < audio_sr * 0.08:
            return ""

        # Resample if needed (w2v2 expects 16kHz)
        if audio_sr != 16000:
            import torchaudio
            chunk_t = torch.tensor(chunk).unsqueeze(0)
            chunk_t = torchaudio.functional.resample(chunk_t, audio_sr, 16000)
            chunk = chunk_t.squeeze(0).numpy()

        inputs = w2v2_processor(chunk, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            logits = w2v2_model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        result = w2v2_processor.batch_decode(predicted_ids)[0]

        # Clean eSpeak artifacts
        result = result.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
        return result

    # Window sweep parameters
    base_start = SEGMENT_START
    base_end = SEGMENT_END
    offsets = list(range(-200, 220, 20))  # -200ms to +200ms in 20ms steps

    results = []
    total_combos = len(offsets) * len(offsets)
    combo_count = 0
    start_time = time.time()
    max_time = 600  # 10 minutes
    max_combos = 500

    print(f"\nRunning window sweep: {len(offsets)}x{len(offsets)} = {total_combos} combinations")
    print(f"Base window: {base_start:.3f}s - {base_end:.3f}s ({(base_end-base_start)*1000:.0f}ms)")
    print(f"Target IPA: {' '.join(TARGET_IPA)}")
    print()

    for start_offset_ms in offsets:
        for end_offset_ms in offsets:
            # Check limits
            if combo_count >= max_combos:
                print(f"Reached max {max_combos} combinations")
                break
            if time.time() - start_time > max_time:
                print(f"Reached max {max_time}s timeout")
                break

            start = base_start + start_offset_ms / 1000
            end = base_end + end_offset_ms / 1000

            if end - start < 0.1:
                continue  # Skip too short

            combo_count += 1

            # Run backends
            allo_ipa = run_allosaurus(start, end)
            w2v2_ipa = run_wav2vec2(start, end)

            # Score
            allo_score, allo_matches = score_ipa(allo_ipa, TARGET_IPA)
            w2v2_score, w2v2_matches = score_ipa(w2v2_ipa, TARGET_IPA)

            results.append({
                "start_offset_ms": start_offset_ms,
                "end_offset_ms": end_offset_ms,
                "window_ms": round((end - start) * 1000),
                "allo_ipa": allo_ipa,
                "allo_score": allo_score,
                "allo_matches": allo_matches,
                "w2v2_ipa": w2v2_ipa,
                "w2v2_score": w2v2_score,
                "w2v2_matches": w2v2_matches,
            })

            if combo_count % 50 == 0:
                print(f"  {combo_count}/{total_combos} combinations...")
        else:
            continue
        break

    elapsed = time.time() - start_time
    print(f"\nCompleted {combo_count} combinations in {elapsed:.1f}s")

    # Find best results
    best_allo = max(results, key=lambda x: (x["allo_score"], -abs(x["start_offset_ms"])))
    best_w2v2 = max(results, key=lambda x: (x["w2v2_score"], -abs(x["start_offset_ms"])))

    print("\n" + "="*80)
    print("BEST RESULTS")
    print("="*80)
    print(f"\nTarget: {' '.join(TARGET_IPA)}")
    print(f"\nAllosaurus best:")
    print(f"  Window: start{best_allo['start_offset_ms']:+d}ms, end{best_allo['end_offset_ms']:+d}ms ({best_allo['window_ms']}ms)")
    print(f"  IPA: {best_allo['allo_ipa']}")
    print(f"  Score: {best_allo['allo_score']:.2f} ({best_allo['allo_matches']}/7 phones)")

    print(f"\nwav2vec2 best:")
    print(f"  Window: start{best_w2v2['start_offset_ms']:+d}ms, end{best_w2v2['end_offset_ms']:+d}ms ({best_w2v2['window_ms']}ms)")
    print(f"  IPA: {best_w2v2['w2v2_ipa']}")
    print(f"  Score: {best_w2v2['w2v2_score']:.2f} ({best_w2v2['w2v2_matches']}/7 phones)")

    # Top 10 for each backend
    print("\n" + "="*80)
    print("TOP 10 ALLOSAURUS")
    print("="*80)
    top_allo = sorted(results, key=lambda x: -x["allo_score"])[:10]
    print(f"{'Start':>8} {'End':>8} {'Window':>8} {'Score':>6} IPA")
    for r in top_allo:
        print(f"{r['start_offset_ms']:+8d} {r['end_offset_ms']:+8d} {r['window_ms']:>8d} {r['allo_score']:>6.2f} {r['allo_ipa']}")

    print("\n" + "="*80)
    print("TOP 10 WAV2VEC2")
    print("="*80)
    top_w2v2 = sorted(results, key=lambda x: -x["w2v2_score"])[:10]
    print(f"{'Start':>8} {'End':>8} {'Window':>8} {'Score':>6} IPA")
    for r in top_w2v2:
        print(f"{r['start_offset_ms']:+8d} {r['end_offset_ms']:+8d} {r['window_ms']:>8d} {r['w2v2_score']:>6.2f} {r['w2v2_ipa']}")

    return {
        "results": results,
        "best_allo": best_allo,
        "best_w2v2": best_w2v2,
        "combo_count": combo_count,
        "elapsed_s": elapsed,
    }


@app.local_entrypoint()
def main():
    """Run the window sweep on Modal."""
    from pathlib import Path
    import json

    vocals_path = Path("validation_video/Hernán-1-1.vocals.wav")
    if not vocals_path.exists():
        print(f"Vocals file not found: {vocals_path}")
        return

    print(f"Loading vocals: {vocals_path}")
    audio_bytes = vocals_path.read_bytes()
    print(f"Uploading {len(audio_bytes)/1e6:.1f}MB to Modal...")

    result = run_window_sweep.remote(audio_bytes, 16000)

    # Save results
    output_path = Path("validation_video/window_sweep_capitan.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

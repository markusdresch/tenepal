#!/usr/bin/env python3
"""Allosaurus Blank Bias Sweep - Reduce CTC blank dominance.

Test different blank_bias values to recover phonemes.

Usage:
    modal run scripts/allo_bias_sweep.py
"""

import modal
import json
from pathlib import Path

app = modal.App("allo-bias-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install(
        "allosaurus",
        "soundfile",
        "numpy",
        "torch",
    )
)

volume = modal.Volume.from_name("tenepal-models", create_if_missing=True)
CACHE_DIR = "/tenepal-models"

# Test segments from the 5-minute analysis
TEST_SEGMENTS = [
    {"cue": 7, "start": 36.042, "end": 36.497, "expected": "k a p i t a n", "text": "¡Capitán!"},
    {"cue": 8, "start": 37.999, "end": 38.539, "expected": "d o n d e e s t a", "text": "¿Dónde está?"},
    {"cue": 9, "start": 39.619, "end": 40.193, "expected": "n o k j e r e s a l i r", "text": "No quiere salir"},
    {"cue": 4, "start": 31.941, "end": 32.582, "expected": "l i s t o b j e n", "text": "Listo, ¡bien!"},
    {"cue": 15, "start": 60.0, "end": 65.0, "expected": "", "text": "¿Por qué tienes que pelear..."},
]

# Bias values to test
BIAS_VALUES = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, float('inf')]


@app.function(
    image=image,
    gpu="T4",
    volumes={CACHE_DIR: volume},
    timeout=600,
)
def run_bias_sweep(audio_bytes: bytes, segments: list):
    """Run Allosaurus with different blank bias values."""
    import numpy as np
    import soundfile as sf
    import torch
    import io
    import tempfile

    print("Loading audio...")
    audio_data, audio_sr = sf.read(io.BytesIO(audio_bytes))
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    print(f"Audio: {len(audio_data)/audio_sr:.1f}s, sr={audio_sr}")

    # Initialize Allosaurus
    print("\nInitializing Allosaurus...")
    import allosaurus
    from allosaurus.app import read_recognizer
    from allosaurus.audio import read_audio
    from allosaurus.am.utils import move_to_tensor
    from pathlib import Path as AlloPath

    allo_pkg_dir = AlloPath(allosaurus.__file__).parent / "pretrained"
    allo_cache = Path(CACHE_DIR) / "allosaurus" / "pretrained"
    if allo_cache.exists() and not allo_pkg_dir.exists():
        allo_pkg_dir.symlink_to(allo_cache)

    model = read_recognizer()
    print("Allosaurus ready")

    # Get blank index
    phone_list = model.lm.inventory.unit.id_to_unit
    blank_idx = None
    for idx, phone in phone_list.items():
        if phone == "<blk>":
            blank_idx = idx
            break

    print(f"Blank index: {blank_idx}")
    print(f"Phone inventory: {len(phone_list)} phones")

    results = []

    for seg in segments:
        cue = seg["cue"]
        start = seg["start"]
        end = seg["end"]
        expected = seg["expected"]
        text = seg["text"]

        print(f"\n{'='*60}")
        print(f"Cue {cue}: \"{text}\" ({start:.3f}-{end:.3f}s)")
        print(f"Expected: {expected}")
        print(f"{'='*60}")

        # Extract audio chunk
        start_idx = int(start * audio_sr)
        end_idx = int(end * audio_sr)
        chunk = audio_data[start_idx:end_idx]

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, audio_sr)
            temp_path = f.name

        # Get features and logits
        audio_allo = read_audio(temp_path)
        feat = model.pm.compute(audio_allo)
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)

        tensor_feat, tensor_feat_len = move_to_tensor([feats, feat_len], model.config.device_id)

        with torch.no_grad():
            tensor_lprobs = model.am(tensor_feat, tensor_feat_len)

        if model.config.device_id >= 0:
            lprobs = tensor_lprobs.cpu().numpy()[0]
        else:
            lprobs = tensor_lprobs.numpy()[0]

        # Original output (no bias)
        original = model.recognize(temp_path, lang_id='ipa', topk=1)

        seg_result = {
            "cue": cue,
            "text": text,
            "expected": expected,
            "start": start,
            "end": end,
            "original": original,
            "results": {}
        }

        print(f"\n{'Bias':<8} {'Blank%':<8} {'Decoded IPA':<60}")
        print("-" * 80)

        for bias in BIAS_VALUES:
            # Apply bias to logits
            modified_lprobs = lprobs.copy()

            if bias == float('inf'):
                # Remove blank entirely
                modified_lprobs[:, blank_idx] = -float('inf')
            else:
                # Subtract bias from blank logit
                modified_lprobs[:, blank_idx] -= bias

            # Decode: greedy argmax then collapse
            decoded_indices = np.argmax(modified_lprobs, axis=1)

            # Count blank frames
            blank_count = np.sum(decoded_indices == blank_idx)
            blank_pct = 100 * blank_count / len(decoded_indices)

            # Convert to phones and collapse
            phones = []
            prev_idx = -1
            for idx in decoded_indices:
                if idx != prev_idx:
                    if idx != blank_idx:
                        phone = phone_list.get(idx, f"?{idx}")
                        phones.append(phone)
                    prev_idx = idx

            decoded_ipa = " ".join(phones)

            bias_str = "inf" if bias == float('inf') else f"{bias:.1f}"
            print(f"{bias_str:<8} {blank_pct:>5.1f}%   {decoded_ipa[:60]}")

            seg_result["results"][bias_str] = {
                "blank_pct": round(blank_pct, 1),
                "decoded": decoded_ipa,
                "n_phones": len(phones),
            }

        results.append(seg_result)

    return results


@app.local_entrypoint()
def main():
    """Run bias sweep on Modal."""

    vocals_path = Path("validation_video/Hernán-1-1.vocals.wav")
    if not vocals_path.exists():
        print(f"Vocals file not found: {vocals_path}")
        return

    print(f"Loading vocals: {vocals_path}")
    audio_bytes = vocals_path.read_bytes()
    print(f"Uploading {len(audio_bytes)/1e6:.1f}MB to Modal...")

    # Run sweep
    results = run_bias_sweep.remote(audio_bytes, TEST_SEGMENTS)

    # Save results
    output_dir = Path("validation_video/analysis")
    output_file = output_dir / "allo_bias_sweep.txt"

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ALLOSAURUS BLANK BIAS SWEEP RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"Cue {r['cue']}: \"{r['text']}\"\n")
            f.write(f"Time: {r['start']:.3f}-{r['end']:.3f}s\n")
            f.write(f"Expected: {r['expected']}\n")
            f.write(f"Original: {r['original']}\n\n")

            f.write(f"{'Bias':<8} {'Blank%':<8} {'#Phones':<8} {'Decoded IPA'}\n")
            f.write("-" * 80 + "\n")

            for bias_str, data in r["results"].items():
                f.write(f"{bias_str:<8} {data['blank_pct']:>5.1f}%  {data['n_phones']:>5}    {data['decoded']}\n")

            f.write("\n" + "=" * 80 + "\n\n")

        # Summary
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write("Recommended bias: Check which value gives best phoneme match to expected.\n")
        f.write("- bias=0: Current behavior (high blank %)\n")
        f.write("- bias=2-3: Moderate reduction, may recover missed phonemes\n")
        f.write("- bias=inf: No blanks, all phonemes (may be noisy)\n")

    print(f"\nSaved: {output_file}")

    # Also save as JSON
    json_file = output_dir / "allo_bias_sweep.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\nCue {r['cue']}: {r['text']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Original: {r['original']}")
        for bias_str in ["2.0", "3.0", "inf"]:
            if bias_str in r["results"]:
                data = r["results"][bias_str]
                print(f"  bias={bias_str}: {data['decoded'][:50]}... ({data['blank_pct']:.0f}% blank)")


if __name__ == "__main__":
    main()

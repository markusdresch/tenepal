#!/usr/bin/env python3
"""
Cut clean audio clips around detected high-confidence ejectives.

These clips can be used for:
1. Validation/ground truth annotation
2. Training ejective classifiers
3. Demonstrating Maya ejective detection
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for tenepal_modal import
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf


def find_high_confidence_ejectives(audio_path: str, use_w2v2: bool = True) -> list[dict]:
    """Find ejectives where all 3 methods agree."""
    from tenepal_modal import EjectiveDetector

    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    detector = EjectiveDetector(use_w2v2=use_w2v2)
    result = detector.detect_ejectives(audio, sr, 0.0, verbose=False)

    # All-3-agree candidates
    high_conf = [
        c for c in result["candidates"]
        if c["heuristic_ejective"] and c["sklearn_ejective"] and c["w2v2_ejective"]
    ]

    return high_conf, audio, sr


def cut_clip(
    audio: np.ndarray,
    sr: int,
    center_time: float,
    window_s: float = 0.5,
) -> np.ndarray:
    """Cut a clip centered on a timestamp."""
    start_s = max(0, int((center_time - window_s / 2) * sr))
    end_s = min(len(audio), int((center_time + window_s / 2) * sr))
    return audio[start_s:end_s]


def main():
    parser = argparse.ArgumentParser(description="Cut clips around high-confidence ejectives")
    parser.add_argument("--input-dir", default="validation_video/maya_samples/",
                       help="Directory with vocals.wav files")
    parser.add_argument("--output-dir", default="validation_video/ejective_clips/",
                       help="Output directory for clips")
    parser.add_argument("--window", type=float, default=0.5,
                       help="Clip duration in seconds (centered on ejective)")
    parser.add_argument("--no-w2v2", action="store_true",
                       help="Disable wav2vec2 (faster, less accurate)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = sorted(input_path.glob("*.vocals.wav"))
    print(f"Processing {len(samples)} samples...")

    manifest = []
    total_clips = 0

    for sample in samples:
        print(f"\n{sample.stem}:")
        high_conf, audio, sr = find_high_confidence_ejectives(
            str(sample), use_w2v2=not args.no_w2v2
        )

        if not high_conf:
            print("  No high-confidence ejectives found")
            continue

        base_name = sample.stem.replace(".vocals", "")

        for i, c in enumerate(high_conf):
            clip = cut_clip(audio, sr, c["local_time"], args.window)

            clip_name = f"{base_name}_ej{i+1:02d}_{c['local_time']:.2f}s.wav"
            clip_path = output_path / clip_name

            sf.write(str(clip_path), clip, sr)

            clip_info = {
                "source": sample.name,
                "clip": clip_name,
                "time": c["time"],
                "local_time": c["local_time"],
                "features": {
                    "closure_ms": c["closure_ms"],
                    "burst_rel_db": c["burst_rel_db"],
                    "vot_ms": c["vot_ms"],
                    "has_creak": c["has_creak"],
                    "f0_drop": c["f0_drop"],
                },
                "votes": {
                    "heuristic": c["heuristic_ejective"],
                    "sklearn": c["sklearn_ejective"],
                    "w2v2": c["w2v2_ejective"],
                }
            }
            manifest.append(clip_info)
            total_clips += 1

            print(f"  {c['local_time']:.2f}s → {clip_name}")

    # Save manifest (convert numpy types to Python types)
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return obj

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(convert_types(manifest), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Extracted {total_clips} high-confidence ejective clips")
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Allosaurus CTC Debug + Parselmouth Acoustic Analysis on Modal GPU.

Deep dive into WHY Allosaurus fails on clear speech segments.
Extracts frame-level CTC logits and compares with Parselmouth acoustic features.

Usage:
    modal run scripts/allo_debug_parselmouth.py
"""

import modal
import json
from pathlib import Path

app = modal.App("allo-debug-parselmouth")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install(
        "allosaurus",
        "soundfile",
        "numpy",
        "torch",
        "praat-parselmouth",
    )
)

volume = modal.Volume.from_name("tenepal-models", create_if_missing=True)
CACHE_DIR = "/tenepal-models"

# Segments to analyze
SEGMENTS = [
    {"name": "Capitán", "start": 35.991, "end": 36.531, "expected": "k a p i t a n"},
    {"name": "Listo_bien", "start": 31.941, "end": 32.582, "expected": "l i s t o b j e n"},
    {"name": "Dónde_está", "start": 37.982, "end": 38.556, "expected": "d o n d e e s t a"},
    {"name": "No_quiere_salir", "start": 39.602, "end": 40.463, "expected": "n o k j e r e s a l i r"},
]


@app.function(
    image=image,
    gpu="T4",
    volumes={CACHE_DIR: volume},
    timeout=900,
)
def analyze_segments(audio_bytes: bytes):
    """Run Allo CTC debug + Parselmouth analysis on Modal GPU."""
    import numpy as np
    import soundfile as sf
    import torch
    import parselmouth
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

    # Get phone inventory
    phone_list = model.lm.inventory.unit.id_to_unit
    print(f"Phone inventory: {len(phone_list)} phones")

    results = {}

    for seg in SEGMENTS:
        print(f"\n{'='*80}")
        print(f"Analyzing: {seg['name']} ({seg['start']:.3f}s - {seg['end']:.3f}s)")
        print(f"Expected: {seg['expected']}")
        print(f"{'='*80}")

        # Extract audio chunk
        start_idx = int(seg["start"] * audio_sr)
        end_idx = int(seg["end"] * audio_sr)
        chunk = audio_data[start_idx:end_idx]
        duration_ms = len(chunk) / audio_sr * 1000

        # === PART A: Allosaurus CTC Logits ===
        print("\n--- Part A: Allosaurus CTC Frame Analysis ---")

        # Save chunk to temp file for Allosaurus
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, audio_sr)
            temp_path = f.name

        # Get features
        audio_allo = read_audio(temp_path)
        feat = model.pm.compute(audio_allo)
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)

        tensor_feat, tensor_feat_len = move_to_tensor([feats, feat_len], model.config.device_id)

        # Get logits (log probabilities)
        with torch.no_grad():
            tensor_lprobs = model.am(tensor_feat, tensor_feat_len)

        if model.config.device_id >= 0:
            lprobs = tensor_lprobs.cpu().numpy()[0]
        else:
            lprobs = tensor_lprobs.numpy()[0]

        num_frames = lprobs.shape[0]
        num_phones = lprobs.shape[1]
        frame_time_ms = duration_ms / num_frames

        print(f"CTC frames: {num_frames}, phones: {num_phones}, frame_time: {frame_time_ms:.1f}ms")

        # Also get the final decoded output
        final_output = model.recognize(temp_path, lang_id='ipa', topk=1)
        print(f"Final decoded: {final_output}")

        # Frame-by-frame analysis
        allo_frames = []
        print(f"\n{'Time':>6} {'Top1':>12} {'Top2':>12} {'Top3':>12} {'Top4':>12} {'Top5':>12}")
        print("-" * 78)

        for frame_idx in range(num_frames):
            frame_lprobs = lprobs[frame_idx]
            # Convert to probabilities
            probs = np.exp(frame_lprobs)
            # Get top 5
            top_indices = np.argsort(probs)[::-1][:5]
            top_probs = probs[top_indices]
            top_phones = [phone_list.get(idx, f"?{idx}") for idx in top_indices]

            time_ms = frame_idx * frame_time_ms

            frame_data = {
                "time_ms": round(time_ms, 1),
                "top5": [(top_phones[i], round(float(top_probs[i]), 4)) for i in range(5)],
            }
            allo_frames.append(frame_data)

            # Print row
            cols = [f"{top_phones[i]}({top_probs[i]:.2f})" for i in range(5)]
            print(f"{time_ms:>6.0f} {cols[0]:>12} {cols[1]:>12} {cols[2]:>12} {cols[3]:>12} {cols[4]:>12}")

        # === PART B: Parselmouth Acoustic Features ===
        print("\n--- Part B: Parselmouth Acoustic Features ---")

        # Create Sound object
        snd = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(audio_sr))

        # Intensity
        intensity = snd.to_intensity(time_step=0.01)
        intensity_times = intensity.xs()
        intensity_values = np.array([intensity.get_value(t) for t in intensity_times])

        # Pitch
        pitch = snd.to_pitch(time_step=0.01)
        pitch_times = pitch.xs()
        pitch_values = np.array([pitch.get_value_at_time(t) for t in pitch_times])

        # Formants
        formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5)
        formant_times = np.arange(0, snd.duration, 0.01)

        pm_frames = []
        print(f"\n{'Time':>6} {'Int(dB)':>8} {'Pitch':>8} {'F1':>8} {'F2':>8} {'Voiced':>8} {'Interp':>15}")
        print("-" * 78)

        for i, t in enumerate(formant_times):
            if t > snd.duration:
                break

            # Intensity at this time
            int_val = intensity.get_value(t) if t <= intensity_times[-1] else 0

            # Pitch at this time
            pitch_val = pitch.get_value_at_time(t)
            voiced = not np.isnan(pitch_val) and pitch_val > 0

            # Formants
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)

            # Interpretation
            if not voiced:
                if int_val > 60:
                    interp = "burst/fric"
                elif int_val > 45:
                    interp = "closure"
                else:
                    interp = "silence"
            else:
                # Identify vowel by F1/F2
                if not np.isnan(f1) and not np.isnan(f2):
                    if f1 > 600 and 1000 < f2 < 1500:
                        interp = "/a/"
                    elif f1 < 400 and f2 > 2000:
                        interp = "/i/"
                    elif f1 < 400 and f2 < 1000:
                        interp = "/u/"
                    elif 400 < f1 < 600 and f2 > 1500:
                        interp = "/e/"
                    elif 400 < f1 < 600 and f2 < 1200:
                        interp = "/o/"
                    else:
                        interp = "vowel?"
                else:
                    interp = "voiced"

            frame_data = {
                "time_ms": round(t * 1000, 1),
                "intensity_db": round(float(int_val), 1) if not np.isnan(int_val) else None,
                "pitch_hz": round(float(pitch_val), 1) if voiced else None,
                "f1_hz": round(float(f1), 1) if not np.isnan(f1) else None,
                "f2_hz": round(float(f2), 1) if not np.isnan(f2) else None,
                "voiced": voiced,
                "interpretation": interp,
            }
            pm_frames.append(frame_data)

            # Print row
            int_str = f"{int_val:.0f}" if not np.isnan(int_val) else "-"
            pitch_str = f"{pitch_val:.0f}" if voiced else "-"
            f1_str = f"{f1:.0f}" if not np.isnan(f1) else "-"
            f2_str = f"{f2:.0f}" if not np.isnan(f2) else "-"
            voiced_str = "yes" if voiced else "no"

            print(f"{t*1000:>6.0f} {int_str:>8} {pitch_str:>8} {f1_str:>8} {f2_str:>8} {voiced_str:>8} {interp:>15}")

        # === Summary ===
        print("\n--- Summary ---")

        # Count blank frames in Allo
        blank_count = sum(1 for f in allo_frames if f["top5"][0][0] == "<blk>")
        blank_pct = 100 * blank_count / len(allo_frames)
        print(f"Allo blank frames: {blank_count}/{len(allo_frames)} ({blank_pct:.1f}%)")

        # Top non-blank phones
        phone_counts = {}
        for f in allo_frames:
            for phone, prob in f["top5"]:
                if phone != "<blk>" and prob > 0.1:
                    phone_counts[phone] = phone_counts.get(phone, 0) + 1
        sorted_phones = sorted(phone_counts.items(), key=lambda x: -x[1])[:10]
        print(f"Top non-blank phones: {sorted_phones}")

        # Parselmouth summary
        voiced_count = sum(1 for f in pm_frames if f["voiced"])
        voiced_pct = 100 * voiced_count / len(pm_frames)
        print(f"Parselmouth voiced frames: {voiced_count}/{len(pm_frames)} ({voiced_pct:.1f}%)")

        results[seg["name"]] = {
            "expected": seg["expected"],
            "final_decoded": final_output,
            "duration_ms": duration_ms,
            "allo_frames": allo_frames,
            "pm_frames": pm_frames,
            "blank_pct": blank_pct,
            "voiced_pct": voiced_pct,
        }

    return results


@app.local_entrypoint()
def main():
    """Run analysis on Modal."""
    vocals_path = Path("validation_video/Hernán-1-1.vocals.wav")
    if not vocals_path.exists():
        print(f"Vocals file not found: {vocals_path}")
        return

    print(f"Loading vocals: {vocals_path}")
    audio_bytes = vocals_path.read_bytes()
    print(f"Uploading {len(audio_bytes)/1e6:.1f}MB to Modal...")

    results = analyze_segments.remote(audio_bytes)

    # Save results
    output_dir = Path("validation_video/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "allo_pm_debug.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Create text report
    report_path = output_dir / "capitan_debug.txt"
    with open(report_path, "w") as f:
        for name, data in results.items():
            f.write(f"{'='*80}\n")
            f.write(f"Segment: {name}\n")
            f.write(f"Expected: {data['expected']}\n")
            f.write(f"Decoded: {data['final_decoded']}\n")
            f.write(f"Duration: {data['duration_ms']:.0f}ms\n")
            f.write(f"Blank%: {data['blank_pct']:.1f}%\n")
            f.write(f"Voiced%: {data['voiced_pct']:.1f}%\n")
            f.write(f"{'='*80}\n\n")

            # Allo frames
            f.write("ALLOSAURUS CTC FRAMES:\n")
            f.write(f"{'Time':>6} {'Top1':>15} {'Top2':>15} {'Top3':>15}\n")
            f.write("-" * 60 + "\n")
            for frame in data["allo_frames"]:
                t = frame["time_ms"]
                tops = frame["top5"][:3]
                cols = [f"{p}({prob:.2f})" for p, prob in tops]
                f.write(f"{t:>6.0f} {cols[0]:>15} {cols[1]:>15} {cols[2]:>15}\n")
            f.write("\n")

            # PM frames
            f.write("PARSELMOUTH ACOUSTIC FEATURES:\n")
            f.write(f"{'Time':>6} {'Int':>6} {'Pitch':>6} {'F1':>6} {'F2':>6} {'Interp':>12}\n")
            f.write("-" * 50 + "\n")
            for frame in data["pm_frames"]:
                t = frame["time_ms"]
                int_v = frame["intensity_db"] or 0
                pitch_v = frame["pitch_hz"] or 0
                f1_v = frame["f1_hz"] or 0
                f2_v = frame["f2_hz"] or 0
                interp = frame["interpretation"]
                f.write(f"{t:>6.0f} {int_v:>6.0f} {pitch_v:>6.0f} {f1_v:>6.0f} {f2_v:>6.0f} {interp:>12}\n")
            f.write("\n\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()

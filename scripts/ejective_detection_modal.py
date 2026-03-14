#!/usr/bin/env python3
"""
Modal GPU-accelerated ejective detection using wav2vec2.

Run locally: python scripts/ejective_detection_modal.py --input <audio>
Run on Modal: modal run scripts/ejective_detection_modal.py --input <audio>
"""

import json
import os
import sys
from pathlib import Path

import modal

# Modal app setup
app = modal.App("ejective-detection")

# GPU image with transformers + torch
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "soundfile",
        "numpy",
        "scikit-learn",
    )
)


@app.function(
    image=gpu_image,
    gpu="T4",  # Cheap GPU, sufficient for wav2vec2-base
    timeout=300,
)
def detect_ejectives_w2v2(
    audio_bytes: bytes,
    sample_rate: int,
    stop_candidates: list[dict],
) -> list[dict]:
    """
    Run wav2vec2 embedding extraction and clustering on GPU.

    Args:
        audio_bytes: Raw audio as bytes (float32)
        sample_rate: Audio sample rate
        stop_candidates: List of {burst_time, ...} dicts from local detection

    Returns:
        List of candidates with w2v2_ejective classification added
    """
    import numpy as np
    import torch
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    # Reconstruct audio array
    audio = np.frombuffer(audio_bytes, dtype=np.float32)

    # Load model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()

    # Extract embeddings for each stop candidate
    embeddings = []
    for c in stop_candidates:
        burst_time = c["burst_time"]

        # Extract 100ms window around burst
        start_sample = max(0, int((burst_time - 0.05) * sample_rate))
        end_sample = min(len(audio), int((burst_time + 0.05) * sample_rate))
        chunk = audio[start_sample:end_sample]

        if len(chunk) < sample_rate * 0.02:  # Too short
            embeddings.append(np.zeros(768))
            continue

        # Process through wav2vec2
        inputs = processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pool over time dimension
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)

    embeddings = np.array(embeddings)

    # Cluster to find ejective-like stops
    if len(embeddings) < 5:
        # Not enough data
        for c in stop_candidates:
            c["w2v2_ejective"] = False
        return stop_candidates

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # K-means with 2 clusters (ejective vs non-ejective)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_scaled)

    # Smaller cluster = ejectives (minority)
    cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
    ejective_cluster = 0 if cluster_sizes[0] < cluster_sizes[1] else 1

    for i, c in enumerate(stop_candidates):
        c["w2v2_ejective"] = bool(labels[i] == ejective_cluster)

    ejective_count = sum(1 for c in stop_candidates if c["w2v2_ejective"])
    print(f"Classified {ejective_count}/{len(stop_candidates)} as ejective-like")

    return stop_candidates


@app.function(
    image=gpu_image,
    gpu="T4",
    timeout=600,
)
def process_audio_batch(audio_files: list[dict]) -> list[dict]:
    """
    Process multiple audio files in one GPU session.

    Args:
        audio_files: List of {name, audio_bytes, sample_rate, candidates}

    Returns:
        List of {name, candidates with w2v2 classifications}
    """
    import numpy as np
    import torch
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running batch on: {device}")
    print(f"Processing {len(audio_files)} files")

    # Load model once
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()

    results = []

    for file_data in audio_files:
        name = file_data["name"]
        audio = np.frombuffer(file_data["audio_bytes"], dtype=np.float32)
        sr = file_data["sample_rate"]
        candidates = file_data["candidates"]

        print(f"\n  Processing {name}: {len(candidates)} candidates")

        if len(candidates) < 5:
            for c in candidates:
                c["w2v2_ejective"] = False
            results.append({"name": name, "candidates": candidates})
            continue

        # Extract embeddings
        embeddings = []
        for c in candidates:
            burst_time = c["burst_time"]
            start_sample = max(0, int((burst_time - 0.05) * sr))
            end_sample = min(len(audio), int((burst_time + 0.05) * sr))
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.02:
                embeddings.append(np.zeros(768))
                continue

            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding)

        embeddings = np.array(embeddings)

        # Cluster
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_scaled)

        cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
        ejective_cluster = 0 if cluster_sizes[0] < cluster_sizes[1] else 1

        for i, c in enumerate(candidates):
            c["w2v2_ejective"] = bool(labels[i] == ejective_cluster)

        ejective_count = sum(1 for c in candidates if c["w2v2_ejective"])
        print(f"    → {ejective_count} ejective-like")

        results.append({"name": name, "candidates": candidates})

    return results


# ============================================================================
# Local detection (heuristic + sklearn) - runs on CPU
# ============================================================================

def run_local_detection(audio_path: str) -> tuple[list[dict], bytes, int]:
    """
    Run heuristic and sklearn detection locally.
    Returns candidates and audio data for Modal.
    """
    import numpy as np
    import parselmouth
    import soundfile as sf
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    # Load audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Parselmouth analysis
    sound = parselmouth.Sound(audio_path)
    intensity = sound.to_intensity(minimum_pitch=50, time_step=0.005)
    pitch = sound.to_pitch_cc(time_step=0.01)

    times = intensity.xs()
    values = np.array([intensity.get_value(t) or 0 for t in times])

    # Derivative-based stop detection
    dt = times[1] - times[0] if len(times) > 1 else 0.005
    derivative = np.gradient(values, dt)

    drop_threshold = -80
    rise_threshold = 80
    min_gap = 0.015
    max_gap = 0.120

    raw_candidates = []
    i = 0
    while i < len(derivative) - 5:
        if derivative[i] < drop_threshold:
            drop_time = times[i]
            drop_intensity = values[i]

            for j in range(i + 3, min(i + int(max_gap / dt), len(derivative))):
                if derivative[j] > rise_threshold:
                    rise_time = times[j]
                    gap = rise_time - drop_time

                    if gap >= min_gap:
                        raw_candidates.append({
                            "drop_time": drop_time,
                            "rise_time": rise_time,
                            "drop_intensity": float(drop_intensity),
                            "rise_intensity": float(values[j]),
                            "gap_ms": gap * 1000
                        })
                        i = j
                        break
                if values[j] < 10:
                    break
        i += 1

    # Deduplicate
    if not raw_candidates:
        return [], audio.tobytes(), sr

    deduped = [raw_candidates[0]]
    for c in raw_candidates[1:]:
        if c["drop_time"] - deduped[-1]["drop_time"] > 0.050:
            deduped.append(c)

    # Extract features for each candidate
    candidates = []
    for rc in deduped:
        burst_time = rc["rise_time"]
        closure_start = rc["drop_time"]
        closure_duration = rc["gap_ms"]

        # Burst intensity
        burst_intensity = intensity.get_value(burst_time) or 60

        # Context intensity
        context_start = max(0, closure_start - 0.1)
        context_intensities = [
            intensity.get_value(t)
            for t in np.linspace(context_start, closure_start, 10)
        ]
        context_intensities = [v for v in context_intensities if v is not None]
        context_mean = np.mean(context_intensities) if context_intensities else 60
        burst_relative = burst_intensity - context_mean

        # VOT estimation
        vot = 0
        for dt_val in np.arange(0.005, 0.060, 0.005):
            t = burst_time + dt_val
            if t < sound.xmax:
                f0 = pitch.get_value_at_time(t)
                if f0 is not None and f0 > 0:
                    vot = dt_val * 1000
                    break

        # Creak detection
        has_creak = False
        f0_values = []
        for dt_val in np.arange(-0.02, 0.02, 0.005):
            t = burst_time + dt_val
            if 0 < t < sound.xmax:
                f0 = pitch.get_value_at_time(t)
                if f0 is not None:
                    f0_values.append(f0)
        if len(f0_values) >= 3:
            has_creak = np.std(f0_values) > 20

        # F0 perturbation
        f0_before = pitch.get_value_at_time(closure_start)
        f0_after = pitch.get_value_at_time(burst_time + 0.03)
        f0_perturbation = 0
        if f0_before and f0_after:
            f0_perturbation = f0_before - f0_after

        candidates.append({
            "burst_time": float(burst_time),
            "closure_duration": float(closure_duration),
            "vot": float(vot),
            "burst_intensity": float(burst_intensity),
            "burst_relative_intensity": float(burst_relative),
            "has_creak": bool(has_creak),
            "f0_perturbation": float(f0_perturbation),
            "heuristic_ejective": False,
            "sklearn_ejective": False,
            "w2v2_ejective": False,
        })

    # Heuristic classification
    for c in candidates:
        score = 0
        if c["closure_duration"] >= 60:
            score += 1
        if c["burst_relative_intensity"] >= 6:
            score += 1
        if 0 < c["vot"] <= 40:
            score += 1
        if c["f0_perturbation"] >= 10:
            score += 1
        if c["has_creak"]:
            score += 1
        c["heuristic_ejective"] = score >= 3

    # Sklearn anomaly detection
    if len(candidates) >= 5:
        features = np.array([
            [c["closure_duration"], c["vot"], c["burst_intensity"],
             c["burst_relative_intensity"], 1.0 if c["has_creak"] else 0.0,
             c["f0_perturbation"]]
            for c in candidates
        ])

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        model = IsolationForest(contamination=0.15, random_state=42, n_estimators=100)
        predictions = model.fit_predict(features_scaled)

        for i, c in enumerate(candidates):
            c["sklearn_ejective"] = predictions[i] == -1

    return candidates, audio.tobytes(), sr


# ============================================================================
# Main entry point
# ============================================================================

@app.local_entrypoint()
def main(
    input_dir: str = "validation_video/maya_samples/",
    use_vocals: bool = True,
):
    """Run full ejective detection comparison with Modal GPU for wav2vec2."""
    import time
    import numpy as np

    sample_path = Path(input_dir)

    # Find wav files
    if use_vocals:
        wav_files = sorted(sample_path.glob("*.vocals.wav"))
    else:
        wav_files = [f for f in sorted(sample_path.glob("*.wav"))
                     if ".vocals." not in f.name]

    if not wav_files:
        print(f"No wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} audio files")
    print(f"Using: {'vocals (isolated)' if use_vocals else 'original audio'}")

    # Run local detection for all files
    print("\n[Local] Running heuristic + sklearn detection...")
    t0 = time.time()

    batch_data = []
    for wav_file in wav_files:
        print(f"  {wav_file.name}...", end=" ", flush=True)
        candidates, audio_bytes, sr = run_local_detection(str(wav_file))
        print(f"{len(candidates)} stops")

        batch_data.append({
            "name": wav_file.name,
            "audio_bytes": audio_bytes,
            "sample_rate": sr,
            "candidates": candidates,
        })

    local_time = time.time() - t0
    print(f"Local detection: {local_time:.1f}s")

    # Run wav2vec2 on Modal GPU (batch)
    print("\n[Modal GPU] Running wav2vec2 embedding clustering...")
    t0 = time.time()

    gpu_results = process_audio_batch.remote(batch_data)

    modal_time = time.time() - t0
    print(f"Modal GPU time: {modal_time:.1f}s")

    # Merge results
    results_by_name = {r["name"]: r["candidates"] for r in gpu_results}

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Ejective Detection Comparison (Modal GPU)")
    print("=" * 80)

    print(f"\n{'File':<30} {'Stops':>6} {'Heur':>6} {'SKL':>6} {'W2V2':>6} {'2+':>6}")
    print("-" * 66)

    totals = {"stops": 0, "heur": 0, "skl": 0, "w2v2": 0, "agree": 0}

    for bd in batch_data:
        name = bd["name"]
        candidates = results_by_name.get(name, bd["candidates"])

        heur = sum(1 for c in candidates if c["heuristic_ejective"])
        skl = sum(1 for c in candidates if c["sklearn_ejective"])
        w2v2 = sum(1 for c in candidates if c["w2v2_ejective"])
        agree = sum(1 for c in candidates
                    if sum([c["heuristic_ejective"], c["sklearn_ejective"],
                            c["w2v2_ejective"]]) >= 2)

        short_name = name.replace(".vocals.wav", "").replace(".wav", "")
        print(f"{short_name:<30} {len(candidates):>6} {heur:>6} {skl:>6} {w2v2:>6} {agree:>6}")

        totals["stops"] += len(candidates)
        totals["heur"] += heur
        totals["skl"] += skl
        totals["w2v2"] += w2v2
        totals["agree"] += agree

    print("-" * 66)
    print(f"{'TOTAL':<30} {totals['stops']:>6} {totals['heur']:>6} "
          f"{totals['skl']:>6} {totals['w2v2']:>6} {totals['agree']:>6}")

    print(f"\nTiming: Local={local_time:.1f}s, Modal GPU={modal_time:.1f}s")

    # Save detailed results (convert numpy bools to Python bools)
    output_path = Path("validation_video/maya_samples/ejective_detection_results.json")

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return obj

    output_data = sanitize({
        "files": [
            {
                "name": bd["name"],
                "candidates": results_by_name.get(bd["name"], bd["candidates"])
            }
            for bd in batch_data
        ],
        "totals": totals,
        "timing": {"local_s": local_time, "modal_gpu_s": modal_time}
    })
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    # Allow running without Modal for testing local detection only
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true",
                        help="Run only local detection (no Modal)")
    parser.add_argument("--input", default="validation_video/maya_samples/",
                        help="Input directory or file")
    args = parser.parse_args()

    if args.local_only:
        # Quick local test
        import soundfile as sf
        sample_path = Path(args.input)

        if sample_path.is_file():
            wav_files = [sample_path]
        else:
            wav_files = sorted(sample_path.glob("*.vocals.wav"))

        for wav_file in wav_files:
            print(f"\n{wav_file.name}")
            candidates, _, _ = run_local_detection(str(wav_file))
            heur = sum(1 for c in candidates if c["heuristic_ejective"])
            skl = sum(1 for c in candidates if c["sklearn_ejective"])
            print(f"  Stops: {len(candidates)}, Heuristic: {heur}, Sklearn: {skl}")
    else:
        print("Run with: modal run scripts/ejective_detection_modal.py")

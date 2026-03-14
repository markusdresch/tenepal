#!/usr/bin/env python3
"""Test pyannote speech-separation-ami-1.0 on validation clip."""

import os
import sys
from pathlib import Path

import scipy.io.wavfile
import torch
import torchaudio
from pyannote.audio import Pipeline


def main():
    # Input file
    input_path = Path("validation_video/separation_comparison/e03_20m10s/original_clip.wav")
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    output_dir = Path("validation_video/separation_comparison/method_test/pyannote_sep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("Loading pyannote speech-separation-ami-1.0...")
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: No HF token found, may fail on gated models")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        use_auth_token=hf_token
    )

    # GPU if available
    if torch.cuda.is_available():
        print("Using GPU")
        pipeline.to(torch.device("cuda"))
    else:
        print("Using CPU (slow)")

    # Load audio
    print(f"Loading {input_path}...")
    waveform, sample_rate = torchaudio.load(str(input_path))
    print(f"  Shape: {waveform.shape}, SR: {sample_rate}")

    # Run separation
    print("Running joint diarization + separation...")
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    with ProgressHook() as hook:
        diarization, sources = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            hook=hook
        )

    # Save diarization
    rttm_path = output_dir / "diarization.rttm"
    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)
    print(f"Saved diarization: {rttm_path}")

    # Print speakers
    speakers = diarization.labels()
    print(f"\nDetected {len(speakers)} speakers:")
    for spk in speakers:
        total = diarization.label_duration(spk)
        print(f"  {spk}: {total:.1f}s")

    # Save separated sources
    print(f"\nSaving {len(speakers)} separated sources...")
    for s, speaker in enumerate(speakers):
        out_path = output_dir / f"{speaker}.wav"
        scipy.io.wavfile.write(str(out_path), 16000, sources.data[:, s])
        print(f"  {out_path}")

    # Stats
    print(f"\nOutput sample rate: 16000 Hz")
    print(f"Sources shape: {sources.data.shape}")

    # Save stats
    import json
    stats = {
        "model": "pyannote/speech-separation-ami-1.0",
        "sample_rate": 16000,
        "num_sources": len(speakers),
        "sources": [
            {
                "speaker_id": spk,
                "duration_s": float(diarization.label_duration(spk))
            }
            for spk in speakers
        ]
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {stats_path}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

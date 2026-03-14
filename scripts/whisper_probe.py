"""Whisper Segment Probe — brute-force decode a single audio segment.

Isolates a time range from the vocals file and runs Whisper with every
combination of parameters to find what produces the best transcription.

Usage:
    modal run whisper_probe.py --input validation_video/Hernán-1-1-1.vocals.wav \
        --start 28.38 --end 28.95 --expected "soldado"

    # Wider window with padding:
    modal run whisper_probe.py --input validation_video/Hernán-1-1-1.vocals.wav \
        --start 28.38 --end 28.95 --expected "soldado" --pad 1.0

    # Also probe a different segment:
    modal run whisper_probe.py --input validation_video/Hernán-1-1-1.vocals.wav \
        --start 38.62 --end 39.11 --expected "capitán"
"""

import modal
import os

tenepal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("faster-whisper>=1.0.0", "numpy", "soundfile", "torch")
)

app = modal.App("whisper-probe", image=tenepal_image)
model_cache = modal.Volume.from_name("tenepal-models", create_if_missing=True)
CACHE_DIR = "/tenepal-models"


@app.function(
    gpu="T4",
    timeout=600,
    volumes={CACHE_DIR: model_cache},
)
def probe_segment(
    audio_bytes: bytes,
    start: float,
    end: float,
    expected: str = "",
    pad: float = 0.0,
):
    """Try every reasonable Whisper config on one segment."""
    import numpy as np
    import soundfile as sf
    import tempfile
    import io
    from faster_whisper import WhisperModel

    # Load audio
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Extract segment with optional padding
    pad_start = max(0, start - pad)
    pad_end = min(len(audio_data) / sr, end + pad)
    i0 = int(pad_start * sr)
    i1 = int(pad_end * sr)
    chunk = audio_data[i0:i1]

    # Also make a padded-with-silence version (gives Whisper clean boundaries)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    chunk_silpad = np.concatenate([silence, chunk, silence])

    dur = len(chunk) / sr
    print(f"═══ Whisper Probe ═══")
    print(f"Segment: {start:.3f}–{end:.3f} ({end-start:.3f}s)")
    print(f"With pad: {pad_start:.3f}–{pad_end:.3f} ({dur:.3f}s)")
    print(f"Expected: '{expected}'")
    print(f"Sample rate: {sr}")
    print()

    # Configs to try
    models = ["medium", "large-v2", "large-v3-turbo"]
    configs = []

    for model_name in models:
        # Base: defaults
        configs.append({
            "model": model_name, "label": f"{model_name} / defaults",
            "vad": True, "beam": 5, "temp": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "prompt": None, "cond_prev": True, "audio": "chunk",
        })
        # No VAD
        configs.append({
            "model": model_name, "label": f"{model_name} / no-vad",
            "vad": False, "beam": 5, "temp": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "prompt": None, "cond_prev": True, "audio": "chunk",
        })
        # No VAD + silence padding
        configs.append({
            "model": model_name, "label": f"{model_name} / no-vad+silpad",
            "vad": False, "beam": 5, "temp": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "prompt": None, "cond_prev": True, "audio": "silpad",
        })
        # No VAD + vocab prompt
        configs.append({
            "model": model_name, "label": f"{model_name} / no-vad+prompt",
            "vad": False, "beam": 5, "temp": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "prompt": "soldado, capitán, Cortés, espada, Moctezuma, Marina, Tlaxcala",
            "cond_prev": False, "audio": "chunk",
        })
        # No VAD + prompt + silence padding
        configs.append({
            "model": model_name, "label": f"{model_name} / no-vad+prompt+silpad",
            "vad": False, "beam": 5, "temp": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "prompt": "soldado, capitán, Cortés, espada, Moctezuma, Marina, Tlaxcala",
            "cond_prev": False, "audio": "silpad",
        })
        # High beam
        configs.append({
            "model": model_name, "label": f"{model_name} / no-vad+beam10",
            "vad": False, "beam": 10, "temp": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "prompt": None, "cond_prev": True, "audio": "chunk",
        })
        # Temperature 0 only (deterministic)
        configs.append({
            "model": model_name, "label": f"{model_name} / no-vad+t0",
            "vad": False, "beam": 5, "temp": 0.0,
            "prompt": None, "cond_prev": True, "audio": "chunk",
        })

    results = []
    current_model = None
    whisper = None

    for cfg in configs:
        # Load model only when switching
        if cfg["model"] != current_model:
            import torch
            del whisper
            torch.cuda.empty_cache()
            print(f"Loading {cfg['model']}...")
            whisper = WhisperModel(
                cfg["model"], device="cuda", compute_type="float16",
                download_root=f"{CACHE_DIR}/whisper",
            )
            current_model = cfg["model"]

        # Pick audio variant
        audio_chunk = chunk_silpad if cfg["audio"] == "silpad" else chunk

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_chunk, sr)
            try:
                kwargs = {
                    "vad_filter": cfg["vad"],
                    "beam_size": cfg["beam"],
                    "temperature": cfg["temp"],
                    "word_timestamps": True,
                    "language": None,
                }
                if cfg["prompt"]:
                    kwargs["initial_prompt"] = cfg["prompt"]
                if not cfg["cond_prev"]:
                    kwargs["condition_on_previous_text"] = False

                segs, info = whisper.transcribe(tmp.name, **kwargs)
                texts = []
                for s in segs:
                    texts.append({
                        "text": s.text.strip(),
                        "lang": info.language,
                        "prob": f"{s.avg_logprob:.3f}",
                        "start": f"{s.start:.2f}",
                        "end": f"{s.end:.2f}",
                    })
            except Exception as e:
                texts = [{"text": f"ERROR: {e}", "lang": "?", "prob": "?"}]
            finally:
                os.unlink(tmp.name)

        # Check match
        all_text = " ".join(t["text"] for t in texts).lower()
        hit = expected.lower() in all_text if expected else False
        marker = "✅" if hit else "  "

        result = {
            "label": cfg["label"],
            "hit": hit,
            "segments": texts,
            "all_text": all_text,
        }
        results.append(result)

        seg_str = " | ".join(
            f"[{t['lang']}:{t['prob']}] {t['text']}" for t in texts
        ) if texts else "(empty)"
        print(f"  {marker} {cfg['label']:40s}  →  {seg_str}")

    # Summary
    hits = [r for r in results if r["hit"]]
    print()
    print(f"═══ Summary: {len(hits)}/{len(results)} configs found '{expected}' ═══")
    if hits:
        print("Winners:")
        for r in hits:
            print(f"  ✅ {r['label']:40s} → {r['all_text']}")
    else:
        print("No config produced the expected text.")
        # Show unique outputs
        unique = set(r["all_text"] for r in results if r["all_text"].strip())
        if unique:
            print("Unique outputs seen:")
            for u in sorted(unique):
                count = sum(1 for r in results if r["all_text"] == u)
                print(f"  ({count}×) {u}")

    return results


@app.local_entrypoint()
def main(
    input: str = "validation_video/Hernán-1-1-1.vocals.wav",
    start: float = 28.38,
    end: float = 28.95,
    expected: str = "soldado",
    pad: float = 0.5,
):
    """Probe a segment with all Whisper configs."""
    input_path = os.path.abspath(input)
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Probing {input} [{start:.2f}–{end:.2f}] for '{expected}' (pad={pad}s)")
    probe_segment.remote(
        audio_bytes=audio_bytes,
        start=start,
        end=end,
        expected=expected,
        pad=pad,
    )

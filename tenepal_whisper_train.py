"""Tenepal Whisper Finetuning Pipeline on Modal.

Downloads Puebla-Nahuatl corpus (OpenSLR 92), finetunes Whisper-large-v3
with PEFT/LoRA, and evaluates WER. Everything runs serverless on Modal.

Data: ~150h of Nahuatl speech from Sierra Norte/Nororiental de Puebla
      (Jonathan Amith, SIL, INALI recordings with ELAN transcriptions)

Usage:
    # Full pipeline:
    modal run tenepal_whisper_train.py

    # Individual phases:
    modal run tenepal_whisper_train.py::download_corpus
    modal run tenepal_whisper_train.py::preprocess
    modal run tenepal_whisper_train.py::run_train --epochs 3
    modal run tenepal_whisper_train.py::run_eval

Cost estimate: ~$5-15 for full pipeline (download + preprocess on CPU,
               training on A10G for ~2-4h, eval on A10G for ~30min)
"""

import modal
import os
import json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenSLR 92 — Puebla-Nahuatl (Amith et al.)
OPENSLR_BASE = "https://openslr.trmal.net/resources/92"
MANIFEST_URL = f"{OPENSLR_BASE}/Puebla-Nahuatl-Manifest.tgz"
SPEECH_TRANS_URL = f"{OPENSLR_BASE}/SpeechTranslation_Nahuatl_Manifest.tgz"
AUDIO_PART_URL = f"{OPENSLR_BASE}/Sound-Files-Puebla-Nahuatl.tgz.part{{:02d}}"
NUM_AUDIO_PARTS = 10

# Volume paths
DATA_DIR = "/data"
RAW_DIR = f"{DATA_DIR}/raw"
MANIFEST_DIR = f"{RAW_DIR}/manifest"
AUDIO_DIR = f"{RAW_DIR}/audio"
TRANS_DIR = f"{RAW_DIR}/transcriptions"
PROCESSED_DIR = f"{DATA_DIR}/processed"
SEGMENTS_DIR = f"{PROCESSED_DIR}/segments"
CHECKPOINT_DIR = f"{DATA_DIR}/checkpoints"
MODEL_DIR = f"{DATA_DIR}/model"
RESULTS_DIR = f"{DATA_DIR}/results"

# Training hyperparameters
DEFAULT_MAX_STEPS = 3000  # ~0.45 epochs of 107K samples (effective batch 16)
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-5
WARMUP_STEPS = 200
LOGGING_STEPS = 50
GRAD_ACCUM_STEPS = 2
EVAL_STEPS = 1500
SAVE_STEPS = 1500

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Model
WHISPER_MODEL = "openai/whisper-large-v3"
PROXY_LANGUAGE = "spanish"  # Nahuatl not in Whisper; Spanish is closest proxy
TARGET_SR = 16000

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("tenepal-whisper")
vol = modal.Volume.from_name("tenepal-data", create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "sox", "wget")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers>=4.40",
        "datasets>=2.18",
        "peft>=0.10",
        "bitsandbytes>=0.43",
        "accelerate>=0.28",
        "evaluate",
        "jiwer",
        "librosa",
        "soundfile",
        "huggingface_hub",
        "numpy",
    )
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: str, desc: str = ""):
    """Download a file with wget."""
    import subprocess

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    label = desc or os.path.basename(dest)
    print(f"  [dl] {label} <- {url}")
    subprocess.run(
        ["wget", "-nv", "-O", dest, url],
        check=True,
    )
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  [dl] {label}: {size_mb:.1f} MB")


def _parse_elan(eaf_path: str) -> list:
    """Parse ELAN .eaf file → list of {start_ms, end_ms, text, tier}."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # Build time slot map
    time_slots = {}
    for ts in root.iter("TIME_SLOT"):
        sid = ts.get("TIME_SLOT_ID")
        val = ts.get("TIME_VALUE")
        if sid and val:
            time_slots[sid] = int(val)

    segments = []
    for tier in root.iter("TIER"):
        tier_id = tier.get("TIER_ID", "")
        # Skip translation/comment tiers
        low = tier_id.lower()
        if any(s in low for s in ["spanish", "español", "translat", "comment", "note", "free"]):
            continue

        for ann in tier.iter("ALIGNABLE_ANNOTATION"):
            ref1 = ann.get("TIME_SLOT_REF1")
            ref2 = ann.get("TIME_SLOT_REF2")
            val_elem = ann.find("ANNOTATION_VALUE")

            if ref1 in time_slots and ref2 in time_slots and val_elem is not None:
                text = (val_elem.text or "").strip()
                if text and len(text) > 1:
                    segments.append({
                        "start_ms": time_slots[ref1],
                        "end_ms": time_slots[ref2],
                        "text": text,
                        "tier": tier_id,
                    })

    return segments


def _get_audio_ref(eaf_path: str) -> str | None:
    """Extract audio filename from an ELAN file's MEDIA_DESCRIPTOR."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(eaf_path)
    for md in tree.getroot().iter("MEDIA_DESCRIPTOR"):
        url = md.get("RELATIVE_MEDIA_URL") or md.get("MEDIA_URL") or ""
        if url:
            return os.path.basename(url.replace("file://", ""))
    return None


def _parse_trs(trs_path: str) -> tuple[str | None, list]:
    """Parse a Transcriber .trs file → (audio_filename, segments).

    The .trs format uses <Sync time="..."/> within <Turn> elements.
    Each sync starts a new segment that runs until the next sync or turn end.
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(trs_path)
    except ET.ParseError:
        return None, []

    root = tree.getroot()
    audio_ref = root.get("audio_filename")  # basename without extension

    segments = []
    for turn in root.iter("Turn"):
        speaker = turn.get("speaker", "")
        turn_end = float(turn.get("endTime", "0"))

        # Collect sync points and text within the turn
        syncs = []
        current_text_parts = []

        for elem in turn:
            if elem.tag == "Sync":
                # Save previous segment if we have text
                if syncs and current_text_parts:
                    text = " ".join(current_text_parts).strip()
                    if text and len(text) > 1:
                        segments.append({
                            "start_ms": int(float(syncs[-1]) * 1000),
                            "end_ms": 0,  # filled below
                            "text": text,
                            "tier": speaker,
                        })
                syncs.append(elem.get("time", "0"))
                current_text_parts = []
                # Text after Sync tag
                if elem.tail and elem.tail.strip():
                    current_text_parts.append(elem.tail.strip())
            elif elem.tag == "Event":
                # Skip events, but capture trailing text
                if elem.tail and elem.tail.strip():
                    current_text_parts.append(elem.tail.strip())
            elif elem.tag == "Comment":
                if elem.tail and elem.tail.strip():
                    current_text_parts.append(elem.tail.strip())

        # Don't forget the last segment in the turn
        if syncs and current_text_parts:
            text = " ".join(current_text_parts).strip()
            if text and len(text) > 1:
                segments.append({
                    "start_ms": int(float(syncs[-1]) * 1000),
                    "end_ms": 0,
                    "text": text,
                    "tier": speaker,
                })

        # Also capture text directly inside Turn (before first Sync)
        if turn.text and turn.text.strip() and not syncs:
            text = turn.text.strip()
            if len(text) > 1:
                start_ms = int(float(turn.get("startTime", "0")) * 1000)
                end_ms = int(turn_end * 1000)
                segments.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "tier": speaker,
                })

    # Fix end_ms: each segment ends where the next begins
    for i in range(len(segments) - 1):
        if segments[i]["end_ms"] == 0:
            segments[i]["end_ms"] = segments[i + 1]["start_ms"]
    # Last segment: try to use a reasonable bound
    if segments and segments[-1]["end_ms"] == 0:
        # Use the last turn's endTime if available, otherwise start + 10s
        last_turn_end = 0
        for turn in root.iter("Turn"):
            t = float(turn.get("endTime", "0"))
            if t > last_turn_end:
                last_turn_end = t
        if last_turn_end > segments[-1]["start_ms"] / 1000:
            segments[-1]["end_ms"] = int(last_turn_end * 1000)
        else:
            segments[-1]["end_ms"] = segments[-1]["start_ms"] + 10000

    # Filter out very short segments
    segments = [s for s in segments if s["end_ms"] - s["start_ms"] >= 500]

    return audio_ref, segments


def _log_tree(root_path: str, max_depth: int = 2, prefix: str = "  "):
    """Log directory structure up to max_depth."""
    from pathlib import Path

    root = Path(root_path)
    if not root.exists():
        print(f"{prefix}{root_path}: does not exist")
        return

    def _walk(p, depth):
        if depth > max_depth:
            return
        entries = sorted(p.iterdir()) if p.is_dir() else []
        for e in entries:
            if e.name.startswith("."):
                continue
            if e.is_dir():
                n_files = sum(1 for _ in e.rglob("*") if _.is_file())
                print(f"{prefix}{'  ' * depth}{e.name}/ ({n_files} files)")
                _walk(e, depth + 1)
            else:
                sz = e.stat().st_size
                unit = "KB" if sz < 1024 * 1024 else "MB"
                val = sz / 1024 if unit == "KB" else sz / (1024 * 1024)
                print(f"{prefix}{'  ' * depth}{e.name} ({val:.1f} {unit})")

    _walk(root, 0)


# ---------------------------------------------------------------------------
# Phase 1: Download corpus
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    timeout=28800,  # 8h — large audio download + extraction
    memory=16384,   # 16 GB
)
def download_corpus():
    """Download OpenSLR 92 manifests, parse transcriptions, download audio."""
    import tarfile
    import subprocess
    from pathlib import Path

    for d in [MANIFEST_DIR, AUDIO_DIR, TRANS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Idempotency: check if we already have matched audio + transcriptions
    done = f"{RAW_DIR}/.download_complete"
    if os.path.exists(done):
        stats = json.loads(Path(done).read_text())
        if stats.get("matched_recordings", 0) > 0:
            print(f"[download] Already complete: {json.dumps(stats, indent=2)}")
            return stats
        # Previous run had 0 matches — stale marker, re-run
        print("[download] Previous run had 0 matches, re-running...")
        os.remove(done)

    # --- 1. Main manifest (874 MB) — already extracted from prior run ---
    manifest_tgz = f"{RAW_DIR}/manifest.tgz"
    if not os.path.exists(f"{MANIFEST_DIR}/.extracted"):
        _download_file(MANIFEST_URL, manifest_tgz, "main manifest (874 MB)")
        print("[download] Extracting main manifest...")
        with tarfile.open(manifest_tgz, "r:gz") as tar:
            tar.extractall(MANIFEST_DIR)
        Path(f"{MANIFEST_DIR}/.extracted").touch()
        vol.commit()

    # --- 2. Parse ELAN transcriptions (.eaf) ---
    elan_files = list(Path(MANIFEST_DIR).rglob("*.eaf"))
    print(f"[download] Found {len(elan_files)} ELAN files")

    all_segments = {}  # audio_ref -> segments
    for eaf in elan_files:
        segs = _parse_elan(str(eaf))
        if not segs:
            continue
        ref = _get_audio_ref(str(eaf)) or eaf.stem
        all_segments[ref] = segs

    elan_segs = sum(len(s) for s in all_segments.values())
    print(f"[download] ELAN: {elan_segs} segments from {len(all_segments)} recordings")

    # --- 3. Parse Transcriber files (.trs) — 954 files, richer coverage ---
    trs_files = list(Path(MANIFEST_DIR).rglob("*.trs"))
    print(f"[download] Found {len(trs_files)} Transcriber files")

    trs_count = 0
    trs_seg_count = 0
    for trs in trs_files:
        audio_ref, segs = _parse_trs(str(trs))
        if not segs or not audio_ref:
            continue
        wav_ref = audio_ref + ".wav"
        # Only add if we don't already have ELAN transcription for this recording
        if wav_ref not in all_segments:
            all_segments[wav_ref] = segs
            trs_count += 1
            trs_seg_count += len(segs)

    total_segs = sum(len(s) for s in all_segments.values())
    total_dur_ms = sum(
        s["end_ms"] - s["start_ms"]
        for segs in all_segments.values() for s in segs
    )
    hours = total_dur_ms / (1000 * 3600)
    print(f"[download] TRS added: {trs_seg_count} segments from {trs_count} recordings")
    print(f"[download] Total: {total_segs} segments, {len(all_segments)} recordings, {hours:.1f}h")

    # --- 4. Save transcription index ---
    with open(f"{TRANS_DIR}/index.json", "w") as f:
        json.dump({"elan_segments": all_segments}, f, ensure_ascii=False)
    vol.commit()

    # --- 5. Download audio parts (split tar, ~84 GB total) ---
    # The manifest only has 151 untranscribed Tepetzintla WAVs.
    # ALL transcribed audio is in the split tar archive.
    existing_wavs = set(p.stem for p in Path(AUDIO_DIR).rglob("*.wav"))
    needed = set(Path(k).stem for k in all_segments.keys())
    matched = needed & existing_wavs

    if len(matched) < len(needed) * 0.5:
        print(f"[download] Only {len(matched)}/{len(needed)} recordings have audio")
        print("[download] Downloading audio parts (split tar, ~84 GB)...")
        parts_dir = f"{RAW_DIR}/audio_parts"
        os.makedirs(parts_dir, exist_ok=True)

        for i in range(NUM_AUDIO_PARTS):
            url = AUDIO_PART_URL.format(i)
            path = f"{parts_dir}/part{i:02d}"
            if os.path.exists(path):
                size_gb = os.path.getsize(path) / (1024**3)
                print(f"  [dl] part{i:02d} exists ({size_gb:.1f} GB)")
                continue
            _download_file(url, path, f"audio part {i:02d}/{NUM_AUDIO_PARTS - 1}")
            vol.commit()

        # Extract all audio (we'll match later in preprocess)
        print("[download] Extracting audio from split tar (this takes a while)...")
        # Use --checkpoint to log progress every 1000 records
        cmd = (
            f"cat {parts_dir}/part* | "
            f"tar xzf - -C {AUDIO_DIR} --checkpoint=1000 "
            f"--checkpoint-action=echo='%u records extracted'"
        )
        subprocess.run(cmd, shell=True, check=True)
        vol.commit()

        # Clean up parts to save ~84 GB
        print("[download] Cleaning up tar parts...")
        import shutil
        shutil.rmtree(parts_dir, ignore_errors=True)
        vol.commit()
    else:
        print(f"[download] {len(matched)}/{len(needed)} recordings already have audio")

    # --- 6. Final match count ---
    all_audio = set(p.stem for p in Path(AUDIO_DIR).rglob("*.wav"))
    final_matched = needed & all_audio
    print(f"[download] Audio files: {len(all_audio)}")
    print(f"[download] Matched recordings: {len(final_matched)}/{len(needed)}")

    # Show unmatched sample
    unmatched = needed - all_audio
    if unmatched:
        sample = list(unmatched)[:5]
        print(f"[download] Unmatched sample: {sample}")

    stats = {
        "elan_files": len(elan_files),
        "trs_files": len(trs_files),
        "total_recordings": len(all_segments),
        "total_segments": total_segs,
        "total_hours": round(hours, 1),
        "audio_files": len(all_audio),
        "matched_recordings": len(final_matched),
        "unmatched_recordings": len(unmatched),
    }

    Path(done).write_text(json.dumps(stats))
    vol.commit()
    print(f"\n[download] COMPLETE:\n{json.dumps(stats, indent=2)}")
    return stats


def _parse_speech_trans(st_dir: str) -> list:
    """Parse the SpeechTranslation_Nahuatl_Manifest for aligned segments.

    Tries common formats: ESPnet-style (text + wav.scp + segments),
    TSV, or line-aligned text files.
    """
    from pathlib import Path

    segments = []
    st_root = Path(st_dir)

    # Look for ESPnet-style text file (uttid <tab> transcription)
    for text_file in st_root.rglob("text"):
        if text_file.is_file():
            with open(text_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        segments.append({"utt_id": parts[0], "text": parts[1]})

    # Look for segments file (uttid recording start end)
    seg_file = st_root / "segments"
    if not seg_file.exists():
        for sf in st_root.rglob("segments"):
            if sf.is_file():
                seg_file = sf
                break

    if seg_file.exists():
        seg_map = {}
        with open(seg_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    seg_map[parts[0]] = {
                        "recording": parts[1],
                        "start_s": float(parts[2]),
                        "end_s": float(parts[3]),
                    }
        for seg in segments:
            if seg["utt_id"] in seg_map:
                seg.update(seg_map[seg["utt_id"]])

    # Fallback: TSV files
    if not segments:
        for tsv in st_root.rglob("*.tsv"):
            with open(tsv) as f:
                header = f.readline().strip().split("\t")
                for line in f:
                    vals = line.strip().split("\t")
                    if len(vals) == len(header):
                        segments.append(dict(zip(header, vals)))

    # Fallback: paired .txt files (nah.txt / es.txt)
    if not segments:
        nah_files = list(st_root.rglob("*nah*")) + list(st_root.rglob("*nahuatl*"))
        for nf in nah_files:
            if nf.suffix == ".txt" and nf.is_file():
                with open(nf) as f:
                    for i, line in enumerate(f):
                        text = line.strip()
                        if text:
                            segments.append({"utt_id": f"{nf.stem}_{i:05d}", "text": text})

    return segments


# ---------------------------------------------------------------------------
# Phase 2: Preprocess
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    timeout=14400,  # 4h — resampling hundreds of hours is CPU-intensive
    memory=32768,   # 32 GB for audio processing
)
def preprocess():
    """Segment audio, resample to 16kHz, build train/dev/test splits."""
    import random
    import soundfile as sf
    import librosa
    from pathlib import Path

    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    # Idempotency
    done = f"{PROCESSED_DIR}/.preprocess_complete"
    if os.path.exists(done):
        stats = json.loads(Path(done).read_text())
        print(f"[preprocess] Already complete: {json.dumps(stats, indent=2)}")
        return stats

    vol.reload()

    # Load transcription index from download phase
    index_path = f"{TRANS_DIR}/index.json"
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"{index_path} not found — run download_corpus first"
        )
    index = json.loads(Path(index_path).read_text())
    elan_segments = index["elan_segments"]

    # Build audio file lookup: filename (with/without ext) -> path
    audio_files = {}
    for p in Path(AUDIO_DIR).rglob("*.wav"):
        audio_files[p.name] = str(p)
        audio_files[p.stem] = str(p)

    print(f"[preprocess] {len(audio_files) // 2} audio files available")
    print(f"[preprocess] {len(elan_segments)} recordings with transcriptions")

    # Process each recording → load once at 16kHz, cut segments
    metadata = []
    seg_id = 0
    skipped_no_audio = 0
    skipped_short = 0
    rec_count = 0

    for audio_ref, segments in elan_segments.items():
        # Resolve audio path
        audio_path = audio_files.get(audio_ref)
        if not audio_path:
            # Try common extensions
            for ext in [".wav", ".WAV", ".mp3"]:
                stem = Path(audio_ref).stem
                if stem in audio_files:
                    audio_path = audio_files[stem]
                    break
        if not audio_path:
            skipped_no_audio += 1
            continue

        # Load and resample entire recording once (much faster than per-chunk)
        try:
            audio_data, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"  [preprocess] WARN: failed to load {audio_ref}: {e}")
            skipped_no_audio += 1
            continue

        total_dur = len(audio_data) / TARGET_SR
        is_presegmented = total_dur < 30.0

        if is_presegmented and len(segments) == 1:
            if total_dur < 0.5:
                skipped_short += 1
                continue

            out_path = f"{SEGMENTS_DIR}/seg_{seg_id:06d}.wav"
            sf.write(out_path, audio_data, TARGET_SR)
            metadata.append({
                "path": out_path,
                "sentence": segments[0]["text"],
                "recording": audio_ref,
                "duration_s": round(total_dur, 2),
            })
            seg_id += 1
        else:
            # Cut segments from already-resampled recording
            for seg in segments:
                start_sample = int(seg["start_ms"] / 1000 * TARGET_SR)
                end_sample = int(seg["end_ms"] / 1000 * TARGET_SR)

                if end_sample <= start_sample:
                    continue

                chunk = audio_data[start_sample:end_sample]
                dur = len(chunk) / TARGET_SR
                if dur < 0.5 or dur > 30.0:
                    skipped_short += 1
                    continue

                out_path = f"{SEGMENTS_DIR}/seg_{seg_id:06d}.wav"
                sf.write(out_path, chunk, TARGET_SR)
                metadata.append({
                    "path": out_path,
                    "sentence": seg["text"],
                    "recording": audio_ref,
                    "duration_s": round(dur, 2),
                })
                seg_id += 1

        rec_count += 1
        if rec_count % 50 == 0:
            print(f"  [preprocess] {rec_count} recordings, {seg_id} segments...")
            # Checkpoint every 200 recordings to avoid losing progress
            if rec_count % 200 == 0:
                vol.commit()

    print(f"[preprocess] {len(metadata)} segments extracted")
    print(f"[preprocess] Skipped: {skipped_no_audio} no audio, {skipped_short} too short/long")

    if not metadata:
        raise RuntimeError("No segments extracted — check audio/transcription alignment")

    # --- Split into train/dev/test (by recording for speaker independence) ---
    random.seed(42)
    recordings = list({m["recording"] for m in metadata})
    random.shuffle(recordings)

    n = len(recordings)
    n_test = max(1, int(n * 0.1))
    n_dev = max(1, int(n * 0.1))
    test_recs = set(recordings[:n_test])
    dev_recs = set(recordings[n_test:n_test + n_dev])

    for m in metadata:
        if m["recording"] in test_recs:
            m["split"] = "test"
        elif m["recording"] in dev_recs:
            m["split"] = "dev"
        else:
            m["split"] = "train"

    # Save metadata
    meta_path = f"{PROCESSED_DIR}/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

    # Stats
    from collections import Counter
    split_counts = Counter(m["split"] for m in metadata)
    split_hours = {}
    for split in ["train", "dev", "test"]:
        dur = sum(m["duration_s"] for m in metadata if m["split"] == split)
        split_hours[split] = round(dur / 3600, 2)

    vocab = set()
    for m in metadata:
        vocab.update(m["sentence"].split())

    stats = {
        "total_segments": len(metadata),
        "splits": dict(split_counts),
        "hours": split_hours,
        "total_hours": round(sum(split_hours.values()), 2),
        "recordings": len(recordings),
        "vocab_size": len(vocab),
        "skipped_no_audio": skipped_no_audio,
        "skipped_short": skipped_short,
    }

    stats_path = f"{PROCESSED_DIR}/stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    Path(done).write_text(json.dumps(stats))
    vol.commit()

    print(f"\n[preprocess] COMPLETE:\n{json.dumps(stats, indent=2)}")
    return stats


# ---------------------------------------------------------------------------
# Phase 3: Train
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    gpu="A100",
    timeout=28800,  # 8h
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    max_steps: int = DEFAULT_MAX_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
):
    """Finetune Whisper-large-v3 with LoRA on Nahuatl."""
    import torch
    import soundfile as sf
    import numpy as np
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from transformers import (
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from peft import LoraConfig, get_peft_model

    vol.reload()

    # Check prerequisites
    meta_path = f"{PROCESSED_DIR}/metadata.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found — run preprocess first")

    metadata = json.loads(Path(meta_path).read_text())
    print(f"[train] Loaded {len(metadata)} segments")

    # --- Load processor ---
    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
    tokenizer = WhisperTokenizer.from_pretrained(
        WHISPER_MODEL, language=PROXY_LANGUAGE, task="transcribe",
    )
    processor = WhisperProcessor.from_pretrained(
        WHISPER_MODEL, language=PROXY_LANGUAGE, task="transcribe",
    )

    # --- Custom PyTorch Dataset for on-the-fly feature extraction ---
    # Avoids ~100GB Arrow cache from HF datasets .map()
    class WhisperNahuatlDataset(torch.utils.data.Dataset):
        def __init__(self, items, feat_extractor, tok, sr):
            self.items = items
            self.feat_extractor = feat_extractor
            self.tok = tok
            self.sr = sr

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            audio, _ = sf.read(item["path"])
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            input_features = self.feat_extractor(
                audio, sampling_rate=self.sr,
            ).input_features[0]
            labels = self.tok(item["sentence"]).input_ids
            return {"input_features": input_features, "labels": labels}

    train_items = [m for m in metadata if m["split"] == "train"]
    dev_items = [m for m in metadata if m["split"] == "dev"]
    train_ds = WhisperNahuatlDataset(train_items, feature_extractor, tokenizer, TARGET_SR)
    dev_ds = WhisperNahuatlDataset(dev_items, feature_extractor, tokenizer, TARGET_SR)
    print(f"[train] Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    # --- Data collator ---
    @dataclass
    class WhisperDataCollator:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_feats = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")

            label_feats = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            # Strip BOS token if prepended
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    # --- Load model (fp16 on A100, ~6.2 GB) ---
    print("[train] Loading Whisper-large-v3 (fp16)...")
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # Configure generation
    model.generation_config.language = PROXY_LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --- Apply LoRA ---
    print("[train] Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training args ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=learning_rate,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        fp16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        remove_unused_columns=False,
        label_names=["labels"],
        # predict_with_generate deferred to Phase 4 eval (too slow for training loop)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    # --- Train ---
    eff_batch = batch_size * GRAD_ACCUM_STEPS
    print(f"[train] Starting training: max_steps={max_steps}, batch_size={batch_size}, lr={learning_rate}")
    print(f"[train] Effective batch: {eff_batch}, steps/epoch: ~{len(train_ds) // eff_batch}")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    # Clean old checkpoints to avoid torch version conflicts
    if os.path.exists(CHECKPOINT_DIR):
        import shutil
        for ckpt_dir in sorted(Path(CHECKPOINT_DIR).glob("checkpoint-*")):
            print(f"[train] Removing old checkpoint: {ckpt_dir.name}")
            shutil.rmtree(ckpt_dir)

    result = trainer.train()
    print(f"[train] Training complete: {result.metrics}")

    # --- Save ---
    # LoRA adapter (small, ~60 MB)
    adapter_dir = f"{CHECKPOINT_DIR}/lora-adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"[train] Saved LoRA adapter to {adapter_dir}")

    # Merged model for inference
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[train] Merging LoRA weights into base model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)
    print(f"[train] Saved merged model to {MODEL_DIR}")

    vol.commit()
    print("[train] COMPLETE")
    return result.metrics


# ---------------------------------------------------------------------------
# Phase 4: Evaluate
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    gpu="A100",
    timeout=14400,  # 4h — 2x 13.7K segments sequential inference
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_model(sample_size: int = 500):
    """Compare baseline Whisper vs. finetuned model on Nahuatl test set.

    Args:
        sample_size: Number of test segments to evaluate (0 = all).
                     Default 500 gives statistically significant results
                     while keeping eval under ~30min.
    """
    import torch
    import random
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    import evaluate as hf_evaluate

    vol.reload()

    # Check prerequisites
    meta_path = f"{PROCESSED_DIR}/metadata.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found — run preprocess first")
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"{MODEL_DIR} not found — run train first")

    metadata = json.loads(Path(meta_path).read_text())
    test_items = [m for m in metadata if m["split"] == "test"]
    print(f"[eval] Full test set: {len(test_items)} segments")

    # Sample for speed
    if sample_size > 0 and sample_size < len(test_items):
        random.seed(42)
        test_items = random.sample(test_items, sample_size)
        print(f"[eval] Sampled {sample_size} segments for evaluation")

    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    references = [m["sentence"] for m in test_items]
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)

    def _transcribe_all(model, items, label, language=None):
        """Transcribe items using direct model inference (no pipeline overhead)."""
        model.eval()
        preds = []
        for i, m in enumerate(items):
            audio, _ = sf.read(m["path"])
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            input_features = processor.feature_extractor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt",
            ).input_features.to("cuda", dtype=torch.float16)

            gen_kwargs = {"task": "transcribe", "max_new_tokens": 225}
            if language:
                gen_kwargs["language"] = language

            with torch.no_grad():
                predicted_ids = model.generate(input_features, **gen_kwargs)

            text = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True,
            )[0].strip()
            preds.append(text)

            if i < 5:
                print(f"  {label} [{i}]: {text[:80]}")
                print(f"  Reference [{i}]: {references[i][:80]}")
            if (i + 1) % 100 == 0:
                print(f"  [eval] {label}: {i + 1}/{len(items)} done")
        return preds

    # --- Baseline: Whisper-large-v3 (no finetuning) ---
    print("[eval] Running baseline Whisper-large-v3...")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL, torch_dtype=torch.float16, device_map="auto",
    )
    baseline_preds = _transcribe_all(baseline_model, test_items, "Baseline")
    del baseline_model
    torch.cuda.empty_cache()

    baseline_wer = wer_metric.compute(predictions=baseline_preds, references=references)
    baseline_cer = cer_metric.compute(predictions=baseline_preds, references=references)
    print(f"[eval] Baseline WER: {baseline_wer:.4f}, CER: {baseline_cer:.4f}")

    # --- Finetuned model ---
    print("[eval] Running finetuned model...")
    finetuned_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="auto",
    )
    finetuned_preds = _transcribe_all(
        finetuned_model, test_items, "Finetuned", language=PROXY_LANGUAGE,
    )
    del finetuned_model
    torch.cuda.empty_cache()

    finetuned_wer = wer_metric.compute(predictions=finetuned_preds, references=references)
    finetuned_cer = cer_metric.compute(predictions=finetuned_preds, references=references)
    print(f"[eval] Finetuned WER: {finetuned_wer:.4f}, CER: {finetuned_cer:.4f}")

    # --- Analyze baseline hallucinations ---
    # Detect common hallucination patterns (e.g., outputs in Danish, Dutch)
    hallucination_analysis = _analyze_hallucinations(baseline_preds)

    # --- Compile results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load training stats if available
    train_stats = {}
    stats_path = f"{PROCESSED_DIR}/stats.json"
    if os.path.exists(stats_path):
        train_stats = json.loads(Path(stats_path).read_text())

    results = {
        "model": WHISPER_MODEL,
        "proxy_language": PROXY_LANGUAGE,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "target_modules": LORA_TARGET_MODULES,
        },
        "dataset": {
            "source": "OpenSLR 92 (Puebla-Nahuatl)",
            **train_stats,
        },
        "baseline": {
            "wer": round(baseline_wer, 4),
            "cer": round(baseline_cer, 4),
            "sample_outputs": baseline_preds[:10],
            "hallucination_analysis": hallucination_analysis,
        },
        "finetuned": {
            "wer": round(finetuned_wer, 4),
            "cer": round(finetuned_cer, 4),
            "sample_outputs": finetuned_preds[:10],
        },
        "improvement": {
            "wer_reduction": round(baseline_wer - finetuned_wer, 4),
            "cer_reduction": round(baseline_cer - finetuned_cer, 4),
            "wer_relative_pct": round((1 - finetuned_wer / max(baseline_wer, 1e-6)) * 100, 1),
        },
        "references_sample": references[:10],
        "test_segments_evaluated": len(test_items),
        "test_segments_total": len([m for m in json.loads(Path(meta_path).read_text()) if m["split"] == "test"]),
        "gpu": "A100",
    }

    results_path = f"{RESULTS_DIR}/eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    vol.commit()

    # Print comparison table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':<15} {'Finetuned':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'WER':<20} {baseline_wer:<15.4f} {finetuned_wer:<15.4f} {baseline_wer - finetuned_wer:<+15.4f}")
    print(f"{'CER':<20} {baseline_cer:<15.4f} {finetuned_cer:<15.4f} {baseline_cer - finetuned_cer:<+15.4f}")
    print("=" * 60)
    print(f"Test segments: {len(test_items)}")
    print(f"Results saved to: {results_path}")
    print("[eval] COMPLETE")

    return results


def _analyze_hallucinations(predictions: list) -> dict:
    """Analyze what Whisper hallucinates when given Nahuatl audio."""
    from collections import Counter

    # Common hallucination language indicators
    lang_markers = {
        "danish": ["og", "det", "er", "jeg", "har", "en", "til", "på"],
        "dutch": ["het", "een", "van", "dat", "niet", "ook", "maar"],
        "norwegian": ["og", "det", "er", "som", "har", "til", "på", "med"],
        "spanish": ["que", "de", "los", "las", "del", "una", "por"],
        "english": ["the", "and", "this", "that", "with", "for", "from"],
        "portuguese": ["que", "não", "uma", "para", "com", "dos"],
    }

    lang_scores = Counter()
    empty_count = 0
    repetition_count = 0

    for pred in predictions:
        if not pred.strip():
            empty_count += 1
            continue

        # Check for repetitive hallucination (same phrase repeated)
        words = pred.lower().split()
        if len(words) >= 6:
            first_three = " ".join(words[:3])
            if pred.lower().count(first_three) >= 3:
                repetition_count += 1

        # Score language markers
        word_set = set(words)
        for lang, markers in lang_markers.items():
            hits = sum(1 for m in markers if m in word_set)
            if hits >= 2:
                lang_scores[lang] += 1

    return {
        "language_distribution": dict(lang_scores.most_common()),
        "empty_outputs": empty_count,
        "repetitive_hallucinations": repetition_count,
        "total_predictions": len(predictions),
    }


# ---------------------------------------------------------------------------
# Phase 5: Cross-Language Robustness Test
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    gpu="A100",
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def cross_language_eval():
    """Test baseline vs finetuned on NAH test samples + edge cases.

    Answers key questions:
    1. Does finetuned model produce Nahuatl-like output for NAH audio?
    2. Does baseline hallucinate random languages?
    3. If SPA/ENG edge cases uploaded, does finetuned model break on them?
    """
    import torch
    import random
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    vol.reload()

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"{MODEL_DIR} not found — run train first")

    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)

    # Collect test files
    test_files = []

    # 1. NAH test samples from corpus
    meta_path = f"{PROCESSED_DIR}/metadata.json"
    if os.path.exists(meta_path):
        metadata = json.loads(Path(meta_path).read_text())
        test_items = [m for m in metadata if m["split"] == "test"]
        random.seed(42)
        nah_sample = random.sample(test_items, min(20, len(test_items)))
        for m in nah_sample:
            test_files.append({"path": m["path"], "label": "NAH", "ref": m["sentence"]})

    # 2. Edge case clips if uploaded
    ec_dir = f"{DATA_DIR}/cross_lang_test"
    if os.path.exists(ec_dir):
        ec_catalog_path = f"{ec_dir}/catalog.json"
        if os.path.exists(ec_catalog_path):
            ec_catalog = json.loads(Path(ec_catalog_path).read_text())
            for ec_id, info in ec_catalog.items():
                wav_path = f"{ec_dir}/{ec_id}.wav"
                if os.path.exists(wav_path):
                    test_files.append({
                        "path": wav_path,
                        "label": info["expected_lang"],
                        "ref": info.get("segment_text", ""),
                    })

    print(f"[cross-lang] Testing {len(test_files)} files")

    def _transcribe(model, audio_path):
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        max_samples = 30 * TARGET_SR
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        input_features = processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt",
        ).input_features.to("cuda", dtype=torch.float16)

        with torch.no_grad():
            # Auto-detect language
            ids_auto = model.generate(input_features, task="transcribe", max_new_tokens=225)
            # Force Spanish proxy
            ids_spa = model.generate(input_features, language=PROXY_LANGUAGE,
                                      task="transcribe", max_new_tokens=225)

        text_auto = processor.tokenizer.batch_decode(ids_auto, skip_special_tokens=True)[0].strip()
        text_spa = processor.tokenizer.batch_decode(ids_spa, skip_special_tokens=True)[0].strip()
        return text_auto, text_spa

    results = []
    for model_name, model_path in [("baseline", WHISPER_MODEL), ("finetuned", MODEL_DIR)]:
        print(f"\n{'='*60}")
        print(f"[cross-lang] {model_name.upper()}")
        print(f"{'='*60}")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto",
        )
        model.eval()

        for tf in test_files:
            text_auto, text_spa = _transcribe(model, tf["path"])
            entry = {
                "model": model_name,
                "file": os.path.basename(tf["path"]),
                "expected": tf["label"],
                "reference": tf["ref"][:100],
                "text_auto": text_auto[:200],
                "text_forced_spa": text_spa[:200],
            }
            results.append(entry)
            print(f"\n  [{tf['label']}] {os.path.basename(tf['path'])}")
            print(f"    ref:  {tf['ref'][:80]}")
            print(f"    auto: {text_auto[:80]}")
            print(f"    spa:  {text_spa[:80]}")

        del model
        torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = f"{RESULTS_DIR}/cross_language_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    vol.commit()

    print(f"\n[cross-lang] Saved to {results_path}")
    print("[cross-lang] COMPLETE")
    return results


# ---------------------------------------------------------------------------
# Phase 6: Edge-Case Checkpoint Evaluation
# ---------------------------------------------------------------------------

EVAL_CLIPS_DIR = f"{DATA_DIR}/eval_clips"


@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    gpu="A100",
    timeout=600,
    memory=32768,
)
def eval_checkpoint_gpu(checkpoint: str = "checkpoint-3000"):
    """Transcribe all clips in eval_clips dir with a specific LoRA checkpoint.

    Returns list of {filename, checkpoint, text} dicts.
    """
    import torch
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    vol.reload()

    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)

    # Load model: either merged model or base + LoRA adapter
    if checkpoint == "model":
        print(f"[eval] Loading merged model from {MODEL_DIR}")
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float16, device_map="auto",
        )
    else:
        ckpt_path = f"{CHECKPOINT_DIR}/{checkpoint}"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"{ckpt_path} not found. Available: "
                + str([p.name for p in Path(CHECKPOINT_DIR).iterdir()])
            )
        print(f"[eval] Loading base model + LoRA from {ckpt_path}")
        from peft import PeftModel
        base_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL, torch_dtype=torch.float16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model = model.merge_and_unload()

    model.eval()

    # Transcribe all clips
    clips_path = Path(EVAL_CLIPS_DIR)
    if not clips_path.exists():
        raise FileNotFoundError(f"{EVAL_CLIPS_DIR} not found — upload clips first")

    wav_files = sorted(clips_path.glob("*.wav"))
    print(f"[eval] {len(wav_files)} clips, checkpoint={checkpoint}")

    results = []
    for wav_path in wav_files:
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        input_features = processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt",
        ).input_features.to("cuda", dtype=torch.float16)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language=PROXY_LANGUAGE,
                task="transcribe",
                max_new_tokens=225,
            )
        text = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True,
        )[0].strip()

        entry = {"checkpoint": checkpoint, "filename": wav_path.name, "text": text}
        results.append(entry)
        print(f"  {wav_path.name}: {text[:80]}")

    print(f"[eval] Done — {len(results)} transcriptions")
    return results


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run the full pipeline: download → preprocess → train → evaluate."""
    print("=" * 60)
    print("Tenepal Whisper Finetuning Pipeline")
    print("=" * 60)

    print("\n[1/4] Downloading corpus...")
    dl_stats = download_corpus.remote()
    print(f"  -> {dl_stats.get('total_segments', '?')} segments, "
          f"{dl_stats.get('total_hours', '?')}h audio")

    print("\n[2/4] Preprocessing...")
    pp_stats = preprocess.remote()
    print(f"  -> {pp_stats.get('total_segments', '?')} segments "
          f"({pp_stats.get('hours', {})})")

    print("\n[3/4] Training (this will take a while)...")
    train_metrics = train.remote()
    print(f"  -> {train_metrics}")

    print("\n[4/4] Evaluating...")
    results = evaluate_model.remote()
    print(f"  -> Baseline WER: {results['baseline']['wer']:.4f}")
    print(f"  -> Finetuned WER: {results['finetuned']['wer']:.4f}")
    print(f"  -> WER reduction: {results['improvement']['wer_reduction']:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results: modal volume get tenepal-data data/results/eval_results.json")
    print("=" * 60)


@app.local_entrypoint()
def run_train(
    max_steps: int = DEFAULT_MAX_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
):
    """Train only (assumes download + preprocess already done)."""
    metrics = train.remote(max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate)
    print(f"Training complete: {json.dumps(metrics, indent=2)}")


@app.local_entrypoint()
def run_eval():
    """Evaluate only (assumes train already done)."""
    results = evaluate_model.remote()
    print(f"Results: {json.dumps(results, indent=2)}")


@app.local_entrypoint()
def run_cross_lang():
    """Cross-language robustness test.

    Tests baseline vs finetuned on NAH test segments.
    For SPA/ENG edge cases, first upload:
        modal volume put tenepal-data validation_video/edge_cases/ data/cross_lang_test/
    """
    results = cross_language_eval.remote()
    print(f"\nResults: {json.dumps(results, indent=2)}")


@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    timeout=60,
    memory=2048,
)
def reset_markers():
    """Clear idempotency markers so phases re-run."""
    from pathlib import Path
    markers = [
        f"{RAW_DIR}/.download_complete",
        f"{PROCESSED_DIR}/.preprocess_complete",
    ]
    for m in markers:
        if os.path.exists(m):
            os.remove(m)
            print(f"Removed {m}")
        else:
            print(f"Not found: {m}")
    vol.commit()


@app.local_entrypoint()
def reset():
    """Clear markers so download/preprocess re-run."""
    reset_markers.remote()
    print("Markers cleared.")


@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    timeout=300,
    memory=4096,
)
def debug_alignment():
    """Diagnose audio/transcription name mismatch."""
    from pathlib import Path

    vol.reload()

    index = json.loads(Path(f"{TRANS_DIR}/index.json").read_text())
    elan_keys = list(index["elan_segments"].keys())

    audio_names = sorted(p.name for p in Path(AUDIO_DIR).rglob("*.wav"))
    audio_stems = sorted(p.stem for p in Path(AUDIO_DIR).rglob("*.wav"))

    print(f"ELAN keys ({len(elan_keys)} total), first 10:")
    for k in elan_keys[:10]:
        print(f"  '{k}'")

    print(f"\nAudio files ({len(audio_names)} total), first 10:")
    for n in audio_names[:10]:
        print(f"  '{n}'")

    # Check for partial matches
    matches = 0
    for k in elan_keys:
        k_stem = Path(k).stem
        if k in audio_names or k in audio_stems or k_stem in audio_stems:
            matches += 1
            if matches <= 5:
                print(f"\n  MATCH: '{k}' -> found")

    print(f"\nDirect matches: {matches}/{len(elan_keys)}")

    # Check ELAN audio_refs (the MEDIA_DESCRIPTOR paths)
    audio_refs = index.get("audio_refs", {})
    print(f"\nELAN audio_refs (MEDIA_DESCRIPTOR), first 10:")
    for k, v in list(audio_refs.items())[:10]:
        print(f"  key='{k}'  eaf='{Path(v).name}'")

    # Check if .trs files exist (Transcriber format, not ELAN)
    trs_files = list(Path(MANIFEST_DIR).rglob("*.trs"))
    eaf_files = list(Path(MANIFEST_DIR).rglob("*.eaf"))
    print(f"\n.eaf files: {len(eaf_files)}")
    print(f".trs files: {len(trs_files)}")

    # Check for Tepet .trs or .eaf files (matching the WAVs we have)
    tepet_trs = [f for f in trs_files if "Tepet" in f.name or "tepet" in f.name.lower()]
    tepet_eaf = [f for f in eaf_files if "Tepet" in f.name or "tepet" in f.name.lower()]
    print(f"\nTepetzintla .trs: {len(tepet_trs)}")
    print(f"Tepetzintla .eaf: {len(tepet_eaf)}")
    for f in tepet_trs[:5]:
        print(f"  trs: {f.name}")
    for f in tepet_eaf[:5]:
        print(f"  eaf: {f.name}")

    # Show manifest directory tree 2 levels deep
    print(f"\nManifest full tree:")
    manifest_root = Path(MANIFEST_DIR)
    for d in sorted(manifest_root.rglob("*")):
        if d.is_dir() and not d.name.startswith("."):
            depth = len(d.relative_to(manifest_root).parts)
            if depth <= 3:
                n_eaf = sum(1 for _ in d.glob("*.eaf"))
                n_trs = sum(1 for _ in d.glob("*.trs"))
                n_wav = sum(1 for _ in d.glob("*.wav"))
                print(f"  {'  ' * depth}{d.name}/ (eaf={n_eaf}, trs={n_trs}, wav={n_wav})")

    # Check where the Tepetzintla WAVs live
    print(f"\nTepet WAV parent dirs:")
    tepet_wavs = [p for p in Path(AUDIO_DIR).rglob("*.wav") if "Tepet" in p.name]
    parents = set(str(p.parent) for p in tepet_wavs)
    for par in parents:
        print(f"  {par}")

    # Sample a .trs file to understand format
    if trs_files:
        sample_trs = trs_files[0]
        print(f"\nSample .trs file ({sample_trs.name}):")
        content = sample_trs.read_text(errors="replace")[:2000]
        print(content)

    return {"elan_keys_sample": elan_keys[:20], "audio_names_sample": audio_names[:20]}


@app.local_entrypoint()
def run_eval_checkpoint(
    clips_dir: str = "clips/edge_cases",
    checkpoint: str = "checkpoint-3000",
):
    """Evaluate a single checkpoint on edge-case clips.

    Uploads clips to Modal volume, runs Whisper inference, prints JSONL.

    Usage:
        modal run tenepal_whisper_train.py::run_eval_checkpoint \\
            --clips-dir clips/edge_cases --checkpoint checkpoint-3000
    """
    from pathlib import Path

    clips_path = Path(clips_dir)
    wav_files = sorted(clips_path.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {clips_dir}")
        return

    import subprocess
    import sys

    print(f"[eval] Uploading {len(wav_files)} clips to volume...", file=sys.stderr)
    # Use modal CLI to upload — vol.reload()/commit() can't run from local entrypoint
    # Upload only WAV files (skip results/ subdir)
    for wav in wav_files:
        subprocess.run(
            ["modal", "volume", "put", "tenepal-data", "--force",
             str(wav), f"eval_clips/{wav.name}"],
            check=True, capture_output=True,
        )
    print(f"[eval] Upload complete, running {checkpoint}...", file=sys.stderr)

    results = eval_checkpoint_gpu.remote(checkpoint=checkpoint)

    # Output as JSONL (stdout only — stderr for progress)
    for entry in results:
        print(json.dumps(entry, ensure_ascii=False))


@app.local_entrypoint()
def run_debug():
    """Debug alignment between audio files and ELAN transcriptions."""
    result = debug_alignment.remote()
    print(json.dumps(result, indent=2))

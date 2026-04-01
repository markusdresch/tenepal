"""F005 on Modal: extract wav2vec2 embeddings + train classifier.

Uploads annotated audio segments to volume, runs on CPU, returns results.
Cost: ~$0.10-0.20 (CPU only, ~15 min).

Usage:
    modal run scripts/f005_embedding_langid_modal.py
"""

import json
import modal
import os
import subprocess
import sys
from pathlib import Path

app = modal.App("tenepal-f005")
vol = modal.Volume.from_name("slangophone-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "transformers>=4.40",
        "librosa",
        "soundfile",
        "scikit-learn",
        "numpy",
    )
)

F005_DIR = "/data/f005"
AUDIO_DIR = f"{F005_DIR}/audio"


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=3600,
    memory=16384,
    cpu=4,
)
def run_f005():
    """Extract embeddings + train classifier."""
    import sqlite3
    import time
    from collections import Counter

    import librosa
    import numpy as np
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, balanced_accuracy_score,
    )
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    TARGET_SR = 16000
    WAV2VEC_MODEL = "facebook/wav2vec2-base"

    vol.reload()

    MEDIA_AUDIO_MAP = {
        "Hernán-1-3.mp4": "Hernán-1-3",
        "La-Otra-Conquista_14m15-24m15.mp4": "La-Otra-Conquista_14m15-24m15",
        "La-Otra-Conquista_24m15-34m15.mp4": "La-Otra-Conquista_24m15-34m15",
        "La-Otra-Conquista_34m15-44m25.mp4": "La-Otra-Conquista_34m15-44m25",
        "La-Otra-Conquista_44m25-55m25.mp4": "La-Otra-Conquista_44m25-55m25",
        "La-Otra-Conquista_84m25-94m25.mp4": "La-Otra-Conquista_84m25-94m25",
        "La-Otra-Conquista_test10m.wav": "La-Otra-Conquista_test10m",
        "La-Otra-Conquista_24m15-34m15.wav": "La-Otra-Conquista_24m15-34m15",
        "La-Otra-Conquista_84m25-94m25.wav": "La-Otra-Conquista_84m25-94m25",
    }

    # --- Load annotations ---
    db_path = f"{F005_DIR}/annotations.db"
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT media_file, cue_index, start_s, end_s, correct_lang, correct_speaker,
               pipeline_lang
        FROM annotations
        WHERE correct_lang IS NOT NULL AND correct_lang != ''
          AND start_s IS NOT NULL AND end_s IS NOT NULL
          AND end_s > start_s
    """).fetchall()
    db.close()

    segments = []
    for r in rows:
        stem = MEDIA_AUDIO_MAP.get(r["media_file"])
        if not stem:
            continue
        audio_path = f"{AUDIO_DIR}/{stem}.vocals.wav"
        if not os.path.exists(audio_path):
            audio_path = f"{AUDIO_DIR}/{stem}.wav"
        if not os.path.exists(audio_path):
            continue
        segments.append({
            "audio_path": audio_path,
            "start_s": r["start_s"],
            "end_s": r["end_s"],
            "lang": r["correct_lang"],
            "speaker": r["correct_speaker"] or "",
            "media": r["media_file"],
            "cue_index": r["cue_index"],
            "pipeline_lang": r["pipeline_lang"] or "",
        })

    print(f"[data] {len(segments)} segments")
    print(f"[data] Languages: {dict(Counter(s['lang'] for s in segments))}")

    # --- Extract embeddings ---
    print(f"\n[embed] Loading {WAV2VEC_MODEL}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
    model.eval()

    embeddings = np.zeros((len(segments), 768), dtype=np.float32)
    audio_cache = {}
    t0 = time.time()

    for i, seg in enumerate(segments):
        ap = seg["audio_path"]
        if ap not in audio_cache:
            audio_cache[ap], _ = librosa.load(ap, sr=TARGET_SR, mono=True)
            if len(audio_cache) > 3:
                del audio_cache[next(iter(audio_cache))]

        audio = audio_cache[ap]
        start = int(seg["start_s"] * TARGET_SR)
        end = int(seg["end_s"] * TARGET_SR)
        chunk = audio[start:end]

        if len(chunk) < TARGET_SR * 0.3:
            continue
        if len(chunk) > TARGET_SR * 30:
            chunk = chunk[:TARGET_SR * 30]

        inputs = feature_extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            hidden = model(inputs.input_values).last_hidden_state
            embeddings[i] = hidden.mean(dim=1).squeeze().numpy()

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(segments)}] {rate:.1f} seg/s, ETA {(len(segments)-i-1)/rate:.0f}s")

    elapsed = time.time() - t0
    print(f"[embed] Done: {len(segments)} in {elapsed:.1f}s ({len(segments)/elapsed:.1f} seg/s)")

    # --- Classify ---
    labels = np.array([s["lang"] for s in segments])
    mask = np.any(embeddings != 0, axis=1)
    X, y = embeddings[mask], labels[mask]
    media_arr = [s["media"] for s, m in zip(segments, mask) if m]

    # Ensure plain Python strings (not np.str_) for serialization
    y = np.array([str(v) for v in y])
    print(f"\n[clf] {len(X)} segments, {dict(Counter(y))}")

    y_binary = np.array(["NAH" if l == "NAH" else "OTHER" for l in y])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Multi-class
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    y_pred_multi = cross_val_predict(clf, X, y_enc, cv=cv)
    y_pred_labels = le.inverse_transform(y_pred_multi)

    print("\n" + "=" * 60)
    print("MULTI-CLASS (5-fold CV)")
    print("=" * 60)
    report_multi = classification_report(y, y_pred_labels, zero_division=0)
    print(report_multi)

    # Binary + confidence via predict_proba
    clf_bin = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    y_pred_bin = cross_val_predict(clf_bin, X, y_binary, cv=cv)
    y_proba_bin = cross_val_predict(clf_bin, X, y_binary, cv=cv, method="predict_proba")
    # Fit on all data to get classes_ attribute
    clf_bin.fit(X, y_binary)
    nah_class_idx = list(clf_bin.classes_).index("NAH")

    print("=" * 60)
    print("BINARY: NAH vs OTHER (5-fold CV)")
    print("=" * 60)
    report_bin = classification_report(y_binary, y_pred_bin, zero_division=0)
    print(report_bin)

    cm = confusion_matrix(y_binary, y_pred_bin, labels=["NAH", "OTHER"])
    print(f"              Pred NAH  Pred OTHER")
    print(f"  True NAH    {cm[0,0]:>8}  {cm[0,1]:>10}")
    print(f"  True OTHER  {cm[1,0]:>8}  {cm[1,1]:>10}")

    # Per-media
    print("\n--- Per-media ---")
    per_media = {}
    for mf in sorted(set(media_arr)):
        idx = [i for i, m in enumerate(media_arr) if m == mf]
        correct = sum(1 for i in idx if y_pred_labels[i] == y[i])
        acc = correct / len(idx)
        per_media[mf] = {"accuracy": round(acc, 3), "total": len(idx)}
        print(f"  {mf}: {acc:.1%} ({correct}/{len(idx)})")

    # --- Per-segment predictions with confidence ---
    per_segment = []
    for i, (seg, m) in enumerate(zip(segments, mask)):
        if not m:
            continue
        idx = int(np.sum(mask[:i+1])) - 1
        # Confidence = P(predicted_class) from CV probas
        nah_prob = float(y_proba_bin[idx][nah_class_idx])
        conf = nah_prob if y_pred_bin[idx] == "NAH" else 1.0 - nah_prob
        per_segment.append({
            "media": seg["media"],
            "cue_index": seg.get("cue_index", i),
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "true_lang": str(y[idx]),
            "pred_multi": str(y_pred_labels[idx]),
            "pred_binary": str(y_pred_bin[idx]),
            "f005_confidence": round(conf, 4),
            "f005_nah_prob": round(nah_prob, 4),
            "pipeline_lang": seg.get("pipeline_lang", ""),
            "speaker": seg.get("speaker", ""),
        })

    # Save full results to volume (avoids numpy serialization issues)
    results = {
        "model": WAV2VEC_MODEL,
        "segments": int(len(X)),
        "labels": {k: int(v) for k, v in Counter(y).items()},
        "multi_class": {
            "accuracy": round(float(accuracy_score(y, y_pred_labels)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, y_pred_labels)), 4),
            "report": report_multi,
        },
        "binary_nah_vs_other": {
            "accuracy": round(float(accuracy_score(y_binary, y_pred_bin)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_binary, y_pred_bin)), 4),
            "confusion_matrix": [[int(c) for c in row] for row in cm.tolist()],
            "report": report_bin,
        },
        "per_media": per_media,
        "per_segment": per_segment,
        "extraction_time_s": round(elapsed, 1),
    }
    os.makedirs(F005_DIR, exist_ok=True)
    with open(f"{F005_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    vol.commit()
    print(f"\n[save] Results written to volume: f005/results.json")

    # Return only a small summary (no numpy types)
    return f"done: {len(X)} segments, binary={round(float(accuracy_score(y_binary, y_pred_bin)), 3)}, multi={round(float(accuracy_score(y, y_pred_labels)), 3)}"


@app.local_entrypoint()
def main():
    # Upload audio + DB to volume
    audio_dir = Path("validation_video")
    vocals = sorted(audio_dir.glob("*.vocals.wav"))
    wavs = [audio_dir / "La-Otra-Conquista_test10m.wav"]

    print(f"[upload] {len(vocals)} vocals files + DB to slangophone-data...")
    for f in vocals + wavs:
        if f.exists():
            dest = f"f005/audio/{f.name}"
            subprocess.run(
                ["modal", "volume", "put", "slangophone-data", "--force",
                 str(f), dest],
                check=True, capture_output=True,
            )
    subprocess.run(
        ["modal", "volume", "put", "slangophone-data", "--force",
         "tools/annotator/annotations.db", "f005/annotations.db"],
        check=True, capture_output=True,
    )
    print("[upload] done")

    summary = run_f005.remote()
    print(f"\n{'='*60}")
    print(summary)

    # Download full results from volume
    os.makedirs("output", exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "slangophone-data",
         "f005/results.json", "output/f005_results.json"],
        check=True,
    )
    print(f"Downloaded to output/f005_results.json")

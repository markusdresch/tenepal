"""F005: wav2vec2 Embedding-based Language Identification.

Extracts wav2vec2-base embeddings from annotated segments, trains a linear
classifier (LogisticRegression), evaluates standalone LangID accuracy.

Design: embeddings are cached as .npy so extraction runs once, classifier
iteration is instant.

Usage:
    # Full pipeline (extract + train + eval):
    python scripts/f005_embedding_langid.py

    # Only classifier (if embeddings already cached):
    python scripts/f005_embedding_langid.py --skip-extract

    # With GPU acceleration:
    python scripts/f005_embedding_langid.py --device cuda

Requires: torch, torchaudio, transformers, scikit-learn, librosa, soundfile
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "tools" / "annotator" / "annotations.db"
AUDIO_DIR = REPO_ROOT / "validation_video"
CACHE_DIR = REPO_ROOT / "output" / "f005_embeddings"
RESULTS_PATH = REPO_ROOT / "output" / "f005_results.json"

WAV2VEC_MODEL = "facebook/wav2vec2-base"
TARGET_SR = 16000
EMBED_DIM = 768

# Map media_file from DB → audio stem for .vocals.wav lookup
MEDIA_AUDIO_MAP = {
    "Hernán-1-3.mp4": "Hernán-1-3",
    "La-Otra-Conquista_14m15-24m15.mp4": "La-Otra-Conquista_14m15-24m15",
    "La-Otra-Conquista_24m15-34m15.mp4": "La-Otra-Conquista_24m15-34m15",
    "La-Otra-Conquista_34m15-44m25.mp4": "La-Otra-Conquista_34m15-44m25",
    "La-Otra-Conquista_44m25-55m25.mp4": "La-Otra-Conquista_44m25-55m25",
    "La-Otra-Conquista_84m25-94m25.mp4": "La-Otra-Conquista_84m25-94m25",
    "La-Otra-Conquista_test10m.wav": "La-Otra-Conquista_test10m",
    # .wav refs fall back to the file directly
    "La-Otra-Conquista_24m15-34m15.wav": "La-Otra-Conquista_24m15-34m15",
    "La-Otra-Conquista_84m25-94m25.wav": "La-Otra-Conquista_84m25-94m25",
}


# ---------------------------------------------------------------------------
# Phase 1: Load annotations
# ---------------------------------------------------------------------------

def load_annotations():
    """Load annotated segments from SQLite DB."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT media_file, cue_index, start_s, end_s, correct_lang, correct_speaker
        FROM annotations
        WHERE correct_lang IS NOT NULL AND correct_lang != ''
          AND start_s IS NOT NULL AND end_s IS NOT NULL
          AND end_s > start_s
    """).fetchall()
    db.close()

    segments = []
    for r in rows:
        media = r["media_file"]
        stem = MEDIA_AUDIO_MAP.get(media)
        if not stem:
            continue

        # Prefer .vocals.wav, fall back to .wav
        audio_path = AUDIO_DIR / f"{stem}.vocals.wav"
        if not audio_path.exists():
            audio_path = AUDIO_DIR / f"{stem}.wav"
        if not audio_path.exists():
            continue

        segments.append({
            "audio_path": str(audio_path),
            "start_s": r["start_s"],
            "end_s": r["end_s"],
            "lang": r["correct_lang"],
            "speaker": r["correct_speaker"] or "",
            "media": media,
            "cue_index": r["cue_index"],
        })

    return segments


# ---------------------------------------------------------------------------
# Phase 2: Extract embeddings
# ---------------------------------------------------------------------------

def extract_embeddings(segments, device="cpu"):
    """Extract wav2vec2-base mean-pooled embeddings for each segment."""
    import torch
    import librosa
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    print(f"[embed] Loading {WAV2VEC_MODEL} on {device}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL).to(device)
    model.eval()

    os.makedirs(CACHE_DIR, exist_ok=True)
    embeddings = np.zeros((len(segments), EMBED_DIM), dtype=np.float32)

    # Cache audio files to avoid reloading the same file for every segment
    audio_cache = {}
    t0 = time.time()

    for i, seg in enumerate(segments):
        ap = seg["audio_path"]
        if ap not in audio_cache:
            audio_cache[ap], _ = librosa.load(ap, sr=TARGET_SR, mono=True)
            # Keep max 3 files in cache to limit memory
            if len(audio_cache) > 3:
                oldest = next(iter(audio_cache))
                del audio_cache[oldest]

        audio = audio_cache[ap]
        start_sample = int(seg["start_s"] * TARGET_SR)
        end_sample = int(seg["end_s"] * TARGET_SR)
        chunk = audio[start_sample:end_sample]

        if len(chunk) < TARGET_SR * 0.3:  # skip < 0.3s
            continue

        # Truncate to 30s max
        if len(chunk) > TARGET_SR * 30:
            chunk = chunk[: TARGET_SR * 30]

        inputs = feature_extractor(
            chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)
            hidden = outputs.last_hidden_state  # (1, T, 768)
            embedding = hidden.mean(dim=1).squeeze().cpu().numpy()  # (768,)

        embeddings[i] = embedding

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(segments) - i - 1) / rate
            print(f"  [{i+1}/{len(segments)}] {rate:.1f} seg/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[embed] {len(segments)} segments in {elapsed:.1f}s "
          f"({len(segments)/elapsed:.1f} seg/s)")

    # Save cache
    np.save(CACHE_DIR / "embeddings.npy", embeddings)
    labels = [seg["lang"] for seg in segments]
    with open(CACHE_DIR / "labels.json", "w") as f:
        json.dump(labels, f)
    with open(CACHE_DIR / "segments.json", "w") as f:
        json.dump(segments, f, ensure_ascii=False)

    print(f"[embed] Cached to {CACHE_DIR}/")
    return embeddings, labels


# ---------------------------------------------------------------------------
# Phase 3: Train + Evaluate classifier
# ---------------------------------------------------------------------------

def train_classifier(embeddings, labels):
    """Train LogisticRegression, evaluate with leave-one-group-out by media file."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from collections import Counter

    # Load segments for grouping
    with open(CACHE_DIR / "segments.json") as f:
        segments = json.load(f)

    # Filter out zero embeddings (skipped segments) and rare classes
    mask = np.any(embeddings != 0, axis=1)
    X = embeddings[mask]
    y = np.array(labels)[mask]
    segs_filtered = [s for s, m in zip(segments, mask) if m]

    print(f"\n[clf] {len(X)} segments after filtering")
    print(f"[clf] Label distribution: {dict(Counter(y))}")

    # Binary mode: NAH vs non-NAH (the actual pipeline task)
    y_binary = np.array(["NAH" if l == "NAH" else "OTHER" for l in y])
    print(f"[clf] Binary: {dict(Counter(y_binary))}")

    # --- Stratified K-Fold (simple, robust) ---
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Multi-class
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_pred_multi = cross_val_predict(clf, X, y_enc, cv=cv)
    y_pred_labels = le.inverse_transform(y_pred_multi)

    print("\n" + "=" * 60)
    print("MULTI-CLASS RESULTS (5-fold CV)")
    print("=" * 60)
    report_multi = classification_report(y, y_pred_labels, zero_division=0)
    print(report_multi)

    # Binary NAH vs OTHER
    clf_bin = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    y_pred_binary = cross_val_predict(clf_bin, X, y_binary, cv=cv)

    print("=" * 60)
    print("BINARY RESULTS: NAH vs OTHER (5-fold CV)")
    print("=" * 60)
    report_binary = classification_report(y_binary, y_pred_binary, zero_division=0)
    print(report_binary)

    cm = confusion_matrix(y_binary, y_pred_binary, labels=["NAH", "OTHER"])
    print(f"Confusion matrix (NAH vs OTHER):")
    print(f"              Pred NAH  Pred OTHER")
    print(f"  True NAH    {cm[0,0]:>8}  {cm[0,1]:>10}")
    print(f"  True OTHER  {cm[1,0]:>8}  {cm[1,1]:>10}")

    # Per-media accuracy
    print("\n--- Per-media accuracy ---")
    media_list = [s["media"] for s in segs_filtered]
    media_set = sorted(set(media_list))
    per_media = {}
    for mf in media_set:
        idx = [i for i, m in enumerate(media_list) if m == mf]
        if not idx:
            continue
        correct = sum(1 for i in idx if y_pred_labels[i] == y[i])
        total = len(idx)
        acc = correct / total
        per_media[mf] = {"accuracy": round(acc, 3), "total": total}
        print(f"  {mf}: {acc:.1%} ({correct}/{total})")

    # --- Train final model on all data for later use ---
    clf_final = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf_final.fit(X, y_binary)

    # Save results
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    results = {
        "model": WAV2VEC_MODEL,
        "segments": len(X),
        "labels": dict(Counter(y)),
        "multi_class": {
            "accuracy": round(accuracy_score(y, y_pred_labels), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, y_pred_labels), 4),
            "report": report_multi,
        },
        "binary_nah_vs_other": {
            "accuracy": round(accuracy_score(y_binary, y_pred_binary), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_binary, y_pred_binary), 4),
            "confusion_matrix": cm.tolist(),
            "report": report_binary,
        },
        "per_media": per_media,
    }

    os.makedirs(RESULTS_PATH.parent, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[clf] Results saved to {RESULTS_PATH}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="F005: wav2vec2 embedding LangID")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip embedding extraction, use cached")
    parser.add_argument("--device", default="cpu", help="torch device (cpu/cuda)")
    args = parser.parse_args()

    print("F005: wav2vec2 Embedding-based Language Identification")
    print("=" * 60)

    # Load annotations
    segments = load_annotations()
    print(f"[data] {len(segments)} annotated segments loaded")
    from collections import Counter
    lang_dist = Counter(s["lang"] for s in segments)
    print(f"[data] Languages: {dict(lang_dist)}")

    if args.skip_extract:
        print("[embed] Loading cached embeddings...")
        embeddings = np.load(CACHE_DIR / "embeddings.npy")
        with open(CACHE_DIR / "labels.json") as f:
            labels = json.load(f)
    else:
        embeddings, labels = extract_embeddings(segments, device=args.device)

    results = train_classifier(embeddings, labels)

    print("\n" + "=" * 60)
    print(f"Binary (NAH vs OTHER): {results['binary_nah_vs_other']['balanced_accuracy']:.1%} balanced accuracy")
    print(f"Multi-class: {results['multi_class']['balanced_accuracy']:.1%} balanced accuracy")
    print("=" * 60)


if __name__ == "__main__":
    main()

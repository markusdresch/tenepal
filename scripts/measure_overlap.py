#!/usr/bin/env python3
"""Measure overlap detection signals for all annotated segments.

Runs Parselmouth F0 analysis on each segment and writes results to the
annotations DB: overlap_detected, f0_low_ratio, f0_high_ratio, hnr.

Uses the same logic as detect_overlap() in tenepal_modal.py but runs
locally without Modal.

Usage:
    # Run with annotator venv (has parselmouth):
    tools/annotator/.venv/bin/python scripts/measure_overlap.py
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "tools" / "annotator" / "annotations.db"
AUDIO_DIR = REPO_ROOT / "validation_video"

MEDIA_AUDIO_MAP = {
    "Hernán-1-3.mp4": "hernan/Hernán-1-3",
    "La-Otra-Conquista_14m15-24m15.mp4": "loc/La-Otra-Conquista_14m15-24m15",
    "La-Otra-Conquista_24m15-34m15.mp4": "loc/La-Otra-Conquista_24m15-34m15",
    "La-Otra-Conquista_34m15-44m25.mp4": "loc/La-Otra-Conquista_34m15-44m25",
    "La-Otra-Conquista_44m25-55m25.mp4": "loc/La-Otra-Conquista_44m25-55m25",
    "La-Otra-Conquista_84m25-94m25.mp4": "loc/La-Otra-Conquista_84m25-94m25",
    "La-Otra-Conquista_test10m.wav": "loc/La-Otra-Conquista_test10m",
    "La-Otra-Conquista_24m15-34m15.wav": "loc/La-Otra-Conquista_24m15-34m15",
    "La-Otra-Conquista_84m25-94m25.wav": "loc/La-Otra-Conquista_84m25-94m25",
}

# Same thresholds as tenepal_modal.py detect_overlap()
F0_LOW_BAND = (80, 180)
F0_HIGH_BAND = (180, 350)
F0_MIN_RATIO = 0.20
HNR_THRESHOLD = 10.0


def detect_overlap(audio, sr, start_s, end_s):
    """Replicate detect_overlap() from tenepal_modal.py."""
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    chunk = audio[start_sample:end_sample]

    if len(chunk) < sr * 0.3:
        return False, {"f0_low_ratio": 0, "f0_high_ratio": 0, "hnr": 0, "bimodal": False}

    snd = parselmouth.Sound(chunk, sampling_frequency=sr)

    # F0 analysis
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=400)
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]

    if len(voiced) < 5:
        return False, {"f0_low_ratio": 0, "f0_high_ratio": 0, "hnr": 0, "bimodal": False}

    total_voiced = len(voiced)
    low_count = np.sum((voiced >= F0_LOW_BAND[0]) & (voiced < F0_LOW_BAND[1]))
    high_count = np.sum((voiced >= F0_HIGH_BAND[0]) & (voiced <= F0_HIGH_BAND[1]))
    low_ratio = low_count / total_voiced
    high_ratio = high_count / total_voiced

    bimodal = low_ratio > F0_MIN_RATIO and high_ratio > F0_MIN_RATIO

    # HNR
    harmonicity = snd.to_harmonicity(time_step=0.01)
    hnr_values = harmonicity.values[0]
    hnr_valid = hnr_values[hnr_values > -200]
    mean_hnr = float(np.mean(hnr_valid)) if len(hnr_valid) > 0 else 20.0
    low_hnr = mean_hnr < HNR_THRESHOLD

    # Intensity variation
    intensity = snd.to_intensity(time_step=0.01)
    int_values = intensity.values[0]
    int_cv = float(np.std(int_values) / max(np.mean(int_values), 0.001))
    high_int_var = int_cv > 0.15

    score = (bimodal * 0.5) + (low_hnr * 0.3) + (high_int_var * 0.2)
    is_overlap = bimodal and score >= 0.5

    return is_overlap, {
        "f0_low_ratio": round(low_ratio, 4),
        "f0_high_ratio": round(high_ratio, 4),
        "hnr": round(mean_hnr, 2),
        "bimodal": bimodal,
        "low_hnr": low_hnr,
        "int_cv": round(int_cv, 4),
        "score": round(score, 3),
    }


def main():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    # Add columns if missing
    existing_cols = [r[1] for r in db.execute("PRAGMA table_info(annotations)").fetchall()]
    for col, typ in [
        ("overlap_detected", "BOOLEAN"),
        ("overlap_f0_low_ratio", "FLOAT"),
        ("overlap_f0_high_ratio", "FLOAT"),
        ("overlap_hnr", "FLOAT"),
        ("overlap_score", "FLOAT"),
    ]:
        if col not in existing_cols:
            db.execute(f"ALTER TABLE annotations ADD COLUMN {col} {typ}")
    db.commit()

    rows = db.execute("""
        SELECT id, media_file, cue_index, start_s, end_s, correct_lang, correct_speaker
        FROM annotations
        WHERE start_s IS NOT NULL AND end_s IS NOT NULL AND end_s > start_s
        ORDER BY media_file, start_s
    """).fetchall()

    print(f"Segments to measure: {len(rows)}")

    # Group by media for efficient audio loading
    from collections import defaultdict
    by_media = defaultdict(list)
    for r in rows:
        by_media[r["media_file"]].append(r)

    updated = 0
    skipped = 0
    t0 = time.time()

    for media_file, segments in by_media.items():
        stem = MEDIA_AUDIO_MAP.get(media_file)
        if not stem:
            skipped += len(segments)
            continue

        audio_path = AUDIO_DIR / f"{stem}.vocals.wav"
        if not audio_path.exists():
            audio_path = AUDIO_DIR / f"{stem}.wav"
        if not audio_path.exists():
            print(f"  SKIP {media_file}: no audio at {audio_path}")
            skipped += len(segments)
            continue

        print(f"  Loading {audio_path.name}...", end="", flush=True)
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        print(f" {len(segments)} segments")

        for seg in segments:
            # Use ±2s context window to simulate turn-level detection
            ctx = 2.0
            ctx_start = max(0, seg["start_s"] - ctx)
            ctx_end = min(len(audio) / sr, seg["end_s"] + ctx)
            is_ov, details = detect_overlap(audio, sr, ctx_start, ctx_end)

            db.execute("""
                UPDATE annotations SET
                    overlap_detected = ?,
                    overlap_f0_low_ratio = ?,
                    overlap_f0_high_ratio = ?,
                    overlap_hnr = ?,
                    overlap_score = ?
                WHERE id = ?
            """, (
                is_ov,
                details["f0_low_ratio"],
                details["f0_high_ratio"],
                details["hnr"],
                details.get("score", 0),
                seg["id"],
            ))
            updated += 1

        db.commit()

    elapsed = time.time() - t0
    print(f"\nDone: {updated} updated, {skipped} skipped, {elapsed:.1f}s")

    # Quick stats
    flagged = db.execute("SELECT COUNT(*) FROM annotations WHERE overlap_detected = 1").fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM annotations WHERE overlap_detected IS NOT NULL").fetchone()[0]
    print(f"Overlap flagged: {flagged}/{total} ({flagged/total:.1%})")

    # False positive analysis: flagged as overlap but single speaker
    false_pos = db.execute("""
        SELECT correct_speaker, correct_lang, COUNT(*) as cnt,
               AVG(overlap_f0_low_ratio) as avg_low, AVG(overlap_f0_high_ratio) as avg_high,
               AVG(overlap_hnr) as avg_hnr
        FROM annotations
        WHERE overlap_detected = 1
          AND correct_speaker IS NOT NULL AND correct_speaker != ''
        GROUP BY correct_speaker, correct_lang
        ORDER BY cnt DESC
    """).fetchall()

    print(f"\nOverlap-flagged segments by speaker+lang:")
    for r in false_pos[:15]:
        print(f"  {r[0]:30s} {r[1]:>4}: {r[2]:>3} flagged "
              f"(low={r[3]:.2f} high={r[4]:.2f} hnr={r[5]:.1f})")

    db.close()


if __name__ == "__main__":
    main()

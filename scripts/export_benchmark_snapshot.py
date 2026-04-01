#!/usr/bin/env python3
"""Export canonical benchmark snapshots from the annotator DB.

Creates versioned GT JSON files + a public_benchmarks.json report.
The DB is the single source of truth for all published numbers.

Usage:
    python scripts/export_benchmark_snapshot.py
    python scripts/export_benchmark_snapshot.py --version v2
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "tools" / "annotator" / "annotations.db"
SNAPSHOTS_DIR = REPO_ROOT / "benchmarks" / "snapshots"
REPORTS_DIR = REPO_ROOT / "benchmarks" / "reports"
SRT_DIR = REPO_ROOT / "eq_comparison_results"

# Which SRT is the current best pipeline config?
BEST_SRT = "13_v7_morphology_expansion.srt"

# Media groups for benchmark reporting
HERNAN_MEDIA = ["Hernán-1-3.mp4"]
LOC_MEDIA = [
    "La-Otra-Conquista_14m15-24m15.mp4",
    "La-Otra-Conquista_24m15-34m15.mp4",
    "La-Otra-Conquista_34m15-44m25.mp4",
    "La-Otra-Conquista_44m25-55m25.mp4",
    "La-Otra-Conquista_84m25-94m25.mp4",
]


def load_db_annotations():
    """Load all annotations from DB grouped by media."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT media_file, cue_index, start_s, end_s,
               correct_lang, correct_speaker, pipeline_lang,
               f005_pred_lang, f005_confidence, overlap
        FROM annotations
        WHERE correct_lang IS NOT NULL AND correct_lang != ''
        ORDER BY media_file, cue_index
    """).fetchall()
    db.close()
    return [dict(r) for r in rows]


def parse_srt(srt_path):
    """Parse SRT → {cue_index: lang}."""
    text = Path(srt_path).read_text(encoding="utf-8")
    preds = {}
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            cue = int(lines[0].strip())
        except ValueError:
            continue
        content = " ".join(lines[2:])
        m = re.match(r"\[(\w+)(?:\|[^\]]+)?\]", content)
        lang = m.group(1).lower() if m else "oth"
        lang = {
            "nahuatl": "nah", "español": "spa", "spanish": "spa",
            "english": "eng", "other": "oth", "silence": "sil",
        }.get(lang, lang)
        preds[cue] = lang
    return preds


def compute_accuracy(annotations, srt_preds, subset_name):
    """Compute accuracy metrics for a set of annotations vs SRT predictions.

    Returns dict with all reproducible metrics.
    """
    matched = []
    for ann in annotations:
        cue = ann["cue_index"]
        true = ann["correct_lang"].lower()
        pred = srt_preds.get(cue)
        if pred is not None:
            matched.append({"cue": cue, "true": true, "pred": pred})

    if not matched:
        return {"error": "no matched segments", "subset": subset_name}

    total = len(matched)
    gt_dist = Counter(m["true"] for m in matched)

    # Multi-class: all segments
    correct_all = sum(1 for m in matched if m["true"] == m["pred"])

    # Multi-class: excl UNK (UNK segments are not scorable)
    scorable = [m for m in matched if m["true"] != "unk"]
    correct_excl_unk = sum(1 for m in scorable if m["true"] == m["pred"])

    # NAH+SPA only (the primary task — is it Nahuatl or Spanish?)
    nah_spa = [m for m in matched if m["true"] in ("nah", "spa")]
    correct_nah_spa = sum(1 for m in nah_spa if m["true"] == m["pred"])

    # Binary: NAH vs non-NAH (excl UNK)
    binary = [m for m in scorable]
    binary_correct = sum(
        1 for m in binary
        if (m["true"] == "nah") == (m["pred"] == "nah")
    )

    # Per-class precision/recall for NAH
    nah_tp = sum(1 for m in scorable if m["true"] == "nah" and m["pred"] == "nah")
    nah_fp = sum(1 for m in scorable if m["true"] != "nah" and m["pred"] == "nah")
    nah_fn = sum(1 for m in scorable if m["true"] == "nah" and m["pred"] != "nah")
    nah_prec = nah_tp / (nah_tp + nah_fp) if (nah_tp + nah_fp) else 0
    nah_rec = nah_tp / (nah_tp + nah_fn) if (nah_tp + nah_fn) else 0

    # Confusion matrix (scorable only)
    confusion = defaultdict(Counter)
    for m in scorable:
        confusion[m["true"]][m["pred"]] += 1

    return {
        "subset": subset_name,
        "total_segments": total,
        "gt_distribution": dict(gt_dist),
        "unk_segments": gt_dist.get("unk", 0),
        "metrics": {
            "multiclass_all": {
                "accuracy": round(correct_all / total, 4),
                "correct": correct_all,
                "total": total,
            },
            "multiclass_excl_unk": {
                "accuracy": round(correct_excl_unk / len(scorable), 4) if scorable else None,
                "correct": correct_excl_unk,
                "total": len(scorable),
                "note": "UNK ground truth excluded (unscorable interjections)",
            },
            "nah_spa_subset": {
                "accuracy": round(correct_nah_spa / len(nah_spa), 4) if nah_spa else None,
                "correct": correct_nah_spa,
                "total": len(nah_spa),
                "note": "Only NAH+SPA segments (primary task)",
            },
            "binary_nah_detection": {
                "accuracy": round(binary_correct / len(binary), 4) if binary else None,
                "correct": binary_correct,
                "total": len(binary),
                "nah_precision": round(nah_prec, 4),
                "nah_recall": round(nah_rec, 4),
                "note": "Is it NAH or not? (excl UNK)",
            },
        },
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Export benchmark snapshots")
    parser.add_argument("--version", default="v2", help="Snapshot version tag")
    args = parser.parse_args()

    version = args.version
    today = date.today().isoformat()

    print(f"Exporting benchmark snapshot {version} ({today})")
    print(f"DB: {DB_PATH}")
    print()

    # Load all annotations
    annotations = load_db_annotations()
    print(f"Total annotations: {len(annotations)}")

    # Group by media
    by_media = defaultdict(list)
    for ann in annotations:
        by_media[ann["media_file"]].append(ann)

    # --- Export GT snapshots ---
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Hernán snapshot
    hernan_anns = []
    for mf in HERNAN_MEDIA:
        hernan_anns.extend(by_media.get(mf, []))

    hernan_gt = [{
        "cue_index": a["cue_index"],
        "correct_lang": a["correct_lang"],
        "correct_speaker": a["correct_speaker"],
        "start_s": a.get("start_s"),
        "end_s": a.get("end_s"),
    } for a in hernan_anns]

    hernan_path = SNAPSHOTS_DIR / f"hernan_gt_{version}.json"
    with open(hernan_path, "w") as f:
        json.dump(hernan_gt, f, indent=2, ensure_ascii=False)
    print(f"Hernán GT: {len(hernan_gt)} segments → {hernan_path}")

    # LOC snapshot
    loc_anns = []
    for mf in LOC_MEDIA:
        loc_anns.extend(by_media.get(mf, []))

    loc_gt = [{
        "media_file": a["media_file"],
        "cue_index": a["cue_index"],
        "correct_lang": a["correct_lang"],
        "start_s": a.get("start_s"),
        "end_s": a.get("end_s"),
    } for a in loc_anns]

    loc_path = SNAPSHOTS_DIR / f"loc_gt_{version}.json"
    with open(loc_path, "w") as f:
        json.dump(loc_gt, f, indent=2, ensure_ascii=False)
    print(f"LOC GT: {len(loc_gt)} segments → {loc_path}")

    # --- Evaluate against best SRT ---
    srt_path = SRT_DIR / BEST_SRT
    if not srt_path.exists():
        print(f"\nWARN: {srt_path} not found, skipping SRT evaluation")
        srt_preds = {}
    else:
        srt_preds = parse_srt(srt_path)
        print(f"\nSRT predictions: {len(srt_preds)} segments from {BEST_SRT}")

    hernan_result = compute_accuracy(hernan_anns, srt_preds, "hernan")
    # LOC doesn't have a matching SRT (different cue indices per clip)
    # We'd need per-clip SRTs — for now report DB-level stats only

    # --- DB-level stats for LOC ---
    loc_dist = Counter(a["correct_lang"] for a in loc_anns)
    loc_pipe_correct = sum(
        1 for a in loc_anns
        if a["pipeline_lang"] and a["correct_lang"].upper() == a["pipeline_lang"].upper()
        and a["correct_lang"].lower() != "unk"
    )
    loc_scorable = sum(1 for a in loc_anns if a["correct_lang"].lower() != "unk")

    # --- Build public report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "_meta": {
            "generated": today,
            "version": version,
            "db_source": str(DB_PATH),
            "note": "Canonical benchmark numbers. All claims in README/PAPER must match these values.",
        },
        "langid_pipeline": {
            "hernan": {
                "gt_snapshot": f"benchmarks/snapshots/hernan_gt_{version}.json",
                "prediction_artifact": f"eq_comparison_results/{BEST_SRT}",
                "eval_command": f"python evaluate.py benchmarks/snapshots/hernan_gt_{version}.json eq_comparison_results/{BEST_SRT}",
                **hernan_result,
            },
            "loc": {
                "gt_snapshot": f"benchmarks/snapshots/loc_gt_{version}.json",
                "total_segments": len(loc_anns),
                "gt_distribution": dict(loc_dist),
                "note": "LOC uses per-clip SRTs; combined eval requires clip-level matching",
            },
        },
        "asr_quality": {
            "whisper_finetuning": {
                "dataset": "OpenSLR 92 (Puebla-Nahuatl)",
                "model": "whisper-large-v3 + LoRA (checkpoint-3000)",
                "test_segments": 500,
                "baseline_cer": 1.0761,
                "finetuned_cer": 0.696,
                "cer_reduction": "35%",
                "source": "slangophone-data volume: results/eval_results.json",
            },
        },
        "experimental": {
            "_note": "NOT for public claims until pipeline integration",
            "f005_embedding_langid": {
                "model": "facebook/wav2vec2-base",
                "segments": 1006,
                "binary_balanced_accuracy": 0.851,
                "multiclass_accuracy": 0.783,
                "oracle_with_pipeline": 0.869,
            },
        },
    }

    report_path = REPORTS_DIR / "public_benchmarks.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nPublic report → {report_path}")

    # Summary
    m = hernan_result.get("metrics", {})
    print(f"\n{'='*60}")
    print(f"HERNÁN BENCHMARK ({version})")
    print(f"{'='*60}")
    for name, data in m.items():
        if data and "accuracy" in data and data["accuracy"] is not None:
            print(f"  {name:30s}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
    print(f"\n  GT: {hernan_result.get('gt_distribution', {})}")
    print(f"  SRT: {BEST_SRT}")

    return report


if __name__ == "__main__":
    main()

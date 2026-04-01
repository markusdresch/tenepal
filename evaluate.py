"""Evaluate Tenepal SRT output against ground truth JSON.

Usage:
    python evaluate.py <ground_truth.json> <pipeline_output.srt>

Ground truth formats:
    - DB export: [{cue_index, correct_lang, start_s, end_s, ...}]
    - Legacy:    {segments: [{lang, start, end, done}, ...]}

Metrics:
    - Segment accuracy: % segments with correct lang tag
    - Duration-weighted accuracy: % of correctly classified film time
    - Weighted error cost: asymmetric costs (NAH→SPA ≠ SPA→NAH)
    - Precision/Recall per language (NAH, SPA, ENG, OTH)
    - Confusion matrix: what gets tagged as what
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict


# --- Asymmetric error cost matrix ---
# NAH→SPA (losing indigenous language) is worse than SPA→NAH (hallucinating it).
# X→UNK (honest abstention) is cheaper than a wrong answer.
# UNK ground truth → any prediction is free (humans couldn't tell either).
ERROR_COSTS = {
    # (actual, predicted) → cost
    # Indigenous language loss (worst errors)
    ("nah", "spa"): 2.0,     # Losing Nahuatl as Spanish
    ("nah", "oth"): 1.5,     # Losing Nahuatl as unknown
    ("nah", "unk"): 0.5,     # Abstaining on Nahuatl (honest, but still a miss)
    ("may", "spa"): 2.0,     # Losing Maya as Spanish
    ("may", "oth"): 1.5,     # Losing Maya as unknown
    ("may", "unk"): 0.5,     # Abstaining on Maya
    # Hallucinating indigenous (bad but less bad)
    ("spa", "nah"): 1.5,     # Hallucinating Nahuatl where Spanish
    ("spa", "may"): 1.5,     # Hallucinating Maya where Spanish
    # All other misclassifications
    "_default": 1.0,
    # Honest abstention (better than wrong)
    "_to_unk": 0.5,           # Any lang → UNK (abstaining)
    # UNK ground truth (unknowable → free pass)
    "_from_unk": 0.0,         # UNK → anything (not scorable)
}


def get_error_cost(actual, predicted):
    """Look up asymmetric error cost for a misclassification pair."""
    if actual == predicted:
        return 0.0
    if actual == "unk":
        return ERROR_COSTS["_from_unk"]
    if predicted == "unk":
        return ERROR_COSTS.get((actual, "unk"), ERROR_COSTS["_to_unk"])
    return ERROR_COSTS.get((actual, predicted), ERROR_COSTS["_default"])


def _parse_srt_time(ts: str) -> float:
    """Parse SRT timestamp '00:01:23,456' → seconds."""
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_srt(path):
    """Parse SRT file into segments with lang tags + timestamps, keyed by cue index."""
    text = Path(path).read_text(encoding="utf-8")
    segments = {}
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            cue_idx = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        start_s = _parse_srt_time(time_match.group(1))
        end_s = _parse_srt_time(time_match.group(2))
        content = " ".join(lines[2:])
        lang = "oth"
        tag_match = re.match(r"\[(\w+)(?:\|[^\]]+)?\]", content)
        if tag_match:
            lang = tag_match.group(1).lower()
        lang_map = {
            "nahuatl": "nah", "español": "spa", "spanish": "spa",
            "english": "eng", "deutsch": "deu", "???": "oth",
            "other": "oth", "silence": "sil",
        }
        lang = lang_map.get(lang, lang)
        segments[cue_idx] = {
            "lang": lang, "text": content[:60],
            "start_s": start_s, "end_s": end_s,
        }
    return segments


def load_ground_truth(path):
    """Load ground truth. Returns dict: cue_index → {lang, start_s, end_s}."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        # Legacy format: {segments: [{lang, start, end, done}, ...]}
        segs = data.get("segments", [])
        return {
            i + 1: {"lang": s["lang"].lower(), "start_s": s.get("start", 0), "end_s": s.get("end", 0)}
            for i, s in enumerate(segs) if s.get("done", False)
        }
    # DB export: [{cue_index, correct_lang, start_s, end_s, ...}]
    gt = {}
    for item in data:
        cue = item.get("cue_index")
        lang = (item.get("correct_lang") or "").lower()
        if cue is not None and lang:
            gt[cue] = {
                "lang": lang,
                "start_s": item.get("start_s", 0) or 0,
                "end_s": item.get("end_s", 0) or 0,
            }
    return gt


def evaluate(gt_path, srt_path):
    gt_map = load_ground_truth(gt_path)
    pred_map = load_srt(srt_path)

    if not gt_map:
        print("No annotated segments in ground truth!")
        return

    matched = set(gt_map.keys()) & set(pred_map.keys())
    unmatched_gt = set(gt_map.keys()) - set(pred_map.keys())
    unmatched_pred = set(pred_map.keys()) - set(gt_map.keys())

    print(f"Ground truth: {len(gt_map)} annotated segments")
    print(f"Pipeline:     {len(pred_map)} predicted segments")
    print(f"Matched:      {len(matched)} by cue index")
    if unmatched_gt:
        print(f"Unmatched GT: {len(unmatched_gt)} (no SRT cue)")
    if unmatched_pred:
        print(f"Extra SRT:    {len(unmatched_pred)} (no GT annotation)")
    print()

    # --- Language Classification ---
    correct = 0
    total = 0
    total_excl_unk = 0
    correct_excl_unk = 0
    confusion = defaultdict(Counter)
    per_lang = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_cost = 0.0
    max_possible_cost = 0.0

    # Duration-weighted tracking
    dur_correct_all = 0.0
    dur_total_all = 0.0
    dur_correct_excl_unk = 0.0
    dur_total_excl_unk = 0.0
    dur_correct_nah_spa = 0.0
    dur_total_nah_spa = 0.0
    nah_spa_correct = 0
    nah_spa_total = 0

    for cue in sorted(matched):
        gt_lang = gt_map[cue]["lang"]
        pred_lang = pred_map[cue]["lang"]
        # Duration: prefer GT timestamps, fall back to SRT
        gt_info = gt_map[cue]
        srt_info = pred_map[cue]
        dur = gt_info.get("end_s", 0) - gt_info.get("start_s", 0)
        if dur <= 0:
            dur = srt_info.get("end_s", 0) - srt_info.get("start_s", 0)
        dur = max(dur, 0)

        total += 1
        is_correct = pred_lang == gt_lang
        confusion[gt_lang][pred_lang] += 1

        cost = get_error_cost(gt_lang, pred_lang)
        total_cost += cost
        max_possible_cost += 2.0 if gt_lang != "unk" else 0.0

        if is_correct:
            correct += 1
            per_lang[gt_lang]["tp"] += 1
        else:
            per_lang[gt_lang]["fn"] += 1
            per_lang[pred_lang]["fp"] += 1

        # Duration tracking
        dur_total_all += dur
        if is_correct:
            dur_correct_all += dur

        if gt_lang != "unk":
            total_excl_unk += 1
            dur_total_excl_unk += dur
            if is_correct:
                correct_excl_unk += 1
                dur_correct_excl_unk += dur

        if gt_lang in ("nah", "spa"):
            nah_spa_total += 1
            dur_total_nah_spa += dur
            if is_correct:
                nah_spa_correct += 1
                dur_correct_nah_spa += dur

    # Cost for unmatched GT (treat as OTH prediction)
    for cue in unmatched_gt:
        gt_lang = gt_map[cue]["lang"]
        cost = get_error_cost(gt_lang, "oth")
        total_cost += cost
        max_possible_cost += 2.0 if gt_lang != "unk" else 0.0
        per_lang[gt_lang]["fn"] += 1
        confusion[gt_lang]["(none)"] += 1

    # --- Print Results ---
    accuracy = correct / max(total, 1)
    accuracy_excl_unk = correct_excl_unk / max(total_excl_unk, 1)
    weighted_score = 1.0 - (total_cost / max(max_possible_cost, 1))
    dur_acc_all = dur_correct_all / max(dur_total_all, 0.001)
    dur_acc_excl_unk = dur_correct_excl_unk / max(dur_total_excl_unk, 0.001)
    dur_acc_nah_spa = dur_correct_nah_spa / max(dur_total_nah_spa, 0.001)
    nah_spa_acc = nah_spa_correct / max(nah_spa_total, 1)

    print(f"{'='*60}")
    print(f"SEGMENT ACCURACY:  {correct}/{total} = {accuracy:.1%}")
    unk_gt_count = sum(1 for g in gt_map.values() if g["lang"] == "unk")
    if unk_gt_count > 0:
        print(f"  excl. UNK:       {correct_excl_unk}/{total_excl_unk} = {accuracy_excl_unk:.1%} ({unk_gt_count} UNK skipped)")
    if nah_spa_total:
        print(f"  NAH+SPA only:    {nah_spa_correct}/{nah_spa_total} = {nah_spa_acc:.1%}")
    print(f"DURATION-WEIGHTED: {dur_correct_all:.0f}s/{dur_total_all:.0f}s = {dur_acc_all:.1%}")
    if dur_total_excl_unk > 0:
        print(f"  excl. UNK:       {dur_correct_excl_unk:.0f}s/{dur_total_excl_unk:.0f}s = {dur_acc_excl_unk:.1%}")
    if dur_total_nah_spa > 0:
        print(f"  NAH+SPA only:    {dur_correct_nah_spa:.0f}s/{dur_total_nah_spa:.0f}s = {dur_acc_nah_spa:.1%}")
    print(f"WEIGHTED SCORE:    {weighted_score:.1%} (cost={total_cost:.1f}/{max_possible_cost:.1f})")
    print(f"{'='*60}")

    # Error cost breakdown
    print(f"\nError Cost Breakdown:")
    cost_by_pair = defaultdict(lambda: {"count": 0, "cost": 0.0})
    for gt_lang, preds in confusion.items():
        for pred_lang, count in preds.items():
            if gt_lang != pred_lang and pred_lang != "(none)":
                c = get_error_cost(gt_lang, pred_lang)
                if c > 0:
                    pair = f"{gt_lang.upper()}->{pred_lang.upper()}"
                    cost_by_pair[pair]["count"] += count
                    cost_by_pair[pair]["cost"] += c * count
    if cost_by_pair:
        sorted_pairs = sorted(cost_by_pair.items(), key=lambda x: -x[1]["cost"])
        print(f"  {'Error':>15} {'Count':>6} {'Unit':>6} {'Total':>8}")
        print(f"  {'-'*40}")
        for pair, info in sorted_pairs[:10]:
            unit_cost = info["cost"] / max(info["count"], 1)
            print(f"  {pair:>15} {info['count']:>6} {unit_cost:>5.1f}x {info['cost']:>7.1f}")

    # Per-language precision/recall
    print(f"\n{'Language':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 55)
    all_langs = sorted(set(list(per_lang.keys()) + [g["lang"] for g in gt_map.values()]))
    for lang in all_langs:
        s = per_lang[lang]
        prec = s["tp"] / max(s["tp"] + s["fp"], 1)
        rec = s["tp"] / max(s["tp"] + s["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 0.001)
        print(f"{lang.upper():<10} {prec:>7.1%} {rec:>7.1%} {f1:>7.1%} {s['tp']:>5} {s['fp']:>5} {s['fn']:>5}")

    # Confusion matrix
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    pred_langs = sorted(set(
        pl for row in confusion.values() for pl in row.keys()
    ))
    header = f"{'':>10}" + "".join(f"{p.upper():>8}" for p in pred_langs)
    print(header)
    print("-" * len(header))
    for gt_lang in sorted(confusion.keys()):
        row = f"{gt_lang.upper():>10}"
        for pl in pred_langs:
            count = confusion[gt_lang].get(pl, 0)
            cell = f"{count}" if count else "·"
            row += f"{cell:>8}"
        print(row)

    return {
        "segment_accuracy": accuracy,
        "segment_accuracy_excl_unk": accuracy_excl_unk,
        "segment_accuracy_nah_spa": nah_spa_acc,
        "duration_accuracy": dur_acc_all,
        "duration_accuracy_excl_unk": dur_acc_excl_unk,
        "duration_accuracy_nah_spa": dur_acc_nah_spa,
        "weighted_score": weighted_score,
        "total_cost": total_cost,
        "total": total,
        "correct": correct,
        "nah_spa_total": nah_spa_total,
        "nah_spa_correct": nah_spa_correct,
        "per_lang": dict(per_lang),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <ground_truth.json> <pipeline_output.srt>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])

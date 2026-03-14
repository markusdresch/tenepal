"""Hallucination statistics for Whisper NAH evaluation.

Computes confusion matrices, hallucination rates, and 95% Wilson score confidence
intervals from corpus evaluation results (baseline.json).

Whisper "hallucination" in this context: Whisper assigns a non-NAH ISO-639-1 code
(e.g. "da", "nl", "no") to audio that is ground-truth Nahuatl (NAH).  Because
Whisper was not trained on Nahuatl, it maps NAH speech to phonetically similar
European languages — most commonly Danish, Dutch, and Norwegian.

Usage (CLI):
    # Formatted table (default)
    python -m tools.corpus.hallucination_stats

    # JSON output
    python -m tools.corpus.hallucination_stats --json

    # Specific results file
    python -m tools.corpus.hallucination_stats --results tools/corpus/results/baseline.json

    # Save JSON stats to file
    python -m tools.corpus.hallucination_stats --output stats.json

Usage (module):
    from tools.corpus.hallucination_stats import (
        compute_hallucination_stats,
        format_confusion_matrix,
        wilson_ci,
    )
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_RESULTS = REPO_ROOT / "tools" / "corpus" / "results" / "baseline.json"

# ─── ISO 639-1 name mapping for common Whisper hallucination targets ──────────

_LANG_NAMES: dict[str, str] = {
    "da": "Danish",
    "nl": "Dutch",
    "no": "Norwegian",
    "sv": "Swedish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "hu": "Hungarian",
    "sk": "Slovak",
    "hr": "Croatian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "sl": "Slovenian",
    "bg": "Bulgarian",
    "uk": "Ukrainian",
    "ru": "Russian",
    "nah": "Nahuatl",
    "nah-naz": "Nahuatl (Zacatlan)",
    "unknown": "unknown",
}


# ─── Wilson score confidence interval ─────────────────────────────────────────


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    The Wilson score interval has better coverage than the naive Wald interval,
    especially for small samples and extreme proportions.

    Formula (two-sided):
        centre = (p + z^2 / (2n)) / (1 + z^2 / n)
        margin = z * sqrt(p*(1-p)/n + z^2/(4*n^2)) / (1 + z^2/n)
        lower  = centre - margin
        upper  = centre + margin

    Args:
        successes:  Number of successes (must be <= total)
        total:      Total number of trials
        z:          Z-score for desired confidence level (default 1.96 = 95%)

    Returns:
        (lower, upper) bounds in [0.0, 1.0].  Returns (0.0, 1.0) when total == 0.
    """
    if total == 0:
        return (0.0, 1.0)

    n = total
    p = successes / n
    z2 = z * z

    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)

    return (round(lower, 6), round(upper, 6))


# ─── Core statistics computation ──────────────────────────────────────────────


def compute_hallucination_stats(results_json: dict[str, Any]) -> dict[str, Any]:
    """Compute Whisper hallucination statistics from corpus evaluation results.

    Reads per_sample entries from the evaluation results dict (same schema as
    baseline.json produced by tools.corpus.evaluate).  Samples with skipped=True
    or whisper_detected_lang="unknown" are excluded from statistics.

    Args:
        results_json:  Parsed evaluation results dict (as written by evaluate_corpus)

    Returns:
        dict with keys:
          n_evaluated         — samples with whisper_detected_lang != "unknown"
          n_hallucinated      — whisper_detected_lang not in ("nah", "nah-naz", "unknown")
          hallucination_rate  — n_hallucinated / n_evaluated
          hallucination_ci_95 — Wilson CI tuple (lower, upper)
          confusion_matrix    — dict[lang_code, {count, pct, ci_95}]
          top_hallucination_langs — [(lang, count), ...] sorted descending
          correct_detections  — whisper returned a NAH variant code (rare)
          unknown_detections  — whisper_detected_lang == "unknown"
    """
    per_sample: list[dict] = results_json.get("per_sample", [])

    # Collect counts
    lang_counts: dict[str, int] = {}
    n_unknown = 0
    n_total_with_whisper = 0  # samples that had whisper output (not unknown)

    # NAH codes: Whisper won't return "nah" but treat it as non-hallucination
    _nah_codes = {"nah", "nah-naz", "nah-nci"}

    for sample in per_sample:
        if sample.get("skipped", False):
            continue

        wlang = sample.get("whisper_detected_lang", "unknown")
        if wlang == "unknown":
            n_unknown += 1
            continue

        n_total_with_whisper += 1
        lang_counts[wlang] = lang_counts.get(wlang, 0) + 1

    n_evaluated = n_total_with_whisper
    n_correct = sum(lang_counts.get(code, 0) for code in _nah_codes)
    n_hallucinated = n_evaluated - n_correct

    hallucination_rate = n_hallucinated / n_evaluated if n_evaluated > 0 else 0.0
    hallucination_ci = wilson_ci(n_hallucinated, n_evaluated)

    # Build confusion matrix: NAH input -> Whisper detected language
    confusion_matrix: dict[str, dict] = {}
    for lang, count in lang_counts.items():
        pct = count / n_evaluated if n_evaluated > 0 else 0.0
        ci = wilson_ci(count, n_evaluated)
        confusion_matrix[lang] = {
            "count": count,
            "pct": round(pct, 6),
            "ci_95": ci,
        }

    # Top hallucination languages (exclude NAH codes, sort by count descending)
    top_langs = sorted(
        [(lang, cnt) for lang, cnt in lang_counts.items() if lang not in _nah_codes],
        key=lambda x: -x[1],
    )

    return {
        "n_evaluated": n_evaluated,
        "n_hallucinated": n_hallucinated,
        "hallucination_rate": round(hallucination_rate, 6),
        "hallucination_ci_95": hallucination_ci,
        "confusion_matrix": confusion_matrix,
        "top_hallucination_langs": top_langs,
        "correct_detections": n_correct,
        "unknown_detections": n_unknown,
    }


# ─── Formatted output ─────────────────────────────────────────────────────────


def format_confusion_matrix(stats: dict[str, Any]) -> str:
    """Format hallucination statistics as a human-readable text table.

    Args:
        stats:  Output of compute_hallucination_stats()

    Returns:
        Multi-line string suitable for printing to terminal or writing to file.
    """
    lines: list[str] = []

    n_eval = stats["n_evaluated"]
    n_hall = stats["n_hallucinated"]
    n_correct = stats["correct_detections"]
    n_unknown = stats["unknown_detections"]
    rate = stats["hallucination_rate"]
    ci_lo, ci_hi = stats["hallucination_ci_95"]

    lines.append("Whisper Hallucination Analysis (NAH Input)")
    lines.append("=" * 50)
    lines.append(f"Evaluated:          {n_eval} samples (all ground truth = NAH)")
    lines.append(f"Unknown (no Whisper output): {n_unknown}")
    lines.append(f"Correct (Whisper -> NAH):    {n_correct}")
    lines.append(f"Hallucinated:       {n_hall}")
    lines.append(
        f"Hallucination rate: {rate * 100:.1f}%  "
        f"(95% CI: {ci_lo * 100:.1f}%-{ci_hi * 100:.1f}%)"
    )

    if stats["confusion_matrix"]:
        lines.append("")
        lines.append("NAH -> Whisper Detected Language:")

        # Sort by count descending
        cm_sorted = sorted(
            stats["confusion_matrix"].items(),
            key=lambda kv: -kv[1]["count"],
        )
        for lang, info in cm_sorted:
            lang_name = _LANG_NAMES.get(lang, lang)
            count = info["count"]
            pct = info["pct"] * 100
            ci_l, ci_h = info["ci_95"]
            label = f"{lang} ({lang_name})"
            lines.append(
                f"  {label:<22} {count:>4}  ({pct:5.1f}%)  "
                f"[CI: {ci_l * 100:.1f}%-{ci_h * 100:.1f}%]"
            )

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Whisper hallucination statistics from corpus evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Formatted table (default)
  python -m tools.corpus.hallucination_stats

  # JSON output
  python -m tools.corpus.hallucination_stats --json

  # Specify results file
  python -m tools.corpus.hallucination_stats --results tools/corpus/results/baseline.json

  # Save JSON stats to file
  python -m tools.corpus.hallucination_stats --output stats.json
        """,
    )
    parser.add_argument(
        "--results",
        default=str(_DEFAULT_RESULTS),
        help=f"Path to evaluation results JSON (default: {_DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output computed stats as JSON instead of formatted table",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON stats to this file path (optional)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}", file=sys.stderr)
        return 1

    with open(results_path, encoding="utf-8") as f:
        results_json = json.load(f)

    # Handle placeholder results gracefully
    res = results_json.get("results", {})
    if res.get("accuracy") is None and not results_json.get("per_sample"):
        print("No evaluation data available yet.")
        print(f"  Status: {results_json.get('status', 'unknown')}")
        corpus_stats = results_json.get("corpus_stats", {})
        if corpus_stats:
            print(f"  Samples indexed: {corpus_stats.get('total_samples', '?')}")
            print(f"  Audio available: {corpus_stats.get('audio_available', 0)}")
        print()
        print("Run evaluation first:")
        print("  python -m tools.corpus.evaluate --limit 30 --output tools/corpus/results/")
        return 0

    stats = compute_hallucination_stats(results_json)

    # Optionally write JSON stats to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(stats, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Stats written to: {output_path}")

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print(format_confusion_matrix(stats))

    return 0


if __name__ == "__main__":
    sys.exit(main())

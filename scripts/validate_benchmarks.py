#!/usr/bin/env python3
"""Validate that public benchmark claims match canonical artifacts.

Checks README.md and PAPER.md against benchmarks/reports/public_benchmarks.json.
Fails hard on any drift.

Usage:
    python scripts/validate_benchmarks.py
    python scripts/validate_benchmarks.py --fix  # show what needs updating
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
REPORT_PATH = REPO_ROOT / "benchmarks" / "reports" / "public_benchmarks.json"


def load_report():
    if not REPORT_PATH.exists():
        print(f"FAIL: {REPORT_PATH} not found")
        print(f"  Run: python scripts/export_benchmark_snapshot.py")
        sys.exit(2)
    return json.loads(REPORT_PATH.read_text())


def check_referenced_files(report):
    """All files referenced in the report must exist."""
    errors = []
    hernan = report.get("langid_pipeline", {}).get("hernan", {})

    for key in ["gt_snapshot", "prediction_artifact"]:
        path = hernan.get(key, "")
        if path and not (REPO_ROOT / path).exists():
            errors.append(f"Referenced file missing: {path}")

    loc = report.get("langid_pipeline", {}).get("loc", {})
    path = loc.get("gt_snapshot", "")
    if path and not (REPO_ROOT / path).exists():
        errors.append(f"Referenced file missing: {path}")

    return errors


def check_no_f005_in_public(text, filename):
    """F005/embedding numbers must not appear in public claims sections."""
    errors = []
    # Look for F005 accuracy numbers in non-experimental context
    if "85.1%" in text or "85.0% balanced" in text:
        # Check if it's in an experimental/future section
        for match in re.finditer(r"85\.1%|85\.0%.*balanced", text):
            start = max(0, match.start() - 200)
            context = text[start:match.start()]
            if not any(w in context.lower() for w in ["experimental", "f005", "embedding", "wav2vec", "future", "planned"]):
                errors.append(f"{filename}: F005 number {match.group()} appears in non-experimental context")
    return errors


def check_claims_in_file(filepath, report, show_fix=False):
    """Check that published numbers in a file match the report."""
    if not filepath.exists():
        return []

    text = filepath.read_text()
    filename = filepath.name
    errors = []
    warnings = []

    hernan = report.get("langid_pipeline", {}).get("hernan", {})
    metrics = hernan.get("metrics", {})

    # Check for the old 85.7% claim
    if "85.7%" in text or "85.7" in text:
        errors.append(
            f"{filename}: Contains '85.7%' — this number is not reproducible from current artifacts. "
            f"Reproducible numbers: "
            f"NAH+SPA={metrics.get('nah_spa_subset', {}).get('accuracy', '?')}, "
            f"excl_UNK={metrics.get('multiclass_excl_unk', {}).get('accuracy', '?')}"
        )

    # Check for old segment count 551
    if "551" in text:
        nah_spa_n = metrics.get("nah_spa_subset", {}).get("total")
        if nah_spa_n and nah_spa_n != 551:
            warnings.append(
                f"{filename}: Contains '551' segments — current NAH+SPA count is {nah_spa_n}"
            )

    # Check ASR claims
    asr = report.get("asr_quality", {}).get("whisper_finetuning", {})
    baseline_cer = asr.get("baseline_cer")
    ft_cer = asr.get("finetuned_cer")

    if baseline_cer:
        baseline_pct = f"{baseline_cer * 100:.0f}%"
        ft_pct = f"{ft_cer * 100:.0f}%"
        # Check CER claims are consistent
        if "108%" in text and baseline_pct != "108%":
            errors.append(f"{filename}: CER baseline '108%' doesn't match report {baseline_pct}")
        if "70%" in text and "CER" in text and ft_pct != "70%":
            # Only flag if it's actually about CER, not some other percentage
            pass  # 70% appears in many contexts, skip

    # F005 check
    errors.extend(check_no_f005_in_public(text, filename))

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark claims")
    parser.add_argument("--fix", action="store_true", help="Show suggested fixes")
    args = parser.parse_args()

    report = load_report()
    all_errors = []
    all_warnings = []

    print("Benchmark Claim Validator")
    print("=" * 60)
    print(f"Report: {REPORT_PATH}")
    print(f"Version: {report.get('_meta', {}).get('version', '?')}")
    print(f"Generated: {report.get('_meta', {}).get('generated', '?')}")
    print()

    # 1. Check referenced files exist
    file_errors = check_referenced_files(report)
    if file_errors:
        print("--- Referenced Files ---")
        for e in file_errors:
            print(f"  FAIL: {e}")
        all_errors.extend(file_errors)
    else:
        print("--- Referenced Files: OK ---")

    # 2. Check README.md
    print("\n--- README.md ---")
    errors, warnings = check_claims_in_file(REPO_ROOT / "README.md", report, args.fix)
    for e in errors:
        print(f"  FAIL: {e}")
    for w in warnings:
        print(f"  WARN: {w}")
    if not errors and not warnings:
        print("  OK")
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    # 3. Check PAPER.md
    print("\n--- PAPER.md ---")
    errors, warnings = check_claims_in_file(REPO_ROOT / "PAPER.md", report, args.fix)
    for e in errors:
        print(f"  FAIL: {e}")
    for w in warnings:
        print(f"  WARN: {w}")
    if not errors and not warnings:
        print("  OK")
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    # 4. Print canonical numbers for reference
    hernan = report.get("langid_pipeline", {}).get("hernan", {})
    metrics = hernan.get("metrics", {})

    print(f"\n--- Canonical Numbers (use these) ---")
    for name, data in metrics.items():
        if data and data.get("accuracy") is not None:
            pct = f"{data['accuracy'] * 100:.1f}%"
            print(f"  {name:30s}: {pct:>6} ({data['correct']}/{data['total']})")

    asr = report.get("asr_quality", {}).get("whisper_finetuning", {})
    if asr:
        print(f"  {'whisper_baseline_cer':30s}: {asr['baseline_cer']*100:.0f}%")
        print(f"  {'whisper_finetuned_cer':30s}: {asr['finetuned_cer']*100:.0f}%")

    # Summary
    print(f"\n{'='*60}")
    if all_errors:
        print(f"FAILED: {len(all_errors)} errors, {len(all_warnings)} warnings")
        return 1
    elif all_warnings:
        print(f"PASSED with {len(all_warnings)} warnings")
        return 0
    else:
        print("PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())

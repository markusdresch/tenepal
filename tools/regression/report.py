#!/usr/bin/env python3
"""
Tenepal regression report generation and A/B comparison.

Usage:
    # Compare two reports:
    python tools/regression/report.py baseline.json current.json

    # Programmatic usage:
    from tools.regression.report import generate_report, compare_reports, format_summary
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path


REPORT_SCHEMA_VERSION = "1.0"


def get_git_commit() -> str | None:
    """Get current git commit hash (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def generate_report(runner_result: dict, git_commit: str | None = None) -> dict:
    """Convert runner result to full report with metadata.

    Args:
        runner_result: Output from run_regression()
        git_commit: Optional git SHA (auto-detected if None)

    Returns:
        Full report dict matching REPORT_SCHEMA
    """
    if git_commit is None:
        git_commit = get_git_commit()

    run_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Calculate by_language stats
    by_language: dict[str, dict[str, int]] = {}
    for result in runner_result.get("results", []):
        # Skip results without expected_lang (from dry-run mode)
        if "expected_lang" not in result:
            continue

        lang = result["expected_lang"]
        if lang not in by_language:
            by_language[lang] = {"total": 0, "passed": 0}

        by_language[lang]["total"] += 1
        if result.get("passed"):
            by_language[lang]["passed"] += 1

    total_clips = runner_result.get("total_clips", 0)
    passed = runner_result.get("passed", 0)
    pass_rate = passed / total_clips if total_clips > 0 else 0.0

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "git_commit": git_commit,
        "manifest_version": runner_result.get("manifest_version", "unknown"),
        "modal_args": runner_result.get("modal_args", {}),
        "summary": {
            "total_clips": total_clips,
            "passed": passed,
            "failed": runner_result.get("failed", 0),
            "skipped": runner_result.get("skipped", 0),
            "pass_rate": round(pass_rate, 3),
            "duration_s": round(runner_result.get("duration_s", 0), 1),
            "by_language": by_language,
        },
        "results": runner_result.get("results", []),
    }


def save_report(report: dict, output_dir: str = "tools/regression/reports") -> str:
    """Save report to JSON file.

    Filename: regression_{timestamp}.json

    Args:
        report: Report dict from generate_report()
        output_dir: Directory for report files

    Returns:
        Path to saved report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use timestamp from run_id for filename (replace colons for filesystem safety)
    timestamp = report["run_id"].replace(":", "-").replace("T", "_")
    filename = f"regression_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(filepath)


def compare_reports(report_a: dict, report_b: dict) -> dict:
    """Compare two regression reports.

    Args:
        report_a: Baseline report (A)
        report_b: Current report (B)

    Returns:
        Comparison dict with regressions, improvements, and deltas
    """
    a_run_id = report_a.get("run_id", "unknown")
    b_run_id = report_b.get("run_id", "unknown")

    a_summary = report_a.get("summary", {})
    b_summary = report_b.get("summary", {})

    a_pass_rate = a_summary.get("pass_rate", 0)
    b_pass_rate = b_summary.get("pass_rate", 0)

    # Build lookup dicts by clip ID
    a_results = {r["id"]: r for r in report_a.get("results", [])}
    b_results = {r["id"]: r for r in report_b.get("results", [])}

    a_ids = set(a_results.keys())
    b_ids = set(b_results.keys())

    common_ids = a_ids & b_ids
    new_clips = list(b_ids - a_ids)
    removed_clips = list(a_ids - b_ids)

    regressions = []
    improvements = []
    unchanged = []

    for clip_id in common_ids:
        a_passed = a_results[clip_id].get("passed", False)
        b_passed = b_results[clip_id].get("passed", False)

        if a_passed and not b_passed:
            regressions.append(clip_id)
        elif not a_passed and b_passed:
            improvements.append(clip_id)
        else:
            unchanged.append(clip_id)

    return {
        "a_run_id": a_run_id,
        "b_run_id": b_run_id,
        "a_pass_rate": a_pass_rate,
        "b_pass_rate": b_pass_rate,
        "pass_rate_delta": round(b_pass_rate - a_pass_rate, 3),
        "regressions": sorted(regressions),
        "improvements": sorted(improvements),
        "unchanged": sorted(unchanged),
        "new_clips": sorted(new_clips),
        "removed_clips": sorted(removed_clips),
    }


def format_summary(report: dict, show_failures: bool = True) -> str:
    """Format report as human-readable summary.

    Args:
        report: Report dict from generate_report()
        show_failures: If True, list failed clips

    Returns:
        Formatted summary string
    """
    run_id = report.get("run_id", "unknown")
    git_commit = report.get("git_commit", "unknown")
    summary = report.get("summary", {})

    total = summary.get("total_clips", 0)
    passed = summary.get("passed", 0)
    pass_rate = summary.get("pass_rate", 0)
    duration = summary.get("duration_s", 0)
    by_language = summary.get("by_language", {})

    lines = [
        "=" * 60,
        f"Regression Report: {run_id} ({git_commit})",
        "=" * 60,
        f"Summary: {passed}/{total} passed ({pass_rate:.1%})",
        f"Duration: {duration:.1f}s",
        "",
        "By Language:",
    ]

    for lang in sorted(by_language.keys()):
        stats = by_language[lang]
        lang_total = stats.get("total", 0)
        lang_passed = stats.get("passed", 0)
        lang_rate = lang_passed / lang_total if lang_total > 0 else 0
        lines.append(f"  {lang}: {lang_passed}/{lang_total} ({lang_rate:.0%})")

    if show_failures:
        failed_clips = [
            r for r in report.get("results", [])
            if not r.get("passed") and not r.get("skipped")
        ]
        if failed_clips:
            lines.append("")
            lines.append("Failed:")
            for clip in failed_clips:
                clip_id = clip.get("id", "unknown")
                reasons = clip.get("fail_reasons", ["unknown reason"])
                reason_str = "; ".join(reasons[:2])  # Limit to first 2 reasons
                lines.append(f"  - {clip_id}: {reason_str}")

    lines.append("=" * 60)

    return "\n".join(lines)


def format_comparison(comparison: dict) -> str:
    """Format comparison result as human-readable string.

    Args:
        comparison: Result from compare_reports()

    Returns:
        Formatted comparison string
    """
    lines = [
        f"Comparing {comparison['a_run_id']} -> {comparison['b_run_id']}",
        f"Pass rate: {comparison['a_pass_rate']:.1%} -> {comparison['b_pass_rate']:.1%} ({comparison['pass_rate_delta']:+.1%})",
    ]

    if comparison["regressions"]:
        lines.append(f"REGRESSIONS: {comparison['regressions']}")

    if comparison["improvements"]:
        lines.append(f"Improvements: {comparison['improvements']}")

    if comparison["new_clips"]:
        lines.append(f"New clips: {comparison['new_clips']}")

    if comparison["removed_clips"]:
        lines.append(f"Removed clips: {comparison['removed_clips']}")

    unchanged_count = len(comparison["unchanged"])
    lines.append(f"Unchanged: {unchanged_count} clips")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare regression reports")
    parser.add_argument("report_a", nargs="?", help="First report (baseline)")
    parser.add_argument("report_b", nargs="?", help="Second report (current)")
    args = parser.parse_args()

    if not args.report_a or not args.report_b:
        parser.print_help()
        print("\nUsage: python tools/regression/report.py baseline.json current.json")
        exit(1)

    # Load and compare
    with open(args.report_a) as f:
        a = json.load(f)
    with open(args.report_b) as f:
        b = json.load(f)

    comparison = compare_reports(a, b)
    print(format_comparison(comparison))

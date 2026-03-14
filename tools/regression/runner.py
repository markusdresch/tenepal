#!/usr/bin/env python3
"""
Tenepal regression test runner.

Usage:
    python tools/regression/runner.py --dry-run        # Validate manifest only
    python tools/regression/runner.py -v               # Run with verbose output
    python tools/regression/runner.py --whisper-model small  # Use smaller model
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def load_manifest(path: str = "tools/regression/manifest.json") -> dict:
    """Load and validate test manifest.

    Args:
        path: Path to manifest JSON file

    Returns:
        Parsed manifest dict

    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest schema is invalid
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Validate required fields
    required_fields = ["version", "clips"]
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")

    clip_required = ["id", "source", "start_s", "end_s"]
    for i, clip in enumerate(manifest["clips"]):
        for field in clip_required:
            if field not in clip:
                raise ValueError(f"Clip {i} missing required field: {field}")

    return manifest


def extract_clip(
    source: str, start_s: float, end_s: float, pad_s: float = 0.5
) -> bytes:
    """Extract audio clip using ffmpeg, return as bytes.

    Args:
        source: Path to source audio/video file
        start_s: Start time in seconds
        end_s: End time in seconds
        pad_s: Padding to add before/after (default 0.5s)

    Returns:
        WAV audio bytes (16kHz mono PCM)

    Raises:
        RuntimeError: If ffmpeg fails
        FileNotFoundError: If source file doesn't exist
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    # Apply padding
    actual_start = max(0, start_s - pad_s)
    actual_end = end_s + pad_s

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-i",
        str(source_path),
        "-ss",
        str(actual_start),
        "-to",
        str(actual_end),
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:500]}")

    return result.stdout


def evaluate_clip(clip_def: dict, output_text: str) -> dict:
    """Evaluate clip output against expectations.

    Args:
        clip_def: Clip definition from manifest
        output_text: Full output text (SRT content or similar)

    Returns:
        {
            "id": clip_id,
            "passed": bool,
            "expect_present_results": [{"pattern": str, "found": bool}],
            "expect_absent_results": [{"pattern": str, "found": bool}],
            "fail_reasons": [str],
            "output_preview": str (first 200 chars)
        }
    """
    clip_id = clip_def["id"]
    expect_present = clip_def.get("expect_present", [])
    expect_absent = clip_def.get("expect_absent", [])

    present_results = []
    absent_results = []
    fail_reasons = []

    # Check expect_present patterns - ALL must be found
    for pattern in expect_present:
        found = bool(re.search(pattern, output_text, re.IGNORECASE))
        present_results.append({"pattern": pattern, "found": found})
        if not found:
            fail_reasons.append(f"Missing expected pattern: {pattern}")

    # Check expect_absent patterns - NONE should be found
    for pattern in expect_absent:
        found = bool(re.search(pattern, output_text, re.IGNORECASE))
        absent_results.append({"pattern": pattern, "found": found})
        if found:
            fail_reasons.append(f"Found forbidden pattern: {pattern}")

    passed = len(fail_reasons) == 0

    return {
        "id": clip_id,
        "passed": passed,
        "expect_present_results": present_results,
        "expect_absent_results": absent_results,
        "fail_reasons": fail_reasons,
        "output_preview": output_text[:200] if output_text else "(empty)",
    }


def run_regression(
    manifest_path: str = "tools/regression/manifest.json",
    modal_args: dict | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    save_report: bool = True,
    report_dir: str = "tools/regression/reports",
) -> dict:
    """Run all regression tests.

    Args:
        manifest_path: Path to test manifest
        modal_args: Override args for Modal (e.g., {"whisper_model": "small"})
        dry_run: If True, just validate manifest without running
        verbose: Print progress for each clip
        save_report: If True, save JSON report after run
        report_dir: Directory for report files

    Returns:
        {
            "run_id": str (timestamp),
            "manifest_version": str,
            "total_clips": int,
            "passed": int,
            "failed": int,
            "skipped": int,
            "duration_s": float,
            "results": [clip_result_dicts],
            "modal_args": dict,
            "report_path": str (if save_report=True)
        }
    """
    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and validate manifest
    manifest = load_manifest(manifest_path)
    clips = manifest["clips"]
    total_clips = len(clips)

    if verbose:
        print(f"[regression] Loaded manifest v{manifest['version']} with {total_clips} clips")

    if dry_run:
        # Just validate manifest and source files
        results = []
        skipped = 0
        for clip in clips:
            source_exists = Path(clip["source"]).exists()
            if not source_exists:
                skipped += 1
                results.append({
                    "id": clip["id"],
                    "passed": False,
                    "fail_reasons": [f"Source file not found: {clip['source']}"],
                    "skipped": True,
                })
                if verbose:
                    print(f"  [SKIP] {clip['id']}: source not found")
            else:
                results.append({
                    "id": clip["id"],
                    "passed": True,
                    "fail_reasons": [],
                    "skipped": False,
                })
                if verbose:
                    print(f"  [OK] {clip['id']}: source exists")

        duration = time.time() - start_time
        return {
            "run_id": run_id,
            "manifest_version": manifest["version"],
            "total_clips": total_clips,
            "passed": total_clips - skipped,
            "failed": 0,
            "skipped": skipped,
            "duration_s": duration,
            "results": results,
            "modal_args": modal_args or {},
            "dry_run": True,
        }

    # Lazy import Modal to avoid init overhead in dry-run mode
    try:
        from tenepal_modal import process_film
    except ImportError as e:
        print(f"[regression] ERROR: Cannot import Modal pipeline: {e}")
        print("[regression] Make sure you're running from project root with Modal configured")
        sys.exit(1)

    # Prepare modal args with defaults
    default_modal_args = {
        "whisper_model": "medium",
        "use_demucs": False,  # Skip demucs for speed on short clips
        "phoneme_compare": True,
        "whisper_rescue": True,
        "phone_vote": True,
    }
    if modal_args:
        default_modal_args.update(modal_args)

    results = []
    passed = 0
    failed = 0
    skipped = 0

    timeout_per_clip = manifest.get("timeout_per_clip_s", 60)

    for i, clip in enumerate(clips):
        clip_id = clip["id"]
        if verbose:
            print(f"[{i+1}/{total_clips}] {clip_id}...")

        # Check source exists
        if not Path(clip["source"]).exists():
            skipped += 1
            results.append({
                "id": clip_id,
                "passed": False,
                "fail_reasons": [f"Source file not found: {clip['source']}"],
                "skipped": True,
            })
            if verbose:
                print(f"  [SKIP] Source not found: {clip['source']}")
            continue

        try:
            # Extract clip
            clip_bytes = extract_clip(
                clip["source"],
                clip["start_s"],
                clip["end_s"],
                clip.get("pad_s", 0.5),
            )

            # Process through Modal
            clip_start = time.time()
            srt_output = process_film.remote(
                audio_bytes=clip_bytes,
                filename=f"{clip_id}.wav",
                **default_modal_args,
            )
            clip_duration = time.time() - clip_start

            # Evaluate results
            result = evaluate_clip(clip, srt_output)
            result["duration_s"] = clip_duration
            result["skipped"] = False
            result["expected_lang"] = clip.get("expected_lang", "UNK")
            results.append(result)

            if result["passed"]:
                passed += 1
                if verbose:
                    print(f"  [PASS] {clip_duration:.1f}s")
            else:
                failed += 1
                if verbose:
                    print(f"  [FAIL] {', '.join(result['fail_reasons'])}")

        except Exception as e:
            failed += 1
            results.append({
                "id": clip_id,
                "passed": False,
                "fail_reasons": [f"Exception: {str(e)[:200]}"],
                "skipped": False,
                "error": True,
            })
            if verbose:
                print(f"  [ERROR] {e}")

    duration = time.time() - start_time

    result = {
        "run_id": run_id,
        "manifest_version": manifest["version"],
        "total_clips": total_clips,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration_s": duration,
        "results": results,
        "modal_args": default_modal_args,
        "dry_run": False,
    }

    # Generate and save report if requested
    if save_report:
        from tools.regression.report import generate_report, save_report as _save_report
        report = generate_report(result)
        report_path = _save_report(report, report_dir)
        result["report_path"] = report_path
        if verbose:
            print(f"[regression] Report saved: {report_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Tenepal regression tests")
    parser.add_argument("--manifest", default="tools/regression/manifest.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model size")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of summary")
    parser.add_argument("--no-report", action="store_true", help="Skip saving report")
    parser.add_argument("--report-dir", default="tools/regression/reports", help="Report output directory")
    args = parser.parse_args()

    result = run_regression(
        manifest_path=args.manifest,
        modal_args={"whisper_model": args.whisper_model},
        dry_run=args.dry_run,
        verbose=args.verbose,
        save_report=not args.no_report and not args.dry_run,
        report_dir=args.report_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Use formatted summary for non-dry-run results
        if not result.get("dry_run"):
            from tools.regression.report import format_summary, generate_report
            report = generate_report(result)
            print(format_summary(report))
            if result.get("report_path"):
                print(f"\nReport saved: {result['report_path']}")
        else:
            # Simpler output for dry-run
            print(f"\n{'='*60}")
            print(f"DRY-RUN: {result['passed']}/{result['total_clips']} passed")
            if result["skipped"] > 0:
                print(f"Skipped: {result['skipped']} (missing source files)")
            print(f"Duration: {result['duration_s']:.1f}s")

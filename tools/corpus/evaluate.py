"""Corpus evaluator: runs Tenepal pipeline on Amith NAH corpus and computes accuracy.

Evaluation measures how often the pipeline correctly identifies Nahuatl (NAH)
on the ground-truth-labeled Amith Zacatlan/Tepetzintla corpus.

Usage (CLI):
    # Validate manifest — no audio required
    python -m tools.corpus.evaluate --dry-run

    # Evaluate first 30 samples via Modal
    python -m tools.corpus.evaluate --limit 30 --output tools/corpus/results/

    # Full evaluation (all 55 samples)
    python -m tools.corpus.evaluate --output tools/corpus/results/

    # Local evaluation (no Modal)
    python -m tools.corpus.evaluate --local --limit 10 --output tools/corpus/results/

Usage (module):
    from tools.corpus.evaluate import evaluate_corpus, CorpusResult
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.corpus.hallucination_stats import compute_hallucination_stats


# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_MANIFEST = REPO_ROOT / "tools" / "corpus" / "manifest.json"
_DEFAULT_RESULTS_DIR = REPO_ROOT / "tools" / "corpus" / "results"

EVALUATOR_VERSION = "1.0"


# ─── CorpusResult dataclass ───────────────────────────────────────────────────

@dataclass
class CorpusResult:
    """Evaluation result for one corpus sample.

    Attributes:
        sample_id               Unique sample identifier from manifest
        ground_truth            Ground truth language label (always NAH)
        detected_lang           Language detected by pipeline (NAH, SPA, OTH, etc.)
        correct                 Whether detected_lang == ground_truth
        confidence              Language confidence score from pipeline (0.0-1.0)
        processing_time_s       Wall-clock time to process sample (seconds)
        whisper_hallucination   True if Whisper assigned a non-NAH language to NAH audio
        whisper_detected_lang   Language Whisper inferred before hallucination filtering
        error                   Error message if pipeline raised an exception
        skipped                 True if audio not available or sample was skipped
    """

    sample_id: str
    ground_truth: str
    detected_lang: str
    correct: bool
    confidence: float
    processing_time_s: float
    whisper_hallucination: bool
    whisper_detected_lang: str
    error: str = ""
    skipped: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Manifest loading ─────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> dict:
    """Load and validate corpus manifest.

    Args:
        manifest_path: Path to manifest.json produced by tools.corpus.index

    Returns:
        Parsed manifest dict

    Raises:
        FileNotFoundError: If manifest does not exist
        ValueError: If manifest schema is invalid
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    required_fields = ["version", "corpus", "samples"]
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")

    sample_required = ["id", "audio_path", "ground_truth_lang", "audio_available"]
    for i, sample in enumerate(manifest["samples"]):
        for field in sample_required:
            if field not in sample:
                raise ValueError(f"Sample {i} missing required field: {field}")

    return manifest


# ─── Language extraction from pipeline output ─────────────────────────────────

def _extract_dominant_language(srt_output: str) -> tuple[str, float]:
    """Extract the dominant language label from SRT pipeline output.

    The SRT output contains language tags embedded in segment lines.
    This function counts occurrences of each language tag and returns
    the most frequent non-OTH tag (falling back to OTH if all are OTH).

    Args:
        srt_output: Raw SRT text from the pipeline

    Returns:
        (dominant_lang, confidence) where confidence is the fraction of
        segments assigned to the dominant language.
    """
    import re

    if not srt_output or not srt_output.strip():
        return "OTH", 0.0

    # Match language tags in SRT lines — e.g. "[NAH]", "[SPA]", "[OTH]", "[MAY]"
    lang_pattern = re.compile(r"\[(NAH|SPA|OTH|MAY|LAT|DEU|ENG|FRA|ITA|UNK)\]")
    tags = lang_pattern.findall(srt_output)

    if not tags:
        return "OTH", 0.0

    # Count occurrences
    from collections import Counter
    counts = Counter(tags)
    total = len(tags)

    # Find dominant: prefer non-OTH languages
    non_oth = {lang: cnt for lang, cnt in counts.items() if lang != "OTH"}
    if non_oth:
        dominant = max(non_oth, key=non_oth.get)  # type: ignore[arg-type]
        confidence = non_oth[dominant] / total
    else:
        dominant = "OTH"
        confidence = counts["OTH"] / total

    return dominant, round(confidence, 4)


def _extract_whisper_lang(srt_output: str) -> str:
    """Extract Whisper's detected language from debug lines in SRT output.

    The pipeline embeds Whisper language info in debug comments or metadata
    lines when available. Falls back to "unknown" if not present.

    Args:
        srt_output: Raw SRT text from the pipeline

    Returns:
        ISO 639-1 language code (e.g. "da", "nl", "nah") or "unknown"
    """
    import re

    if not srt_output:
        return "unknown"

    # Look for whisper_lang= patterns in debug output
    m = re.search(r"whisper_lang[=:]\s*([a-z]{2,5})", srt_output, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Look for "lang: XX" in SRT comments
    m = re.search(r"#\s*lang:\s*([a-z]{2,5})", srt_output, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return "unknown"


# ─── Main evaluation function ─────────────────────────────────────────────────

def evaluate_corpus(
    manifest_path: Path,
    results_dir: Path,
    whisper_model: str = "small",
    sample_limit: int | None = None,
    dry_run: bool = False,
    local: bool = False,
    verbose: bool = False,
) -> dict:
    """Evaluate pipeline on corpus samples and compute NAH accuracy.

    Iterates through the corpus manifest, runs each available audio file
    through the Tenepal pipeline (Modal or local), and computes:
    - Overall NAH detection accuracy
    - Confusion matrix (NAH -> detected language)
    - Whisper hallucination rate and language distribution

    Args:
        manifest_path:  Path to manifest.json from tools.corpus.index
        results_dir:    Directory to write evaluation results
        whisper_model:  Whisper model size ("tiny", "small", "medium", "large")
        sample_limit:   Maximum samples to evaluate (None = all available)
        dry_run:        If True, validate manifest and report counts; no pipeline run
        local:          If True, use local src/tenepal instead of Modal
        verbose:        Print per-sample progress

    Returns:
        dict with shape:
        {
            "version": "1.0",
            "timestamp": "...",
            "config": {...},
            "results": {
                "accuracy": float,
                "n_samples": int,
                "n_correct": int,
                "n_skipped": int,
                "confusion_matrix": {"NAH->NAH": int, ...},
                "whisper_hallucination_rate": float,
                "whisper_hallucination_langs": {"da": int, ...},
                "mean_confidence": float,
                "mean_processing_time_s": float,
            },
            "per_sample": [CorpusResult.to_dict(), ...]
        }

    Side effects:
        Writes two files to results_dir after evaluation (when n_evaluated > 0):
        - baseline.json: full evaluation results (accuracy, confusion matrix, per-sample)
        - hallucination_stats.json: Wilson-CI confusion matrix for Whisper hallucination
          patterns (which languages Whisper assigns to NAH audio), generated by
          tools.corpus.hallucination_stats.compute_hallucination_stats()
    """
    manifest = load_manifest(manifest_path)
    all_samples = manifest["samples"]

    # Filter to available audio only
    available = [s for s in all_samples if s.get("audio_available", False)]

    if sample_limit is not None:
        available = available[:sample_limit]

    n_total_manifest = len(all_samples)
    n_available = len(available)

    if verbose:
        print(f"[evaluate] Manifest: {n_total_manifest} total samples, "
              f"{n_available} audio available")
        print(f"[evaluate] Evaluating {len(available)} samples")
        if dry_run:
            print("[evaluate] Dry run — pipeline will NOT be invoked")

    # ── Dry-run: just report manifest state ───────────────────────────────────
    if dry_run:
        n_unavailable = n_total_manifest - n_available
        print()
        print(f"  Corpus:           {manifest.get('corpus', 'unknown')}")
        print(f"  Total samples:    {n_total_manifest}")
        print(f"  Audio available:  {n_available}")
        print(f"  Audio missing:    {n_unavailable}")
        if sample_limit:
            print(f"  Would evaluate:   min({sample_limit}, {n_available}) = "
                  f"{min(sample_limit, n_available)}")
        else:
            print(f"  Would evaluate:   {n_available}")
        print()
        if n_unavailable > 0:
            print(f"  NOTE: {n_unavailable} samples lack audio.")
            print("  Download corpus audio with: scripts/download_amith_corpus.py")
            print("  Requires: MOZILLA_DC_TOKEN env var (CC-BY-ND-4.0 license)")
        return {
            "version": EVALUATOR_VERSION,
            "dry_run": True,
            "manifest": str(manifest_path),
            "n_total": n_total_manifest,
            "n_available": n_available,
            "n_unavailable": n_unavailable,
            "would_evaluate": min(sample_limit or n_available, n_available),
        }

    # ── Full evaluation ───────────────────────────────────────────────────────
    if not available:
        print("[evaluate] WARNING: No audio available. Creating placeholder results.")
        return _create_placeholder_results(manifest, results_dir, whisper_model, sample_limit)

    # Lazy import pipeline
    process_film_fn = _load_pipeline(local=local)

    per_sample: list[CorpusResult] = []
    n_correct = 0
    n_skipped = 0
    whisper_hallucination_count = 0
    whisper_lang_distribution: dict[str, int] = {}
    confidence_sum = 0.0
    time_sum = 0.0

    for i, sample in enumerate(available):
        sample_id = sample["id"]
        ground_truth = sample["ground_truth_lang"]
        audio_path = Path(sample["audio_path"])

        if verbose:
            print(f"[{i+1}/{len(available)}] {sample_id[:60]}...")

        # Verify audio exists
        if not audio_path.exists():
            n_skipped += 1
            per_sample.append(CorpusResult(
                sample_id=sample_id,
                ground_truth=ground_truth,
                detected_lang="SKIP",
                correct=False,
                confidence=0.0,
                processing_time_s=0.0,
                whisper_hallucination=False,
                whisper_detected_lang="unknown",
                error=f"Audio file not found: {audio_path}",
                skipped=True,
            ))
            if verbose:
                print(f"  [SKIP] Audio not found: {audio_path}")
            continue

        # Run pipeline
        t_start = time.time()
        try:
            srt_output = _run_pipeline(
                process_film_fn=process_film_fn,
                audio_path=audio_path,
                sample_id=sample_id,
                whisper_model=whisper_model,
                local=local,
            )
            processing_time = time.time() - t_start

            detected_lang, confidence = _extract_dominant_language(srt_output)
            whisper_lang = _extract_whisper_lang(srt_output)

            is_correct = detected_lang == ground_truth

            # Whisper hallucination: Whisper assigned a non-NAH language to NAH audio
            # "nah" would be correct; anything else (da, nl, no, etc.) is a hallucination
            is_whisper_hallucination = whisper_lang not in ("nah", "unknown", "NAH")

            if is_correct:
                n_correct += 1
            if is_whisper_hallucination:
                whisper_hallucination_count += 1
                whisper_lang_distribution[whisper_lang] = (
                    whisper_lang_distribution.get(whisper_lang, 0) + 1
                )

            confidence_sum += confidence
            time_sum += processing_time

            result = CorpusResult(
                sample_id=sample_id,
                ground_truth=ground_truth,
                detected_lang=detected_lang,
                correct=is_correct,
                confidence=confidence,
                processing_time_s=round(processing_time, 2),
                whisper_hallucination=is_whisper_hallucination,
                whisper_detected_lang=whisper_lang,
            )

            if verbose:
                status = "PASS" if is_correct else "FAIL"
                print(f"  [{status}] detected={detected_lang} conf={confidence:.2f} "
                      f"whisper={whisper_lang} time={processing_time:.1f}s")

        except Exception as exc:
            processing_time = time.time() - t_start
            n_skipped += 1
            error_msg = str(exc)[:300]
            per_sample.append(CorpusResult(
                sample_id=sample_id,
                ground_truth=ground_truth,
                detected_lang="ERROR",
                correct=False,
                confidence=0.0,
                processing_time_s=round(processing_time, 2),
                whisper_hallucination=False,
                whisper_detected_lang="unknown",
                error=error_msg,
                skipped=True,
            ))
            if verbose:
                print(f"  [ERROR] {error_msg[:100]}")
            continue

        per_sample.append(result)

    # ── Compute aggregate metrics ─────────────────────────────────────────────
    n_evaluated = len(per_sample) - n_skipped
    accuracy = n_correct / n_evaluated if n_evaluated > 0 else 0.0
    whisper_hallucination_rate = (
        whisper_hallucination_count / n_evaluated if n_evaluated > 0 else 0.0
    )
    mean_confidence = confidence_sum / n_evaluated if n_evaluated > 0 else 0.0
    mean_time = time_sum / n_evaluated if n_evaluated > 0 else 0.0

    # Build confusion matrix (all entries for NAH ground truth)
    confusion_matrix: dict[str, int] = {}
    for r in per_sample:
        if not r.skipped:
            key = f"{r.ground_truth}->{r.detected_lang}"
            confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    output = {
        "version": EVALUATOR_VERSION,
        "timestamp": timestamp,
        "config": {
            "manifest": str(manifest_path),
            "whisper_model": whisper_model,
            "sample_limit": sample_limit,
            "sample_count": n_evaluated,
            "local": local,
        },
        "results": {
            "accuracy": round(accuracy, 4),
            "n_samples": n_evaluated,
            "n_correct": n_correct,
            "n_skipped": n_skipped,
            "confusion_matrix": confusion_matrix,
            "whisper_hallucination_rate": round(whisper_hallucination_rate, 4),
            "whisper_hallucination_langs": dict(
                sorted(whisper_lang_distribution.items(), key=lambda x: -x[1])
            ),
            "mean_confidence": round(mean_confidence, 4),
            "mean_processing_time_s": round(mean_time, 2),
        },
        "per_sample": [r.to_dict() for r in per_sample],
    }

    # Write results
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = results_dir / "baseline.json"
    baseline_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"\n[evaluate] Results written to {baseline_path}")

    # Compute and write hallucination statistics
    if n_evaluated > 0:
        hall_stats = compute_hallucination_stats(output)
        hall_stats_path = results_dir / "hallucination_stats.json"
        hall_stats_path.write_text(
            json.dumps(hall_stats, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        if verbose:
            from tools.corpus.hallucination_stats import format_confusion_matrix
            print()
            print(format_confusion_matrix(hall_stats))

    return output


# ─── Placeholder results (no audio available) ─────────────────────────────────

def _create_placeholder_results(
    manifest: dict,
    results_dir: Path,
    whisper_model: str,
    sample_limit: int | None,
) -> dict:
    """Create placeholder results structure when no audio is available.

    Captures the manifest state and documents what's needed for full evaluation.
    This allows PAPER.md planning to proceed with known sample counts even
    before corpus audio is downloaded.
    """
    n_total = len(manifest["samples"])
    n_available = sum(1 for s in manifest["samples"] if s.get("audio_available", False))
    n_would_eval = min(sample_limit or n_total, n_available)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    output = {
        "version": EVALUATOR_VERSION,
        "timestamp": timestamp,
        "status": "PLACEHOLDER — audio not yet downloaded",
        "config": {
            "manifest": str(_DEFAULT_MANIFEST),
            "whisper_model": whisper_model,
            "sample_limit": sample_limit,
            "sample_count": 0,
            "local": False,
        },
        "corpus_stats": {
            "total_samples": n_total,
            "audio_available": n_available,
            "audio_missing": n_total - n_available,
            "would_evaluate_at_limit": n_would_eval,
        },
        "download_instructions": {
            "script": "scripts/download_amith_corpus.py",
            "env_var": "MOZILLA_DC_TOKEN",
            "license": "CC-BY-ND-4.0",
            "source": "Mozilla Common Voice / Amith Zacatlan-Tepetzintla corpus",
            "steps": [
                "1. Accept CC-BY-ND-4.0 license at Mozilla Data Collective",
                "2. Export MOZILLA_DC_TOKEN=<your-token>",
                "3. Run: python scripts/download_amith_corpus.py",
                "4. Then: python -m tools.corpus.evaluate --limit 30 --output tools/corpus/results/",
            ],
        },
        "results": {
            "accuracy": None,
            "n_samples": 0,
            "n_correct": 0,
            "n_skipped": n_total,
            "confusion_matrix": {},
            "whisper_hallucination_rate": None,
            "whisper_hallucination_langs": {},
            "mean_confidence": None,
            "mean_processing_time_s": None,
        },
        "per_sample": [],
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = results_dir / "baseline.json"
    baseline_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[evaluate] Placeholder results written to {baseline_path}")
    print(f"[evaluate] {n_total} samples indexed, {n_available} audio available.")
    print("[evaluate] Download corpus audio to run full evaluation.")

    return output


# ─── Pipeline loader ──────────────────────────────────────────────────────────

def _load_pipeline(local: bool) -> Any:
    """Lazily import and return the pipeline function.

    Args:
        local: If True, load from src/tenepal; otherwise load Modal function

    Returns:
        Callable that processes audio and returns SRT text
    """
    if local:
        try:
            # Local pipeline — direct import from src
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from tenepal.pipeline import process_audio  # type: ignore[import]
            return process_audio
        except ImportError as exc:
            raise ImportError(
                f"Local pipeline not available: {exc}\n"
                "Make sure you're running from project root with venv activated."
            ) from exc
    else:
        try:
            from tenepal_modal import process_film  # type: ignore[import]
            return process_film
        except ImportError as exc:
            raise ImportError(
                f"Modal pipeline not available: {exc}\n"
                "Run: pip install modal && modal token new"
            ) from exc


def _run_pipeline(
    process_film_fn: Any,
    audio_path: Path,
    sample_id: str,
    whisper_model: str,
    local: bool,
) -> str:
    """Run the pipeline on a single audio file and return SRT output.

    Args:
        process_film_fn: Pipeline callable (Modal remote or local)
        audio_path:      Path to audio file
        sample_id:       Sample identifier for logging
        whisper_model:   Whisper model size
        local:           Whether running local pipeline

    Returns:
        SRT text output from pipeline
    """
    audio_bytes = audio_path.read_bytes()

    if local:
        # Local pipeline: direct call
        result = process_film_fn(
            audio_bytes=audio_bytes,
            filename=f"{sample_id}.wav",
            whisper_model=whisper_model,
        )
    else:
        # Modal: call remote
        result = process_film_fn.remote(
            audio_bytes=audio_bytes,
            filename=f"{sample_id}.wav",
            whisper_model=whisper_model,
            use_demucs=False,  # Skip demucs for corpus speed
            phoneme_compare=False,
            whisper_rescue=False,
            phone_vote=False,
        )

    # process_film returns {"srt": str, "stats": dict} — extract SRT text
    if isinstance(result, dict):
        return result.get("srt", "")
    return result  # fallback: treat as plain string (local pipeline compatibility)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Tenepal NAH detection accuracy on Amith corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate manifest (no audio required)
  python -m tools.corpus.evaluate --dry-run

  # Evaluate first 30 samples via Modal
  python -m tools.corpus.evaluate --limit 30 --output tools/corpus/results/

  # Full evaluation (all available samples)
  python -m tools.corpus.evaluate --output tools/corpus/results/

  # Local evaluation (no Modal, requires src/tenepal)
  python -m tools.corpus.evaluate --local --limit 10

  # JSON output
  python -m tools.corpus.evaluate --dry-run --json
        """,
    )
    parser.add_argument(
        "--manifest",
        default=str(_DEFAULT_MANIFEST),
        help=f"Path to corpus manifest (default: {_DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_RESULTS_DIR),
        help=f"Output directory for results (default: {_DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all available)",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model size: tiny|small|medium|large (default: small)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and report sample counts; do not run pipeline",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local src/tenepal pipeline instead of Modal",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample progress",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON to stdout",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    results_dir = Path(args.output)

    try:
        result = evaluate_corpus(
            manifest_path=manifest_path,
            results_dir=results_dir,
            whisper_model=args.whisper_model,
            sample_limit=args.limit,
            dry_run=args.dry_run,
            local=args.local,
            verbose=args.verbose,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.dry_run:
        # Summary already printed inside evaluate_corpus
        return 0

    # Print summary
    res = result.get("results", {})
    print()
    print("=" * 60)
    print("NAH CORPUS EVALUATION RESULTS")
    print("=" * 60)
    accuracy = res.get("accuracy")
    n_samples = res.get("n_samples", 0)
    n_correct = res.get("n_correct", 0)
    n_skipped = res.get("n_skipped", 0)

    if accuracy is None:
        print(f"Status     : PLACEHOLDER (audio not downloaded)")
        print(f"Samples    : {result.get('corpus_stats', {}).get('total_samples', '?')} indexed")
    else:
        print(f"Accuracy   : {accuracy * 100:.1f}%  ({n_correct}/{n_samples} correct)")
        print(f"Skipped    : {n_skipped}")
        hall_rate = res.get("whisper_hallucination_rate")
        if hall_rate is not None:
            print(f"Whisper hallucination rate: {hall_rate * 100:.1f}%")
        hall_langs = res.get("whisper_hallucination_langs", {})
        if hall_langs:
            top = sorted(hall_langs.items(), key=lambda x: -x[1])[:5]
            print(f"Top Whisper hallucination langs: "
                  f"{', '.join(f'{l}({n})' for l, n in top)}")
        conf = res.get("mean_confidence")
        if conf is not None:
            print(f"Mean confidence: {conf:.3f}")
        mean_t = res.get("mean_processing_time_s")
        if mean_t is not None:
            print(f"Mean processing time: {mean_t:.1f}s/sample")
        cm = res.get("confusion_matrix", {})
        if cm:
            print()
            print("Confusion matrix (NAH -> detected):")
            for key, count in sorted(cm.items()):
                print(f"  {key}: {count}")

    # Print hallucination statistics if available
    if accuracy is not None and n_samples > 0:
        hall_stats = compute_hallucination_stats(result)
        from tools.corpus.hallucination_stats import format_confusion_matrix
        print()
        print(format_confusion_matrix(hall_stats))

    output_path = results_dir / "baseline.json"
    print()
    print(f"Results saved to: {output_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

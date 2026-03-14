#!/usr/bin/env python3
"""Evaluate Nahuatl lexicon matches from existing fused-IPA SRT files."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Support running as: python scripts/modal_lexicon_eval.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_nahuatl_lexicon_class():
    """Load NahuatlLexicon without importing the full tenepal package tree."""
    module_path = SRC_DIR / "tenepal" / "language" / "nahuatl_lexicon.py"
    module_name = "nahuatl_lexicon_local"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.NahuatlLexicon


NahuatlLexicon = _load_nahuatl_lexicon_class()

LANG_RE = re.compile(r"\[([A-Z]{3})(?:\|[^\]]+)?\]")
FUSED_RE = re.compile(r"^\s*♫?\s*fused:\s*(.+?)\s*$", re.IGNORECASE)


def parse_nah_segments(srt_path: Path) -> list[dict]:
    """Parse NAH-tagged segments with fused IPA from an SRT file."""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    segments: list[dict] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        cue_id = lines[0].strip()
        timestamp = lines[1].strip()
        content_lines = lines[2:]
        header = content_lines[0]

        lang_match = LANG_RE.search(header)
        if not lang_match:
            continue
        lang = lang_match.group(1)
        if lang != "NAH":
            continue

        fused = None
        for line in content_lines:
            match = FUSED_RE.match(line)
            if match:
                fused = match.group(1).strip()
                break
        if not fused:
            continue

        phonemes = [p for p in fused.split() if p]
        if not phonemes:
            continue

        segments.append(
            {
                "file": str(srt_path),
                "file_name": srt_path.name,
                "cue_id": cue_id,
                "timestamp": timestamp,
                "lang": lang,
                "phonemes": phonemes,
            }
        )

    return segments


def run_eval(srt_pattern: str, min_freq: int, output_path: Path) -> dict:
    """Run lexicon subsequence matching and return full report data."""
    srt_files = sorted(Path(p) for p in glob.glob(srt_pattern))
    if not srt_files:
        raise FileNotFoundError(f"No SRT files matched pattern: {srt_pattern}")

    lexicon_path = SRC_DIR / "tenepal" / "data" / "nah_lexicon_merged.json"
    lexicon = NahuatlLexicon(lexicon_path=lexicon_path, min_freq=min_freq)

    all_segments: list[dict] = []
    for srt_file in srt_files:
        all_segments.extend(parse_nah_segments(srt_file))

    total_matches = 0
    segments_with_matches = 0
    unique_words: set[str] = set()
    segment_rows: list[dict] = []

    for segment in all_segments:
        matches = lexicon.match_subsequence(segment["phonemes"])
        match_rows = []
        for m in matches:
            total_matches += 1
            unique_words.add(m.word)
            match_rows.append(
                {
                    "word": m.word,
                    "score": round(m.score, 6),
                    "position": m.start_idx,
                    "length": m.length,
                }
            )

        if match_rows:
            segments_with_matches += 1

        segment_rows.append(
            {
                "file": segment["file"],
                "file_name": segment["file_name"],
                "cue_id": segment["cue_id"],
                "timestamp": segment["timestamp"],
                "lang": segment["lang"],
                "phonemes": segment["phonemes"],
                "matches": match_rows,
            }
        )

    total_segments = len(all_segments)
    coverage = (segments_with_matches / total_segments * 100.0) if total_segments else 0.0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "srt_pattern": srt_pattern,
            "min_freq": min_freq,
            "output": str(output_path),
        },
        "summary": {
            "files_matched": len(srt_files),
            "nah_segments": total_segments,
            "segments_with_matches": segments_with_matches,
            "coverage_percent": round(coverage, 2),
            "total_matches": total_matches,
            "unique_words_count": len(unique_words),
            "unique_words": sorted(unique_words),
        },
        "segments": segment_rows,
    }
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local post-processing: evaluate NAH lexicon matches from fused-IPA SRT."
    )
    parser.add_argument(
        "--srt",
        default="validation_video/Hernán-*.srt",
        help='Glob pattern for input SRT files (default: "validation_video/Hernán-*.srt").',
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=50,
        help="Minimum corpus frequency for NahuatlLexicon entries (default: 50).",
    )
    parser.add_argument(
        "--output",
        default="validation_video/modal_lexicon_eval.json",
        help="Path to output JSON report.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    output_path = Path(args.output)

    try:
        report = run_eval(args.srt, args.min_freq, output_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"Saved report: {output_path}")
    print(f"NAH segments: {summary['nah_segments']}")
    print(f"Segments with matches: {summary['segments_with_matches']}")
    print(f"Coverage: {summary['coverage_percent']}%")
    print(f"Total matches: {summary['total_matches']}")
    print(f"Unique words: {summary['unique_words_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

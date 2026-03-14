#!/usr/bin/env python3
"""Minimal regression harness for fixed Hernan NAH subset.

Computes:
- LID accuracy from language tags in predicted SRT files
- WER/CER against fixture reference_text (when available)

Usage:
  python tests/regression/eval_regression.py \
    --fixture tests/regression/fixtures/hernan_nah_25.json \
    --prediction validation_video/Hernán-1-1-1.srt \
    --prediction validation_video/Hernán-1-1-2.srt \
    --output tests/regression/reports/hernan_nah_25_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KNOWN_LANGS = {"NAH", "SPA", "MAY", "LAT", "OTH", "ENG", "DEU", "FRA", "ITA", "UNK"}


@dataclass
class Cue:
    start_s: float
    end_s: float
    lang: str
    text: str


def parse_time(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    seconds, millis = rest.split(".")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def normalize_text(text: str) -> str:
    cleaned = text.casefold()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Keep letters/numbers/whitespace plus a small IPA/orthography-friendly range.
    cleaned = re.sub(r"[^\w\s\u00C0-\u024F\u0250-\u02AF'’ːˑˈˌ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def edit_distance(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        curr = [i]
        for j, token_b in enumerate(seq_b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(
                min(
                    prev[j] + 1,       # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost # substitution
                )
            )
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> tuple[float | None, int, int]:
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens:
        return None, 0, 0
    edits = edit_distance(ref_tokens, hyp_tokens)
    return edits / len(ref_tokens), edits, len(ref_tokens)


def char_error_rate(reference: str, hypothesis: str) -> tuple[float | None, int, int]:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return None, 0, 0
    edits = edit_distance(ref_chars, hyp_chars)
    return edits / len(ref_chars), edits, len(ref_chars)


def _parse_lang(tag_content: str) -> str:
    parts = [p.strip().upper() for p in tag_content.split("|")]
    for part in parts:
        if part in KNOWN_LANGS:
            return part
    return "UNK"


def parse_srt(path: Path) -> list[Cue]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    cues: list[Cue] = []
    for block in [b for b in text.strip().split("\n\n") if b.strip()]:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        m = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1].strip(),
        )
        if not m:
            continue
        cue_text = " ".join(lines[2:]).strip()
        tag_match = re.search(r"\[(.*?)\]", cue_text)
        lang = "UNK"
        if tag_match:
            lang = _parse_lang(tag_match.group(1))
            cue_text = cue_text[tag_match.end():].strip()
        cue_text = cue_text.replace("[LLM]", "").strip()
        # Drop backend debug tails when present.
        for marker in ("♫allo:", "♬allo:", "♫w2v2:", "♬w2v2:"):
            if marker in cue_text:
                cue_text = cue_text.split(marker)[0].strip()
        cues.append(
            Cue(
                start_s=parse_time(m.group(1)),
                end_s=parse_time(m.group(2)),
                lang=lang,
                text=cue_text,
            )
        )
    return cues


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def source_key_from_audio(path_str: str) -> str:
    return Path(path_str).stem.casefold()


def choose_prediction_file(source_audio: str, prediction_files: list[Path]) -> Path | None:
    source_key = source_key_from_audio(source_audio)
    ranked: list[tuple[int, Path]] = []
    for path in prediction_files:
        stem = path.stem.casefold()
        score = 0
        if source_key in stem:
            score = len(source_key)
        elif stem in source_key:
            score = len(stem)
        ranked.append((score, path))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if ranked and ranked[0][0] > 0:
        return ranked[0][1]
    return prediction_files[0] if prediction_files else None


def evaluate(fixture_path: Path, prediction_files: list[Path]) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    segments = fixture.get("segments", [])

    parsed_predictions = {str(path): parse_srt(path) for path in prediction_files}

    per_segment = []
    lid_correct = 0
    lid_total = 0
    wer_edits = wer_ref_words = 0
    cer_edits = cer_ref_chars = 0

    for seg in segments:
        prediction_file = choose_prediction_file(seg["source_audio"], prediction_files)
        prediction_cues = parsed_predictions.get(str(prediction_file), []) if prediction_file else []

        best = None
        best_overlap = 0.0
        for cue in prediction_cues:
            ov = overlap_seconds(seg["start_s"], seg["end_s"], cue.start_s, cue.end_s)
            if ov > best_overlap:
                best_overlap = ov
                best = cue

        pred_lang = best.lang if best else "NONE"
        pred_text = best.text if best else ""
        matched = best is not None and best_overlap > 0.0

        lid_total += 1
        if pred_lang == str(seg["expected_lang"]).upper():
            lid_correct += 1

        seg_wer, seg_wer_edits, seg_wer_total = word_error_rate(seg.get("reference_text", ""), pred_text)
        seg_cer, seg_cer_edits, seg_cer_total = char_error_rate(seg.get("reference_text", ""), pred_text)
        wer_edits += seg_wer_edits
        wer_ref_words += seg_wer_total
        cer_edits += seg_cer_edits
        cer_ref_chars += seg_cer_total

        per_segment.append(
            {
                "id": seg["id"],
                "source_audio": seg["source_audio"],
                "expected_lang": str(seg["expected_lang"]).upper(),
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "prediction_file": str(prediction_file) if prediction_file else None,
                "matched": matched,
                "overlap_s": round(best_overlap, 3),
                "pred_lang": pred_lang,
                "reference_text": seg.get("reference_text", ""),
                "pred_text": pred_text,
                "wer": None if seg_wer is None else round(seg_wer, 4),
                "cer": None if seg_cer is None else round(seg_cer, 4),
            }
        )

    summary = {
        "n_segments": len(segments),
        "n_predictions": sum(len(v) for v in parsed_predictions.values()),
        "lid_accuracy": round((lid_correct / lid_total) if lid_total else 0.0, 4),
        "lid_correct": lid_correct,
        "lid_total": lid_total,
        "wer": None if wer_ref_words == 0 else round(wer_edits / wer_ref_words, 4),
        "wer_edits": wer_edits,
        "wer_ref_words": wer_ref_words,
        "cer": None if cer_ref_chars == 0 else round(cer_edits / cer_ref_chars, 4),
        "cer_edits": cer_edits,
        "cer_ref_chars": cer_ref_chars,
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_path": str(fixture_path),
        "prediction_files": [str(path) for path in prediction_files],
        "summary": summary,
        "per_segment": per_segment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed regression subset with WER/CER + LID accuracy.")
    parser.add_argument(
        "--fixture",
        default="tests/regression/fixtures/hernan_nah_25.json",
        help="Path to regression fixture JSON",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction SRT path (pass multiple --prediction for multi-audio fixtures)",
    )
    parser.add_argument(
        "--output",
        default="tests/regression/reports/hernan_nah_25_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    prediction_files = [Path(p) for p in args.prediction]
    for path in prediction_files:
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    report = evaluate(fixture_path, prediction_files)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]
    print(f"Saved report: {out_path}")
    print(
        "LID accuracy: "
        f"{summary['lid_correct']}/{summary['lid_total']} = {summary['lid_accuracy']:.1%}"
    )
    print(f"WER: {summary['wer'] if summary['wer'] is not None else 'n/a'}")
    print(f"CER: {summary['cer'] if summary['cer'] is not None else 'n/a'}")


if __name__ == "__main__":
    main()

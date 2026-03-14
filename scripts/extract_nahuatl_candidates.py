#!/usr/bin/env python3
"""Extract likely Nahuatl passages from OCR-readable PDFs.

This is a heuristic marker extractor, not a full language identifier.
It prefers recall (find more candidates) over precision.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


SPANISH_STOPWORDS = {
    "de", "la", "el", "los", "las", "y", "que", "en", "un", "una", "por", "para",
    "con", "se", "es", "del", "al", "como", "más", "mas", "su", "sus", "ya",
    "pero", "si", "sí", "no", "le", "lo", "a", "o", "u", "yo", "tu", "tú", "mi",
    "me", "te", "nos", "vos", "fue", "era", "son", "ser", "ha", "han",
}

NAH_FUNCTION_TOKENS = {
    "in", "yn", "auh", "ca", "nican", "inic", "ye", "amo", "niman", "ompa",
    "tlaca", "mexica", "tenochtitlan", "teotl", "teochichimeca", "huehuetque",
}


WORD_RE = re.compile(r"[a-záéíóúüñç'\-]+", re.IGNORECASE)


def run_pdftotext(pdf_path: Path) -> str:
    cmd = ["pdftotext", "-layout", str(pdf_path), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return proc.stdout.decode("utf-8", errors="ignore")


def normalize_token(token: str) -> str:
    return token.lower().strip(".,;:!?¡¿()[]{}\"“”'`")


def score_paragraph(text: str) -> tuple[float, dict[str, float]]:
    tokens = [normalize_token(t) for t in WORD_RE.findall(text)]
    tokens = [t for t in tokens if t]
    if len(tokens) < 5:
        return 0.0, {"token_count": len(tokens)}

    spanish_hits = sum(1 for t in tokens if t in SPANISH_STOPWORDS)
    function_hits = sum(1 for t in tokens if t in NAH_FUNCTION_TOKENS)

    cluster_hits = 0
    for t in tokens:
        if "tl" in t:
            cluster_hits += 1
        if "tz" in t:
            cluster_hits += 1
        if "hu" in t:
            cluster_hits += 1
        if t.endswith("tzin"):
            cluster_hits += 2
        if t.startswith(("cuauh", "quauh", "teo", "xoch", "tla")):
            cluster_hits += 1

    spanish_ratio = spanish_hits / max(len(tokens), 1)

    # Positive signals from Nahuatl-like morphology and function tokens.
    score = function_hits * 2.0 + cluster_hits * 0.6
    # Penalize strongly Spanish-looking prose.
    score -= spanish_ratio * 6.0
    # Mild boost for long dense lexical strings (often glossary-like lines).
    score += min(len(tokens), 30) * 0.03

    details = {
        "token_count": len(tokens),
        "spanish_hits": spanish_hits,
        "function_hits": function_hits,
        "cluster_hits": cluster_hits,
        "spanish_ratio": round(spanish_ratio, 4),
        "score": round(score, 4),
    }
    return score, details


def extract_candidates(pdf_path: Path, min_score: float) -> list[dict]:
    text = run_pdftotext(pdf_path)
    pages = text.split("\f")
    rows: list[dict] = []

    for i, page in enumerate(pages, start=1):
        lines = [ln.rstrip() for ln in page.splitlines()]
        # Group contiguous non-empty lines as a paragraph.
        buff: list[str] = []
        for ln in lines + [""]:
            if ln.strip():
                buff.append(ln.strip())
                continue

            if buff:
                para = " ".join(buff)
                buff = []
                score, details = score_paragraph(para)
                if score >= min_score:
                    rows.append(
                        {
                            "pdf": str(pdf_path),
                            "page": i,
                            "score": round(score, 4),
                            "text": para,
                            "details": details,
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract likely Nahuatl passages from PDF text.")
    parser.add_argument("--pdf", action="append", required=True, help="Input PDF path (repeatable)")
    parser.add_argument("--out-dir", default="codices/extracted/nahuatl_markers", help="Output directory")
    parser.add_argument("--min-score", type=float, default=4.5, help="Minimum heuristic score")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: list[dict] = []
    for pdf in args.pdf:
        pdf_path = Path(pdf)
        rows = extract_candidates(pdf_path, args.min_score)
        combined.extend(rows)

        stem = pdf_path.stem
        jsonl_path = out_dir / f"{stem}.nahuatl_candidates.jsonl"
        tsv_path = out_dir / f"{stem}.nahuatl_candidates.tsv"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with tsv_path.open("w", encoding="utf-8") as f:
            f.write("page\tscore\ttext\n")
            for row in rows:
                text = row["text"].replace("\t", " ").replace("\n", " ")
                f.write(f'{row["page"]}\t{row["score"]}\t{text}\n')

        print(f"{pdf_path}: {len(rows)} candidates")
        print(f"  {jsonl_path}")
        print(f"  {tsv_path}")

    combined.sort(key=lambda r: (-r["score"], r["pdf"], r["page"]))
    combined_path = out_dir / "combined.nahuatl_candidates.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Combined: {len(combined)} candidates")
    print(f"  {combined_path}")


if __name__ == "__main__":
    main()

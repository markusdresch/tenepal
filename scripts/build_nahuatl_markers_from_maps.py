#!/usr/bin/env python3
"""Build Nahuatl marker tables from editable Epitran-style maps.

This script applies longest-match grapheme rules from CSV maps to marker tokens,
then exports IPA-oriented marker lists for modern and classical Nahuatl.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    grapheme: str
    ipa: str


def load_rules(path: Path) -> list[Rule]:
    rules: list[Rule] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = (row.get("grapheme") or "").strip()
            ipa = (row.get("ipa") or "").strip()
            if not g or not ipa:
                continue
            rules.append(Rule(grapheme=g.lower(), ipa=ipa))
    # Longest-match-first.
    rules.sort(key=lambda r: len(r.grapheme), reverse=True)
    return rules


def transliterate(token: str, rules: list[Rule]) -> tuple[str, int]:
    out: list[str] = []
    i = 0
    unknown = 0
    s = token.lower()
    while i < len(s):
        matched = False
        for r in rules:
            if s.startswith(r.grapheme, i):
                out.append(r.ipa)
                i += len(r.grapheme)
                matched = True
                break
        if matched:
            continue
        ch = s[i]
        if ch.isalpha():
            unknown += 1
        i += 1
    return (" ".join(x for x in out if x), unknown)


def load_markers(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            token = parts[0].strip()
            count = int(parts[1])
            file_count = int(parts[2])
            if token:
                rows.append({"token": token, "count": count, "file_count": file_count})
    return rows


def build_variant(
    rows: list[dict], rules: list[Rule], out_prefix: Path
) -> dict:
    token_rows: list[dict] = []
    ipa_counts: Counter[str] = Counter()
    unknown_rows: list[dict] = []

    for row in rows:
        token = row["token"]
        ipa, unknown = transliterate(token, rules)
        if ipa:
            token_rows.append(
                {
                    "token": token,
                    "count": row["count"],
                    "file_count": row["file_count"],
                    "ipa": ipa,
                    "unknown_chars": unknown,
                }
            )
            ipa_counts[ipa] += row["count"]
        if unknown > 0:
            unknown_rows.append(
                {
                    "token": token,
                    "count": row["count"],
                    "file_count": row["file_count"],
                    "unknown_chars": unknown,
                }
            )

    token_path = out_prefix.with_suffix(".token_to_ipa.tsv")
    with token_path.open("w", encoding="utf-8") as f:
        f.write("token\tcount\tfile_count\tipa\tunknown_chars\n")
        for r in sorted(token_rows, key=lambda x: (-x["count"], x["token"])):
            f.write(
                f"{r['token']}\t{r['count']}\t{r['file_count']}\t{r['ipa']}\t{r['unknown_chars']}\n"
            )

    ipa_path = out_prefix.with_suffix(".ipa_markers.tsv")
    with ipa_path.open("w", encoding="utf-8") as f:
        f.write("ipa\tcount\n")
        for ipa, count in ipa_counts.most_common():
            f.write(f"{ipa}\t{count}\n")

    unknown_path = out_prefix.with_suffix(".unknown.tsv")
    with unknown_path.open("w", encoding="utf-8") as f:
        f.write("token\tcount\tfile_count\tunknown_chars\n")
        for r in sorted(unknown_rows, key=lambda x: (-x["count"], x["token"])):
            f.write(f"{r['token']}\t{r['count']}\t{r['file_count']}\t{r['unknown_chars']}\n")

    return {
        "token_to_ipa_tsv": str(token_path),
        "ipa_markers_tsv": str(ipa_path),
        "unknown_tsv": str(unknown_path),
        "tokens_total": len(rows),
        "tokens_with_ipa": len(token_rows),
        "tokens_with_unknown": len(unknown_rows),
        "unique_ipa_markers": len(ipa_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nahuatl IPA markers from map CSV rules.")
    parser.add_argument(
        "--input",
        default="codices/extracted/zacatlan_markers/markers_nahuatl_strict.tsv",
        help="Input marker TSV (token_proxy,count,file_count).",
    )
    parser.add_argument(
        "--modern-map",
        default="src/tenepal/data/epitran_maps/nah-modern.csv",
        help="Modern Nahuatl map CSV.",
    )
    parser.add_argument(
        "--classical-map",
        default="src/tenepal/data/epitran_maps/nah-classical.csv",
        help="Classical Nahuatl map CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="codices/extracted/zacatlan_markers/map_markers",
        help="Output directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_markers(input_path)
    modern_rules = load_rules(Path(args.modern_map))
    classical_rules = load_rules(Path(args.classical_map))

    modern_summary = build_variant(rows, modern_rules, out_dir / "nah_modern")
    classical_summary = build_variant(rows, classical_rules, out_dir / "nah_classical")

    summary = {
        "input": str(input_path),
        "rows": len(rows),
        "modern_map": str(Path(args.modern_map)),
        "classical_map": str(Path(args.classical_map)),
        "modern": modern_summary,
        "classical": classical_summary,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {summary_path}")
    for section in ("modern", "classical"):
        print(f"  {section}: {summary[section]['ipa_markers_tsv']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure language distribution from SRT files for A/B comparison.

Usage:
    python scripts/measure_lang_distribution.py results/Hernán-1-1.baseline.srt
    python scripts/measure_lang_distribution.py results/*.srt
"""
import re
import sys
from pathlib import Path

SPA_IPA_PHONEMES = {"b", "d", "ɡ", "ɲ", "ɾ", "β", "ð", "ɣ"}
NAH_MARKERS = {"tɬ", "kʷ", "ʔ", "ɬ"}
MAY_MARKERS = {"kʼ", "tʼ", "tsʼ", "pʼ"}


def analyze_srt(path: str) -> dict:
    text = Path(path).read_text()
    blocks = re.split(r"\n\n+", text.strip())

    lang_counts: dict[str, int] = {}
    oth_total = 0
    oth_with_spa_ipa = 0
    oth_examples: list[str] = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        header = lines[2]
        m = re.match(r"^\[([A-Z]+)\|", header)
        if not m:
            continue

        lang = m.group(1)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        if lang == "OTH":
            oth_total += 1
            ipa_lines = [l for l in lines if l.startswith("♫fused:") or l.startswith("♫w2v2:")]
            ipa_text = " ".join(ipa_lines)
            tokens = set(re.findall(r"[a-zɡɲɾβðɣʔɬ]+", ipa_text))
            spa_hits = len(tokens & SPA_IPA_PHONEMES)
            nah_hits = sum(1 for m2 in NAH_MARKERS if m2 in ipa_text)
            may_hits = sum(1 for m2 in MAY_MARKERS if m2 in ipa_text)

            if spa_hits >= 2 and nah_hits == 0 and may_hits == 0:
                oth_with_spa_ipa += 1
                if len(oth_examples) < 5:
                    oth_examples.append(f"  {lines[1]}  {header}")

    total = sum(lang_counts.values())
    return {
        "path": path,
        "total": total,
        "lang_counts": lang_counts,
        "oth_total": oth_total,
        "oth_with_spa_ipa": oth_with_spa_ipa,
        "oth_examples": oth_examples,
    }


def print_report(results: list[dict]):
    print("=" * 80)
    print("LANGUAGE DISTRIBUTION REPORT")
    print("=" * 80)

    all_langs = set()
    for r in results:
        all_langs.update(r["lang_counts"].keys())
    lang_order = ["SPA", "OTH", "NAH", "MAY", "LAT", "ENG"]
    lang_order = [l for l in lang_order if l in all_langs]
    lang_order += sorted(all_langs - set(lang_order))

    totals = {l: 0 for l in lang_order}
    total_segs = 0
    total_oth = 0
    total_oth_spa = 0

    for r in results:
        name = Path(r["path"]).stem
        total = r["total"]
        total_segs += total
        total_oth += r["oth_total"]
        total_oth_spa += r["oth_with_spa_ipa"]

        parts = []
        for lang in lang_order:
            c = r["lang_counts"].get(lang, 0)
            totals[lang] = totals.get(lang, 0) + c
            pct = c / total * 100 if total else 0
            parts.append(f"{lang}={c:3d}({pct:4.1f}%)")

        spa_leak = f"OTH-as-SPA={r['oth_with_spa_ipa']}/{r['oth_total']}"
        print(f"\n{name} ({total} segs)")
        print(f"  {' | '.join(parts)}")
        print(f"  {spa_leak}")

        if r["oth_examples"]:
            print("  Examples of likely-SPA tagged as OTH:")
            for ex in r["oth_examples"]:
                print(f"    {ex}")

    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"TOTALS ({total_segs} segs)")
        parts = []
        for lang in lang_order:
            c = totals[lang]
            pct = c / total_segs * 100 if total_segs else 0
            parts.append(f"{lang}={c}({pct:.1f}%)")
        print(f"  {' | '.join(parts)}")
        print(f"  OTH-as-SPA (likely missed Spanish): {total_oth_spa}/{total_oth} "
              f"({total_oth_spa/total_oth*100:.1f}% of OTH)" if total_oth else "")
        print(f"  → Potential SPA recovery: +{total_oth_spa} segments")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <srt_file> [srt_file ...]")
        sys.exit(1)

    results = []
    for path in sys.argv[1:]:
        if not Path(path).exists():
            print(f"WARN: {path} not found, skipping")
            continue
        results.append(analyze_srt(path))

    print_report(results)

#!/usr/bin/env python3
"""Extract edge cases from all SRT files for language tag validation.

Flags:
1. tɬ in non-NAH — SPA/OTH/MAY with tɬ in fused IPA (likely misclassified NAH)
2. Multi-NAH-word clusters — non-NAH with 3+ NAH lexicon words (≥4 phonemes)
3. NAH without markers — NAH-tagged with 0 lexicon matches AND no tɬ
4. kʷ in non-NAH — another potential NAH marker

Usage:
    python scripts/extract_edge_cases.py
    python scripts/extract_edge_cases.py --output validation_video/edge_cases_all.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_lexicon(min_freq: int = 50, min_phonemes: int = 4) -> list[dict]:
    """Load NAH lexicon with filtering."""
    lexicon_path = PROJECT_ROOT / "src/tenepal/data/nah_lexicon_merged.json"
    with open(lexicon_path) as f:
        entries = json.load(f)
    return [
        e for e in entries
        if (e.get("source") == "curated" or e.get("freq", 0) >= min_freq)
        and len(e["ipa"]) >= min_phonemes
    ]


# IPA normalization
IPA_MODS = frozenset("ːʲʰˤʼ")
ALLOPHONE = {"ð": "d", "β": "b", "ɣ": "ɡ", "ɸ": "f", "ɻ": "ɾ", "r": "ɾ"}


def norm(tok: str) -> str:
    tok = "".join(c for c in tok if c not in IPA_MODS)
    return ALLOPHONE.get(tok, tok)


def find_lexicon_matches(phonemes: list[str], lexicon: list[dict], threshold: float = 0.75) -> list[str]:
    """Find NAH lexicon words in phoneme sequence."""
    if not phonemes or len(phonemes) < 4:
        return []
    normalized = [norm(p) for p in phonemes]
    matches = []
    for entry in lexicon:
        pattern = [norm(p) for p in entry["ipa"]]
        for start in range(max(1, len(normalized) - len(pattern) - 1)):
            window = normalized[start:start + len(pattern)]
            if len(window) < 4:
                continue
            overlap = sum(1 for a, b in zip(window, pattern) if a == b)
            score = overlap / max(len(window), len(pattern))
            if score >= threshold:
                matches.append(entry["word"])
                break
    return matches


def has_tl(phonemes: list[str]) -> bool:
    """Check if tɬ is in phoneme sequence."""
    return "tɬ" in phonemes


def has_kw(phonemes: list[str]) -> bool:
    """Check if kʷ is in phoneme sequence."""
    return "kʷ" in phonemes or "kw" in phonemes


# SRT parsing
LANG_RE = re.compile(r"\[([A-Z]{3})(?:\|[^\]]+)?\]")
FUSED_RE = re.compile(r"♫?\s*fused:\s*(.+)", re.I)
TIME_RE = re.compile(r"(\d+:\d+:\d+[,\.]\d+)\s*-->\s*(\d+:\d+:\d+[,\.]\d+)")


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT file and extract segments with language tags and fused IPA."""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments = []

    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Get cue number
        try:
            cue_num = int(lines[0].strip())
        except ValueError:
            cue_num = i + 1

        # Get timestamp
        time_match = TIME_RE.search(lines[1]) if len(lines) > 1 else None
        timestamp = time_match.group(1) if time_match else "?"

        content = "\n".join(lines[2:])

        # Get language tag
        lang_match = LANG_RE.search(content)
        lang = lang_match.group(1) if lang_match else "UNK"

        # Get fused IPA
        fused_match = FUSED_RE.search(content)
        if not fused_match:
            continue
        phonemes = fused_match.group(1).strip().split()

        segments.append({
            "file": srt_path.name,
            "cue": cue_num,
            "timestamp": timestamp,
            "lang": lang,
            "phonemes": phonemes,
        })

    return segments


def classify_edge_case(seg: dict, lexicon: list[dict]) -> list[str]:
    """Classify segment into edge case categories."""
    reasons = []
    lang = seg["lang"]
    phonemes = seg["phonemes"]

    has_tl_marker = has_tl(phonemes)
    has_kw_marker = has_kw(phonemes)
    matches = find_lexicon_matches(phonemes, lexicon)

    # Store for later use
    seg["has_tl"] = has_tl_marker
    seg["has_kw"] = has_kw_marker
    seg["lexicon_matches"] = matches

    # 1. tɬ in non-NAH
    if has_tl_marker and lang != "NAH":
        reasons.append("tl_in_non_nah")

    # 2. Multi-NAH-word clusters in non-NAH
    if len(matches) >= 3 and lang != "NAH":
        reasons.append("multi_nah_words_in_non_nah")

    # 3. NAH without markers
    if lang == "NAH" and not has_tl_marker and len(matches) == 0:
        reasons.append("nah_without_markers")

    # 4. kʷ in non-NAH
    if has_kw_marker and lang != "NAH":
        reasons.append("kw_in_non_nah")

    return reasons


def main():
    parser = argparse.ArgumentParser(description="Extract edge cases from SRT files")
    parser.add_argument("--output", "-o", default="validation_video/edge_cases_all.json",
                        help="Output JSON file")
    parser.add_argument("--srt-dir", default="validation_video",
                        help="Directory containing SRT files")
    args = parser.parse_args()

    srt_dir = PROJECT_ROOT / args.srt_dir
    srt_files = sorted(srt_dir.glob("Hernán-*.srt"))

    print(f"Scanning {len(srt_files)} SRT files...")

    # Load lexicon (≥4 phonemes for multi-word detection)
    lexicon = load_lexicon(min_freq=50, min_phonemes=4)
    print(f"Lexicon: {len(lexicon)} entries (min_freq=50, min_phonemes=4)")

    # Process all files
    all_edge_cases = []
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for srt_path in srt_files:
        segments = parse_srt(srt_path)
        episode = srt_path.stem.replace("Hernán-", "E")

        for seg in segments:
            reasons = classify_edge_case(seg, lexicon)
            if reasons:
                seg["reasons"] = reasons
                all_edge_cases.append(seg)

                for reason in reasons:
                    summary[episode][reason][seg["lang"]] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("EDGE CASE SUMMARY")
    print("=" * 80)

    # Aggregate by reason
    by_reason = defaultdict(lambda: defaultdict(int))
    for seg in all_edge_cases:
        for reason in seg["reasons"]:
            by_reason[reason][seg["lang"]] += 1

    print(f"\n{'Reason':<30} | {'SPA':<5} | {'MAY':<5} | {'OTH':<5} | {'NAH':<5} | {'Total':<5}")
    print("-" * 70)
    for reason in ["tl_in_non_nah", "multi_nah_words_in_non_nah", "nah_without_markers", "kw_in_non_nah"]:
        counts = by_reason[reason]
        total = sum(counts.values())
        print(f"{reason:<30} | {counts.get('SPA', 0):<5} | {counts.get('MAY', 0):<5} | {counts.get('OTH', 0):<5} | {counts.get('NAH', 0):<5} | {total:<5}")

    # Per-episode breakdown
    print(f"\n{'Episode':<15} | {'tl_non_nah':<12} | {'multi_words':<12} | {'nah_no_mark':<12} | {'kw_non_nah':<12}")
    print("-" * 70)
    for episode in sorted(summary.keys()):
        ep_data = summary[episode]
        tl = sum(ep_data.get("tl_in_non_nah", {}).values())
        multi = sum(ep_data.get("multi_nah_words_in_non_nah", {}).values())
        nah_no = sum(ep_data.get("nah_without_markers", {}).values())
        kw = sum(ep_data.get("kw_in_non_nah", {}).values())
        print(f"{episode:<15} | {tl:<12} | {multi:<12} | {nah_no:<12} | {kw:<12}")

    # Save JSON
    output_path = PROJECT_ROOT / args.output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "srt_files": len(srt_files),
            "lexicon_entries": len(lexicon),
            "min_freq": 50,
            "min_phonemes": 4,
        },
        "summary": {
            reason: dict(counts) for reason, counts in by_reason.items()
        },
        "by_episode": {
            ep: {r: dict(langs) for r, langs in reasons.items()}
            for ep, reasons in summary.items()
        },
        "edge_cases": [
            {
                "file": seg["file"],
                "cue": seg["cue"],
                "timestamp": seg["timestamp"],
                "lang": seg["lang"],
                "reasons": seg["reasons"],
                "has_tl": seg["has_tl"],
                "has_kw": seg["has_kw"],
                "lexicon_matches": seg["lexicon_matches"],
                "phonemes": " ".join(seg["phonemes"][:20]) + ("..." if len(seg["phonemes"]) > 20 else ""),
            }
            for seg in all_edge_cases
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nTotal edge cases: {len(all_edge_cases)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

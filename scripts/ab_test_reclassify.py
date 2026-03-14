#!/usr/bin/env python3
"""A/B test language tag reclassification based on edge case detection.

Version A (baseline): Current language tags as-is
Version B (with overrides):
  - Any segment with tɬ → force NAH
  - Any non-NAH with 3+ long NAH words → force NAH
  - Any NAH with 0 markers and no tɬ → flag as NAH? (uncertain)

Usage:
    python scripts/ab_test_reclassify.py
    python scripts/ab_test_reclassify.py --export  # Export B-version SRTs
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_edge_cases() -> dict:
    """Load edge cases from JSON."""
    path = PROJECT_ROOT / "validation_video/edge_cases_all.json"
    with open(path) as f:
        return json.load(f)


def parse_srt_blocks(srt_path: Path) -> list[dict]:
    """Parse SRT into blocks preserving original content."""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(\n\s*\n)", text)

    result = []
    for i in range(0, len(blocks), 2):
        block = blocks[i].strip()
        separator = blocks[i + 1] if i + 1 < len(blocks) else "\n\n"
        if not block:
            continue

        lines = block.split("\n")
        if len(lines) < 3:
            result.append({"raw": block, "separator": separator, "cue": None})
            continue

        try:
            cue_num = int(lines[0].strip())
        except ValueError:
            cue_num = None

        result.append({
            "raw": block,
            "separator": separator,
            "cue": cue_num,
            "lines": lines,
        })

    return result


def apply_reclassification(block: dict, changes: dict) -> str:
    """Apply language tag change to a block."""
    if block["cue"] is None or block["cue"] not in changes:
        return block["raw"]

    change = changes[block["cue"]]
    old_tag = change["old"]
    new_tag = change["new"]

    # Replace [OLD|...] with [NEW|...]
    lines = block["lines"].copy()
    for i, line in enumerate(lines):
        if f"[{old_tag}|" in line or f"[{old_tag}]" in line:
            lines[i] = re.sub(
                rf"\[{old_tag}(\|[^\]]+)?\]",
                f"[{new_tag}\\1]",
                line
            )
            break

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="A/B test reclassification")
    parser.add_argument("--export", action="store_true",
                        help="Export B-version SRT files")
    parser.add_argument("--output-dir", default="validation_video/reclassified",
                        help="Output directory for reclassified SRTs")
    args = parser.parse_args()

    # Load edge cases
    data = load_edge_cases()
    edge_cases = data["edge_cases"]

    print(f"Loaded {len(edge_cases)} edge cases")

    # Build change map: file -> cue -> change
    changes_by_file = defaultdict(dict)

    # Version B rules
    for case in edge_cases:
        file_name = case["file"]
        cue = case["cue"]
        old_lang = case["lang"]
        reasons = case["reasons"]

        new_lang = old_lang
        change_reason = None

        # Rule 1: tɬ in non-NAH → force NAH
        if "tl_in_non_nah" in reasons:
            new_lang = "NAH"
            change_reason = "tl_detected"

        # Rule 2: 3+ NAH words in non-NAH → force NAH
        elif "multi_nah_words_in_non_nah" in reasons:
            new_lang = "NAH"
            change_reason = "multi_nah_words"

        # Rule 3: NAH without markers → flag as uncertain (NAH?)
        elif "nah_without_markers" in reasons:
            new_lang = "NAH?"
            change_reason = "no_markers"

        # Rule 4: kʷ alone is not enough to force NAH (too common in Spanish loans)
        # Skip kw_in_non_nah unless combined with other markers

        if new_lang != old_lang:
            changes_by_file[file_name][cue] = {
                "old": old_lang,
                "new": new_lang,
                "reason": change_reason,
            }

    # Count changes
    total_changes = sum(len(changes) for changes in changes_by_file.values())

    # Summary by change type
    change_counts = defaultdict(lambda: defaultdict(int))
    for file_name, changes in changes_by_file.items():
        episode = file_name.replace("Hernán-", "E").replace(".srt", "")
        for cue, change in changes.items():
            key = f"{change['old']}→{change['new']}"
            change_counts[episode][key] += 1

    print("\n" + "=" * 80)
    print("A/B TEST COMPARISON")
    print("=" * 80)
    print(f"\nVersion A: Original tags (baseline)")
    print(f"Version B: With reclassification rules")
    print(f"\nTotal segments changed: {total_changes}")

    # Change type summary
    type_totals = defaultdict(int)
    for episode, types in change_counts.items():
        for change_type, count in types.items():
            type_totals[change_type] += count

    print(f"\n{'Change Type':<20} | {'Count':<6}")
    print("-" * 30)
    for change_type in sorted(type_totals.keys()):
        print(f"{change_type:<20} | {type_totals[change_type]:<6}")

    # Per-episode delta
    print(f"\n{'Episode':<15} | {'SPA→NAH':<10} | {'OTH→NAH':<10} | {'MAY→NAH':<10} | {'NAH→NAH?':<10} | {'Total':<6}")
    print("-" * 75)

    for episode in sorted(change_counts.keys()):
        types = change_counts[episode]
        spa = types.get("SPA→NAH", 0)
        oth = types.get("OTH→NAH", 0)
        may = types.get("MAY→NAH", 0)
        nah_uncertain = types.get("NAH→NAH?", 0)
        total = sum(types.values())
        print(f"{episode:<15} | {spa:<10} | {oth:<10} | {may:<10} | {nah_uncertain:<10} | {total:<6}")

    # Grand totals
    print("-" * 75)
    spa_total = type_totals.get("SPA→NAH", 0)
    oth_total = type_totals.get("OTH→NAH", 0)
    may_total = type_totals.get("MAY→NAH", 0)
    nah_uncertain_total = type_totals.get("NAH→NAH?", 0)
    print(f"{'TOTAL':<15} | {spa_total:<10} | {oth_total:<10} | {may_total:<10} | {nah_uncertain_total:<10} | {total_changes:<6}")

    # Impact summary
    print(f"\n" + "=" * 80)
    print("IMPACT SUMMARY")
    print("=" * 80)
    print(f"  • {spa_total} SPA segments → NAH (likely misclassified Spanish dialogue)")
    print(f"  • {oth_total} OTH segments → NAH (previously unknown, now identified)")
    print(f"  • {may_total} MAY segments → NAH (possible MAY↔NAH confusion)")
    print(f"  • {nah_uncertain_total} NAH segments flagged as uncertain (no supporting markers)")

    # Export B-version SRTs if requested
    if args.export:
        output_dir = PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting B-version SRTs to {output_dir}...")

        srt_dir = PROJECT_ROOT / "validation_video"
        for srt_path in sorted(srt_dir.glob("Hernán-*.srt")):
            file_name = srt_path.name
            changes = changes_by_file.get(file_name, {})

            if not changes:
                # No changes, just copy
                shutil.copy(srt_path, output_dir / file_name)
                continue

            # Parse and apply changes
            blocks = parse_srt_blocks(srt_path)
            output_lines = []

            for block in blocks:
                new_content = apply_reclassification(block, changes)
                output_lines.append(new_content)
                output_lines.append(block["separator"])

            # Write modified SRT
            output_path = output_dir / file_name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("".join(output_lines))

            print(f"  {file_name}: {len(changes)} changes")

        print(f"\nExported {len(list(output_dir.glob('*.srt')))} SRT files")

    # Save comparison JSON
    comparison_path = PROJECT_ROOT / "validation_video/ab_test_comparison.json"
    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version_a": "original_tags",
        "version_b": "with_reclassification",
        "rules": [
            "tɬ in non-NAH → NAH",
            "3+ NAH words (≥4 phonemes) in non-NAH → NAH",
            "NAH without markers → NAH? (uncertain)",
        ],
        "total_changes": total_changes,
        "by_change_type": dict(type_totals),
        "by_episode": {ep: dict(types) for ep, types in change_counts.items()},
        "changes": {
            file_name: {
                str(cue): change for cue, change in changes.items()
            }
            for file_name, changes in changes_by_file.items()
        },
    }

    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved comparison to {comparison_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compute_backend_agreement.py

Parses Hernán-1-1-1.srt and computes per-segment Allosaurus/wav2vec2 agreement stats.
Used for PAPER.md Section 5.3 backend agreement analysis.
"""

import sys
from pathlib import Path


def parse_srt_blocks(path: Path):
    """Parse SRT file into blocks, each a list of lines."""
    blocks = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "" and current:
                blocks.append(current)
                current = []
            else:
                current.append(line)
    if current:
        blocks.append(current)
    return blocks


def extract_ipa(block_lines, prefix):
    """Extract IPA tokens from a block line starting with prefix (e.g. '♫allo:')."""
    for line in block_lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            tokens_str = stripped[len(prefix):].strip()
            return tokens_str.split()
    return None


def compute_agreement(allo_tokens, w2v2_tokens):
    """
    Compute:
    - exact_match: fraction of aligned positions (up to min length) that are identical
    - length_ratio: min/max length
    - agreement: exact_match * length_ratio
    """
    n_allo = len(allo_tokens)
    n_w2v2 = len(w2v2_tokens)
    if n_allo == 0 or n_w2v2 == 0:
        return None

    min_len = min(n_allo, n_w2v2)
    max_len = max(n_allo, n_w2v2)

    matches = sum(1 for i in range(min_len) if allo_tokens[i] == w2v2_tokens[i])
    exact_match = matches / min_len
    length_ratio = min_len / max_len
    agreement = exact_match * length_ratio

    return exact_match, length_ratio, agreement


def main():
    srt_path = Path(__file__).parent.parent / "validation_video" / "Hernán-1-1-1.srt"
    if not srt_path.exists():
        print(f"ERROR: SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)

    blocks = parse_srt_blocks(srt_path)

    exact_matches = []
    length_ratios = []
    agreements = []
    divergent = 0

    for block in blocks:
        allo = extract_ipa(block, "♫allo:")
        w2v2 = extract_ipa(block, "♫w2v2:")

        if allo is None or w2v2 is None:
            continue  # mono-backend segment, skip

        result = compute_agreement(allo, w2v2)
        if result is None:
            continue

        em, lr, ag = result
        exact_matches.append(em)
        length_ratios.append(lr)
        agreements.append(ag)

        if em < 0.25:
            divergent += 1

    n_pairs = len(agreements)
    if n_pairs == 0:
        print("ERROR: No dual-backend segments found.", file=sys.stderr)
        sys.exit(1)

    mean_em = sum(exact_matches) / n_pairs
    mean_lr = sum(length_ratios) / n_pairs
    mean_ag = sum(agreements) / n_pairs
    diverge_pct = divergent / n_pairs * 100

    print("Backend Agreement Report — Hernán-1-1-1.srt")
    print(f"N segments with both backends: {n_pairs}")
    print(f"Mean token exact-match rate:   {mean_em:.2f} ({mean_em*100:.1f}%)")
    print(f"Mean length-ratio:             {mean_lr:.2f}")
    print(f"Mean agreement score:          {mean_ag:.2f}")
    print(f"Divergent segments (<25% token match): {divergent} ({diverge_pct:.1f}%)")


if __name__ == "__main__":
    main()

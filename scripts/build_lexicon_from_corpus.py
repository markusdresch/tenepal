#!/usr/bin/env python3
"""Build Nahuatl lexicon entries from Zacatlan corpus extraction.

Reads the extracted corpus markers, filters out Spanish loanwords and short tokens,
converts Zacatlan orthography to IPA using existing G2P maps, and outputs lexicon
JSON files compatible with NahuatlLexicon.

Usage:
    python scripts/build_lexicon_from_corpus.py

Outputs:
    - src/tenepal/data/nah_lexicon_corpus_only.json  (corpus entries only)
    - src/tenepal/data/nah_lexicon_merged.json       (corpus + curated)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tenepal.phoneme.text_to_ipa import NahuatlMapG2P


# Paths
CORPUS_DIR = PROJECT_ROOT / "codices" / "extracted" / "zacatlan_markers"
MARKERS_TSV = CORPUS_DIR / "markers_nahuatl.tsv"
LOANWORDS_TSV = CORPUS_DIR / "loanwords_es.tsv"
DATA_DIR = PROJECT_ROOT / "src" / "tenepal" / "data"
EXISTING_LEXICON = DATA_DIR / "nah_lexicon.json"
OUTPUT_CORPUS_ONLY = DATA_DIR / "nah_lexicon_corpus_only.json"
OUTPUT_MERGED = DATA_DIR / "nah_lexicon_merged.json"

# Filtering constants
MIN_TOKEN_LENGTH = 3  # Skip tokens <= 2 chars
MIN_FILE_COUNT = 2    # Skip tokens appearing in < 2 files
SOURCE_CORPUS = "amith-zacatlan"
SOURCE_CURATED = "curated"


def load_tsv(path: Path) -> list[tuple[str, int, int]]:
    """Load TSV file with token, count, file_count columns."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                token = parts[0]
                try:
                    count = int(parts[1])
                    file_count = int(parts[2])
                    entries.append((token, count, file_count))
                except ValueError:
                    continue
    return entries


def load_loanwords(path: Path) -> set[str]:
    """Load Spanish loanwords to exclude."""
    loanwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            parts = line.strip().split("\t")
            if parts:
                loanwords.add(parts[0].lower())
    return loanwords


def load_existing_lexicon(path: Path) -> list[dict]:
    """Load existing hand-curated lexicon."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    # Mark as curated source if not already set
    for entry in entries:
        if "source" not in entry:
            entry["source"] = SOURCE_CURATED
    return entries


def normalize_ipa_phonemes(phonemes: list[str]) -> list[str]:
    """Post-process IPA phonemes to canonical lexicon format.

    - Splits onset+vowel combinations (tɬa → tɬ + a)
    - Combines vowel + ː into single long vowel (a + ː → aː)
    """
    VOWELS = {"a", "e", "i", "o", "u"}
    ONSETS = {"tɬ", "ts", "tʃ", "kʷ"}

    # Pass 1: Split onset+vowel combinations
    split_result = []
    for p in phonemes:
        split_done = False
        for onset in ONSETS:
            if p.startswith(onset) and len(p) > len(onset):
                rest = p[len(onset):]
                # Check if rest is vowel(s)
                if all(c in VOWELS for c in rest):
                    split_result.append(onset)
                    for v in rest:
                        split_result.append(v)
                    split_done = True
                    break

        if not split_done:
            split_result.append(p)

    # Pass 2: Combine vowel + ː into long vowel
    result = []
    i = 0
    while i < len(split_result):
        p = split_result[i]

        # Check if next phoneme is ː (length marker)
        if i + 1 < len(split_result) and split_result[i + 1] == "ː":
            if p in VOWELS:
                result.append(p + "ː")
                i += 2
                continue

        result.append(p)
        i += 1

    return result


def convert_to_ipa(token: str, g2p: NahuatlMapG2P) -> list[str]:
    """Convert Zacatlan orthography token to IPA phoneme list."""
    raw = g2p.convert(token, "nah")
    return normalize_ipa_phonemes(raw)


def validate_known_words(g2p: NahuatlMapG2P) -> bool:
    """Spot-check known word conversions."""
    test_cases = [
        ("wa:n", ["w", "aː", "n"]),
        ("a:mo", ["aː", "m", "o"]),
        ("nochi", ["n", "o", "tʃ", "i"]),
        ("tle:n", ["tɬ", "eː", "n"]),
    ]

    print("\n=== G2P Validation ===")
    all_passed = True
    for word, expected in test_cases:
        result = convert_to_ipa(word, g2p)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"  {status} {word} → {result} (expected: {expected})")
        if not passed:
            all_passed = False

    return all_passed


def build_corpus_entries(
    markers: list[tuple[str, int, int]],
    loanwords: set[str],
    g2p: NahuatlMapG2P,
) -> list[dict]:
    """Filter markers and convert to lexicon entries."""
    entries = []
    skipped_short = 0
    skipped_loanword = 0
    skipped_file_count = 0
    skipped_empty_ipa = 0

    for token, count, file_count in markers:
        # Skip short tokens
        if len(token) <= 2:
            skipped_short += 1
            continue

        # Skip loanwords (case-insensitive)
        if token.lower() in loanwords:
            skipped_loanword += 1
            continue

        # Skip low file count
        if file_count < MIN_FILE_COUNT:
            skipped_file_count += 1
            continue

        # Convert to IPA
        ipa = convert_to_ipa(token, g2p)
        if not ipa:
            skipped_empty_ipa += 1
            continue

        entry = {
            "word": token,
            "ipa": ipa,
            "gloss": None,
            "source": SOURCE_CORPUS,
            "freq": count,
            "file_count": file_count,
        }
        entries.append(entry)

    print(f"\n=== Filtering Summary ===")
    print(f"  Total markers: {len(markers)}")
    print(f"  Skipped (short ≤2): {skipped_short}")
    print(f"  Skipped (loanword): {skipped_loanword}")
    print(f"  Skipped (file_count <{MIN_FILE_COUNT}): {skipped_file_count}")
    print(f"  Skipped (empty IPA): {skipped_empty_ipa}")
    print(f"  Corpus entries: {len(entries)}")

    return entries


def merge_lexicons(
    corpus_entries: list[dict],
    curated_entries: list[dict],
) -> list[dict]:
    """Merge corpus entries with curated entries, curated takes precedence."""
    # Index curated by word
    curated_words = {e["word"].lower() for e in curated_entries}

    # Start with curated
    merged = list(curated_entries)

    # Add corpus entries not in curated (by word match)
    added = 0
    for entry in corpus_entries:
        if entry["word"].lower() not in curated_words:
            merged.append(entry)
            added += 1

    print(f"\n=== Merge Summary ===")
    print(f"  Curated entries: {len(curated_entries)}")
    print(f"  Corpus entries added: {added}")
    print(f"  Total merged: {len(merged)}")

    return merged


def print_summary(entries: list[dict], title: str) -> None:
    """Print lexicon summary with top entries by frequency."""
    print(f"\n=== {title} ===")
    print(f"  Total entries: {len(entries)}")

    # Count by source
    by_source: dict[str, int] = {}
    for e in entries:
        src = e.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    print(f"  By source:")
    for src, count in sorted(by_source.items()):
        print(f"    {src}: {count}")

    # Top 20 by frequency
    with_freq = [e for e in entries if e.get("freq")]
    with_freq.sort(key=lambda x: x.get("freq", 0), reverse=True)

    print(f"\n  Top 20 by frequency:")
    for e in with_freq[:20]:
        ipa_str = " ".join(e["ipa"])
        print(f"    {e['word']:15} ({e.get('freq', 0):5}x, {e.get('file_count', 0):2} files) → [{ipa_str}]")


def save_lexicon(entries: list[dict], path: Path) -> None:
    """Save lexicon entries to JSON file."""
    # Sort by frequency (descending) for corpus entries, then alphabetically
    def sort_key(e: dict) -> tuple:
        freq = e.get("freq", 0)
        return (-freq, e["word"].lower())

    sorted_entries = sorted(entries, key=sort_key)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_entries, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {path}")


def test_lexicon_loading(path: Path) -> bool:
    """Test that NahuatlLexicon can load the generated file."""
    try:
        from tenepal.language.nahuatl_lexicon import NahuatlLexicon

        lexicon = NahuatlLexicon(lexicon_path=path)
        # Try a simple match
        test_ipa = ["w", "aː", "n"]
        match = lexicon.match(test_ipa)

        print(f"\n=== Load Test: {path.name} ===")
        print(f"  Loaded {len(lexicon._entries)} entries")
        if match:
            print(f"  Test match ['w', 'aː', 'n'] → {match.word} (score: {match.score:.2f})")
        else:
            print(f"  Test match ['w', 'aː', 'n'] → no match found")

        return True
    except Exception as e:
        print(f"\n=== Load Test FAILED: {path.name} ===")
        print(f"  Error: {e}")
        return False


def main() -> int:
    """Main entry point."""
    print("Building Nahuatl lexicon from Zacatlan corpus...")

    # Check input files exist
    if not MARKERS_TSV.exists():
        print(f"ERROR: Markers file not found: {MARKERS_TSV}")
        return 1
    if not LOANWORDS_TSV.exists():
        print(f"ERROR: Loanwords file not found: {LOANWORDS_TSV}")
        return 1

    # Initialize G2P converter
    g2p = NahuatlMapG2P(variant="modern")

    # Validate known words
    if not validate_known_words(g2p):
        print("\nWARNING: G2P validation failed for some known words")

    # Load data
    print("\nLoading corpus data...")
    markers = load_tsv(MARKERS_TSV)
    print(f"  Loaded {len(markers)} markers")

    loanwords = load_loanwords(LOANWORDS_TSV)
    print(f"  Loaded {len(loanwords)} Spanish loanwords")

    curated = load_existing_lexicon(EXISTING_LEXICON)
    print(f"  Loaded {len(curated)} curated entries")

    # Build corpus entries
    corpus_entries = build_corpus_entries(markers, loanwords, g2p)

    # Merge with curated
    merged_entries = merge_lexicons(corpus_entries, curated)

    # Print summaries
    print_summary(corpus_entries, "Corpus-Only Lexicon")
    print_summary(merged_entries, "Merged Lexicon")

    # Save outputs
    print("\n=== Saving Lexicons ===")
    save_lexicon(corpus_entries, OUTPUT_CORPUS_ONLY)
    save_lexicon(merged_entries, OUTPUT_MERGED)

    # Test loading
    test_lexicon_loading(OUTPUT_MERGED)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

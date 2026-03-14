#!/usr/bin/env python3
"""Quick lexicon performance comparison on E03 NAH segments.

Loads IPA from existing SRT files, runs match_subsequence with
old (20 entries) vs new (various min_freq) lexicon configs.
No GPU needed — pure CPU string matching.
"""
import json
import re
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRT_DIR = _PROJECT_ROOT / "validation_video"
LEXICON_PATH = _PROJECT_ROOT / "src" / "tenepal" / "data"

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tenepal.language.nahuatl_lexicon import NahuatlLexicon


def extract_nah_segments_from_srt(srt_path: Path) -> list[dict]:
    """Extract NAH-tagged segments with their IPA phonemes from SRT."""
    text = srt_path.read_text(encoding="utf-8")
    segments = []
    
    # Split into SRT blocks
    blocks = text.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        
        # Check for NAH tag (format: [NAH|SPEAKER_XX])
        content = "\n".join(lines[2:])
        if "[NAH|" not in content and "[NAH]" not in content:
            continue
        
        # Extract fused IPA if present
        fused_match = re.search(r'fused:\s*([^\n\]]+)', content)
        if fused_match:
            ipa_str = fused_match.group(1).strip()
            phonemes = ipa_str.split()
            if len(phonemes) >= 2:
                segments.append({
                    "file": srt_path.name,
                    "phonemes": phonemes,
                    "text": content[:80],
                })
    
    return segments


def run_comparison():
    # Collect NAH segments from E03 SRTs
    srt_files = sorted(SRT_DIR.glob("Hernán-1-3*.srt"))
    if not srt_files:
        # Fallback: all SRTs
        srt_files = sorted(SRT_DIR.glob("Hernán-*.srt"))
    
    all_segments = []
    for srt in srt_files:
        all_segments.extend(extract_nah_segments_from_srt(srt))
    
    print(f"{'='*60}")
    print(f"LEXICON COMPARISON: {len(all_segments)} NAH segments from {len(srt_files)} SRTs")
    print(f"{'='*60}")
    
    if not all_segments:
        print("ERROR: No NAH segments found!")
        return
    
    # Test configs: (label, min_freq_or_none, lexicon_path_or_none)
    configs = [
        ("OLD (20 curated)", None, LEXICON_PATH / "nah_lexicon.json"),
        ("NEW min_freq=50 (140)", 50, None),
        ("NEW min_freq=25 (262)", 25, None),
        ("NEW min_freq=10 (612)", 10, None),
        ("NEW min_freq=5 (1229)", 5, None),
        ("NEW min_freq=3 (2024)", 3, None),
    ]
    
    for label, min_freq, custom_path in configs:
        if custom_path:
            lex = NahuatlLexicon(lexicon_path=custom_path)
        else:
            lex = NahuatlLexicon(min_freq=min_freq)
        
        entry_count = len(lex._entries)
        matched_segs = 0
        total_matches = 0
        total_score = 0.0
        unique_words = set()
        
        t0 = time.perf_counter()
        
        for seg in all_segments:
            matches = lex.match_subsequence(seg["phonemes"])
            if matches:
                matched_segs += 1
                total_matches += len(matches)
                total_score += sum(m.score for m in matches)
                unique_words.update(m.word for m in matches)
        
        elapsed = time.perf_counter() - t0
        
        print(f"\n[{label}]")
        print(f"    Entries: {entry_count}")
        print(f"    Segments matched: {matched_segs}/{len(all_segments)} ({100*matched_segs/len(all_segments):.1f}%)")
        print(f"    Total matches: {total_matches}")
        print(f"    Total score: {total_score:.1f}")
        print(f"    Unique words: {len(unique_words)} — {sorted(unique_words)[:15]}...")
        print(f"    Time: {elapsed:.2f}s")
    
    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    run_comparison()

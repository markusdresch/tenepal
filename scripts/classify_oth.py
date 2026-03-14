#!/usr/bin/env python3
"""Eliminate OTH tags with full language classification hierarchy.

Classification Hierarchy (in order of authority):
1. Acoustic hard markers (tɬ→NAH, kʷ→likely NAH, ejectives→MAY)
2. Phonotactic score (bigram model, >0.5 margin wins)
3. Lexicon matches (3+ NAH words ≥4 phonemes, score ≥0.9 → NAH)
4. Whisper LLM text analysis (Spanish/Latin/English patterns)
5. Segment length filter (<0.3s with no markers → NON)
6. Fallback: phonotactic best guess

Usage:
    python scripts/classify_oth.py               # Analyze only
    python scripts/classify_oth.py --export      # Generate corrected SRTs
    python scripts/classify_oth.py --all-segs    # Also validate existing tags
"""

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Latin words for LAT detection
LATIN_WORDS = {
    "dominus", "deus", "pater", "filius", "spiritus", "sanctus", "amen",
    "baptizo", "ego", "te", "in", "nomine", "patris", "filii", "gloria",
    "christe", "kyrie", "eleison", "sancta", "maria", "ora", "pro", "nobis",
    "ave", "gratia", "plena", "benedicta", "tu", "mulieribus", "jesus",
    "crucifixus", "resurrexit", "credo", "corpus", "sanguis", "mea", "culpa",
    "peccata", "mundi", "agnus", "dei", "requiem", "aeternam", "dona", "eis",
}

# English words for ENG detection
ENGLISH_WORDS = {
    "the", "and", "is", "are", "was", "were", "have", "has", "had", "will",
    "would", "could", "should", "what", "where", "when", "why", "how", "who",
    "this", "that", "with", "from", "they", "them", "their", "there", "here",
    "next", "episode", "previously", "watch", "subscribe", "amazon", "prime",
    "coming", "soon", "don't", "miss", "season", "finale", "now", "streaming",
}

# Language priors — Hernán is mostly Spanish with NAH dialogue, little MAY
# These correct for corpus size bias (MAY has smallest training set)
LANGUAGE_PRIORS = {"NAH": 0.28, "SPA": 0.55, "MAY": 0.12}

# Nahuatl orthographic patterns in LLM text
NAH_PATTERNS = [
    r"tl[aeiou]",  # tla, tle, tli, etc.
    r"tz[aeiou]",  # tza, tze, etc.
    r"hu[aeiou]",  # hua, hue, etc.
    r"xo[a-z]",    # xochitl, etc.
    r"cuau",       # cuauhtemoc
    r"qui",        # quetzal
]


def load_bigrams() -> dict:
    """Load phonotactic bigram model."""
    path = PROJECT_ROOT / "src/tenepal/data/phonotactic_bigrams.json"
    with open(path) as f:
        data = json.load(f)
    return data


def load_lexicon() -> list[dict]:
    """Load NAH lexicon."""
    path = PROJECT_ROOT / "src/tenepal/data/nah_lexicon_merged.json"
    with open(path) as f:
        return json.load(f)


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT into structured segments."""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    segments = []

    # Split by double newline
    blocks = re.split(r'\n\s*\n', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split('\n')
        if len(lines) < 3:
            continue

        try:
            cue = int(lines[0].strip())
        except ValueError:
            continue

        # Parse timestamp
        ts_match = re.match(r'(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)', lines[1])
        if not ts_match:
            continue

        start_ts, end_ts = ts_match.groups()
        duration = parse_timestamp(end_ts) - parse_timestamp(start_ts)

        # Parse content line (language, speaker, text)
        content_line = lines[2] if len(lines) > 2 else ""
        lang_match = re.match(r'\[(\w+)\|([^\]]+)\]', content_line)

        lang = lang_match.group(1) if lang_match else "UNK"
        speaker = lang_match.group(2) if lang_match else "UNKNOWN"

        # Extract LLM text
        llm_text = ""
        if "[LLM]" in content_line:
            llm_text = content_line.split("[LLM]")[-1].strip()
        elif lang_match:
            llm_text = content_line[lang_match.end():].strip()
        else:
            llm_text = content_line

        # Extract IPA lines
        fused_ipa = ""
        allo_ipa = ""
        w2v2_ipa = ""

        for line in lines[3:]:
            if line.startswith("♫fused:"):
                fused_ipa = line.replace("♫fused:", "").strip()
            elif line.startswith("♫allo:"):
                allo_ipa = line.replace("♫allo:", "").strip()
            elif line.startswith("♫w2v2:"):
                w2v2_ipa = line.replace("♫w2v2:", "").strip()

        segments.append({
            "cue": cue,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration": duration,
            "original_lang": lang,
            "speaker": speaker,
            "llm_text": llm_text,
            "fused_ipa": fused_ipa,
            "allo_ipa": allo_ipa,
            "w2v2_ipa": w2v2_ipa,
            "raw_block": block,
        })

    return segments


def parse_timestamp(ts: str) -> float:
    """Convert timestamp to seconds."""
    parts = ts.replace(',', '.').split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def has_tl(ipa: str) -> bool:
    """Check for tɬ (voiceless lateral affricate) - NAH hard marker."""
    # tɬ, t͡ɬ, tˡ all count
    return bool(re.search(r't[͡]?ɬ|tˡ', ipa))


def has_ejectives(ipa: str) -> int:
    """Count ejective consonants - MAY markers."""
    # kʼ, pʼ, tʼ, tʃʼ
    ejectives = re.findall(r"[kptb]ʼ|t͡?[ʃɕ]ʼ", ipa)
    return len(ejectives)


def has_kw(ipa: str) -> bool:
    """Check for kʷ - NAH hint (but weaker than tɬ)."""
    return "kʷ" in ipa or "kw" in ipa


def score_phonotactics(ipa: str, bigrams: dict) -> dict[str, float]:
    """Score IPA against phonotactic bigram models."""
    phones = ipa.split()
    if len(phones) < 2:
        return {"SPA": 0, "NAH": 0, "MAY": 0}

    # Add start/end markers
    phones = ["^"] + phones + ["$"]

    scores = {}
    config = bigrams.get("config", {})
    alpha = config.get("alpha", 0.01)

    for lang in ["SPA", "NAH", "MAY"]:
        lang_bigrams = bigrams.get("bigrams", {}).get(lang, {})
        total = sum(lang_bigrams.values()) + alpha * 1000  # smoothing

        log_prob = 0
        for i in range(len(phones) - 1):
            bigram = f"{phones[i]},{phones[i+1]}"
            count = lang_bigrams.get(bigram, 0)
            prob = (count + alpha) / total
            log_prob += math.log(prob)

        # Normalize by length and add language prior
        raw_score = log_prob / (len(phones) - 1)
        prior = LANGUAGE_PRIORS.get(lang, 0.1)
        scores[lang] = raw_score + math.log(prior)

    return scores


def match_lexicon(ipa: str, lexicon: list[dict], min_phonemes: int = 4, min_score: float = 0.9) -> list[str]:
    """Find NAH lexicon matches in IPA."""
    phones = ipa.split()
    if len(phones) < min_phonemes:
        return []

    matches = []
    phones_str = " ".join(phones)

    for entry in lexicon:
        word_ipa = entry.get("ipa", [])
        if len(word_ipa) < min_phonemes:
            continue

        word_str = " ".join(word_ipa)
        # Check if word IPA is substring of segment IPA
        if word_str in phones_str:
            matches.append(entry.get("word", word_str))

    return matches


def analyze_llm_text(text: str) -> dict:
    """Analyze LLM text for language hints."""
    text_lower = text.lower()
    words = re.findall(r'\b[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+\b', text_lower)

    result = {
        "latin_words": [],
        "english_words": [],
        "nah_patterns": [],
        "spanish_confidence": 0,
    }

    for word in words:
        if word in LATIN_WORDS:
            result["latin_words"].append(word)
        if word in ENGLISH_WORDS:
            result["english_words"].append(word)

    for pattern in NAH_PATTERNS:
        if re.search(pattern, text_lower):
            result["nah_patterns"].append(pattern)

    # Spanish confidence: common Spanish patterns
    spanish_hints = ["que", "de", "la", "el", "en", "es", "no", "se", "lo", "por", "con", "una", "los", "del", "para"]
    spanish_count = sum(1 for w in words if w in spanish_hints)
    if words:
        result["spanish_confidence"] = spanish_count / len(words)

    return result


def classify_segment(seg: dict, bigrams: dict, lexicon: list[dict]) -> dict:
    """Apply classification hierarchy to a segment."""
    fused = seg["fused_ipa"]
    llm_text = seg["llm_text"]
    duration = seg["duration"]

    # Hierarchy result
    result = {
        "original": seg["original_lang"],
        "new_tag": None,
        "rule": None,
        "confidence": 0,
        "details": {},
    }

    # Rule 1: Acoustic hard markers (highest priority)
    if has_tl(fused):
        result["new_tag"] = "NAH"
        result["rule"] = "acoustic_tl"
        result["confidence"] = 1.0
        result["details"]["tl_found"] = True
        return result

    ejective_count = has_ejectives(fused)
    if ejective_count >= 2:
        result["new_tag"] = "MAY"
        result["rule"] = "acoustic_ejectives"
        result["confidence"] = 0.9
        result["details"]["ejectives"] = ejective_count
        return result

    # Rule 2: Phonotactic scoring
    scores = score_phonotactics(fused, bigrams)
    result["details"]["phonotactic_scores"] = scores

    if scores:
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_lang, best_score = sorted_langs[0]
        second_score = sorted_langs[1][1] if len(sorted_langs) > 1 else -999
        margin = best_score - second_score

        if margin > 0.5:
            result["new_tag"] = best_lang
            result["rule"] = "phonotactic_high_margin"
            result["confidence"] = min(0.9, 0.5 + margin)
            result["details"]["margin"] = margin
            return result

    # Rule 3: Lexicon matches
    matches = match_lexicon(fused, lexicon)
    if len(matches) >= 3:
        result["new_tag"] = "NAH"
        result["rule"] = "lexicon_3plus"
        result["confidence"] = 0.85
        result["details"]["lexicon_matches"] = matches
        return result

    # Rule 4: LLM text analysis
    llm_analysis = analyze_llm_text(llm_text)
    result["details"]["llm_analysis"] = llm_analysis

    if len(llm_analysis["latin_words"]) >= 2:
        result["new_tag"] = "LAT"
        result["rule"] = "llm_latin"
        result["confidence"] = 0.8
        return result

    if len(llm_analysis["english_words"]) >= 2:
        result["new_tag"] = "ENG"
        result["rule"] = "llm_english"
        result["confidence"] = 0.8
        return result

    if llm_analysis["nah_patterns"]:
        # NAH patterns in text
        if has_kw(fused):  # Combined with kʷ
            result["new_tag"] = "NAH"
            result["rule"] = "llm_nah_pattern_plus_kw"
            result["confidence"] = 0.75
            return result

    if llm_analysis["spanish_confidence"] > 0.3:
        result["new_tag"] = "SPA"
        result["rule"] = "llm_spanish_confidence"
        result["confidence"] = 0.7
        result["details"]["spanish_conf"] = llm_analysis["spanish_confidence"]
        return result

    # Rule 5: Segment length filter
    if duration < 0.3:
        result["new_tag"] = "NON"
        result["rule"] = "short_segment"
        result["confidence"] = 0.6
        return result

    # Rule 6: Fallback - phonotactic best guess (even with small margin)
    if scores:
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_lang, _ = sorted_langs[0]
        second_score = sorted_langs[1][1] if len(sorted_langs) > 1 else -999
        margin = best_score - second_score

        result["new_tag"] = best_lang
        result["rule"] = "phonotactic_fallback"
        result["confidence"] = 0.4 + min(0.3, margin)
        result["details"]["margin"] = margin
        return result

    # Ultimate fallback
    result["new_tag"] = "OTH"
    result["rule"] = "no_classification"
    result["confidence"] = 0
    return result


def write_corrected_srt(segments: list[dict], output_path: Path, classifications: dict):
    """Write corrected SRT with updated language tags."""
    lines = []

    for seg in segments:
        cue = seg["cue"]
        classification = classifications.get(cue, {})
        new_tag = classification.get("new_tag", seg["original_lang"])

        # Reconstruct block with new tag
        raw = seg["raw_block"]
        old_tag = seg["original_lang"]

        # Replace [OLD|SPEAKER] with [NEW|SPEAKER]
        new_block = re.sub(
            rf'\[{re.escape(old_tag)}\|([^\]]+)\]',
            f'[{new_tag}|\\1]',
            raw
        )

        lines.append(new_block)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="OTH elimination with full hierarchy")
    parser.add_argument("--export", action="store_true", help="Generate corrected SRTs")
    parser.add_argument("--all-segs", action="store_true", help="Validate all segments, not just OTH")
    parser.add_argument("--output-dir", default="validation_video/corrected", help="Output directory")
    args = parser.parse_args()

    # Load resources
    print("Loading bigrams and lexicon...")
    bigrams = load_bigrams()
    lexicon = load_lexicon()
    print(f"  Loaded {len(lexicon)} lexicon entries")

    # Find all SRT files
    srt_dir = PROJECT_ROOT / "validation_video"
    srt_files = sorted(srt_dir.glob("Hernán-1-*.srt"))
    # Filter to main files (not .buggy, not .fixed duplicates)
    srt_files = [f for f in srt_files if ".buggy" not in f.name]
    # Prefer .fixed versions where they exist
    final_files = []
    for f in srt_files:
        fixed = f.with_suffix("").with_suffix(".fixed.srt")
        if fixed.exists():
            final_files.append(fixed)
        elif ".fixed" not in f.name:
            final_files.append(f)
    srt_files = sorted(set(final_files))

    print(f"\nProcessing {len(srt_files)} SRT files...")

    # Stats
    all_classifications = {}
    stats = {
        "by_episode": {},
        "totals": defaultdict(int),
        "original_totals": defaultdict(int),
        "confusion": defaultdict(lambda: defaultdict(int)),
        "oth_details": [],
    }

    output_dir = PROJECT_ROOT / args.output_dir
    if args.export:
        output_dir.mkdir(parents=True, exist_ok=True)

    for srt_path in srt_files:
        episode = srt_path.stem.replace("Hernán-", "E").replace(".fixed", "")
        print(f"\n{'='*60}")
        print(f"Episode: {episode}")
        print(f"{'='*60}")

        segments = parse_srt(srt_path)
        print(f"  Total segments: {len(segments)}")

        episode_stats = defaultdict(int)
        episode_original = defaultdict(int)
        episode_classifications = {}

        for seg in segments:
            original = seg["original_lang"]
            episode_original[original] += 1
            stats["original_totals"][original] += 1

            # Classify if OTH or --all-segs
            if original == "OTH" or args.all_segs:
                result = classify_segment(seg, bigrams, lexicon)
                new_tag = result["new_tag"]

                episode_classifications[seg["cue"]] = result
                episode_stats[new_tag] += 1
                stats["totals"][new_tag] += 1
                stats["confusion"][original][new_tag] += 1

                # Track OTH details
                if original == "OTH":
                    stats["oth_details"].append({
                        "episode": episode,
                        "cue": seg["cue"],
                        "new_tag": new_tag,
                        "rule": result["rule"],
                        "confidence": result["confidence"],
                        "llm_text": seg["llm_text"][:50],
                    })
            else:
                # Keep original
                episode_stats[original] += 1
                stats["totals"][original] += 1
                stats["confusion"][original][original] += 1

        stats["by_episode"][episode] = dict(episode_stats)
        all_classifications[srt_path.name] = episode_classifications

        # Episode summary
        oth_orig = episode_original.get("OTH", 0)
        oth_remaining = episode_stats.get("OTH", 0)
        print(f"  OTH: {oth_orig} → {oth_remaining} ({oth_orig - oth_remaining} reclassified)")
        print(f"  Distribution: {dict(sorted(episode_stats.items()))}")

        # Export if requested
        if args.export:
            output_path = output_dir / srt_path.name.replace(".fixed", "")
            write_corrected_srt(segments, output_path, episode_classifications)
            print(f"  Exported: {output_path.name}")

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    print("\n--- Original Tag Distribution ---")
    for tag, count in sorted(stats["original_totals"].items()):
        print(f"  {tag}: {count}")

    print("\n--- New Tag Distribution ---")
    total_segs = sum(stats["totals"].values())
    for tag, count in sorted(stats["totals"].items()):
        pct = 100 * count / total_segs if total_segs else 0
        print(f"  {tag}: {count} ({pct:.1f}%)")

    # OTH survival
    oth_original = stats["original_totals"].get("OTH", 0)
    oth_remaining = stats["totals"].get("OTH", 0)
    print(f"\n--- OTH ELIMINATION ---")
    print(f"  Original OTH: {oth_original}")
    print(f"  Remaining OTH: {oth_remaining}")
    print(f"  Eliminated: {oth_original - oth_remaining}")
    print(f"  Target: <50 (Goal: <20)")
    print(f"  Status: {'PASS' if oth_remaining < 50 else 'FAIL'}")

    # Confusion matrix
    if args.all_segs:
        print("\n--- Confusion Matrix (Original → Predicted) ---")
        tags = sorted(set(stats["confusion"].keys()) | set(t for d in stats["confusion"].values() for t in d.keys()))
        header = "      " + " ".join(f"{t:>6}" for t in tags)
        print(header)
        for orig in tags:
            row = f"{orig:>5} " + " ".join(f"{stats['confusion'][orig].get(t, 0):>6}" for t in tags)
            print(row)

    # OTH reclassification summary
    print("\n--- OTH Reclassification by Rule ---")
    rule_counts = defaultdict(int)
    for detail in stats["oth_details"]:
        rule_counts[detail["rule"]] += 1
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count}")

    # Per-episode breakdown
    print("\n--- Per-Episode Tag Counts ---")
    header = "Episode    " + " ".join(f"{t:>5}" for t in ["SPA", "NAH", "MAY", "ENG", "LAT", "NON", "OTH"])
    print(header)
    print("-" * len(header))
    for ep, counts in sorted(stats["by_episode"].items()):
        row = f"{ep:<10} " + " ".join(f"{counts.get(t, 0):>5}" for t in ["SPA", "NAH", "MAY", "ENG", "LAT", "NON", "OTH"])
        print(row)

    # Save results
    results_path = PROJECT_ROOT / "validation_video/oth_elimination_results.json"
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "original_totals": dict(stats["original_totals"]),
        "new_totals": dict(stats["totals"]),
        "by_episode": stats["by_episode"],
        "oth_details": stats["oth_details"],
        "confusion_matrix": {k: dict(v) for k, v in stats["confusion"].items()},
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    if args.export:
        print(f"Corrected SRTs exported to {output_dir}/")


if __name__ == "__main__":
    main()

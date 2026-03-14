#!/usr/bin/env python3
"""Phonotactic bigram model for language identification.

Builds a simple bigram frequency model from pipeline IPA (SRT fused lines).
No ML — just Counter() with Laplace smoothing.

Usage:
    python scripts/phonotactic_model.py                    # Build model + evaluate
    python scripts/phonotactic_model.py --export           # Export model JSON
    python scripts/phonotactic_model.py --reduce-oth       # OTH reduction analysis
    python scripts/phonotactic_model.py --confusion        # Full confusion matrix
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Phoneme normalization
NORMALIZE_MAP = {
    # Length marks
    "ː": "",
    # Laterals/rhotics
    "l": "r", "ɾ": "r", "ɹ": "r", "ʁ": "r", "ʀ": "r",
    # Labials
    "v": "b", "β": "b", "ɸ": "f",
    # Affricates (preserve as single tokens)
    "tɬ": "TL", "t͡ɬ": "TL",
    "tʃ": "ch", "t͡ʃ": "ch", "tɕ": "ch", "t͡ɕ": "ch",
    "ts": "ts", "t͡s": "ts",
    # Labialized
    "kʷ": "KW", "kw": "KW", "ɡʷ": "GW", "gʷ": "GW",
    # Glottal
    "ʔ": "Q",
    # Ejectives (preserve markers)
    "kʼ": "k'", "pʼ": "p'", "tʼ": "t'", "tsʼ": "ts'", "tʃʼ": "ch'",
    # Allophones
    "ð": "d", "ɣ": "g", "ɡ": "g",
    # Vowel normalization
    "ɑ": "a", "ɒ": "o", "ɔ": "o", "ɛ": "e", "ɪ": "i", "ʊ": "u", "ʌ": "a",
    "æ": "a", "ə": "e", "ɨ": "i", "ɯ": "u", "ɤ": "o",
}

# Multi-char patterns to check first
MULTI_CHAR_PATTERNS = ["tɬ", "t͡ɬ", "tʃ", "t͡ʃ", "tɕ", "t͡ɕ", "ts", "t͡s",
                        "kʷ", "kw", "ɡʷ", "gʷ", "kʼ", "pʼ", "tʼ", "tsʼ", "tʃʼ"]


def normalize_phoneme(p: str) -> str:
    """Normalize a single phoneme token."""
    # Check multi-char patterns first
    for pattern in MULTI_CHAR_PATTERNS:
        if pattern in p:
            p = p.replace(pattern, NORMALIZE_MAP.get(pattern, pattern))

    # Strip modifiers
    p = p.replace("ː", "").replace("ʲ", "").replace("ʰ", "").replace("ˤ", "").replace("ʼ", "'")

    # Apply single-char mappings
    result = ""
    for c in p:
        result += NORMALIZE_MAP.get(c, c)

    return result.lower() if result else ""


def get_bigrams(phonemes: list[str]) -> list[tuple[str, str]]:
    """Extract bigrams from phoneme list with start/end markers."""
    normalized = [normalize_phoneme(p) for p in phonemes if p.strip()]
    normalized = [p for p in normalized if p]  # Remove empty

    if len(normalized) < 2:
        return []

    # Add boundary markers
    tokens = ["^"] + normalized + ["$"]
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def has_tl(phonemes: list[str]) -> bool:
    """Check if tɬ present."""
    return any("tɬ" in p or "TL" in normalize_phoneme(p) for p in phonemes)


def has_ejective(phonemes: list[str]) -> bool:
    """Check for Maya ejective markers."""
    ejective_patterns = ["kʼ", "pʼ", "tʼ", "k'", "p'", "t'", "tsʼ", "tʃʼ"]
    return any(any(e in p for e in ejective_patterns) for p in phonemes)


# SRT parsing
LANG_RE = re.compile(r"\[([A-Z]{3}\??)\|")
FUSED_RE = re.compile(r"♫?\s*fused:\s*(.+)", re.I)


def parse_srt_segments(srt_path: Path) -> list[dict]:
    """Parse SRT file for language tags and fused IPA."""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        content = "\n".join(lines[2:])

        # Get language tag
        lang_match = LANG_RE.search(content)
        lang = lang_match.group(1).replace("?", "") if lang_match else "UNK"

        # Get fused IPA
        fused_match = FUSED_RE.search(content)
        if not fused_match:
            continue
        phonemes = fused_match.group(1).strip().split()

        if len(phonemes) >= 2:
            segments.append({
                "file": srt_path.name,
                "lang": lang,
                "phonemes": phonemes,
            })

    return segments


class PhonotacticModel:
    """Simple bigram frequency model for language ID."""

    def __init__(self, alpha: float = 0.01, vocab_size: int = 50):
        self.alpha = alpha
        self.V2 = vocab_size ** 2  # Smoothing denominator
        self.bigram_counts: dict[str, Counter] = defaultdict(Counter)
        self.total_counts: dict[str, int] = defaultdict(int)

    def train(self, lang: str, phonemes: list[str]) -> None:
        """Add bigrams from a segment to language model."""
        bigrams = get_bigrams(phonemes)
        for bg in bigrams:
            key = f"{bg[0]},{bg[1]}"
            self.bigram_counts[lang][key] += 1
            self.total_counts[lang] += 1

    def score(self, lang: str, phonemes: list[str]) -> float:
        """Score a segment for a language using log-likelihood."""
        bigrams = get_bigrams(phonemes)
        if not bigrams:
            return float("-inf")

        counts = self.bigram_counts[lang]
        total = self.total_counts[lang]

        log_prob = 0.0
        for bg in bigrams:
            key = f"{bg[0]},{bg[1]}"
            count = counts[key]
            # Laplace smoothing
            prob = (count + self.alpha) / (total + self.alpha * self.V2)
            log_prob += math.log(prob)

        return log_prob / len(bigrams)  # Normalize by length

    def predict(self, phonemes: list[str]) -> tuple[str, float, dict[str, float]]:
        """Predict language and return scores."""
        scores = {}
        for lang in self.bigram_counts.keys():
            scores[lang] = self.score(lang, phonemes)

        if not scores:
            return "UNK", 0.0, {}

        best_lang = max(scores, key=lambda l: scores[l])
        return best_lang, scores[best_lang], scores

    def export(self) -> dict:
        """Export model as JSON-serializable dict."""
        return {
            lang: dict(counts)
            for lang, counts in self.bigram_counts.items()
        }


def main():
    parser = argparse.ArgumentParser(description="Phonotactic bigram model")
    parser.add_argument("--export", action="store_true", help="Export model JSON")
    parser.add_argument("--reduce-oth", action="store_true", help="OTH reduction analysis")
    parser.add_argument("--confusion", action="store_true", help="Full confusion matrix")
    args = parser.parse_args()

    srt_dir = PROJECT_ROOT / "validation_video"
    srt_files = sorted(srt_dir.glob("Hernán-*.srt"))

    print(f"Loading {len(srt_files)} SRT files...")

    # Collect all segments
    all_segments = []
    for srt_path in srt_files:
        segments = parse_srt_segments(srt_path)
        all_segments.extend(segments)

    print(f"Total segments: {len(all_segments)}")

    # Count by original tag
    tag_counts = Counter(seg["lang"] for seg in all_segments)
    print(f"Original tags: {dict(tag_counts)}")

    # Build model from NAH, SPA, MAY segments
    # Apply tɬ reclassification during training
    model = PhonotacticModel(alpha=0.01, vocab_size=50)
    training_counts = Counter()

    for seg in all_segments:
        lang = seg["lang"]
        phonemes = seg["phonemes"]

        # tɬ reclassification: if tɬ present, train as NAH
        if has_tl(phonemes) and lang != "NAH":
            lang = "NAH"

        # Only train on NAH, SPA, MAY (not OTH/UNK)
        if lang in ("NAH", "SPA", "MAY"):
            model.train(lang, phonemes)
            training_counts[lang] += 1

    print(f"\nTraining segments (with tɬ reclassification): {dict(training_counts)}")
    print(f"Total bigrams: NAH={model.total_counts['NAH']}, SPA={model.total_counts['SPA']}, MAY={model.total_counts['MAY']}")

    # ========== TASK 2: OTH Reduction ==========
    if args.reduce_oth or args.confusion:
        print("\n" + "=" * 70)
        print("OTH REDUCTION ANALYSIS")
        print("=" * 70)

        oth_segments = [seg for seg in all_segments if seg["lang"] == "OTH"]
        print(f"\nOriginal OTH segments: {len(oth_segments)}")

        reclassified = {"NAH": 0, "SPA": 0, "MAY": 0, "OTH": 0}
        reclassification_reasons = Counter()

        for seg in oth_segments:
            phonemes = seg["phonemes"]
            new_lang = "OTH"
            reason = "no_clear_signal"

            # Rule 1: tɬ → NAH
            if has_tl(phonemes):
                new_lang = "NAH"
                reason = "tl_detected"

            # Rule 2: Ejectives → MAY
            elif has_ejective(phonemes):
                new_lang = "MAY"
                reason = "ejective_detected"

            # Rule 3: Model score
            else:
                pred_lang, best_score, scores = model.predict(phonemes)
                sorted_scores = sorted(scores.values(), reverse=True)

                if len(sorted_scores) >= 2:
                    margin = sorted_scores[0] - sorted_scores[1]
                    if margin > 1.5:  # Clear winner
                        new_lang = pred_lang
                        reason = f"model_score_{pred_lang}"

            seg["predicted"] = new_lang
            reclassified[new_lang] += 1
            reclassification_reasons[reason] += 1

        print(f"\nOTH reclassification results:")
        print(f"  → NAH: {reclassified['NAH']}")
        print(f"  → SPA: {reclassified['SPA']}")
        print(f"  → MAY: {reclassified['MAY']}")
        print(f"  → OTH (unchanged): {reclassified['OTH']}")

        print(f"\nReasons:")
        for reason, count in reclassification_reasons.most_common():
            print(f"  {reason}: {count}")

        reduction_pct = 100 * (1 - reclassified["OTH"] / len(oth_segments)) if oth_segments else 0
        print(f"\nOTH reduction: {len(oth_segments)} → {reclassified['OTH']} ({reduction_pct:.1f}% reduced)")

    # ========== TASK 3: Confusion Matrix ==========
    if args.confusion:
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX (all segments)")
        print("=" * 70)

        # Predict all segments
        confusion = defaultdict(lambda: defaultdict(int))
        per_episode = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for seg in all_segments:
            true_lang = seg["lang"]
            phonemes = seg["phonemes"]
            episode = seg["file"].replace("Hernán-", "E").replace(".srt", "")

            # Apply same rules as OTH reduction
            if has_tl(phonemes):
                pred_lang = "NAH"
            elif has_ejective(phonemes) and true_lang in ("OTH", "MAY"):
                pred_lang = "MAY"
            else:
                pred_lang, _, _ = model.predict(phonemes)

            confusion[true_lang][pred_lang] += 1
            per_episode[episode][true_lang][pred_lang] += 1

        # Print aggregated confusion matrix
        langs = ["NAH", "SPA", "MAY", "OTH"]
        print(f"\n{'True\\Pred':<10}", end="")
        for pred in langs:
            print(f"{pred:<8}", end="")
        print("Accuracy")
        print("-" * 50)

        for true in langs:
            print(f"{true:<10}", end="")
            row_total = sum(confusion[true].values())
            for pred in langs:
                print(f"{confusion[true][pred]:<8}", end="")
            correct = confusion[true][true]
            acc = 100 * correct / row_total if row_total > 0 else 0
            print(f"{acc:.1f}%")

        # Per-episode accuracy
        print(f"\nPer-episode accuracy:")
        print(f"{'Episode':<15} | {'NAH':<8} | {'SPA':<8} | {'MAY':<8} | {'OTH':<8}")
        print("-" * 55)

        for episode in sorted(per_episode.keys()):
            ep_conf = per_episode[episode]
            accs = []
            for lang in langs:
                total = sum(ep_conf[lang].values())
                correct = ep_conf[lang][lang]
                acc = 100 * correct / total if total > 0 else 0
                accs.append(f"{acc:.0f}%")
            print(f"{episode:<15} | {accs[0]:<8} | {accs[1]:<8} | {accs[2]:<8} | {accs[3]:<8}")

    # ========== Export Model ==========
    if args.export:
        model_path = PROJECT_ROOT / "src/tenepal/data/phonotactic_bigrams.json"
        model_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "alpha": 0.01,
                "vocab_size": 50,
                "training_segments": dict(training_counts),
            },
            "bigrams": model.export(),
            "totals": dict(model.total_counts),
        }

        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"\nExported model to {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

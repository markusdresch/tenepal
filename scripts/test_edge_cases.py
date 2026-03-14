#!/usr/bin/env python3
"""
Edge Case Testing Script for Tenepal Pipeline

Tests edge cases with different configurations to find optimal settings.
Uses the same backend as the main pipeline (local, not Modal).

Usage:
    python scripts/test_edge_cases.py                    # Test all cases
    python scripts/test_edge_cases.py --case EC001      # Test specific case
    python scripts/test_edge_cases.py --compare         # Compare configurations
    python scripts/test_edge_cases.py --batch-settings  # Test setting variations
"""

import argparse
import json
import sys
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

EDGE_CASES_DIR = Path("validation_video/edge_cases")
CATALOG_FILE = EDGE_CASES_DIR / "catalog.json"
RESULTS_FILE = EDGE_CASES_DIR / "test_results.json"


@dataclass
class TestConfig:
    """Configuration for a pipeline test run"""
    name: str = "default"

    # Phoneme backend settings
    phoneme_backend: str = "allosaurus"  # allosaurus, wav2vec2, both
    use_phone_voting: bool = True

    # Language detection settings
    lang_conf_threshold: float = 0.6
    spa_threshold: float = 2.25
    may_threshold: float = 1.5
    nah_threshold: float = 0.0

    # Maya/ejective settings
    min_ejectives_for_may: int = 2
    require_acoustic_ejectives: bool = False

    # Segment handling
    min_segment_duration: float = 0.3
    merge_gap_threshold: float = 0.0  # 0 = no merging

    # Spanish leak detection
    check_spanish_leak_in_llm: bool = False
    spanish_leak_threshold: float = 0.35

    # Padding for clips
    padding_before: float = 0.0
    padding_after: float = 0.0


def load_catalog() -> Dict[str, Any]:
    """Load edge case catalog"""
    if not CATALOG_FILE.exists():
        print("No catalog found. Run 'python scripts/edge_case_tester.py extract' first.")
        sys.exit(1)

    with open(CATALOG_FILE) as f:
        return json.load(f)


def get_phonemes_allosaurus(wav_path: str) -> str:
    """Get IPA phonemes using Allosaurus"""
    try:
        from allosaurus.app import read_recognizer
        model = read_recognizer()
        return model.recognize(wav_path)
    except Exception as e:
        print(f"    Allosaurus error: {e}")
        return ""


def get_phonemes_wav2vec2(wav_path: str) -> str:
    """Get IPA phonemes using Wav2Vec2"""
    try:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import librosa

        model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)

        audio, sr = librosa.load(wav_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

        return transcription
    except Exception as e:
        print(f"    Wav2Vec2 error: {e}")
        return ""


def identify_language_custom(
    ipa: str,
    config: TestConfig
) -> Tuple[str, float]:
    """
    Custom language identification with configurable thresholds.
    Returns (lang, confidence).
    """
    if not ipa or not ipa.strip():
        return "other", 0.0

    phones = ipa.split()
    if len(phones) < 2:
        return "other", 0.0

    # Simplified scoring based on phoneme patterns
    scores = {"spa": 0.0, "eng": 0.0, "may": 0.0, "nah": 0.0}

    # Spanish indicators
    spa_phones = {"a", "e", "i", "o", "u", "r", "ɾ", "l", "n", "s", "t", "d", "k", "p", "b", "m"}
    # English indicators
    eng_phones = {"ə", "ɪ", "æ", "ʊ", "ð", "θ", "ŋ", "ɹ", "w", "j", "eɪ", "aʊ", "oʊ"}
    # Maya ejectives
    may_ejectives = {"tʼ", "kʼ", "pʼ", "t'", "k'", "p'", "ʔ"}
    # Nahuatl markers
    nah_markers = {"tɬ", "tl", "ts", "ʃ", "tʃ"}

    # Count ejectives
    ejective_count = sum(1 for p in phones if any(ej in p for ej in may_ejectives))

    # Score each language
    for phone in phones:
        if phone in spa_phones or any(c in phone for c in spa_phones):
            scores["spa"] += 0.3
        if phone in eng_phones or any(c in phone for c in eng_phones):
            scores["eng"] += 0.4
        if any(ej in phone for ej in may_ejectives):
            scores["may"] += 0.5
        if any(m in phone for m in nah_markers):
            scores["nah"] += 0.4

    # Apply thresholds
    if scores["may"] > config.may_threshold and ejective_count >= config.min_ejectives_for_may:
        return "may", scores["may"]
    if scores["nah"] > config.nah_threshold:
        return "nah", scores["nah"]
    if scores["eng"] > 4.0:  # English threshold
        return "eng", scores["eng"]
    if scores["spa"] > config.spa_threshold:
        return "spa", scores["spa"]

    # Fallback: highest score above 0
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best, scores[best]

    return "other", 0.0


def detect_spanish_leak(text: str) -> bool:
    """Check if text contains Spanish words that leaked into OTH"""
    import re

    spanish_patterns = [
        r'\bpregunta',
        r'\bprimer[oa]?',
        r'\bmentida',
        r'\benemigo',
        r'\bsolo\s*repite',
        r'\bhermano',
        r'\bamigo',
        r'\bseñor',
        r'\bdios',
        r'\btierra',
        r'\bmuerte',
        r'\bguerr',
        r'\bcapitan',
        r'\brepro',  # reproches
    ]

    text_lower = text.lower()
    for pattern in spanish_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def run_pipeline_test(
    wav_path: Path,
    config: TestConfig
) -> Dict[str, Any]:
    """
    Run a single pipeline test with given configuration.
    Returns results dict.
    """
    import librosa

    results = {
        "config": config.name,
        "wav": str(wav_path),
        "ipa_allo": "",
        "ipa_w2v2": "",
        "ipa_fused": "",
        "lang": "other",
        "conf": 0.0,
        "ejective_count": 0,
        "spanish_leak": False,
        "error": None,
    }

    try:
        # Check duration
        duration = librosa.get_duration(path=str(wav_path))
        if duration < config.min_segment_duration:
            results["error"] = f"Too short: {duration:.2f}s < {config.min_segment_duration}s"
            return results

        # Get phonemes based on backend setting
        if config.phoneme_backend in ("allosaurus", "both"):
            results["ipa_allo"] = get_phonemes_allosaurus(str(wav_path))

        if config.phoneme_backend in ("wav2vec2", "both"):
            results["ipa_w2v2"] = get_phonemes_wav2vec2(str(wav_path))

        # Fuse if both backends used
        if config.use_phone_voting and results["ipa_allo"] and results["ipa_w2v2"]:
            # Simple fusion: prefer wav2vec2 for clarity
            results["ipa_fused"] = results["ipa_w2v2"]
        else:
            results["ipa_fused"] = results["ipa_allo"] or results["ipa_w2v2"]

        # Count ejectives
        may_ejectives = {"tʼ", "kʼ", "pʼ", "t'", "k'", "p'"}
        ipa = results["ipa_fused"]
        results["ejective_count"] = sum(1 for ej in may_ejectives if ej in ipa)

        # Identify language
        lang, conf = identify_language_custom(ipa, config)
        results["lang"] = lang.upper()
        results["conf"] = round(conf, 3)

        # Check Spanish leak if enabled
        if config.check_spanish_leak_in_llm:
            # Would check LLM output here, but we're testing IPA only
            results["spanish_leak"] = detect_spanish_leak(ipa)

    except Exception as e:
        results["error"] = str(e)

    return results


def test_case(
    case_id: str,
    catalog: Dict[str, Any],
    config: TestConfig,
    verbose: bool = True
) -> Dict[str, Any]:
    """Test a single edge case"""
    if case_id not in catalog:
        return {"error": f"Unknown case: {case_id}"}

    case = catalog[case_id]
    wav_path = EDGE_CASES_DIR / f"{case_id}.wav"

    if not wav_path.exists():
        return {"error": f"WAV not found: {wav_path}"}

    if verbose:
        print(f"\n--- {case_id}: {case['description'][:50]}... ---")
        print(f"    Expected: {case['expected_lang']} | Currently: {case['actual_lang']}")

    result = run_pipeline_test(wav_path, config)
    result["case_id"] = case_id
    result["expected"] = case["expected_lang"]
    result["original"] = case["actual_lang"]

    # Determine if pass/fail
    if result["lang"] == case["expected_lang"]:
        result["status"] = "PASS"
    elif result["lang"] == case["actual_lang"]:
        result["status"] = "SAME"  # Same as before (no improvement)
    else:
        result["status"] = "CHANGED"  # Different but not expected

    if verbose:
        status_icon = {"PASS": "✅", "SAME": "⚪", "CHANGED": "🔶"}.get(result["status"], "❓")
        print(f"    Result: {result['lang']} (conf: {result['conf']})")
        print(f"    Ejectives: {result['ejective_count']}")
        print(f"    IPA: {result['ipa_fused'][:60]}...")
        print(f"    {status_icon} {result['status']}")

    return result


def run_all_tests(config: TestConfig, verbose: bool = True) -> List[Dict[str, Any]]:
    """Run tests on all edge cases"""
    catalog = load_catalog()

    print(f"\n{'='*80}")
    print(f"TESTING {len(catalog)} EDGE CASES")
    print(f"Config: {config.name}")
    print(f"Backend: {config.phoneme_backend}, Voting: {config.use_phone_voting}")
    print(f"Thresholds - SPA: {config.spa_threshold}, MAY: {config.may_threshold}")
    print(f"Min ejectives for MAY: {config.min_ejectives_for_may}")
    print(f"{'='*80}")

    results = []
    for case_id in sorted(catalog.keys()):
        result = test_case(case_id, catalog, config, verbose)
        results.append(result)

    # Summary
    passes = sum(1 for r in results if r.get("status") == "PASS")
    same = sum(1 for r in results if r.get("status") == "SAME")
    changed = sum(1 for r in results if r.get("status") == "CHANGED")
    errors = sum(1 for r in results if r.get("error"))

    print(f"\n{'='*80}")
    print(f"SUMMARY: {passes} PASS, {same} SAME, {changed} CHANGED, {errors} ERRORS")
    print(f"{'='*80}")

    return results


def compare_configurations():
    """Compare different configurations side by side"""
    configs = [
        TestConfig(name="baseline"),
        TestConfig(name="no_voting", use_phone_voting=False),
        TestConfig(name="wav2vec2_only", phoneme_backend="wav2vec2"),
        TestConfig(name="lower_spa_thresh", spa_threshold=1.8),
        TestConfig(name="higher_may_thresh", may_threshold=2.0),
        TestConfig(name="require_3_ejectives", min_ejectives_for_may=3),
        TestConfig(name="skip_short_05s", min_segment_duration=0.5),
        TestConfig(name="spanish_leak_check", check_spanish_leak_in_llm=True),
    ]

    catalog = load_catalog()
    all_results = {}

    for config in configs:
        print(f"\n\n{'#'*80}")
        print(f"# CONFIG: {config.name}")
        print(f"{'#'*80}")

        results = run_all_tests(config, verbose=False)
        all_results[config.name] = results

    # Comparison table
    print(f"\n\n{'='*80}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Case':<8} | " + " | ".join(f"{c.name[:12]:<12}" for c in configs))
    print("-" * (10 + 15 * len(configs)))

    for case_id in sorted(catalog.keys()):
        row = f"{case_id:<8} | "
        for config in configs:
            result = next((r for r in all_results[config.name] if r.get("case_id") == case_id), {})
            status = result.get("status", "ERR")
            lang = result.get("lang", "?")
            icon = {"PASS": "✅", "SAME": "⚪", "CHANGED": "🔶", "ERR": "❌"}.get(status, "?")
            row += f"{icon}{lang:<10} | "
        print(row)

    # Summary row
    print("-" * (10 + 15 * len(configs)))
    row = f"{'PASS':<8} | "
    for config in configs:
        passes = sum(1 for r in all_results[config.name] if r.get("status") == "PASS")
        row += f"{passes:<12} | "
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Edge Case Pipeline Tester")
    parser.add_argument("--case", help="Test specific case ID")
    parser.add_argument("--compare", action="store_true", help="Compare configurations")
    parser.add_argument("--config", default="baseline", help="Config name to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.compare:
        compare_configurations()
    elif args.case:
        catalog = load_catalog()
        config = TestConfig(name=args.config)
        test_case(args.case, catalog, config, verbose=True)
    else:
        config = TestConfig(name=args.config)
        run_all_tests(config, verbose=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Edge Case Testing on Modal

Tests edge cases with different pipeline configurations on Modal GPU.
Uses the same backend as production but with configurable settings.

Usage:
    python scripts/test_edge_cases_modal.py                     # Test all with baseline
    python scripts/test_edge_cases_modal.py --case EC001       # Test specific case
    python scripts/test_edge_cases_modal.py --compare          # Compare all configs
    python scripts/test_edge_cases_modal.py --config strict_ejectives
"""

import modal
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

EDGE_CASES_DIR = Path("validation_video/edge_cases")
CATALOG_FILE = EDGE_CASES_DIR / "catalog.json"
RESULTS_DIR = EDGE_CASES_DIR / "modal_results"

# Modal app setup
app = modal.App("tenepal-edge-case-tester")

# Reuse the main pipeline's image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "allosaurus>=1.0.2",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
    )
)


@dataclass
class ModalTestConfig:
    """Configuration for Modal pipeline test"""
    name: str = "baseline"

    # Phoneme backend
    phoneme_backend: str = "allosaurus"  # allosaurus, wav2vec2, both
    use_phone_voting: bool = True

    # Language thresholds
    lang_conf_threshold: float = 0.6
    spa_threshold: float = 2.25
    may_threshold: float = 1.5

    # Ejective detection
    min_ejectives_for_may: int = 2
    require_w2v2_ejective_confirm: bool = False
    ejective_consensus_threshold: int = 2

    # Segment handling
    min_segment_duration: float = 0.0

    # Spanish leak
    check_spanish_leak_llm: bool = False


# Test configurations to compare
CONFIGS = {
    "baseline": ModalTestConfig(name="baseline"),

    "strict_ejectives": ModalTestConfig(
        name="strict_ejectives",
        min_ejectives_for_may=3,
        require_w2v2_ejective_confirm=True,
        ejective_consensus_threshold=3,
    ),

    "lower_spa_thresh": ModalTestConfig(
        name="lower_spa_thresh",
        spa_threshold=1.8,
    ),

    "lower_conf_thresh": ModalTestConfig(
        name="lower_conf_thresh",
        lang_conf_threshold=0.5,
    ),

    "higher_may_thresh": ModalTestConfig(
        name="higher_may_thresh",
        may_threshold=2.5,
    ),

    "spanish_leak_check": ModalTestConfig(
        name="spanish_leak_check",
        check_spanish_leak_llm=True,
    ),

    "skip_short": ModalTestConfig(
        name="skip_short",
        min_segment_duration=0.5,
    ),

    "no_ejective_boost": ModalTestConfig(
        name="no_ejective_boost",
        min_ejectives_for_may=99,  # Effectively disable
    ),

    "w2v2_only": ModalTestConfig(
        name="w2v2_only",
        phoneme_backend="wav2vec2",
    ),

    "combined_fixes": ModalTestConfig(
        name="combined_fixes",
        min_ejectives_for_may=3,
        require_w2v2_ejective_confirm=True,
        lang_conf_threshold=0.5,
        check_spanish_leak_llm=True,
        min_segment_duration=0.3,
    ),
}


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_segment_modal(
    audio_bytes: bytes,
    config_dict: dict,
) -> dict:
    """
    Test a single audio segment on Modal with given config.
    Returns dict with results.
    """
    import tempfile
    import torch
    import librosa

    result = {
        "ipa_allo": "",
        "ipa_w2v2": "",
        "lang": "other",
        "conf": 0.0,
        "ejective_count": 0,
        "ejective_details": {},
        "spanish_leak": False,
        "error": None,
    }

    try:
        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            wav_path = f.name

        # Check duration
        duration = librosa.get_duration(path=wav_path)
        result["duration"] = round(duration, 2)

        min_dur = config_dict.get("min_segment_duration", 0.0)
        if duration < min_dur:
            result["error"] = f"Too short: {duration:.2f}s"
            result["lang"] = "SKIP"
            return result

        # Get Allosaurus phonemes
        backend = config_dict.get("phoneme_backend", "allosaurus")

        if backend in ("allosaurus", "both"):
            try:
                from allosaurus.app import read_recognizer
                model = read_recognizer()
                result["ipa_allo"] = model.recognize(wav_path)
            except Exception as e:
                result["ipa_allo"] = f"ERROR: {e}"

        if backend in ("wav2vec2", "both"):
            try:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
                processor = Wav2Vec2Processor.from_pretrained(model_name)
                model = Wav2Vec2ForCTC.from_pretrained(model_name)

                audio, sr = librosa.load(wav_path, sr=16000)
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

                with torch.no_grad():
                    logits = model(inputs.input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    result["ipa_w2v2"] = processor.batch_decode(predicted_ids)[0]
            except Exception as e:
                result["ipa_w2v2"] = f"ERROR: {e}"

        # Use primary IPA
        ipa = result["ipa_allo"] if result["ipa_allo"] and not result["ipa_allo"].startswith("ERROR") else result["ipa_w2v2"]

        if not ipa or ipa.startswith("ERROR"):
            result["lang"] = "OTHER"
            result["error"] = "No IPA output"
            return result

        # Count ejectives
        ejectives = {"tʼ": 0, "kʼ": 0, "pʼ": 0, "t'": 0, "k'": 0, "p'": 0, "ʔ": 0}
        for ej in ejectives:
            ejectives[ej] = ipa.count(ej)
        result["ejective_count"] = sum(v for k, v in ejectives.items() if k != "ʔ")
        result["ejective_details"] = {k: v for k, v in ejectives.items() if v > 0}

        # W2V2 ejective check if required
        w2v2_ejectives = 0
        if config_dict.get("require_w2v2_ejective_confirm") and result["ipa_w2v2"]:
            for ej in ["tʼ", "kʼ", "pʼ", "t'", "k'", "p'"]:
                w2v2_ejectives += result["ipa_w2v2"].count(ej)
            result["w2v2_ejectives"] = w2v2_ejectives

        # Language identification
        lang, conf = identify_language_modal(
            ipa,
            config_dict,
            result["ejective_count"],
            w2v2_ejectives if config_dict.get("require_w2v2_ejective_confirm") else None,
        )
        result["lang"] = lang.upper()
        result["conf"] = round(conf, 3)

        # Spanish leak check
        if config_dict.get("check_spanish_leak_llm"):
            result["spanish_leak"] = check_spanish_leak(ipa)
            if result["spanish_leak"] and result["lang"] == "OTH":
                result["lang"] = "SPA"
                result["spanish_leak_override"] = True

        # Cleanup
        import os
        os.unlink(wav_path)

    except Exception as e:
        result["error"] = str(e)

    return result


def identify_language_modal(
    ipa: str,
    config: dict,
    ejective_count: int,
    w2v2_ejectives: Optional[int],
) -> tuple:
    """Language identification with Modal config"""
    if not ipa or not ipa.strip():
        return "other", 0.0

    phones = ipa.split()
    if len(phones) < 2:
        return "other", 0.0

    scores = {"spa": 0.0, "eng": 0.0, "may": 0.0, "nah": 0.0}

    # Phoneme sets
    spa_phones = {"a", "e", "i", "o", "u", "r", "ɾ", "l", "n", "s", "t", "d", "k", "p", "b", "m"}
    eng_phones = {"ə", "ɪ", "æ", "ʊ", "ð", "θ", "ŋ", "ɹ", "w", "j"}
    nah_markers = {"tɬ", "tl", "ts", "ʃ", "tʃ"}

    for phone in phones:
        if any(c in phone for c in spa_phones):
            scores["spa"] += 0.3
        if any(c in phone for c in eng_phones):
            scores["eng"] += 0.4
        if any(m in phone for m in nah_markers):
            scores["nah"] += 0.4

    # Maya ejective scoring with config
    min_ej = config.get("min_ejectives_for_may", 2)
    may_thresh = config.get("may_threshold", 1.5)

    # Check if ejectives meet requirements
    ejectives_valid = ejective_count >= min_ej
    if config.get("require_w2v2_ejective_confirm") and w2v2_ejectives is not None:
        ejectives_valid = ejectives_valid and w2v2_ejectives >= 1

    if ejectives_valid:
        scores["may"] = ejective_count * 0.5

    # Apply thresholds
    spa_thresh = config.get("spa_threshold", 2.25)

    if scores["may"] > may_thresh and ejectives_valid:
        return "may", scores["may"]
    if scores["nah"] > 0.0 and any(m in ipa for m in nah_markers):
        return "nah", scores["nah"]
    if scores["eng"] > 4.0:
        return "eng", scores["eng"]
    if scores["spa"] > spa_thresh:
        return "spa", scores["spa"]

    # Confidence gate
    conf_thresh = config.get("lang_conf_threshold", 0.6)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best, scores[best]

    return "other", 0.0


def check_spanish_leak(text: str) -> bool:
    """Check for Spanish patterns in IPA/text"""
    import re
    patterns = [
        r'ment', r'prim', r'enem', r'sold', r'capit', r'señor',
        r'amig', r'dios', r'tierr', r'muert', r'guerr', r'herm',
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def load_catalog() -> dict:
    """Load edge case catalog"""
    with open(CATALOG_FILE) as f:
        return json.load(f)


def test_case_modal(
    case_id: str,
    catalog: dict,
    config: ModalTestConfig,
    verbose: bool = True,
) -> dict:
    """Test a single case on Modal"""
    if case_id not in catalog:
        return {"error": f"Unknown case: {case_id}"}

    case = catalog[case_id]
    wav_path = EDGE_CASES_DIR / f"{case_id}.wav"

    if not wav_path.exists():
        return {"error": f"WAV not found: {wav_path}"}

    if verbose:
        print(f"\n--- {case_id}: {case['description'][:50]}... ---")
        print(f"    Expected: {case['expected_lang']} | Original: {case['actual_lang']}")

    # Read audio
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    # Run on Modal
    result = test_segment_modal.remote(audio_bytes, asdict(config))
    result["case_id"] = case_id
    result["expected"] = case["expected_lang"]
    result["original"] = case["actual_lang"]
    result["config"] = config.name

    # Determine status
    if result.get("lang") == case["expected_lang"]:
        result["status"] = "PASS"
    elif result.get("lang") == case["actual_lang"]:
        result["status"] = "SAME"
    else:
        result["status"] = "CHANGED"

    if verbose:
        status_icon = {"PASS": "✅", "SAME": "⚪", "CHANGED": "🔶", "SKIP": "⏭️"}.get(result["status"], "❓")
        print(f"    Result: {result.get('lang', '?')} (conf: {result.get('conf', 'N/A')})")
        print(f"    Ejectives: {result.get('ejective_count', 0)} {result.get('ejective_details', {})}")
        if result.get("w2v2_ejectives") is not None:
            print(f"    W2V2 ejectives: {result.get('w2v2_ejectives')}")
        if result.get("spanish_leak_override"):
            print(f"    Spanish leak detected → SPA")
        print(f"    IPA: {result.get('ipa_allo', '')[:60]}...")
        print(f"    {status_icon} {result['status']}")

    return result


def run_all_tests(config: ModalTestConfig, verbose: bool = True) -> list:
    """Run all edge case tests with given config"""
    catalog = load_catalog()

    print(f"\n{'='*80}")
    print(f"MODAL TESTING - {len(catalog)} EDGE CASES")
    print(f"Config: {config.name}")
    print(f"Backend: {config.phoneme_backend}")
    print(f"Thresholds - SPA: {config.spa_threshold}, MAY: {config.may_threshold}")
    print(f"Min ejectives: {config.min_ejectives_for_may}, W2V2 confirm: {config.require_w2v2_ejective_confirm}")
    print(f"Spanish leak check: {config.check_spanish_leak_llm}")
    print(f"{'='*80}")

    results = []
    for case_id in sorted(catalog.keys()):
        result = test_case_modal(case_id, catalog, config, verbose)
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


def compare_configs_modal():
    """Compare all configurations on Modal"""
    catalog = load_catalog()
    all_results = {}

    configs_to_test = [
        "baseline",
        "strict_ejectives",
        "lower_spa_thresh",
        "lower_conf_thresh",
        "higher_may_thresh",
        "no_ejective_boost",
        "spanish_leak_check",
        "combined_fixes",
    ]

    for config_name in configs_to_test:
        config = CONFIGS[config_name]
        print(f"\n\n{'#'*80}")
        print(f"# CONFIG: {config_name}")
        print(f"{'#'*80}")

        results = run_all_tests(config, verbose=True)
        all_results[config_name] = results

    # Comparison table
    print(f"\n\n{'='*80}")
    print("MODAL CONFIGURATION COMPARISON")
    print(f"{'='*80}")

    header = f"{'Case':<8} | {'Expected':<6} | {'Original':<6} | "
    header += " | ".join(f"{c[:10]:<10}" for c in configs_to_test)
    print(f"\n{header}")
    print("-" * len(header))

    for case_id in sorted(catalog.keys()):
        case = catalog[case_id]
        row = f"{case_id:<8} | {case['expected_lang']:<6} | {case['actual_lang']:<6} | "
        for config_name in configs_to_test:
            result = next((r for r in all_results[config_name] if r.get("case_id") == case_id), {})
            status = result.get("status", "ERR")
            lang = result.get("lang", "?")
            icon = {"PASS": "✅", "SAME": "⚪", "CHANGED": "🔶", "SKIP": "⏭️"}.get(status, "❌")
            row += f"{icon}{lang:<8} | "
        print(row)

    # Summary row
    print("-" * len(header))
    row = f"{'PASS':<8} | {'':<6} | {'':<6} | "
    for config_name in configs_to_test:
        passes = sum(1 for r in all_results[config_name] if r.get("status") == "PASS")
        row += f"{passes:<10} | "
    print(row)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'comparison.json'}")

    # Best config analysis
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS BY CASE")
    print(f"{'='*80}")

    for case_id in sorted(catalog.keys()):
        case = catalog[case_id]
        best_configs = []
        for config_name in configs_to_test:
            result = next((r for r in all_results[config_name] if r.get("case_id") == case_id), {})
            if result.get("status") == "PASS":
                best_configs.append(config_name)

        if best_configs:
            print(f"{case_id}: {', '.join(best_configs)}")
        else:
            print(f"{case_id}: NO CONFIG PASSES (expected: {case['expected_lang']})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Modal Edge Case Tester")
    parser.add_argument("--case", help="Test specific case ID")
    parser.add_argument("--compare", action="store_true", help="Compare all configurations")
    parser.add_argument("--config", default="baseline", help="Config name (see CONFIGS dict)")
    parser.add_argument("--list-configs", action="store_true", help="List available configs")

    args = parser.parse_args()

    if args.list_configs:
        print("Available configurations:")
        for name, cfg in CONFIGS.items():
            print(f"  {name}:")
            print(f"    - ejectives: min={cfg.min_ejectives_for_may}, w2v2_confirm={cfg.require_w2v2_ejective_confirm}")
            print(f"    - thresholds: spa={cfg.spa_threshold}, may={cfg.may_threshold}, conf={cfg.lang_conf_threshold}")
            print(f"    - spanish_leak: {cfg.check_spanish_leak_llm}")
        return

    with app.run():
        if args.compare:
            compare_configs_modal()
        elif args.case:
            catalog = load_catalog()
            config = CONFIGS.get(args.config, CONFIGS["baseline"])
            test_case_modal(args.case, catalog, config, verbose=True)
        else:
            config = CONFIGS.get(args.config, CONFIGS["baseline"])
            run_all_tests(config, verbose=True)


if __name__ == "__main__":
    main()

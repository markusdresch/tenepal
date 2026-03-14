#!/usr/bin/env python3
"""
Edge Case Tester for Tenepal Pipeline

Extracts small WAV clips from problematic segments and tests
different pipeline configurations to find optimal settings.

Usage:
    python scripts/edge_case_tester.py catalog     # List all edge cases
    python scripts/edge_case_tester.py extract     # Extract WAV clips
    python scripts/edge_case_tester.py test        # Run tests on clips
    python scripts/edge_case_tester.py test --case EC001  # Test specific case
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

EDGE_CASES_DIR = Path("validation_video/edge_cases")
CATALOG_FILE = EDGE_CASES_DIR / "catalog.json"

@dataclass
class EdgeCase:
    """An edge case for testing"""
    id: str
    description: str
    source_wav: str
    start_time: float  # seconds
    end_time: float    # seconds
    expected_lang: str  # SPA, ENG, MAY, NAH, OTH
    actual_lang: str    # What pipeline currently outputs
    issue_type: str     # e.g., "false_positive_may", "oth_spanish_leak"
    segment_text: str   # Original text from SRT
    notes: str = ""
    padding_before: float = 0.5  # Extra context before
    padding_after: float = 0.5   # Extra context after

# Edge case catalog - add cases here
EDGE_CASES = [
    # EC001: English promo tagged as MAY (Episode 6 end)
    EdgeCase(
        id="EC001",
        description="English promo/trailer tagged as Maya due to ejective markers in IPA",
        source_wav="validation_video/Hernán-1-6.wav",
        start_time=2642.639,  # 00:44:02.639
        end_time=2655.514,    # 00:44:15.514
        expected_lang="ENG",
        actual_lang="MAY",
        issue_type="false_positive_may_english",
        segment_text="[LLM] In English: I need you to help me with something.",
        notes="13s segment, clearly English IPA but ejective tʼ markers triggered MAY",
    ),
    # EC002: Spanish "Mentidaske" tagged as OTH (Episode 8)
    EdgeCase(
        id="EC002",
        description="Spanish sentence 'Mentiras que solo repite...' tagged OTH",
        source_wav="validation_video/Hernán-1-8.wav",
        start_time=101.162,  # 00:01:41.162
        end_time=102.850,    # 00:01:42.850
        expected_lang="SPA",
        actual_lang="OTH",
        issue_type="oth_spanish_leak",
        segment_text="[LLM] Mentidaske solorrepite mbhostusanemigos",
        notes="Clear Spanish words mangled together",
    ),
    # EC003: Spanish continuation "ñireproaes" = "ni reproches" (Episode 8)
    EdgeCase(
        id="EC003",
        description="Spanish word continuation split by VAD, tagged OTH",
        source_wav="validation_video/Hernán-1-8.wav",
        start_time=293.740,  # 00:04:53.740
        end_time=294.432,    # 00:04:54.432
        expected_lang="SPA",
        actual_lang="OTH",
        issue_type="oth_spanish_leak_split",
        segment_text="[LLM] ñireproaes",
        notes="Continuation of 'ni reproches' from previous segment",
        padding_before=2.0,  # Need context
    ),
    # EC004: Short segment gibberish (Episode 8)
    EdgeCase(
        id="EC004",
        description="Very short segment produces LLM gibberish",
        source_wav="validation_video/Hernán-1-8.wav",
        start_time=37.290,   # 00:00:37.290
        end_time=37.729,     # 00:00:37.729
        expected_lang="SPA",
        actual_lang="OTH",
        issue_type="short_segment_gibberish",
        segment_text="Short segment",
        notes="0.44s segment, too short for reliable classification",
    ),
    # EC005: Kapitan instead of capitán (Episode 8)
    EdgeCase(
        id="EC005",
        description="German-style orthography 'Kapitan' instead of 'capitán'",
        source_wav="validation_video/Hernán-1-8.wav",
        start_time=50.0,     # approximate - around segment 9
        end_time=52.0,
        expected_lang="SPA",
        actual_lang="SPA",
        issue_type="orthography_error",
        segment_text="Kapitan",
        notes="Whisper outputs German orthography for Spanish word",
    ),
    # EC006: MAY false positive with Spanish articles (Episode 8)
    EdgeCase(
        id="EC006",
        description="MAY tag with Spanish 'El' and 'las' in text",
        source_wav="validation_video/Hernán-1-8.wav",
        start_time=473.0,    # ~00:07:53 segment 120
        end_time=480.0,
        expected_lang="SPA",
        actual_lang="MAY",
        issue_type="false_positive_may_spanish",
        segment_text="El apagarlas fubat'oas mesikasinosere el mos misto",
        notes="Contains Spanish articles, ejective t' is LLM artifact",
    ),
    # EC007: NAH/MAY code-switching cluster (Episode 8, 36:00)
    EdgeCase(
        id="EC007",
        description="Dense MAY/NAH code-switching cluster for validation",
        source_wav="validation_video/Hernán-1-8.wav",
        start_time=2156.0,   # 00:35:56
        end_time=2205.0,     # 00:36:45
        expected_lang="MIX",  # Actually mixed MAY/NAH
        actual_lang="MIX",
        issue_type="code_switching_cluster",
        segment_text="Dense 45s cluster of MAY/NAH switching",
        notes="Scientifically interesting - verify correctness",
        padding_before=2.0,
        padding_after=2.0,
    ),
    # EC008: Spanish with LLM fallback instead of Whisper (Episode 8)
    EdgeCase(
        id="EC008",
        description="SPA segment using LLM instead of Whisper text",
        source_wav="validation_video/Hernán-1-8.wav",
        start_time=48.0,     # around segment 7
        end_time=50.0,
        expected_lang="SPA",
        actual_lang="SPA",
        issue_type="spa_llm_fallback",
        segment_text="[SPA|SPEAKER_10] [LLM] c i b a ɒ s ɪ",
        notes="Whisper suppressed, fell back to LLM IPA output",
    ),
    # EC009: daprime ros laprime n = de primeros, la primera (Episode 8)
    EdgeCase(
        id="EC009",
        description="Spanish toast phrase tagged OTH",
        source_wav="validation_video/Hernán-1-8.wav",
        start_time=70.112,   # 00:01:10.112
        end_time=70.922,     # 00:01:10.922
        expected_lang="SPA",
        actual_lang="OTH",
        issue_type="oth_spanish_leak",
        segment_text="[LLM] daprime ros laprime n",
        notes="'de primeros, la primera' - toast context",
        padding_before=3.0,
    ),
    # EC010: Episode 7 high OTH rate investigation
    EdgeCase(
        id="EC010",
        description="Episode 7 sample - investigate 55% OTH rate",
        source_wav="validation_video/Hernán-1-7.wav",
        start_time=60.0,
        end_time=120.0,
        expected_lang="SPA",
        actual_lang="MIX",
        issue_type="high_oth_rate",
        segment_text="Sample from Ep7 with unusually high OTH",
        notes="39% LLM fallback, 55% OTH - why?",
    ),
]


def extract_wav_clip(case: EdgeCase, output_dir: Path) -> Path:
    """Extract a WAV clip for an edge case using ffmpeg"""
    output_file = output_dir / f"{case.id}.wav"

    start = max(0, case.start_time - case.padding_before)
    duration = (case.end_time - case.start_time) + case.padding_before + case.padding_after

    cmd = [
        "ffmpeg", "-y",
        "-i", case.source_wav,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz for speech models
        "-ac", "1",      # mono
        output_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting {case.id}: {result.stderr}")
        return None

    return output_file


def catalog_cases():
    """Print catalog of all edge cases"""
    print(f"\n{'='*80}")
    print("EDGE CASE CATALOG")
    print(f"{'='*80}\n")

    by_type = {}
    for case in EDGE_CASES:
        by_type.setdefault(case.issue_type, []).append(case)

    for issue_type, cases in by_type.items():
        print(f"\n## {issue_type.upper()} ({len(cases)} cases)")
        print("-" * 60)
        for case in cases:
            duration = case.end_time - case.start_time
            print(f"  {case.id}: {case.description}")
            print(f"       Source: {case.source_wav}")
            print(f"       Time: {case.start_time:.1f}s - {case.end_time:.1f}s ({duration:.1f}s)")
            print(f"       Expected: {case.expected_lang} | Actual: {case.actual_lang}")
            print(f"       Text: {case.segment_text[:60]}...")
            print()

    print(f"\nTotal: {len(EDGE_CASES)} edge cases")


def extract_all_clips():
    """Extract WAV clips for all edge cases"""
    EDGE_CASES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {len(EDGE_CASES)} edge case clips...")

    extracted = []
    for case in EDGE_CASES:
        if not Path(case.source_wav).exists():
            print(f"  SKIP {case.id}: Source not found: {case.source_wav}")
            continue

        output = extract_wav_clip(case, EDGE_CASES_DIR)
        if output:
            duration = case.end_time - case.start_time + case.padding_before + case.padding_after
            print(f"  OK {case.id}: {output.name} ({duration:.1f}s)")
            extracted.append(case)

    # Save catalog
    catalog = {case.id: asdict(case) for case in extracted}
    with open(CATALOG_FILE, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f"\nExtracted {len(extracted)} clips to {EDGE_CASES_DIR}/")
    print(f"Catalog saved to {CATALOG_FILE}")


def run_test(case_id: Optional[str] = None, config: dict = None):
    """
    Run pipeline tests on edge case clips.

    Config options:
        - segment_padding: float (default 0.5)
        - merge_threshold: float (combine segments closer than this)
        - lang_conf_threshold: float (default 0.6)
        - require_ejectives: int (min ejectives for MAY)
        - skip_short: float (skip segments shorter than this)
    """
    if config is None:
        config = {}

    # Load catalog
    if not CATALOG_FILE.exists():
        print("No catalog found. Run 'extract' first.")
        return

    with open(CATALOG_FILE) as f:
        catalog = json.load(f)

    cases_to_test = [case_id] if case_id else list(catalog.keys())

    print(f"\n{'='*80}")
    print(f"TESTING {len(cases_to_test)} EDGE CASES")
    print(f"Config: {config}")
    print(f"{'='*80}\n")

    for cid in cases_to_test:
        if cid not in catalog:
            print(f"Unknown case: {cid}")
            continue

        case_data = catalog[cid]
        wav_file = EDGE_CASES_DIR / f"{cid}.wav"

        if not wav_file.exists():
            print(f"  SKIP {cid}: WAV not found")
            continue

        print(f"\n--- Testing {cid}: {case_data['description'][:50]}... ---")
        print(f"    Expected: {case_data['expected_lang']}")
        print(f"    Currently: {case_data['actual_lang']}")

        # Run mini-pipeline test
        result = run_mini_pipeline(wav_file, config)

        if result:
            print(f"    Result: {result['lang']} (conf: {result.get('conf', 'N/A')})")
            print(f"    IPA: {result.get('ipa', 'N/A')[:60]}...")

            if result['lang'] == case_data['expected_lang']:
                print(f"    ✅ PASS")
            else:
                print(f"    ❌ FAIL (expected {case_data['expected_lang']})")


def run_mini_pipeline(wav_path: Path, config: dict) -> dict:
    """
    Run a minimal version of the pipeline on a single WAV file.
    Returns dict with lang, conf, ipa, text.
    """
    try:
        # Import pipeline components
        from tenepal.phoneme_transcriber import PhonemeTranscriber
        from tenepal.language_id import identify_language

        transcriber = PhonemeTranscriber(backend="allosaurus")

        # Get IPA
        ipa = transcriber.transcribe(str(wav_path))

        # Get language
        lang, score = identify_language(ipa)

        # Compute confidence (simplified)
        conf = min(1.0, score / 3.0) if score > 0 else 0.0

        return {
            "lang": lang.upper(),
            "conf": round(conf, 2),
            "ipa": ipa,
            "score": round(score, 2),
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def run_comparison_tests():
    """
    Run the same edge case with different configurations
    to find optimal settings.
    """
    configs = [
        {"name": "baseline", "lang_conf_threshold": 0.6},
        {"name": "lower_conf", "lang_conf_threshold": 0.5},
        {"name": "higher_conf", "lang_conf_threshold": 0.7},
        {"name": "skip_short", "skip_short": 0.5},
        {"name": "require_3_ejectives", "require_ejectives": 3},
    ]

    print("\n" + "="*80)
    print("COMPARISON TESTS")
    print("="*80)

    for config in configs:
        name = config.pop("name")
        print(f"\n### Configuration: {name}")
        print(f"    Settings: {config}")
        run_test(config=config)


def main():
    parser = argparse.ArgumentParser(description="Edge Case Tester")
    parser.add_argument("command", choices=["catalog", "extract", "test", "compare"],
                       help="Command to run")
    parser.add_argument("--case", help="Specific case ID to test")
    parser.add_argument("--config", help="JSON config string for test")

    args = parser.parse_args()

    if args.command == "catalog":
        catalog_cases()
    elif args.command == "extract":
        extract_all_clips()
    elif args.command == "test":
        config = json.loads(args.config) if args.config else {}
        run_test(args.case, config)
    elif args.command == "compare":
        run_comparison_tests()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download Jonathan Amith Zacatlan/Tepetzintla Nahuatl audio corpus.

Corpus: Zacatlan-Tepetzintla Nahuatl (nhi) — 55 recordings by native speakers
Source: Mozilla Data Collective (CC-BY-ND-4.0 license)
URL: https://datacollective.mozillafoundation.org/datasets/cmlcqxjwl01t8mm07wz7c08bz

Usage:
    python scripts/download_amith_corpus.py --dry-run
    python scripts/download_amith_corpus.py --check
    python scripts/download_amith_corpus.py

The script:
    1. Parses all 55 .trs files to extract audio_filename references
    2. Checks which WAV files exist locally in corpus_audio/amith_nah/
    3. In --dry-run mode: lists expected files and download instructions
    4. In live mode: attempts download of missing files (requires auth token)
    5. Verifies at least 30 audio files present for evaluation use

Output directory: corpus_audio/amith_nah/ (gitignored)
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ─── Paths ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_TRS_DIR = REPO_ROOT / "codices" / "Zacatlan-Tepetzintla-Nahuatl-Transcriptions" / "current"
_DEFAULT_AUDIO_DIR = REPO_ROOT / "corpus_audio" / "amith_nah"

# These are set by main() from CLI args; default to constants above.
TRS_DIR: Path = _DEFAULT_TRS_DIR
AUDIO_DIR: Path = _DEFAULT_AUDIO_DIR

# Mozilla Data Collective dataset page
DATASET_URL = "https://datacollective.mozillafoundation.org/datasets/cmlcqxjwl01t8mm07wz7c08bz"
# License under which this corpus is released
LICENSE = "CC-BY-ND-4.0"
# Language code used for ground truth labeling
GROUND_TRUTH_LANG = "NAH"

# Minimum audio files needed for evaluation
MIN_SAMPLES_REQUIRED = 30

# Supported audio extensions (WAV is primary; MP3/FLAC are fallback)
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac"]


# ─── TRS parsing ────────────────────────────────────────────────────────────

def parse_audio_filename(trs_path: Path) -> str | None:
    """Extract audio_filename attribute from a .trs XML file.

    The attribute lives on the root <Trans> element and specifies the
    base filename (without extension) of the corresponding audio recording.
    """
    try:
        # TRS files may use ISO-8859-1 encoding with non-ASCII chars
        raw = trs_path.read_bytes()
        root = ET.fromstring(raw)
        audio_filename = root.attrib.get("audio_filename", "").strip()
        return audio_filename if audio_filename else None
    except ET.ParseError:
        # Fallback: decode with replacement and retry
        text = trs_path.read_text(encoding="iso-8859-1", errors="replace")
        root = ET.fromstring(text)
        audio_filename = root.attrib.get("audio_filename", "").strip()
        return audio_filename if audio_filename else None
    except Exception:
        return None


def get_section_duration(trs_path: Path) -> float:
    """Extract total audio duration from the Section endTime attribute."""
    try:
        raw = trs_path.read_bytes()
        root = ET.fromstring(raw)
        for section in root.iter("Section"):
            end = section.attrib.get("endTime", "0")
            try:
                return float(end)
            except ValueError:
                pass
    except Exception:
        pass
    return 0.0


def get_speakers(trs_path: Path) -> list[str]:
    """Return speaker IDs from the .trs Speakers block."""
    speakers: list[str] = []
    try:
        raw = trs_path.read_bytes()
        root = ET.fromstring(raw)
        for spk in root.iter("Speaker"):
            spk_id = spk.attrib.get("id", "").strip()
            if spk_id:
                speakers.append(spk_id)
    except Exception:
        pass
    return speakers


# ─── Filename metadata ───────────────────────────────────────────────────────

TOPIC_CODES = {
    "Agric": "Agriculture",
    "Botan": "Botany",
    "Comid": "Food",
    "Creer": "Beliefs",
    "Cuent": "Narrative",
    "MatCl": "Material culture",
    "Medic": "Medicine",
    "Narra": "Life history",
    "Tradi": "Traditions",
    "Zoolo": "Zoology",
}

VILLAGE_CODES = {
    "Omitl": "Omitlan",
    "Tengo": "Tenango",
    "Tenti": "Tentizapa",
    "Tepet": "Tepetzintla",
    "Xmlpa": "Xonamalpan",
    "Xonot": "Xonotla",
    "Xtlax": "Xochitlaxco",
}

_TOPIC_RE = re.compile(r"_(" + "|".join(TOPIC_CODES.keys()) + r")_")
_VILLAGE_RE = re.compile(r"^(" + "|".join(VILLAGE_CODES.keys()) + r")_")


def extract_topic(audio_filename: str) -> str:
    """Extract topic category from audio_filename."""
    m = _TOPIC_RE.search(audio_filename)
    if m:
        code = m.group(1)
        return TOPIC_CODES.get(code, code)
    return "Unknown"


def extract_village(audio_filename: str) -> str:
    """Extract village/location from audio_filename prefix."""
    m = _VILLAGE_RE.match(audio_filename)
    if m:
        code = m.group(1)
        return VILLAGE_CODES.get(code, code)
    return "Unknown"


# ─── Corpus inventory ────────────────────────────────────────────────────────

def build_trs_inventory() -> list[dict]:
    """Parse all .trs files and build inventory of expected audio files.

    Returns list of dicts with: trs_path, audio_filename, expected_wav,
    duration_s, speakers, topic, village.
    """
    if not TRS_DIR.exists():
        print(f"ERROR: TRS directory not found: {TRS_DIR}", file=sys.stderr)
        return []

    trs_files = sorted(TRS_DIR.glob("*.trs"))
    if not trs_files:
        print(f"ERROR: No .trs files found in {TRS_DIR}", file=sys.stderr)
        return []

    inventory: list[dict] = []
    for trs_path in trs_files:
        audio_filename = parse_audio_filename(trs_path)
        if not audio_filename:
            print(f"  WARNING: Could not parse audio_filename from {trs_path.name}")
            continue

        duration_s = get_section_duration(trs_path)
        speakers = get_speakers(trs_path)
        topic = extract_topic(audio_filename)
        village = extract_village(audio_filename)

        # Primary expected WAV path
        expected_wav = AUDIO_DIR / f"{audio_filename}.wav"

        inventory.append({
            "trs_path": str(trs_path),
            "trs_name": trs_path.name,
            "audio_filename": audio_filename,
            "expected_wav": str(expected_wav),
            "duration_s": duration_s,
            "speakers": speakers,
            "topic": topic,
            "village": village,
        })

    return inventory


def find_existing_audio(audio_filename: str) -> Path | None:
    """Check if an audio file exists locally (any supported extension)."""
    for ext in AUDIO_EXTENSIONS:
        candidate = AUDIO_DIR / f"{audio_filename}{ext}"
        if candidate.exists():
            return candidate
    return None


# ─── CLI modes ───────────────────────────────────────────────────────────────

def cmd_dry_run(inventory: list[dict]) -> int:
    """Print download plan and instructions without downloading."""
    print("=" * 72)
    print("Amith Zacatlan/Tepetzintla NAH Corpus — Dry Run")
    print("=" * 72)
    print()
    print(f"TRS directory : {TRS_DIR}")
    print(f"Audio output  : {AUDIO_DIR}")
    print(f"TRS files found: {len(inventory)}")
    print()

    existing: list[dict] = []
    missing: list[dict] = []

    for entry in inventory:
        found = find_existing_audio(entry["audio_filename"])
        if found:
            existing.append({**entry, "local_path": str(found)})
        else:
            missing.append(entry)

    print(f"Audio files present locally : {len(existing)}")
    print(f"Audio files missing locally : {len(missing)}")
    print()

    if existing:
        print("--- Existing audio files ---")
        for e in existing:
            dur = f"{e['duration_s']:.0f}s" if e["duration_s"] else "?"
            print(f"  [OK] {e['audio_filename']}  ({dur}, {e['topic']}, {e['village']})")
        print()

    if missing:
        print("--- Missing audio files ---")
        for m in missing:
            dur = f"{m['duration_s']:.0f}s" if m["duration_s"] else "?"
            print(f"  [--] {m['audio_filename']}.wav  ({dur}, {m['topic']}, {m['village']})")
        print()

    print("=" * 72)
    print("DOWNLOAD INSTRUCTIONS")
    print("=" * 72)
    print()
    print("The Zacatlan-Tepetzintla Nahuatl corpus requires accepting the")
    print(f"CC-BY-ND-4.0 license before downloading audio files.")
    print()
    print("Step 1: Accept the license")
    print(f"  Visit: {DATASET_URL}")
    print("  Click 'I agree to the terms' / accept CC-BY-ND-4.0 license.")
    print()
    print("Step 2: Download audio files")
    print("  The dataset page provides download links or a CLI token.")
    print("  Download all WAV files for the 55 recordings listed above.")
    print()
    print("Step 3: Place files in corpus_audio/amith_nah/")
    print(f"  mkdir -p {AUDIO_DIR}")
    print("  # Copy WAV files:")
    print(f"  cp <downloaded>/*.wav {AUDIO_DIR}/")
    print()
    print("Step 4: Verify at least 30 files present")
    print("  python scripts/download_amith_corpus.py --check")
    print()

    # Check if we have enough for evaluation already
    if len(existing) >= MIN_SAMPLES_REQUIRED:
        print(f"STATUS: {len(existing)} audio files present — READY for evaluation.")
        return 0
    else:
        remaining = MIN_SAMPLES_REQUIRED - len(existing)
        print(f"STATUS: Need {remaining} more audio files to meet minimum ({MIN_SAMPLES_REQUIRED}+).")
        return 1


def cmd_check(inventory: list[dict]) -> int:
    """Check which audio files are present and report readiness."""
    print("=" * 72)
    print("Amith Corpus — Audio File Check")
    print("=" * 72)
    print()

    existing: list[dict] = []
    missing: list[dict] = []

    for entry in inventory:
        found = find_existing_audio(entry["audio_filename"])
        if found:
            existing.append({**entry, "local_path": str(found)})
        else:
            missing.append(entry)

    total_dur = sum(e["duration_s"] for e in existing)

    print(f"TRS files parsed    : {len(inventory)}")
    print(f"Audio files present : {len(existing)}")
    print(f"Audio files missing : {len(missing)}")
    print(f"Total duration      : {total_dur / 60:.1f} minutes")
    print()

    if len(existing) >= MIN_SAMPLES_REQUIRED:
        print(f"READY: {len(existing)} samples >= {MIN_SAMPLES_REQUIRED} minimum threshold.")
        print()
        print("Run corpus indexer to build manifest:")
        print("  python -m tools.corpus.index")
        return 0
    else:
        remaining = MIN_SAMPLES_REQUIRED - len(existing)
        print(f"NOT READY: {remaining} more audio files needed.")
        print()
        print("Run --dry-run for download instructions.")
        return 1


def cmd_download(_inventory: list[dict]) -> int:
    """Attempt automated download (requires environment auth token).

    The Mozilla Data Collective API requires an OAuth token obtained after
    accepting the CC-BY-ND-4.0 license on the dataset page.

    To perform automated download:
        1. Accept license at: {DATASET_URL}
        2. Copy your API token from the dashboard
        3. Set environment variable: export MOZILLA_DC_TOKEN=<token>
        4. Re-run: python scripts/download_amith_corpus.py
    """
    import os

    token = os.environ.get("MOZILLA_DC_TOKEN", "").strip()
    if not token:
        print("ERROR: MOZILLA_DC_TOKEN environment variable not set.")
        print()
        print("The Mozilla Data Collective requires license acceptance before")
        print("automated download. To get your token:")
        print()
        print(f"  1. Visit: {DATASET_URL}")
        print("  2. Accept the CC-BY-ND-4.0 license")
        print("  3. Copy your API token from the dashboard settings")
        print("  4. Run: export MOZILLA_DC_TOKEN=<your-token>")
        print("  5. Re-run this script")
        print()
        print("Alternative: Download files manually and place in corpus_audio/amith_nah/")
        print("Then run: python scripts/download_amith_corpus.py --check")
        return 1

    # With token available — attempt download using Mozilla DC API
    # NOTE: The exact API endpoint and file listing mechanism depends on the
    # Data Collective's REST API version. The pattern below is approximate;
    # consult the dataset page for exact download links when token is obtained.
    try:
        import urllib.request
    except ImportError:
        print("ERROR: urllib.request not available")
        return 1

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    inventory = build_trs_inventory()
    missing = [
        e for e in inventory
        if find_existing_audio(e["audio_filename"]) is None
    ]

    if not missing:
        print("All audio files already present. Nothing to download.")
        return 0

    print(f"Attempting download of {len(missing)} missing audio files...")
    print()

    success_count = 0
    for entry in missing:
        audio_filename = entry["audio_filename"]
        target_path = AUDIO_DIR / f"{audio_filename}.wav"

        # Mozilla DC file URL pattern (dataset-specific; adjust when token obtained)
        file_url = (
            f"https://datacollective.mozillafoundation.org/api/datasets/"
            f"cmlcqxjwl01t8mm07wz7c08bz/files/{audio_filename}.wav"
        )

        req = urllib.request.Request(
            file_url,
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            print(f"  Downloading {audio_filename}.wav ...")
            with urllib.request.urlopen(req) as response:
                target_path.write_bytes(response.read())
            print(f"  [OK] Saved to {target_path}")
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] {audio_filename}.wav — {e}")

    print()
    print(f"Downloaded {success_count}/{len(missing)} files.")

    if success_count + (len(inventory) - len(missing)) >= MIN_SAMPLES_REQUIRED:
        print(f"READY: sufficient samples for evaluation.")
        return 0
    else:
        print(f"Insufficient samples. Run --dry-run for manual download instructions.")
        return 1


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download/check Jonathan Amith NAH corpus audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_amith_corpus.py --dry-run    # Show plan + instructions
  python scripts/download_amith_corpus.py --check      # Check what's present
  python scripts/download_amith_corpus.py              # Attempt download (needs MOZILLA_DC_TOKEN)
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show expected files and download instructions; do not download",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which audio files are present and report readiness",
    )
    parser.add_argument(
        "--trs-dir",
        default=str(_DEFAULT_TRS_DIR),
        help=f"Directory containing .trs files (default: {_DEFAULT_TRS_DIR})",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(_DEFAULT_AUDIO_DIR),
        help=f"Output directory for audio files (default: {_DEFAULT_AUDIO_DIR})",
    )
    args = parser.parse_args()

    # Allow overriding paths via CLI (for testing)
    global TRS_DIR, AUDIO_DIR  # noqa: PLW0603 — module-level path override
    TRS_DIR = Path(args.trs_dir)
    AUDIO_DIR = Path(args.audio_dir)

    print()
    inventory = build_trs_inventory()
    if not inventory:
        print("ERROR: Could not build TRS inventory. Check TRS directory.")
        return 2

    print(f"Parsed {len(inventory)} TRS files from {TRS_DIR}")
    print()

    if args.dry_run:
        return cmd_dry_run(inventory)
    elif args.check:
        return cmd_check(inventory)
    else:
        return cmd_download(inventory)


if __name__ == "__main__":
    sys.exit(main())

"""Corpus indexer: links Amith Zacatlan/Tepetzintla TRS files to audio.

Builds a manifest.json with ground truth labels for NAH evaluation.

Usage (module):
    from tools.corpus.index import build_corpus_index, CorpusSample

Usage (CLI):
    python -m tools.corpus.index [--trs-dir ...] [--audio-dir ...] [--output ...]
    python -m tools.corpus.index --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_TRS_DIR = (
    REPO_ROOT
    / "codices"
    / "Zacatlan-Tepetzintla-Nahuatl-Transcriptions"
    / "current"
)
_DEFAULT_AUDIO_DIR = REPO_ROOT / "corpus_audio" / "amith_nah"
_DEFAULT_OUTPUT = REPO_ROOT / "tools" / "corpus" / "manifest.json"

# Supported audio extensions searched in priority order
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac"]

# Ground truth language label — this corpus is monolingual Nahuatl
GROUND_TRUTH_LANG = "NAH"

# Manifest version
MANIFEST_VERSION = "1.0"
CORPUS_ID = "amith-zacatlan-nah"
LICENSE = "CC-BY-ND-4.0"


# ─── Metadata helpers ──────────────────────────────────────────────────────────

TOPIC_CODES: dict[str, str] = {
    "Agric": "Agric",
    "Botan": "Botan",
    "Comid": "Comid",
    "Creer": "Creer",
    "Cuent": "Cuent",
    "MatCl": "MatCl",
    "Medic": "Medic",
    "Narra": "Narra",
    "Tradi": "Tradi",
    "Zoolo": "Zoolo",
}

VILLAGE_CODES: dict[str, str] = {
    "Omitl": "Omitlan",
    "Tengo": "Tenango",
    "Tenti": "Tentizapa",
    "Tepet": "Tepetzintla",
    "Xmlpa": "Xonamalpan",
    "Xonot": "Xonotla",
    "Xtlax": "Xochitlaxco",
}

_TOPIC_RE = re.compile(r"_(" + "|".join(re.escape(k) for k in TOPIC_CODES) + r")_")
_VILLAGE_RE = re.compile(r"(?:^|_)(" + "|".join(re.escape(k) for k in VILLAGE_CODES) + r")_")

# Speaker code pattern — alphanumeric codes like OPH507, VPR522, CSC370, AMH529
# Underscore is a word char so \b doesn't delimit; use look-around instead.
_SPEAKER_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,4}\d{3,4})(?![A-Za-z])")


def extract_topic(audio_filename: str) -> str:
    """Extract topic category code from audio_filename."""
    m = _TOPIC_RE.search(audio_filename)
    return m.group(1) if m else "Unknown"


def extract_village(audio_filename: str) -> str:
    """Extract village name from audio_filename prefix."""
    m = _VILLAGE_RE.search(audio_filename)
    return VILLAGE_CODES.get(m.group(1), m.group(1)) if m else "Unknown"


def extract_speakers_from_filename(audio_filename: str) -> list[str]:
    """Extract speaker codes (e.g. OPH507, VPR522) from audio_filename."""
    return _SPEAKER_RE.findall(audio_filename)


# ─── TRS parsing ───────────────────────────────────────────────────────────────

def _parse_xml_robust(path: Path) -> ET.Element:
    """Parse a TRS XML file, falling back to ISO-8859-1 if raw bytes fail."""
    try:
        return ET.fromstring(path.read_bytes())
    except ET.ParseError:
        text = path.read_text(encoding="iso-8859-1", errors="replace")
        return ET.fromstring(text)


def parse_trs_metadata(path: Path) -> dict:
    """Extract metadata from a .trs file.

    Returns dict with:
        audio_filename, duration_s, speaker_ids, segment_count
    """
    root = _parse_xml_robust(path)

    audio_filename = root.attrib.get("audio_filename", "").strip()

    # Duration from Section endTime
    duration_s = 0.0
    for section in root.iter("Section"):
        try:
            end = float(section.attrib.get("endTime", "0") or "0")
            if end > duration_s:
                duration_s = end
        except (ValueError, TypeError):
            pass

    # Speaker IDs from Speakers block
    speaker_ids: list[str] = []
    for spk in root.iter("Speaker"):
        spk_id = spk.attrib.get("id", "").strip()
        if spk_id:
            speaker_ids.append(spk_id)

    # Segment count (Turn elements — each contains 1+ sync segments)
    segment_count = sum(1 for _ in root.iter("Turn"))

    return {
        "audio_filename": audio_filename,
        "duration_s": round(duration_s, 3),
        "speaker_ids": speaker_ids,
        "segment_count": segment_count,
    }


def find_audio_file(audio_filename: str, audio_dir: Path) -> Path | None:
    """Find an audio file in audio_dir matching any supported extension."""
    for ext in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{audio_filename}{ext}"
        if candidate.exists():
            return candidate
    return None


# ─── CorpusSample dataclass ────────────────────────────────────────────────────

@dataclass
class CorpusSample:
    """One indexed audio recording with ground truth label.

    Attributes:
        id              Unique identifier derived from TRS filename
        audio_path      Path to audio file (relative to repo root, or absolute)
        transcript_path Path to .trs transcript file
        ground_truth_lang Language code for evaluation; always NAH for this corpus
        duration_s      Recording duration in seconds
        speaker_id      Primary speaker code from filename (e.g. OPH507)
        topic           Topic category code (Agric, Comid, Creer, etc.)
        segment_count   Number of Turn elements in TRS (proxy for utterance count)
        village         Village/community where recording was made
        audio_available Whether the audio file exists locally
    """

    id: str
    audio_path: str
    transcript_path: str
    ground_truth_lang: str
    duration_s: float
    speaker_id: str
    topic: str
    segment_count: int
    village: str = "Unknown"
    audio_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Index builder ─────────────────────────────────────────────────────────────

def build_corpus_index(
    trs_dir: Path,
    audio_dir: Path,
    output_path: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Build corpus manifest from TRS files and audio directory.

    Args:
        trs_dir:     Directory containing .trs transcript files
        audio_dir:   Directory containing downloaded WAV/MP3 audio files
        output_path: Where to write manifest.json
        dry_run:     If True, compute manifest but do not write to disk

    Returns:
        Manifest dict with 'samples' list and 'stats' summary.
    """
    if not trs_dir.exists():
        raise FileNotFoundError(f"TRS directory not found: {trs_dir}")

    trs_files = sorted(trs_dir.glob("*.trs"))
    if not trs_files:
        raise ValueError(f"No .trs files found in {trs_dir}")

    samples: list[CorpusSample] = []
    all_speakers: set[str] = set()
    all_topics: set[str] = set()
    total_duration = 0.0

    for trs_path in trs_files:
        try:
            meta = parse_trs_metadata(trs_path)
        except Exception as exc:
            print(f"  WARNING: Failed to parse {trs_path.name}: {exc}", file=sys.stderr)
            continue

        audio_filename = meta["audio_filename"]
        if not audio_filename:
            # Fallback: derive audio filename from TRS filename
            # TRS naming convention: {audio_base}_ed-YYYY-MM-DD.trs
            stem = trs_path.stem
            audio_filename = re.sub(r"_ed[_-]\d{4}-\d{2}-\d{2}$", "", stem)
            audio_filename = re.sub(r"_ed-\d{6}$", "", audio_filename)

        # Determine audio file presence
        audio_path_obj = find_audio_file(audio_filename, audio_dir)
        audio_available = audio_path_obj is not None

        if audio_available:
            audio_path_str = str(audio_path_obj)
        else:
            # Record expected path even if file not yet downloaded
            audio_path_str = str(audio_dir / f"{audio_filename}.wav")

        # Extract metadata from filename conventions
        topic = extract_topic(audio_filename)
        village = extract_village(audio_filename)
        filename_speakers = extract_speakers_from_filename(audio_filename)

        # Primary speaker: first code found in filename, or "UNKNOWN"
        speaker_id = filename_speakers[0] if filename_speakers else "UNKNOWN"

        # Build unique sample ID
        sample_id = audio_filename.replace(" ", "_")

        # Accumulate stats
        all_speakers.update(filename_speakers)
        all_topics.add(topic)
        total_duration += meta["duration_s"]

        sample = CorpusSample(
            id=sample_id,
            audio_path=audio_path_str,
            transcript_path=str(trs_path),
            ground_truth_lang=GROUND_TRUTH_LANG,
            duration_s=meta["duration_s"],
            speaker_id=speaker_id,
            topic=topic,
            segment_count=meta["segment_count"],
            village=village,
            audio_available=audio_available,
        )
        samples.append(sample)

    available_samples = [s for s in samples if s.audio_available]

    manifest = {
        "version": MANIFEST_VERSION,
        "corpus": CORPUS_ID,
        "license": LICENSE,
        "samples": [s.to_dict() for s in samples],
        "stats": {
            "total_samples": len(samples),
            "available_samples": len(available_samples),
            "total_duration_s": round(total_duration, 1),
            "total_duration_min": round(total_duration / 60, 1),
            "speakers": sorted(all_speakers),
            "topics": sorted(all_topics),
            "ground_truth_lang": GROUND_TRUTH_LANG,
            "corpus_note": (
                "Zacatlan-Ahuacatlan-Tepetzintla Nahuatl (nhi) — "
                "monolingual corpus recorded in Puebla, Mexico. "
                "All speakers are native Nahuatl speakers. "
                "Ground truth: NAH for all samples."
            ),
        },
    }

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return manifest


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build corpus manifest from Amith Zacatlan/Tepetzintla TRS files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.corpus.index --dry-run
  python -m tools.corpus.index
  python -m tools.corpus.index --trs-dir codices/.../current --audio-dir corpus_audio/amith_nah
        """,
    )
    parser.add_argument(
        "--trs-dir",
        default=str(_DEFAULT_TRS_DIR),
        help=f"Directory with .trs files (default: {_DEFAULT_TRS_DIR})",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(_DEFAULT_AUDIO_DIR),
        help=f"Directory with audio files (default: {_DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output manifest path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute manifest but do not write to disk; print summary",
    )
    args = parser.parse_args()

    trs_dir = Path(args.trs_dir)
    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output)

    print()
    print("Building corpus index...")
    print(f"  TRS dir   : {trs_dir}")
    print(f"  Audio dir : {audio_dir}")
    print(f"  Output    : {output_path}")
    print(f"  Dry run   : {args.dry_run}")
    print()

    try:
        manifest = build_corpus_index(
            trs_dir, audio_dir, output_path, dry_run=args.dry_run
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    stats = manifest["stats"]
    print(f"Total samples       : {stats['total_samples']}")
    print(f"Available audio     : {stats['available_samples']}")
    print(f"Total duration      : {stats['total_duration_min']} minutes")
    print(f"Ground truth lang   : {stats['ground_truth_lang']}")
    print(f"Speakers found      : {len(stats['speakers'])}")
    print(f"Topics found        : {len(stats['topics'])}")
    print(f"Topics              : {', '.join(stats['topics'])}")
    print()

    if not args.dry_run:
        print(f"Manifest written to : {output_path}")
    else:
        print("Dry run complete — manifest NOT written to disk.")
    print()

    # Report first few samples for verification
    print("Sample entries:")
    for s in manifest["samples"][:3]:
        avail = "[OK]" if s["audio_available"] else "[--]"
        print(
            f"  {avail} {s['id'][:50]} | {s['topic']} | "
            f"{s['duration_s']:.0f}s | {s['ground_truth_lang']}"
        )
    if len(manifest["samples"]) > 3:
        print(f"  ... ({len(manifest['samples']) - 3} more)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

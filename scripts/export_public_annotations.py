#!/usr/bin/env python3
"""Export public-safe annotation artifacts from the local annotator SQLite DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path


SAFE_FIELDS = [
    "media_file",
    "cue_index",
    "segment_id",
    "start_s",
    "end_s",
    "start_time_ms",
    "end_time_ms",
    "pipeline_lang",
    "pipeline_text",
    "pipeline_ipa",
    "pipeline_confidence",
    "correct_lang",
    "correct_speaker",
    "human_transcription",
    "boundary_suspect",
    "start_adjust_ms",
    "end_adjust_ms",
    "boundary_note",
    "overlap",
    "secondary_speaker",
    "split_into",
    "split_from",
    "annotated_at",
]


def slugify_media_name(name: str) -> str:
    stem = unicodedata.normalize("NFKD", Path(name).stem.lower())
    stem = "".join(ch for ch in stem if not unicodedata.combining(ch))
    chars = []
    for ch in stem:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    ext = Path(name).suffix.lower().lstrip(".")
    base = slug.strip("_") or "unknown_media"
    return f"{base}_{ext}" if ext else base


def export_annotations(db_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    query = f"""
        SELECT {", ".join(SAFE_FIELDS)}
        FROM annotations
        WHERE annotated_at IS NOT NULL
        ORDER BY media_file, cue_index, start_time_ms, end_time_ms
    """
    rows = con.execute(query).fetchall()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["media_file"]].append({key: row[key] for key in SAFE_FIELDS})

    manifest = {
        "source_db": "tools/annotator/annotations.db",
        "media_exports": [],
        "notes": [
            "This export excludes freeform annotator notes and local SQLite metadata.",
            "Character slugs are preserved; media files themselves are not redistributed.",
            "Use these JSONL files as the public audit trail instead of shipping annotations.db.",
        ],
    }

    for media_file, media_rows in sorted(grouped.items()):
        out_name = f"{slugify_media_name(media_file)}.jsonl"
        out_path = out_dir / out_name
        with out_path.open("w", encoding="utf-8") as handle:
            for row in media_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        lang_counts: dict[str, int] = defaultdict(int)
        speaker_counts: dict[str, int] = defaultdict(int)
        for row in media_rows:
            lang = row.get("correct_lang") or "UNK"
            speaker = row.get("correct_speaker") or ""
            lang_counts[lang] += 1
            if speaker:
                speaker_counts[speaker] += 1

        manifest["media_exports"].append(
            {
                "media_file": media_file,
                "output": out_name,
                "rows": len(media_rows),
                "languages": dict(sorted(lang_counts.items())),
                "speakers": dict(sorted(speaker_counts.items())),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="tools/annotator/annotations.db",
        help="Path to annotator SQLite database",
    )
    parser.add_argument(
        "--outdir",
        default="benchmarks/annotations",
        help="Directory for public JSONL annotation exports",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    out_dir = Path(args.outdir).resolve()
    export_annotations(db_path, out_dir)
    print(f"Exported public annotations to {out_dir}")


if __name__ == "__main__":
    main()

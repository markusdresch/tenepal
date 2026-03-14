#!/usr/bin/env python3
"""Migrate manual annotations to a new segmentation using timestamp overlap."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Segment:
    cue_index: int
    start_s: float
    end_s: float
    speaker_id: str | None = None
    pipeline_lang: str | None = None
    pipeline_text: str | None = None
    pipeline_ipa: str | None = None
    pipeline_confidence: float | None = None


def parse_srt_time(time_str: str) -> float:
    h, m, s = time_str.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(srt_path: Path) -> list[Segment]:
    content = srt_path.read_text(encoding="utf-8", errors="replace").strip()
    blocks = re.split(r"\n\n+", content)
    out: list[Segment] = []
    for block in blocks:
        lines = [x.rstrip() for x in block.splitlines() if x.strip() != ""]
        if len(lines) < 2:
            continue
        try:
            cue_idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1],
        )
        if not m:
            continue
        start_s = parse_srt_time(m.group(1))
        end_s = parse_srt_time(m.group(2))
        raw = "\n".join(lines[2:]) if len(lines) > 2 else ""
        first_line = lines[2].strip() if len(lines) > 2 else ""
        pipeline_lang = None
        speaker_id = None
        pipeline_text = raw.strip()

        tag = re.match(r"^\[([^\]]+)\]\s*(.*)$", first_line)
        if tag:
            parts = [p.strip() for p in tag.group(1).split("|") if p.strip()]
            if parts and re.fullmatch(r"[A-Z]{3}", parts[0]):
                pipeline_lang = parts[0]
            if len(parts) >= 2 and re.fullmatch(r"SPEAKER_\d+", parts[1]):
                speaker_id = parts[1]
            if speaker_id is None:
                m_spk = re.search(r"SPEAKER_\d+", tag.group(1))
                speaker_id = m_spk.group(0) if m_spk else None
            # Keep only content text (without tag) on first line
            rest_lines = [tag.group(2)] + lines[3:]
            pipeline_text = "\n".join([x.rstrip() for x in rest_lines if x is not None]).strip()
        else:
            m_lang = re.match(r"^\[([A-Z]{3})", first_line)
            pipeline_lang = m_lang.group(1) if m_lang else None
            m_spk = re.search(r"SPEAKER_\d+", first_line)
            speaker_id = m_spk.group(0) if m_spk else None

        ipa_match = re.search(r"♫fused:\s*([^\n]+)", raw)
        conf_match = re.search(r"conf[:\s]+([0-9.]+)", raw, re.IGNORECASE)
        out.append(
            Segment(
                cue_index=cue_idx,
                start_s=start_s,
                end_s=end_s,
                speaker_id=speaker_id,
                pipeline_lang=pipeline_lang,
                pipeline_text=pipeline_text or None,
                pipeline_ipa=ipa_match.group(1).strip() if ipa_match else None,
                pipeline_confidence=float(conf_match.group(1)) if conf_match else None,
            )
        )
    return out


def overlap_duration(
    seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float
) -> float:
    overlap_start = max(seg1_start, seg2_start)
    overlap_end = min(seg1_end, seg2_end)
    if overlap_end <= overlap_start:
        return 0.0
    return overlap_end - overlap_start


def overlap_ratio(
    seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float
) -> float:
    """Overlap ratio relative to seg1 duration."""
    dur = seg1_end - seg1_start
    if dur <= 0:
        return 0.0
    return overlap_duration(seg1_start, seg1_end, seg2_start, seg2_end) / dur


def pick_media_name(conn: sqlite3.Connection, srt_stem: str) -> str:
    row = conn.execute(
        """
        SELECT media_file, COUNT(*) AS n
        FROM annotations
        WHERE media_file LIKE ?
        GROUP BY media_file
        ORDER BY n DESC, media_file
        LIMIT 1
        """,
        (f"%{srt_stem}%",),
    ).fetchone()
    if row:
        return str(row[0])
    return f"{srt_stem}.mp4"


def load_old_annotations(conn: sqlite3.Connection, media_file: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM annotations
        WHERE media_file = ?
          AND annotated_at IS NOT NULL
          AND cue_index > 0
        ORDER BY start_s, end_s
        """,
        (media_file,),
    ).fetchall()


def extract_speaker_from_text(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"SPEAKER_\d+", text)
    return m.group(0) if m else None


def total_duration(ranges: list[tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in ranges)


def calculate_total_overlap(
    ranges_a: list[tuple[float, float]], ranges_b: list[tuple[float, float]]
) -> float:
    total = 0.0
    for a0, a1 in ranges_a:
        for b0, b1 in ranges_b:
            total += overlap_duration(a0, a1, b0, b1)
    return total


def match_speakers(
    old_rows: list[sqlite3.Row],
    new_segments: list[Segment],
    speaker_threshold: float = 0.5,
) -> tuple[dict[str, str], list[dict]]:
    old_speaker_times: dict[str, list[tuple[float, float]]] = defaultdict(list)
    new_speaker_times: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for ann in old_rows:
        old_spk = ann["correct_speaker"] or extract_speaker_from_text(ann["pipeline_text"])
        if old_spk:
            old_speaker_times[old_spk].append((float(ann["start_s"]), float(ann["end_s"])))

    for seg in new_segments:
        if seg.speaker_id:
            new_speaker_times[seg.speaker_id].append((seg.start_s, seg.end_s))

    candidates: list[dict] = []
    for old_spk, old_times in old_speaker_times.items():
        old_dur = total_duration(old_times)
        if old_dur <= 0:
            continue
        for new_spk, new_times in new_speaker_times.items():
            ov = calculate_total_overlap(old_times, new_times)
            ratio_old = ov / old_dur if old_dur > 0 else 0.0
            candidates.append(
                {
                    "old_speaker": old_spk,
                    "new_speaker": new_spk,
                    "overlap_s": ov,
                    "old_duration_s": old_dur,
                    "overlap_ratio_old": ratio_old,
                }
            )

    candidates.sort(key=lambda x: (x["overlap_s"], x["overlap_ratio_old"]), reverse=True)

    mapping: dict[str, str] = {}
    used_new: set[str] = set()
    for row in candidates:
        old_spk = row["old_speaker"]
        new_spk = row["new_speaker"]
        if old_spk in mapping or new_spk in used_new:
            continue
        if row["overlap_ratio_old"] < speaker_threshold:
            continue
        mapping[old_spk] = new_spk
        used_new.add(new_spk)

    return mapping, candidates


def migrate_speaker_metadata(
    conn: sqlite3.Connection,
    media_file: str,
    episode: str,
    speaker_mapping: dict[str, str],
) -> tuple[int, int, int]:
    # 1) Update annotation speaker references.
    ann_updates = 0
    for old_spk, new_spk in speaker_mapping.items():
        if old_spk == new_spk:
            continue
        cur = conn.execute(
            """
            UPDATE annotations
            SET correct_speaker = ?
            WHERE media_file = ? AND correct_speaker = ?
            """,
            (new_spk, media_file, old_spk),
        )
        ann_updates += cur.rowcount
        # Keep overlap metadata consistent if present.
        conn.execute(
            """
            UPDATE annotations
            SET secondary_speaker = ?
            WHERE media_file = ? AND secondary_speaker = ?
            """,
            (new_spk, media_file, old_spk),
        )

    # 2) Migrate speaker_names rows to new IDs.
    names_upserts = 0
    names_deleted = 0
    for old_spk, new_spk in speaker_mapping.items():
        if old_spk == new_spk:
            continue
        old_row = conn.execute(
            """
            SELECT display_name, notes
            FROM speaker_names
            WHERE episode = ? AND speaker_id = ?
            """,
            (episode, old_spk),
        ).fetchone()
        if not old_row:
            continue
        existing_new = conn.execute(
            """
            SELECT display_name, notes
            FROM speaker_names
            WHERE episode = ? AND speaker_id = ?
            """,
            (episode, new_spk),
        ).fetchone()
        new_display = old_row["display_name"] or ""
        new_notes = old_row["notes"] or ""
        if existing_new:
            # Preserve explicitly set target values; fill blanks from old.
            if existing_new["display_name"]:
                new_display = existing_new["display_name"]
            if existing_new["notes"]:
                new_notes = existing_new["notes"]

        conn.execute(
            """
            INSERT INTO speaker_names (episode, speaker_id, display_name, notes, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(episode, speaker_id) DO UPDATE SET
                display_name=excluded.display_name,
                notes=excluded.notes,
                updated_at=excluded.updated_at
            """,
            (episode, new_spk, new_display, new_notes, datetime.now(timezone.utc).isoformat()),
        )
        names_upserts += 1
        cur = conn.execute(
            "DELETE FROM speaker_names WHERE episode = ? AND speaker_id = ?",
            (episode, old_spk),
        )
        names_deleted += cur.rowcount

    return ann_updates, names_upserts, names_deleted


def migrate_annotations(
    old_db: Path,
    new_srt: Path,
    output: Path,
    threshold: float = 0.5,
    speaker_threshold: float = 0.5,
) -> None:
    if not old_db.exists():
        raise FileNotFoundError(f"old DB not found: {old_db}")
    if not new_srt.exists():
        raise FileNotFoundError(f"new SRT not found: {new_srt}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() != old_db.resolve():
        shutil.copy2(old_db, output)

    conn = sqlite3.connect(output)
    conn.row_factory = sqlite3.Row

    srt_stem = new_srt.stem
    media_file = pick_media_name(conn, srt_stem)
    new_segments = parse_srt(new_srt)
    old_rows = load_old_annotations(conn, media_file)
    speaker_mapping, speaker_candidates = match_speakers(old_rows, new_segments, speaker_threshold=speaker_threshold)

    migrated = 0
    unmatched = 0
    kept_empty = 0

    now = datetime.now(timezone.utc).isoformat(sep=" ")
    for seg in new_segments:
        best = None
        best_ratio = 0.0
        best_overlap = 0.0
        for old in old_rows:
            ov = overlap_duration(seg.start_s, seg.end_s, old["start_s"], old["end_s"])
            if ov <= 0:
                continue
            ratio = overlap_ratio(seg.start_s, seg.end_s, old["start_s"], old["end_s"])
            if ratio > best_ratio or (ratio == best_ratio and ov > best_overlap):
                best_ratio = ratio
                best_overlap = ov
                best = old

        segment_id = f"{media_file.replace('.', '_')}_{seg.cue_index:04d}"
        fields = {
            "media_file": media_file,
            "cue_index": seg.cue_index,
            "segment_id": segment_id,
            "start_s": seg.start_s,
            "end_s": seg.end_s,
            "pipeline_lang": seg.pipeline_lang,
            "pipeline_text": seg.pipeline_text,
            "pipeline_ipa": seg.pipeline_ipa,
            "pipeline_confidence": seg.pipeline_confidence,
            "correct_lang": None,
            "correct_speaker": None,
            "notes": None,
            "annotated_at": None,
            "annotator": "migration",
        }

        if best and best_ratio >= threshold:
            fields["correct_lang"] = best["correct_lang"]
            fields["correct_speaker"] = best["correct_speaker"]
            fields["notes"] = best["notes"]
            fields["annotated_at"] = best["annotated_at"] or now
            fields["annotator"] = best["annotator"] or "migration"
            migrated += 1
        else:
            unmatched += 1

        # Keep pipeline-confirmed labels as null corrections to match annotator semantics.
        if fields["correct_lang"] is None:
            kept_empty += 1

        conn.execute(
            """
            INSERT INTO annotations (
                media_file, cue_index, segment_id, start_s, end_s,
                pipeline_lang, pipeline_text, pipeline_ipa, pipeline_confidence,
                correct_lang, correct_speaker, notes, annotated_at, annotator
            )
            VALUES (
                :media_file, :cue_index, :segment_id, :start_s, :end_s,
                :pipeline_lang, :pipeline_text, :pipeline_ipa, :pipeline_confidence,
                :correct_lang, :correct_speaker, :notes, :annotated_at, :annotator
            )
            ON CONFLICT(media_file, cue_index) DO UPDATE SET
                segment_id=excluded.segment_id,
                start_s=excluded.start_s,
                end_s=excluded.end_s,
                pipeline_lang=excluded.pipeline_lang,
                pipeline_text=excluded.pipeline_text,
                pipeline_ipa=excluded.pipeline_ipa,
                pipeline_confidence=excluded.pipeline_confidence,
                correct_lang=excluded.correct_lang,
                correct_speaker=excluded.correct_speaker,
                notes=excluded.notes,
                annotated_at=excluded.annotated_at,
                annotator=excluded.annotator
            """,
            fields,
        )

    conn.commit()

    episode = new_srt.stem
    ann_updates, names_upserts, names_deleted = migrate_speaker_metadata(
        conn=conn,
        media_file=media_file,
        episode=episode,
        speaker_mapping=speaker_mapping,
    )
    conn.commit()
    conn.close()

    print(f"Old DB: {old_db}")
    print(f"New SRT: {new_srt}")
    print(f"Output DB: {output}")
    print(f"Media key: {media_file}")
    print(f"New segments parsed: {len(new_segments)}")
    print(f"Migrated (overlap >= {threshold:.2f}): {migrated}")
    print(f"Unmatched: {unmatched}")
    print(f"Rows with null correct_lang after migration: {kept_empty}")
    print(f"Speaker mapping threshold: {speaker_threshold:.2f}")
    print(f"Speaker mappings found: {len(speaker_mapping)}")
    for old_spk, new_spk in sorted(speaker_mapping.items()):
        print(f"SPEAKER_MAP {old_spk} -> {new_spk}")
    print(f"Speaker metadata: annotation updates={ann_updates}, speaker_names upserts={names_upserts}, deletes={names_deleted}")
    print("Speaker mapping JSON:")
    print(json.dumps(speaker_mapping, ensure_ascii=False, indent=2))

    if speaker_candidates:
        print("Top speaker overlap candidates:")
        for row in speaker_candidates[:20]:
            print(
                f"CAND {row['old_speaker']} -> {row['new_speaker']} "
                f"overlap_s={row['overlap_s']:.3f} ratio_old={row['overlap_ratio_old']:.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate annotations to new segmentation by timestamp overlap")
    ap.add_argument("--old-db", required=True, type=Path)
    ap.add_argument("--new-srt", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--speaker-threshold", type=float, default=0.5)
    args = ap.parse_args()
    migrate_annotations(
        args.old_db,
        args.new_srt,
        args.output,
        threshold=args.threshold,
        speaker_threshold=args.speaker_threshold,
    )


if __name__ == "__main__":
    main()

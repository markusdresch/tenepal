#!/usr/bin/env python3
"""Extract Nahuatl marker lexicon from Zacatlan/Tepetzintla .trs files.

Outputs:
- segments_clean.jsonl: time-aligned utterances with inline loanwords extracted
- markers_nahuatl.tsv: Nahuatl token counts (keeps vowel length ':')
- markers_whisper_proxy.tsv: same tokens normalized for Whisper-like text (':' removed)
- loanwords_es.tsv: Spanish loanword candidates found in *...*/**...** markup
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’:-]+")
STARRED_RE = re.compile(r"\*{1,2}\s*([^*]+?)\s*\*{1,2}")
WS_RE = re.compile(r"\s+")
NAH_CUES = ("tl", "tz", "ts", "kw", "w", "x", "k")
SPANISH_STOPWORDS = {
    "de", "la", "el", "los", "las", "y", "que", "en", "un", "una", "por", "para",
    "con", "se", "es", "del", "al", "como", "más", "mas", "su", "sus", "ya", "pero",
    "si", "sí", "no", "le", "lo", "a", "o", "u", "yo", "tu", "tú", "mi", "me", "te",
    "nos", "vos", "fue", "era", "son", "ser", "ha", "han", "porque", "cuando",
    "donde", "entonces", "este", "igual", "hasta",
}


def clean_ws(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def normalize_token(token: str) -> str:
    token = token.lower().strip(".,;:!?¡¿()[]{}\"“”'`´")
    return token


def normalize_marker_token(token: str) -> str:
    token = normalize_token(token)
    # Keep internal vowel-length marker from the corpus for marker mining.
    return token


def normalize_proxy_token(token: str) -> str:
    # Whisper usually drops vowel-length marks, so build a proxy lexicon.
    token = token.replace(":", "")
    token = token.replace("’", "'")
    return normalize_token(token)


def iter_turn_segments(turn: ET.Element) -> list[dict]:
    turn_start = float(turn.attrib.get("startTime", "0") or 0.0)
    turn_end = float(turn.attrib.get("endTime", "0") or 0.0)
    speaker = turn.attrib.get("speaker", "").strip()

    segments: list[dict] = []
    current_start = turn_start
    parts: list[str] = []

    if turn.text and turn.text.strip():
        parts.append(turn.text)

    for child in list(turn):
        tag = child.tag
        if tag == "Sync":
            sync_time = float(child.attrib.get("time", current_start) or current_start)
            text = clean_ws(" ".join(parts))
            if text and sync_time >= current_start:
                segments.append(
                    {
                        "start": round(current_start, 3),
                        "end": round(sync_time, 3),
                        "speaker": speaker,
                        "text": text,
                    }
                )
            current_start = sync_time
            parts = []
            if child.tail and child.tail.strip():
                parts.append(child.tail)
            continue

        # Ignore annotation payload but keep tail text (actual transcript continuation).
        if child.tail and child.tail.strip():
            parts.append(child.tail)

    final_text = clean_ws(" ".join(parts))
    if final_text and turn_end >= current_start:
        segments.append(
            {
                "start": round(current_start, 3),
                "end": round(turn_end, 3),
                "speaker": speaker,
                "text": final_text,
            }
        )
    return segments


def parse_trs(path: Path) -> list[dict]:
    root = ET.fromstring(path.read_bytes())
    out: list[dict] = []
    for turn in root.iter("Turn"):
        out.extend(iter_turn_segments(turn))
    return out


def extract_loanwords(text: str) -> list[str]:
    raw = [clean_ws(m.group(1).strip()) for m in STARRED_RE.finditer(text)]
    return [x for x in raw if x]


def remove_starred(text: str) -> str:
    # Keep loanword content out of Nahuatl marker extraction.
    return clean_ws(STARRED_RE.sub(" ", text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract marker lexicon from Zacatlan .trs corpus.")
    parser.add_argument(
        "--in-dir",
        default="codices/Zacatlan-Tepetzintla-Nahuatl-Transcriptions/current",
        help="Directory with .trs files",
    )
    parser.add_argument(
        "--out-dir",
        default="codices/extracted/zacatlan_markers",
        help="Output directory",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_counts: Counter[str] = Counter()
    marker_proxy_counts: Counter[str] = Counter()
    loanword_counts: Counter[str] = Counter()
    proxy_to_colon: dict[str, Counter[str]] = defaultdict(Counter)
    marker_files: dict[str, set[str]] = defaultdict(set)
    proxy_files: dict[str, set[str]] = defaultdict(set)
    loanword_files: dict[str, set[str]] = defaultdict(set)

    segment_rows: list[dict] = []

    trs_files = sorted(in_dir.glob("*.trs"))
    for trs_path in trs_files:
        try:
            segments = parse_trs(trs_path)
        except ET.ParseError:
            # Robust fallback for malformed XML bytes: decode with replacement and retry.
            text = trs_path.read_text(encoding="iso-8859-1", errors="replace")
            root = ET.fromstring(text)
            segments = []
            for turn in root.iter("Turn"):
                segments.extend(iter_turn_segments(turn))

        for seg in segments:
            text = seg["text"]
            loanwords = extract_loanwords(text)
            text_no_loan = remove_starred(text)

            for lw in loanwords:
                lw_norm = normalize_proxy_token(lw)
                if lw_norm:
                    loanword_counts[lw_norm] += 1
                    loanword_files[lw_norm].add(trs_path.name)

            for tok in TOKEN_RE.findall(text_no_loan):
                nah_tok = normalize_marker_token(tok)
                if nah_tok and any(ch.isalpha() for ch in nah_tok):
                    marker_counts[nah_tok] += 1
                    marker_files[nah_tok].add(trs_path.name)
                    proxy_tok = normalize_proxy_token(nah_tok)
                    if proxy_tok and any(ch.isalpha() for ch in proxy_tok):
                        marker_proxy_counts[proxy_tok] += 1
                        proxy_files[proxy_tok].add(trs_path.name)
                        proxy_to_colon[proxy_tok][nah_tok] += 1

            segment_rows.append(
                {
                    "source_file": trs_path.name,
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "text_no_loanwords": text_no_loan,
                    "loanwords_marked": loanwords,
                }
            )

    segments_path = out_dir / "segments_clean.jsonl"
    with segments_path.open("w", encoding="utf-8") as f:
        for row in segment_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    marker_path = out_dir / "markers_nahuatl.tsv"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write("token\tcount\tfile_count\n")
        for token, count in marker_counts.most_common():
            f.write(f"{token}\t{count}\t{len(marker_files[token])}\n")

    proxy_path = out_dir / "markers_whisper_proxy.tsv"
    with proxy_path.open("w", encoding="utf-8") as f:
        f.write("token_proxy\tcount\tfile_count\n")
        for token, count in marker_proxy_counts.most_common():
            f.write(f"{token}\t{count}\t{len(proxy_files[token])}\n")

    strict_path = out_dir / "markers_nahuatl_strict.tsv"
    with strict_path.open("w", encoding="utf-8") as f:
        f.write("token_proxy\tcount\tfile_count\n")
        for token, count in marker_proxy_counts.most_common():
            if len(token) < 3:
                continue
            if token in SPANISH_STOPWORDS:
                continue
            if loanword_counts.get(token, 0) > 0:
                continue
            if not any(cue in token for cue in NAH_CUES):
                continue
            f.write(f"{token}\t{count}\t{len(proxy_files[token])}\n")

    bridge_path = out_dir / "marker_bridge_proxy_to_colon.tsv"
    with bridge_path.open("w", encoding="utf-8") as f:
        f.write("token_proxy\ttotal_count\ttop_colon_forms\n")
        for token, count in marker_proxy_counts.most_common():
            forms = proxy_to_colon.get(token)
            if not forms:
                continue
            top = " | ".join(f"{form}:{n}" for form, n in forms.most_common(5))
            f.write(f"{token}\t{count}\t{top}\n")

    loanword_path = out_dir / "loanwords_es.tsv"
    with loanword_path.open("w", encoding="utf-8") as f:
        f.write("loanword\tcount\tfile_count\n")
        for token, count in loanword_counts.most_common():
            f.write(f"{token}\t{count}\t{len(loanword_files[token])}\n")

    summary = {
        "input_files": len(trs_files),
        "segments": len(segment_rows),
        "unique_markers": len(marker_counts),
        "unique_proxy_markers": len(marker_proxy_counts),
        "unique_loanwords": len(loanword_counts),
        "outputs": {
            "segments_clean_jsonl": str(segments_path),
            "markers_nahuatl_tsv": str(marker_path),
            "markers_whisper_proxy_tsv": str(proxy_path),
            "markers_nahuatl_strict_tsv": str(strict_path),
            "marker_bridge_proxy_to_colon_tsv": str(bridge_path),
            "loanwords_es_tsv": str(loanword_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {segments_path}")
    print(f"Wrote: {marker_path}")
    print(f"Wrote: {proxy_path}")
    print(f"Wrote: {strict_path}")
    print(f"Wrote: {bridge_path}")
    print(f"Wrote: {loanword_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()

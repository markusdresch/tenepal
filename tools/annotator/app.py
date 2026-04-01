#!/usr/bin/env python3
"""
Tenepal Annotator — Video/Audio annotation tool with agent integration.

Usage:
    python tools/annotator/app.py --media validation_video/Hernán-1-3.mp4 --srt validation_video/Hernán-1-3.srt
    python tools/annotator/app.py --export annotations.jsonl
"""

import json
import os
import re
import subprocess
import argparse
import unicodedata
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file, Response
from sqlalchemy import text as sa_text

try:
    from flask_sqlalchemy import SQLAlchemy
except ModuleNotFoundError:
    SQLAlchemy = None
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        Float,
        Text,
        Boolean,
        DateTime,
        UniqueConstraint,
        ForeignKey,
        func,
    )
    from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

# Config
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = BASE_DIR.parent.parent  # tenepal root
DB_PATH = BASE_DIR / "annotations.db"
DISCUSSIONS_DIR = BASE_DIR / "discussions"
DISCUSSIONS_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

if SQLAlchemy is not None:
    db = SQLAlchemy(app)
else:
    class _SQLAlchemyFallback:
        """Minimal Flask-SQLAlchemy-compatible wrapper for local environments."""

        def __init__(self, flask_app: Flask):
            uri = flask_app.config["SQLALCHEMY_DATABASE_URI"]
            self.engine = create_engine(uri, future=True)
            self.session = scoped_session(
                sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
            )
            self.Model = declarative_base()
            self.Model.query = self.session.query_property()
            self.func = func

            # Expose SQLAlchemy symbols expected by model declarations.
            self.Column = Column
            self.Integer = Integer
            self.String = String
            self.Float = Float
            self.Text = Text
            self.Boolean = Boolean
            self.DateTime = DateTime
            self.UniqueConstraint = UniqueConstraint
            self.ForeignKey = ForeignKey

            @flask_app.teardown_appcontext
            def shutdown_session(_exception=None):
                self.session.remove()

        def create_all(self):
            self.Model.metadata.create_all(bind=self.engine)

    db = _SQLAlchemyFallback(app)

# Global state for current media
CURRENT_MEDIA = {"path": None, "srt_path": None, "type": None}


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class Annotation(db.Model):
    """Single segment annotation."""
    __tablename__ = "annotations"
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Source identification
    media_file = db.Column(db.String(500), nullable=False)
    cue_index = db.Column(db.Integer, nullable=False)
    segment_id = db.Column(db.String(100), unique=True, nullable=False)
    
    # Timing
    start_s = db.Column(db.Float, nullable=False)
    end_s = db.Column(db.Float, nullable=False)
    start_time_ms = db.Column(db.Integer)
    end_time_ms = db.Column(db.Integer)
    
    # Pipeline output (what Tenepal produced)
    pipeline_lang = db.Column(db.String(10))
    pipeline_text = db.Column(db.Text)
    pipeline_ipa = db.Column(db.Text)
    pipeline_confidence = db.Column(db.Float)
    
    # Human annotation
    correct_lang = db.Column(db.String(10))  # NAH/SPA/MAY/LAT/OTH/UNK
    correct_speaker = db.Column(db.String(50))  # character slug
    human_transcription = db.Column(db.Text)  # What the annotator actually hears
    boundary_suspect = db.Column(db.Boolean, default=False)
    start_adjust_ms = db.Column(db.Integer, default=0)
    end_adjust_ms = db.Column(db.Integer, default=0)
    boundary_note = db.Column(db.String(200))
    notes = db.Column(db.Text)
    merged_into = db.Column(db.String(100))
    overlap = db.Column(db.Boolean, default=False)
    secondary_speaker = db.Column(db.String(50))  # character slug
    split_into = db.Column(db.String(200))
    split_from = db.Column(db.String(100))
    
    # F005 wav2vec2 embedding classifier
    f005_pred_lang = db.Column(db.String(10))
    f005_confidence = db.Column(db.Float)

    # Meta
    annotated_at = db.Column(db.DateTime)
    annotator = db.Column(db.String(50), default="markus")
    
    # Unique constraint
    __table_args__ = (
        db.UniqueConstraint('media_file', 'cue_index', name='_media_cue_uc'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "segment_id": self.segment_id,
            "media_file": self.media_file,
            "cue_index": self.cue_index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "pipeline": {
                "lang": self.pipeline_lang,
                "text": self.pipeline_text,
                "ipa": self.pipeline_ipa,
                "confidence": self.pipeline_confidence
            },
            "f005": {
                "pred_lang": self.f005_pred_lang,
                "confidence": self.f005_confidence
            },
            "annotation": {
                "correct_lang": self.correct_lang,
                "correct_speaker": self.correct_speaker,
                "human_transcription": self.human_transcription,
                "boundary_suspect": self.boundary_suspect,
                "start_adjust_ms": self.start_adjust_ms,
                "end_adjust_ms": self.end_adjust_ms,
                "boundary_note": self.boundary_note,
                "notes": self.notes,
                "merged_into": self.merged_into,
                "overlap": self.overlap,
                "secondary_speaker": self.secondary_speaker,
                "split_into": self.split_into,
                "split_from": self.split_from,
                "annotated_at": self.annotated_at.isoformat() if self.annotated_at else None,
                "annotator": self.annotator
            }
        }


class Discussion(db.Model):
    """Agent discussion about a segment."""
    __tablename__ = "discussions"
    
    id = db.Column(db.Integer, primary_key=True)
    segment_id = db.Column(db.String(100), db.ForeignKey("annotations.segment_id"))
    agent = db.Column(db.String(50))
    question = db.Column(db.Text)
    response_file = db.Column(db.String(500))  # Path to markdown response
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "segment_id": self.segment_id,
            "agent": self.agent,
            "question": self.question,
            "response_file": self.response_file,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Character(db.Model):
    """Stable character registry — survives pipeline reruns."""
    __tablename__ = "characters"

    slug = db.Column(db.String(50), primary_key=True)  # e.g. "hernan", "marina"
    display_name = db.Column(db.String(200), nullable=False)
    scope = db.Column(db.String(100), default="global")  # hernan / la_otra_conquista / global
    category = db.Column(db.String(20), default="named")  # named / background / irrelevant
    notes = db.Column(db.Text)

    def to_dict(self):
        return {
            "slug": self.slug,
            "display_name": self.display_name,
            "scope": self.scope or "global",
            "category": self.category or "named",
            "notes": self.notes or "",
        }


class SpeakerName(db.Model):
    """Per-episode mapping from volatile pipeline SPEAKER_XX to stable character slug."""
    __tablename__ = "speaker_names"

    id = db.Column(db.Integer, primary_key=True)
    episode = db.Column(db.String(200), nullable=False)
    speaker_id = db.Column(db.String(50), nullable=False)  # volatile SPEAKER_XX
    character_slug = db.Column(db.String(50))  # FK to characters.slug
    display_name = db.Column(db.String(200))  # legacy, derived from character
    notes = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("episode", "speaker_id", name="_episode_speaker_uc"),
    )

    def to_dict(self):
        return {
            "episode": self.episode,
            "speaker_id": self.speaker_id,
            "character_slug": self.character_slug or "",
            "display_name": self.display_name or "",
            "notes": self.notes or "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SRT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_srt_time(time_str: str) -> float:
    """Parse SRT timestamp to seconds."""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT file into list of cues."""
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    cues = []
    
    # Split by double newline (cue separator)
    blocks = re.split(r"\n\n+", content.strip())
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        
        # First line is cue index
        try:
            cue_index = int(lines[0].strip())
        except ValueError:
            continue
        
        # Second line is timestamp
        time_match = re.match(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", lines[1])
        if not time_match:
            continue
        
        start_s = parse_srt_time(time_match.group(1))
        end_s = parse_srt_time(time_match.group(2))
        
        # Rest is text
        text = "\n".join(lines[2:]) if len(lines) > 2 else ""
        
        # Extract structured tag if present: [LANG|SPEAKER_XX|STEM_A]
        # Keep backward compatibility with simpler [LANG|SPEAKER_XX].
        lang = None
        speaker = None
        stem = None
        tag_match = re.match(r"\[([^\]]+)\]", text)
        if tag_match:
            parts = [p.strip() for p in tag_match.group(1).split("|") if p.strip()]
            if parts:
                if re.fullmatch(r"[A-Z]{3}", parts[0]):
                    lang = parts[0]
                else:
                    m_lang = re.search(r"\b([A-Z]{3})\b", parts[0])
                    lang = m_lang.group(1) if m_lang else None
            for p in parts[1:]:
                if speaker is None and re.fullmatch(r"SPEAKER_\d+", p):
                    speaker = p
                if stem is None and re.fullmatch(r"STEM_[A-Z0-9_+-]+", p):
                    stem = p
        if speaker is None:
            m_spk = re.search(r"SPEAKER_\d+", text)
            speaker = m_spk.group(0) if m_spk else None
        if stem is None:
            m_stem = re.search(r"STEM_[A-Z0-9_+-]+", text)
            stem = m_stem.group(0) if m_stem else None
        
        # Extract IPA if present (♫fused: ...)
        ipa_match = re.search(r"♫fused:\s*([^\n]+)", text)
        ipa = ipa_match.group(1).strip() if ipa_match else None
        
        # Extract confidence if present
        conf_match = re.search(r"conf[:\s]+([0-9.]+)", text, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else None
        
        # Clean text (remove metadata lines)
        clean_text = re.sub(r"♫[^\n]+\n?", "", text)
        clean_text = re.sub(r"\[LLM\][^\n]+\n?", "", clean_text)
        clean_text = clean_text.strip()
        
        cues.append({
            "index": cue_index,
            "start_s": start_s,
            "end_s": end_s,
            "text": clean_text,
            "lang": lang,
            "speaker": speaker,
            "stem": stem,
            "ipa": ipa,
            "confidence": confidence,
            "raw": text
        })
    
    return cues


NAH_IPA_MARKERS = {"ts", "tɬ", "kʷ", "tʃʼ", "kʼ", "tɕ"}
EJECTIVE_MARKERS = {"kʼ", "tʼ", "tsʼ", "tʃʼ", "pʼ"}


def parse_srt_rich(srt_path: Path) -> list[dict]:
    """Parse SRT with all metadata (3 IPA backends, markers, source tag)."""
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    cues = []
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        try:
            cue_index = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        start_s = parse_srt_time(time_match.group(1))
        end_s = parse_srt_time(time_match.group(2))

        # Content line
        content_line = lines[2] if len(lines) > 2 else ""
        tag_m = re.match(r"\[(\w+)\|([^\]]+)\]\s*(.*)", content_line)
        if tag_m:
            lang = tag_m.group(1).lower()
            speaker = tag_m.group(2)
            text_content = tag_m.group(3)
        else:
            lang = "oth"
            speaker = "?"
            text_content = content_line
        lang_map = {"nahuatl": "nah", "español": "spa", "other": "oth", "silence": "sil"}
        lang = lang_map.get(lang, lang)

        # Source marker: [FT], [LLM], [IPA], etc.
        source = "whisper"
        for marker in ["[FT]", "[LLM]", "[IPA]", "[REC]", "[UNC]"]:
            if marker in text_content:
                source = marker.strip("[]").lower()
                text_content = text_content.replace(marker, "").strip()
                break

        # Parse ♫ metadata lines
        ipa_allo = ipa_w2v2 = ipa_fused = trim = conf = ""
        for l in lines[3:]:
            if l.startswith("♫allo:"):
                ipa_allo = l[6:].strip()
            elif l.startswith("♫w2v2:"):
                ipa_w2v2 = l[6:].strip()
            elif l.startswith("♫fused:"):
                ipa_fused = l[7:].strip()
            elif l.startswith("♫trim:"):
                trim = l[6:].strip()
            elif l.startswith("♫conf:"):
                conf = l[6:].strip()

        fused_tokens = set(ipa_fused.split()) if ipa_fused else set()
        nah_markers = sorted(fused_tokens & NAH_IPA_MARKERS)
        ejectives = sorted(fused_tokens & EJECTIVE_MARKERS)

        cues.append({
            "cue": cue_index,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration": round(end_s - start_s, 2),
            "speaker": speaker,
            "lang": lang,
            "text": text_content,
            "source": source,
            "ipa_allo": ipa_allo,
            "ipa_w2v2": ipa_w2v2,
            "ipa_fused": ipa_fused,
            "trim": trim,
            "conf": conf,
            "nah_markers": nah_markers,
            "ejectives": ejectives,
        })
    return cues


def extract_speaker_tag(text: str | None) -> str | None:
    """Extract speaker tag like SPEAKER_12 from pipeline text."""
    if not text:
        return None
    m = re.search(r"SPEAKER_\d+", text)
    return m.group(0) if m else None


def extract_stem_tag(text: str | None) -> str | None:
    """Extract stem tag like STEM_A from pipeline text."""
    if not text:
        return None
    m = re.search(r"STEM_[A-Z0-9_+-]+", text)
    return m.group(0) if m else None


def sec_to_ms(sec: float | None) -> int:
    if sec is None:
        return 0
    return int(round(float(sec) * 1000.0))


def interval_overlap_score(a0: int, a1: int, b0: int, b1: int) -> float:
    """Overlap score in [0,1], relative to max segment duration."""
    overlap = max(0, min(a1, b1) - max(a0, b0))
    if overlap <= 0:
        return 0.0
    dur_a = max(1, a1 - a0)
    dur_b = max(1, b1 - b0)
    return overlap / float(max(dur_a, dur_b))


def annotation_interval_ms(ann: Annotation) -> tuple[int, int]:
    start_ms = ann.start_time_ms if ann.start_time_ms is not None else sec_to_ms(ann.start_s)
    end_ms = ann.end_time_ms if ann.end_time_ms is not None else sec_to_ms(ann.end_s)
    return int(start_ms), int(end_ms)


def discover_stem_files(media_path: str | None) -> list[dict]:
    """Discover sibling stem WAV files for multi-stem annotator playback."""
    if not media_path:
        return []
    p = Path(media_path)
    if not p.exists():
        return []
    parent = p.parent
    preferred = ["original.wav", "stem_A.wav", "stem_B.wav"]
    stems: list[dict] = []

    for name in preferred:
        fp = parent / name
        if fp.exists():
            stems.append({"name": name, "role": name.rsplit(".", 1)[0].upper(), "filename": fp.name})

    # Fallback: include additional stem_* wavs if present.
    known = {s["filename"] for s in stems}
    for fp in sorted(parent.glob("stem_*.wav")):
        if fp.name in known:
            continue
        role = fp.stem.upper()
        stems.append({"name": fp.name, "role": role, "filename": fp.name})

    return stems


def media_type_for_path(path: Path) -> str:
    return "video" if path.suffix.lower() in [".mp4", ".mkv", ".webm", ".mov"] else "audio"


def set_current_media(media_path: str | Path) -> bool:
    """Set CURRENT_MEDIA from absolute/project-relative path and auto-resolve sibling SRT."""
    p = Path(media_path)
    if not p.is_absolute():
        p = (PROJECT_DIR / p).resolve()
    if not p.exists() or not p.is_file():
        return False

    CURRENT_MEDIA["path"] = str(p)
    CURRENT_MEDIA["type"] = media_type_for_path(p)
    srt = p.with_suffix(".srt")
    CURRENT_MEDIA["srt_path"] = str(srt) if srt.exists() else None
    return True


def resolve_media_selection(raw_path: str | Path | None) -> Path | None:
    """Resolve a UI-selected media path to an in-project absolute path."""
    if raw_path is None:
        return None

    raw = str(raw_path).strip()
    if not raw:
        return None

    normalized = raw.replace("\\", "/")

    # Treat browser-style "/validation_video/foo.mp4" as project-relative.
    if normalized.startswith("/") and not normalized.startswith(str(PROJECT_DIR)):
        normalized = normalized.lstrip("/")

    target = Path(normalized)

    if target.is_absolute():
        try:
            return target.resolve().relative_to(PROJECT_DIR) and target.resolve()
        except ValueError:
            return None

    # Accept browser-provided paths with an accidental leading slash.
    relative = normalized.lstrip("/")
    candidate = (PROJECT_DIR / relative).resolve()
    try:
        candidate.relative_to(PROJECT_DIR)
        return candidate
    except ValueError:
        pass

    # Fallback to discovered media entries in case the client sends a slightly
    # different but semantically equivalent relative path.
    for media in list_media_candidates():
        if media["relpath"] == relative or media["name"] == relative:
            return (PROJECT_DIR / media["relpath"]).resolve()

    return None


def list_media_candidates() -> list[dict]:
    """Discover selectable media files in project validation folders."""
    exts = {".mp4", ".mkv", ".webm", ".mov", ".wav", ".mp3", ".ogg", ".flac"}
    roots = [PROJECT_DIR / "validation_video", PROJECT_DIR]
    files: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue
        for fp in root.rglob("*"):
            if not fp.is_file() or fp.suffix.lower() not in exts:
                continue
            r = fp.resolve()
            if r in seen:
                continue
            seen.add(r)
            files.append(r)

    out = []
    for fp in files:
        try:
            rel = fp.relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            rel = fp.name
        out.append({
            "name": fp.name,
            "relpath": rel,
            "type": media_type_for_path(fp),
        })

    out.sort(key=lambda x: (0 if x["type"] == "video" else 1, x["relpath"]))
    return out


def current_episode_key() -> str:
    """Episode key used for speaker naming profile partitioning."""
    media_path = CURRENT_MEDIA.get("path")
    if not media_path:
        return "default"
    return Path(media_path).stem


def infer_scope_from_episode(episode: str | None) -> str:
    """Map a media/episode key to a reusable character scope."""
    name = _norm_episode_name(episode)
    stem = Path(name).stem

    if stem.startswith("hernan-"):
        return "hernan"
    if stem.startswith("la-otra-conquista"):
        return "la_otra_conquista"
    if stem.startswith("apocalypto"):
        return "apocalypto"
    if stem.startswith("ixcanul"):
        return "ixcanul"
    if stem.startswith("regreso-a-aztlan") or stem.startswith("regreso_a_aztlan"):
        return "regreso_a_aztlan"
    return stem or "global"


def current_character_scope() -> str:
    """Return the character scope for the currently selected media."""
    return infer_scope_from_episode(current_episode_key())


def collect_pipeline_speaker_ids() -> list[str]:
    """Collect volatile SPEAKER_XX IDs from current SRT + DB pipeline_text."""
    speakers: set[str] = set()
    cue_lookup = _build_cue_lookup()
    for cue in cue_lookup.values():
        spk = cue.get("speaker") or extract_speaker_tag(cue.get("text")) or extract_speaker_tag(cue.get("raw"))
        if spk:
            speakers.add(spk)

    media_path = CURRENT_MEDIA.get("path")
    if media_path:
        media_name = Path(media_path).name
        ann_rows = Annotation.query.filter_by(media_file=media_name).all()
        for ann in ann_rows:
            spk = extract_speaker_tag(ann.pipeline_text)
            if spk:
                speakers.add(spk)

    def sort_key(spk: str):
        m = re.search(r"(\d+)", spk or "")
        return (int(m.group(1)) if m else 10_000, spk or "")

    return sorted(speakers, key=sort_key)


def _slugify(name: str) -> str:
    """Convert display name to slug: 'Young Xicontencatl' → 'young_xicontencatl'."""
    s = name.strip().lower()
    s = re.sub(r"[áà]", "a", s)
    s = re.sub(r"[éè]", "e", s)
    s = re.sub(r"[íì]", "i", s)
    s = re.sub(r"[óò]", "o", s)
    s = re.sub(r"[úù]", "u", s)
    s = re.sub(r"[ñ]", "n", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def collect_character_slugs_for_current_media() -> list[str]:
    """Collect character slugs used in annotations for current media."""
    slugs: set[str] = set()
    media_path = CURRENT_MEDIA.get("path")
    if media_path:
        media_name = Path(media_path).name
        ann_rows = Annotation.query.filter_by(media_file=media_name).all()
        for ann in ann_rows:
            if ann.correct_speaker:
                slugs.add(ann.correct_speaker)
            if ann.secondary_speaker:
                slugs.add(ann.secondary_speaker)
    return sorted(slugs)


def speaker_display_name_map(episode: str) -> dict[str, str]:
    """Map pipeline SPEAKER_XX → display name, resolved via character slugs."""
    rows = SpeakerName.query.filter_by(episode=episode).all()
    char_map = {c.slug: c.display_name for c in Character.query.all()}
    result: dict[str, str] = {}
    for r in rows:
        name = ""
        if r.character_slug:
            name = char_map.get(r.character_slug, r.character_slug)
        if not name:
            name = (r.display_name or "").strip()
        if name:
            result[r.speaker_id] = name
    return result


def character_display_name_map() -> dict[str, str]:
    """Map character slug → display name."""
    return {c.slug: c.display_name for c in Character.query.all()}


def characters_for_scope(scope: str) -> list[Character]:
    """Return characters visible for a given media scope."""
    return Character.query.filter(
        Character.scope.in_(["global", scope])
    ).order_by(Character.display_name).all()


def migrate_character_scopes() -> None:
    """Infer character scopes from speaker usage so film casts stay separated."""
    char_cols = db.session.execute(sa_text("PRAGMA table_info(characters)")).fetchall()
    col_names = {row[1] for row in char_cols}
    if "scope" not in col_names:
        db.session.execute(
            sa_text("ALTER TABLE characters ADD COLUMN scope VARCHAR(100) DEFAULT 'global'")
        )
        db.session.commit()
        print("Updated DB schema: added characters.scope")

    rows = db.session.execute(
        sa_text(
            """
            SELECT character_slug, episode
            FROM speaker_names
            WHERE character_slug IS NOT NULL
              AND character_slug != ''
            """
        )
    ).fetchall()

    usage: dict[str, set[str]] = {}
    for slug, episode in rows:
        scope = infer_scope_from_episode(episode)
        if not scope or scope.startswith("source_"):
            continue
        usage.setdefault(slug, set()).add(scope)

    ann_rows = db.session.execute(
        sa_text(
            """
            SELECT media_file, correct_speaker, secondary_speaker
            FROM annotations
            WHERE media_file IS NOT NULL
            """
        )
    ).fetchall()
    for media_file, correct_speaker, secondary_speaker in ann_rows:
        scope = infer_scope_from_episode(media_file)
        if not scope:
            continue
        for slug in (correct_speaker, secondary_speaker):
            if not slug:
                continue
            usage.setdefault(slug, set()).add(scope)

    changed = 0
    for char in Character.query.all():
        scopes = usage.get(char.slug, set())
        target_scope = "global"
        if len(scopes) == 1:
            target_scope = next(iter(scopes))
        elif len(scopes) > 1:
            target_scope = "global"
        elif not char.scope:
            target_scope = "global"
        else:
            target_scope = char.scope

        if (char.scope or "global") != target_scope:
            char.scope = target_scope
            changed += 1

    if changed:
        db.session.commit()
        print(f"Updated character scopes for {changed} characters")


def load_characters_catalog() -> dict:
    """Load optional characters.json catalog."""
    path = BASE_DIR / "characters.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _norm_episode_name(name: str | None) -> str:
    text = unicodedata.normalize("NFKD", (name or "").strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def character_maps_for_episode(episode: str) -> tuple[dict, dict[str, dict]]:
    """Return (characters_by_name, speaker_lookup) from characters.json for episode."""
    catalog = load_characters_catalog()
    file_episode = _norm_episode_name(catalog.get("episode"))
    req_episode = _norm_episode_name(episode)
    if file_episode and req_episode and file_episode != req_episode:
        # Allow fallback when episode contains extension differences.
        file_episode_stem = Path(file_episode).stem
        req_episode_stem = Path(req_episode).stem
        if file_episode_stem != req_episode_stem:
            return {}, {}

    chars = catalog.get("characters") or {}
    if not isinstance(chars, dict):
        return {}, {}

    speaker_lookup: dict[str, dict] = {}
    for char_name, cfg in chars.items():
        if not isinstance(cfg, dict):
            continue
        primary = cfg.get("primary_speaker")
        notes = cfg.get("notes") or ""
        aliases = cfg.get("aliases") or []
        if primary:
            speaker_lookup[str(primary)] = {
                "character": str(char_name),
                "role": "primary",
                "primary_speaker": str(primary),
                "notes": str(notes),
            }
        for alias in aliases:
            if not alias:
                continue
            speaker_lookup[str(alias)] = {
                "character": str(char_name),
                "role": "alias",
                "primary_speaker": str(primary) if primary else "",
                "notes": str(notes),
            }
    return chars, speaker_lookup


# ═══════════════════════════════════════════════════════════════════════════════
# MCP CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def send_to_mezcalmux(agent: str, message: str) -> dict:
    """Send message to agent via mezcalmux MCP (direct tmux approach)."""
    try:
        # Load mezcalmux state to get agent pane
        state_file = PROJECT_DIR / ".mezcalmux" / "state.json"
        if not state_file.exists():
            return {"success": False, "error": "mezcalmux not running (no state file)"}
        
        state = json.loads(state_file.read_text())
        agents = state.get("agents", {})
        
        if agent not in agents:
            return {"success": False, "error": f"Agent '{agent}' not running. Available: {list(agents.keys())}"}
        
        target = agents[agent]
        
        # Send via tmux
        # First send the text
        r1 = subprocess.run(
            ["tmux", "send-keys", "-t", target, "-l", message],
            capture_output=True, text=True
        )
        if r1.returncode != 0:
            return {"success": False, "error": f"tmux send-keys failed: {r1.stderr}"}
        
        # Wait a bit for Claude Code to register
        import time
        time.sleep(0.5)
        
        # Then send Enter
        r2 = subprocess.run(
            ["tmux", "send-keys", "-t", target, "Enter"],
            capture_output=True, text=True
        )
        if r2.returncode != 0:
            return {"success": False, "error": f"tmux Enter failed: {r2.stderr}"}
        
        return {"success": True, "agent": agent, "target": target}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_available_agents() -> list[str]:
    """Get list of running agents from mezcalmux."""
    try:
        state_file = PROJECT_DIR / ".mezcalmux" / "state.json"
        if not state_file.exists():
            return []
        state = json.loads(state_file.read_text())
        return list(state.get("agents", {}).keys())
    except:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.after_request
def add_no_cache_headers(response):
    """Prevent browser from caching HTML/API responses."""
    if "text/html" in response.content_type or "application/json" in response.content_type:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    """Main annotator UI."""
    url_filter = request.args.get("filter", "")
    return render_template("index.html", url_filter=url_filter)


@app.route("/dashboard")
def dashboard():
    """Diagnostic dashboard — compare pipeline output vs ground truth."""
    # Discover available SRT files for comparison
    eq_dir = PROJECT_DIR / "eq_comparison_results"
    srt_files = sorted(eq_dir.glob("*.srt")) if eq_dir.exists() else []
    srt_options = [{"name": f.stem, "path": f.relative_to(PROJECT_DIR).as_posix()} for f in srt_files]

    # Also include current media SRT if loaded
    if CURRENT_MEDIA.get("srt_path"):
        cur = Path(CURRENT_MEDIA["srt_path"])
        try:
            cur_rel = cur.relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            cur_rel = cur.name
        srt_options.insert(0, {"name": f"(current) {cur.stem}", "path": cur_rel})

    # GT file
    gt_file = PROJECT_DIR / "benchmarks" / "snapshots" / "eq_comparison_gt_v1.json"
    gt_path = gt_file.relative_to(PROJECT_DIR).as_posix() if gt_file.exists() else ""

    # Media filename for audio playback
    media_name = Path(CURRENT_MEDIA["path"]).name if CURRENT_MEDIA.get("path") else ""

    return render_template(
        "dashboard.html",
        srt_options=srt_options,
        gt_path=gt_path,
        media_name=media_name,
    )


@app.route("/api/dashboard/segments")
def api_dashboard_segments():
    """Return enriched segments for dashboard: SRT data + GT comparison."""
    srt_rel = request.args.get("srt", "")
    gt_rel = request.args.get("gt", "")

    if not srt_rel:
        return jsonify({"error": "No SRT specified"}), 400

    srt_path = PROJECT_DIR / srt_rel
    if not srt_path.exists():
        return jsonify({"error": f"SRT not found: {srt_rel}"}), 404

    # Parse SRT with all metadata
    cues = parse_srt_rich(srt_path)

    # Load GT if available
    gt_map = {}
    if gt_rel:
        gt_path = PROJECT_DIR / gt_rel
        if gt_path.exists():
            data = json.loads(gt_path.read_text())
            if isinstance(data, list):
                for item in data:
                    c = item.get("cue_index")
                    lang = (item.get("correct_lang") or "").lower()
                    if c is not None and lang:
                        gt_map[c] = lang
            elif isinstance(data, dict):
                for i, s in enumerate(data.get("segments", []), 1):
                    if s.get("done"):
                        gt_map[i] = s["lang"].lower()

    # Overlay DB annotations on top of JSON GT (DB wins — it has latest corrections)
    if CURRENT_MEDIA.get("path"):
        media_name = Path(CURRENT_MEDIA["path"]).name
        anns = Annotation.query.filter(
            Annotation.media_file == media_name,
            Annotation.annotated_at.isnot(None),
            Annotation.correct_lang.isnot(None),
        ).all()
        for ann in anns:
            if ann.cue_index >= 0 and ann.correct_lang:
                gt_map[ann.cue_index] = ann.correct_lang.lower()

    # Load F005 predictions from DB
    f005_map = {}
    _f005_media = Path(CURRENT_MEDIA["path"]).name if CURRENT_MEDIA.get("path") else ""
    if _f005_media:
        f005_anns = Annotation.query.filter(
            Annotation.media_file == _f005_media,
            Annotation.f005_pred_lang.isnot(None),
        ).all()
        for ann in f005_anns:
            f005_map[ann.cue_index] = ann.f005_pred_lang.lower()

    # --- Decision trace: simulate voter logic locally ---
    from collections import defaultdict
    SPA_FUNC_WORDS = {
        "la", "el", "de", "que", "en", "los", "las", "un", "una", "por", "con",
        "del", "al", "se", "su", "más", "mas", "como", "hay", "pero", "ya",
        "no", "es", "le", "lo", "y", "a",
    }
    NAH_EXCLUSIVE = {"tɬ", "kʷ", "ɬ"}
    TWO_PASS_MARKERS = {"ts", "tɬ", "kʷ", "tʃʼ", "kʼ", "tɕ"}

    # Pass 1: simulate inline speaker prior (accumulating in cue order)
    speaker_prior_state = defaultdict(lambda: defaultdict(int))  # spk → {lang: count}
    inline_prior = {}  # cue → {prior_lang, prior_ratio, prior_total}
    for seg in cues:
        spk = seg["speaker"]
        cue = seg["cue"]
        hist = speaker_prior_state[spk]
        total = sum(hist.values())
        prior_lang = ""
        prior_ratio = 0.0
        if total >= 3:
            best_lang = max(hist, key=hist.get)
            ratio = hist[best_lang] / total
            if ratio >= 0.7:
                prior_lang = best_lang
                prior_ratio = ratio
        inline_prior[cue] = {
            "prior_lang": prior_lang,
            "prior_ratio": round(prior_ratio, 2),
            "prior_total": total,
            "prior_counts": dict(hist),
        }
        # Accumulate AFTER recording state (state is what was visible BEFORE this seg)
        speaker_prior_state[spk][seg["lang"]] += 1

    # Pass 2: compute two-pass IPA evidence per speaker
    speaker_ipa_nah = defaultdict(int)
    speaker_total = defaultdict(int)
    for seg in cues:
        spk = seg["speaker"]
        if seg["lang"] == "sil":
            continue
        speaker_total[spk] += 1
        # Check all IPA streams for NAH markers
        all_tokens = set()
        for field in ["ipa_allo", "ipa_w2v2", "ipa_fused"]:
            all_tokens |= set(seg[field].split()) if seg[field] else set()
        if all_tokens & TWO_PASS_MARKERS:
            speaker_ipa_nah[spk] += 1

    two_pass_eligible = {}
    for spk, nah_count in speaker_ipa_nah.items():
        total = speaker_total[spk]
        if total >= 5 and nah_count >= 3 and (nah_count / total) >= 0.15:
            two_pass_eligible[spk] = {
                "nah_evidence": nah_count,
                "total": total,
                "ratio": round(nah_count / total, 2),
            }

    # Merge and compute error info + decision trace
    ERROR_COSTS_LOCAL = {
        ("nah", "spa"): 2.0, ("nah", "oth"): 1.5, ("nah", "unk"): 0.5,
        ("may", "spa"): 2.0, ("may", "oth"): 1.5, ("may", "unk"): 0.5,
        ("spa", "nah"): 1.5, ("spa", "may"): 1.5,
    }
    rows = []
    for seg in cues:
        cue = seg["cue"]
        gt_lang = gt_map.get(cue, "")
        match = None
        error_type = ""
        cost = 0.0
        if gt_lang:
            match = gt_lang == seg["lang"]
            if not match:
                error_type = f"{gt_lang}\u2192{seg['lang']}"
                if gt_lang == "unk":
                    cost = 0.0
                elif seg["lang"] == "unk":
                    cost = ERROR_COSTS_LOCAL.get((gt_lang, "unk"), 0.5)
                else:
                    cost = ERROR_COSTS_LOCAL.get((gt_lang, seg["lang"]), 1.0)

        # Decision trace
        all_ipa_tokens = set()
        for field in ["ipa_allo", "ipa_w2v2", "ipa_fused"]:
            all_ipa_tokens |= set(seg[field].split()) if seg[field] else set()

        text_words = set(w.strip("¿¡?!.,;:\"'").lower() for w in seg["text"].split())
        spa_func_hits = len(text_words & SPA_FUNC_WORDS)
        has_tl = "tɬ" in all_ipa_tokens
        has_nah_excl = bool(all_ipa_tokens & NAH_EXCLUSIVE)
        has_nah_marker = bool(all_ipa_tokens & TWO_PASS_MARKERS)

        # Determine likely decision path
        traces = []
        if seg["source"] == "whisper":
            traces.append(f"whisper-trusted (text has {spa_func_hits} SPA func words)")
        elif seg["source"] in ("ft", "llm"):
            traces.append(f"ipa-path ({seg['source'].upper()} text)")
        else:
            traces.append(f"source={seg['source']}")

        if has_tl:
            if spa_func_hits >= 3:
                traces.append("tɬ detected BUT Spanish guard fired (≥3 func words)")
            else:
                traces.append("tɬ detected → NAH override")
        if has_nah_excl and not has_tl:
            traces.append(f"NAH-exclusive markers: {all_ipa_tokens & NAH_EXCLUSIVE}")
        if has_nah_marker and not has_nah_excl:
            traces.append(f"NAH IPA markers: {all_ipa_tokens & TWO_PASS_MARKERS}")
        if seg["ejectives"]:
            traces.append(f"ejectives: {seg['ejectives']}")

        # Inline prior state
        ip = inline_prior[cue]
        if ip["prior_lang"]:
            traces.append(
                f"inline-prior: {ip['prior_lang'].upper()} "
                f"({ip['prior_ratio']:.0%} of {ip['prior_total']})"
            )
        elif ip["prior_total"] > 0:
            top = max(ip["prior_counts"], key=ip["prior_counts"].get) if ip["prior_counts"] else "?"
            traces.append(
                f"inline-prior: no majority yet "
                f"(best={top.upper()} {ip['prior_counts'].get(top,0)}/{ip['prior_total']})"
            )

        # Two-pass eligibility
        spk = seg["speaker"]
        if spk in two_pass_eligible:
            tp = two_pass_eligible[spk]
            traces.append(
                f"two-pass eligible: {tp['nah_evidence']}/{tp['total']} "
                f"IPA-NAH ({tp['ratio']:.0%})"
            )
            if seg["lang"] == "spa":
                traces.append("→ two-pass would override SPA→NAH if conf < 0.7")

        if spa_func_hits > 0:
            traces.append(f"SPA text hints: {text_words & SPA_FUNC_WORDS}")

        f005_lang = f005_map.get(cue, "")
        rows.append({**seg, "gt_lang": gt_lang, "match": match,
                      "error_type": error_type, "cost": cost,
                      "decision": traces,
                      "prior": ip,
                      "f005_lang": f005_lang})

    return jsonify(rows)


@app.route("/api/status")
def api_status():
    """Get current status."""
    media_options = list_media_candidates()
    if not CURRENT_MEDIA.get("path") and media_options:
        set_current_media(media_options[0]["relpath"])

    episode = current_episode_key()
    _, speaker_lookup = character_maps_for_episode(episode)
    media_payload = dict(CURRENT_MEDIA)
    if CURRENT_MEDIA.get("path"):
        try:
            media_payload["relpath"] = Path(CURRENT_MEDIA["path"]).resolve().relative_to(PROJECT_DIR).as_posix()
        except Exception:
            media_payload["relpath"] = Path(CURRENT_MEDIA["path"]).name
    return jsonify({
        "media": media_payload,
        "available_media": media_options,
        "stems": discover_stem_files(CURRENT_MEDIA.get("path")),
        "characters_loaded": bool(speaker_lookup),
        "agents": get_available_agents(),
        "db_path": str(DB_PATH)
    })


@app.route("/api/media/select", methods=["POST"])
def api_media_select():
    """Switch currently loaded media in annotator."""
    data = request.json or {}
    relpath = data.get("relpath") or data.get("path")
    if not relpath:
        return jsonify({"error": "relpath is required"}), 400

    target = resolve_media_selection(relpath)
    if target is None:
        return jsonify({"error": "Invalid relative path"}), 400

    if not set_current_media(target):
        return jsonify({"error": f"Media not found: {relpath}"}), 404

    return jsonify({
        "success": True,
        "media": {
            "path": CURRENT_MEDIA.get("path"),
            "relpath": Path(CURRENT_MEDIA["path"]).resolve().relative_to(PROJECT_DIR).as_posix() if CURRENT_MEDIA.get("path") else None,
            "srt_path": CURRENT_MEDIA.get("srt_path"),
            "type": CURRENT_MEDIA.get("type"),
        }
    })


@app.route("/api/segments")
def api_segments():
    """Get all segments from current SRT with annotations."""
    if not CURRENT_MEDIA["srt_path"]:
        return jsonify({"error": "No SRT loaded"}), 400
    
    srt_path = Path(CURRENT_MEDIA["srt_path"])
    if not srt_path.exists():
        return jsonify({"error": f"SRT not found: {srt_path}"}), 404
    
    cues = parse_srt(srt_path)
    media_name = Path(CURRENT_MEDIA["path"]).name
    cues_by_idx = {c["index"]: c for c in cues}

    # Timestamp-based matching for source cues (cue_index >= 0).
    source_annotations = Annotation.query.filter(
        Annotation.media_file == media_name,
        Annotation.cue_index >= 0,
        Annotation.annotated_at.isnot(None),
    ).all()
    annotations = _match_annotations_to_cues_by_time(cues, source_annotations, min_score=0.5)

    # Sync matched annotations to current SRT timestamps/indices.
    # Prevents the "steal" bug where saving a newly annotated cue overwrites
    # an old-timestamped row that was being matched to a different cue.
    matched_ids = {ann.id for ann in annotations.values()}
    need_sync = []
    for cue_idx, ann in annotations.items():
        cue = cues_by_idx.get(cue_idx)
        if not cue:
            continue
        c_start = sec_to_ms(cue["start_s"])
        c_end = sec_to_ms(cue["end_s"])
        if ann.cue_index != cue_idx or ann.start_time_ms != c_start or ann.end_time_ms != c_end:
            need_sync.append((ann, cue_idx, cue))
    if need_sync:
        try:
            # Phase 1: Park rows that need cue_index changes to avoid UNIQUE conflicts.
            for ann, _cue_idx, _cue in need_sync:
                if ann.cue_index != _cue_idx:
                    ann.cue_index = -ann.id
                    ann.segment_id = f"_sync_{ann.id}"
            db.session.flush()
            # Phase 2: Apply final values.
            for ann, cue_idx, cue in need_sync:
                ann.cue_index = cue_idx
                ann.start_s = cue["start_s"]
                ann.end_s = cue["end_s"]
                ann.start_time_ms = sec_to_ms(cue["start_s"])
                ann.end_time_ms = sec_to_ms(cue["end_s"])
                ann.segment_id = f"{media_name.replace('.', '_')}_{cue_idx:04d}"
            db.session.commit()
            app.logger.info("Synced %d annotation timestamps to current SRT", len(need_sync))
        except Exception:
            db.session.rollback()
            app.logger.warning("Failed to sync annotation timestamps", exc_info=True)

    # Recover parked/stale annotations (cue_index < 0) by matching to unassigned cues.
    parked = Annotation.query.filter(
        Annotation.media_file == media_name,
        Annotation.cue_index < 0,
        Annotation.annotated_at.isnot(None),
    ).all()
    if parked:
        assigned_cues = set(annotations.keys())
        recovered = 0
        for ann in parked:
            a0, a1 = annotation_interval_ms(ann)
            if a1 <= a0:
                continue
            best_cue, best_score = None, 0.0
            for cue in cues:
                if cue["index"] in assigned_cues:
                    continue
                c0, c1 = sec_to_ms(cue["start_s"]), sec_to_ms(cue["end_s"])
                score = interval_overlap_score(a0, a1, c0, c1)
                if score > best_score:
                    best_score = score
                    best_cue = cue["index"]
            if best_cue and best_score >= 0.5:
                cue = cues_by_idx[best_cue]
                ann.cue_index = best_cue
                ann.start_s = cue["start_s"]
                ann.end_s = cue["end_s"]
                ann.start_time_ms = sec_to_ms(cue["start_s"])
                ann.end_time_ms = sec_to_ms(cue["end_s"])
                ann.segment_id = f"{media_name.replace('.', '_')}_{best_cue:04d}"
                annotations[best_cue] = ann
                assigned_cues.add(best_cue)
                recovered += 1
        if recovered:
            try:
                db.session.commit()
                app.logger.info("Recovered %d parked annotations", recovered)
            except Exception:
                db.session.rollback()

    result = []
    for cue in cues:
        segment_id = f"{media_name.replace('.', '_')}_{cue['index']:04d}"
        
        ann = annotations.get(cue["index"])
        is_annotated = bool(ann and ann.annotated_at is not None)
        ann_payload = ann.to_dict()["annotation"] if ann else None

        result.append({
            "index": cue["index"],
            "segment_id": segment_id,
            "start_s": cue["start_s"],
            "end_s": cue["end_s"],
            "pipeline": {
                "lang": cue["lang"],
                "text": cue["text"],
                "ipa": cue["ipa"],
                "confidence": cue["confidence"],
                "raw": cue["raw"],
                "speaker": cue.get("speaker") or extract_speaker_tag(cue["text"]) or extract_speaker_tag(cue["raw"]),
                "stem": cue.get("stem") or extract_stem_tag(cue["text"]) or extract_stem_tag(cue["raw"]),
            },
            "annotation": ann_payload,
            "has_annotation": is_annotated,
            "is_merged_part": bool(ann and ann.merged_into),
            "is_merged_segment": False,
            "is_split_segment": bool(ann and ann.split_from),
            "is_split_source": bool(ann and ann.split_into),
        })

    merged_rows = Annotation.query.filter(
        Annotation.media_file == media_name,
        Annotation.cue_index < 0,
        ~Annotation.segment_id.like("_stale_%"),
        ~Annotation.segment_id.like("_sync_%"),
    ).all()
    for ann in merged_rows:
        result.append({
            "index": ann.cue_index,
            "segment_id": ann.segment_id,
            "start_s": ann.start_s,
            "end_s": ann.end_s,
            "pipeline": {
                "lang": ann.pipeline_lang,
                "text": ann.pipeline_text,
                "ipa": ann.pipeline_ipa,
                "confidence": ann.pipeline_confidence,
                "raw": ann.pipeline_text or "",
                "speaker": extract_speaker_tag(ann.pipeline_text),
                "stem": extract_stem_tag(ann.pipeline_text),
            },
            "annotation": ann.to_dict()["annotation"],
            "has_annotation": True,
            "is_merged_part": False,
            "is_merged_segment": True,
            "is_split_segment": bool(ann.split_from),
            "is_split_source": bool(ann.split_into),
        })

    result.sort(key=lambda x: (x["start_s"], x["end_s"]))

    # Optional filter: ?filter=unk → only segments where pipeline says UNK
    # ?filter=disagree → pipeline lang ≠ annotation lang
    lang_filter = request.args.get("filter", "").lower()
    if lang_filter:
        filtered = []
        for seg in result:
            pipe_lang = (seg["pipeline"].get("lang") or "").lower()
            ann_lang = ((seg.get("annotation") or {}).get("correct_lang") or "").lower()
            if lang_filter == "disagree":
                if ann_lang and pipe_lang and ann_lang != pipe_lang:
                    filtered.append(seg)
            else:
                if pipe_lang == lang_filter:
                    filtered.append(seg)
        result = filtered

    return jsonify(result)


def _next_merged_cue_index(media_name: str) -> int:
    """Allocate a synthetic negative cue index for merged segments."""
    min_existing = db.session.query(db.func.min(Annotation.cue_index)).filter_by(media_file=media_name).scalar()
    if min_existing is None or min_existing >= 0:
        return -1
    return int(min_existing) - 1


def _next_synthetic_cue_index(media_name: str) -> int:
    """Allocate a synthetic negative cue index for derived segments."""
    return _next_merged_cue_index(media_name)


def _build_cue_lookup() -> dict[int, dict]:
    """Build cue index lookup from current SRT for pipeline fallback data."""
    if not CURRENT_MEDIA["srt_path"]:
        return {}
    srt_path = Path(CURRENT_MEDIA["srt_path"])
    if not srt_path.exists():
        return {}
    cues = parse_srt(srt_path)
    return {c["index"]: c for c in cues}


def _match_annotations_to_cues_by_time(
    cues: list[dict],
    annotations: list[Annotation],
    min_score: float = 0.5,
) -> dict[int, Annotation]:
    """Greedy one-to-one time overlap matching between cue list and annotation rows."""
    cue_ms = {
        c["index"]: (sec_to_ms(c.get("start_s")), sec_to_ms(c.get("end_s")))
        for c in cues
    }
    candidates: list[tuple[float, int, int, int]] = []
    for ann in annotations:
        a0, a1 = annotation_interval_ms(ann)
        if a1 <= a0:
            continue
        for cue in cues:
            c0, c1 = cue_ms[cue["index"]]
            if c1 <= c0:
                continue
            score = interval_overlap_score(a0, a1, c0, c1)
            if score > 0:
                overlap = max(0, min(a1, c1) - max(a0, c0))
                candidates.append((score, overlap, cue["index"], ann.id))

    # Highest overlap first, then highest normalized score.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    assigned_cues: set[int] = set()
    assigned_ann: set[int] = set()
    ann_by_id = {a.id: a for a in annotations}
    result: dict[int, Annotation] = {}
    for score, _overlap, cue_idx, ann_id in candidates:
        if score < min_score:
            continue
        if cue_idx in assigned_cues or ann_id in assigned_ann:
            continue
        ann = ann_by_id.get(ann_id)
        if not ann:
            continue
        assigned_cues.add(cue_idx)
        assigned_ann.add(ann_id)
        result[cue_idx] = ann
    return result


def _ensure_annotation_row(media_name: str, cue_index: int, cue_lookup: dict[int, dict]) -> Annotation:
    """Ensure annotation row exists for a cue index."""
    ann = Annotation.query.filter_by(media_file=media_name, cue_index=cue_index).first()
    if ann:
        return ann

    cue = cue_lookup.get(cue_index, {})
    segment_id = f"{media_name.replace('.', '_')}_{cue_index:04d}"
    ann = Annotation(
        media_file=media_name,
        cue_index=cue_index,
        segment_id=segment_id,
        start_s=cue.get("start_s", 0.0),
        end_s=cue.get("end_s", 0.0),
        start_time_ms=sec_to_ms(cue.get("start_s", 0.0)),
        end_time_ms=sec_to_ms(cue.get("end_s", 0.0)),
        pipeline_lang=cue.get("lang"),
        pipeline_text=cue.get("text"),
        pipeline_ipa=cue.get("ipa"),
        pipeline_confidence=cue.get("confidence"),
    )
    db.session.add(ann)
    return ann


@app.route("/api/segments/<int:cue_index>", methods=["POST"])
def api_save_segment(cue_index: int):
    """Save annotation for a segment."""
    if not CURRENT_MEDIA["path"]:
        return jsonify({"error": "No media loaded"}), 400
    
    data = request.json
    media_name = Path(CURRENT_MEDIA["path"]).name
    segment_id = f"{media_name.replace('.', '_')}_{cue_index:04d}"
    start_s = float(data.get("start_s", 0) or 0)
    end_s = float(data.get("end_s", 0) or 0)
    start_ms = sec_to_ms(start_s)
    end_ms = sec_to_ms(end_s)
    
    # Find by timestamp first (stable across diarization/speaker-ID churn), then cue index.
    found_by_timestamp = False
    ann = Annotation.query.filter_by(
        media_file=media_name,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
    ).first()
    if ann:
        found_by_timestamp = True
    if not ann:
        candidate = Annotation.query.filter_by(media_file=media_name, cue_index=cue_index).first()
        if candidate:
            # Only reuse this row if its timestamps are close to the current cue.
            # If timestamps diverged (old pipeline run), the row is already matched
            # to a different cue by time-overlap — reusing it would steal that match.
            c_start = candidate.start_time_ms or sec_to_ms(candidate.start_s)
            c_end = candidate.end_time_ms or sec_to_ms(candidate.end_s)
            if c_start and c_end:
                score = interval_overlap_score(c_start, c_end, start_ms, end_ms)
            else:
                score = 1.0  # no timestamps stored, safe to reuse
            if score >= 0.3:
                ann = candidate
            else:
                # Stale row from old pipeline run — reassign its cue_index so it
                # doesn't block the new annotation.  The load-time sync will
                # eventually match it by timestamp to the correct cue.
                candidate.cue_index = -candidate.id  # park it out of the way
                candidate.segment_id = f"_stale_{candidate.id}"
                db.session.flush()
    if not ann:
        ann = Annotation(
            media_file=media_name,
            cue_index=cue_index,
            segment_id=segment_id,
            start_s=start_s,
            end_s=end_s,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        db.session.add(ann)
    else:
        ann.start_s = start_s
        ann.end_s = end_s
        ann.start_time_ms = start_ms
        ann.end_time_ms = end_ms
        ann.cue_index = cue_index
        # Don't update segment_id when found by timestamp — cue indices shift
        # across pipeline reruns, causing UNIQUE constraint collisions
        if not found_by_timestamp:
            ann.segment_id = segment_id
    
    # Update pipeline data if provided
    if "pipeline" in data:
        ann.pipeline_lang = data["pipeline"].get("lang")
        ann.pipeline_text = data["pipeline"].get("text")
        ann.pipeline_ipa = data["pipeline"].get("ipa")
        ann.pipeline_confidence = data["pipeline"].get("confidence")
    
    # Update annotation
    if "annotation" in data:
        a = data["annotation"]
        selected_lang = a.get("correct_lang")

        # Guard: don't overwrite existing correct_lang with empty value.
        # This prevents the "blink" bug where switching filters auto-saves
        # a segment with no selection, wiping out the previous annotation.
        if not selected_lang and ann.correct_lang:
            # Keep existing annotation, only update non-lang fields if provided
            pass
        else:
            # Always store the selected language (even if it matches pipeline).
            # This makes "confirmed correct" distinguishable from "not annotated".
            ann.correct_lang = selected_lang

        ann.correct_speaker = a.get("correct_speaker")
        ann.human_transcription = a.get("human_transcription")
        ann.boundary_suspect = a.get("boundary_suspect", False)
        ann.start_adjust_ms = a.get("start_adjust_ms", 0)
        ann.end_adjust_ms = a.get("end_adjust_ms", 0)
        ann.boundary_note = a.get("boundary_note")
        ann.notes = a.get("notes")
        ann.overlap = a.get("overlap", False)
        ann.secondary_speaker = a.get("secondary_speaker")
        ann.annotated_at = datetime.utcnow()
        ann.annotator = a.get("annotator", "markus")
    
    db.session.commit()
    
    return jsonify({"success": True, "segment_id": segment_id})


@app.route("/api/segments/split", methods=["POST"])
def api_split_segment():
    """Split one source segment into two synthetic segments."""
    if not CURRENT_MEDIA["path"]:
        return jsonify({"error": "No media loaded"}), 400

    data = request.json or {}
    cue_index = data.get("cue_index")
    split_at = data.get("split_at")
    second_speaker = data.get("second_speaker")

    if cue_index is None or split_at is None:
        return jsonify({"error": "cue_index and split_at are required"}), 400

    try:
        cue_index = int(cue_index)
        split_at = float(split_at)
    except (TypeError, ValueError):
        return jsonify({"error": "cue_index must be int and split_at must be float"}), 400

    if cue_index <= 0:
        return jsonify({"error": "Only source SRT segments can be split"}), 400

    media_name = Path(CURRENT_MEDIA["path"]).name
    cue_lookup = _build_cue_lookup()
    src = _ensure_annotation_row(media_name, cue_index, cue_lookup)

    if split_at <= src.start_s or split_at >= src.end_s:
        return jsonify({"error": "split_at must be inside the segment range"}), 400

    src_speaker = src.correct_speaker or extract_speaker_tag(src.pipeline_text) or extract_speaker_tag(cue_lookup.get(cue_index, {}).get("text"))
    merged_text = src.pipeline_text or cue_lookup.get(cue_index, {}).get("text") or ""
    merged_ipa = src.pipeline_ipa or cue_lookup.get(cue_index, {}).get("ipa") or ""

    # Keep simple text/ipa carry-over for manual annotation; exact sub-segmentation is manual.
    seg_a_id = f"{media_name.replace('.', '_')}_SPLIT_{cue_index:04d}_A"
    seg_b_id = f"{media_name.replace('.', '_')}_SPLIT_{cue_index:04d}_B"

    # Remove previous split artifacts if repeated.
    for sid in (seg_a_id, seg_b_id):
        old = Annotation.query.filter_by(media_file=media_name, segment_id=sid).first()
        if old:
            db.session.delete(old)
    db.session.flush()

    cue_a = _next_synthetic_cue_index(media_name)
    cue_b = cue_a - 1

    text_a = merged_text
    text_b = merged_text
    if src_speaker:
        text_a = f"[SPLIT-A|{src_speaker}] {merged_text}".strip()
    if second_speaker:
        text_b = f"[SPLIT-B|{second_speaker}] {merged_text}".strip()

    ann_a = Annotation(
        media_file=media_name,
        cue_index=cue_a,
        segment_id=seg_a_id,
        start_s=src.start_s,
        end_s=split_at,
        start_time_ms=sec_to_ms(src.start_s),
        end_time_ms=sec_to_ms(split_at),
        pipeline_lang=src.pipeline_lang,
        pipeline_text=text_a,
        pipeline_ipa=merged_ipa,
        pipeline_confidence=src.pipeline_confidence,
        notes=f"SPLIT from segment {cue_index} at {split_at:.3f}s (A)",
        split_from=src.segment_id,
        correct_speaker=src_speaker,
        annotated_at=datetime.utcnow(),
        annotator="split-tool",
    )
    ann_b = Annotation(
        media_file=media_name,
        cue_index=cue_b,
        segment_id=seg_b_id,
        start_s=split_at,
        end_s=src.end_s,
        start_time_ms=sec_to_ms(split_at),
        end_time_ms=sec_to_ms(src.end_s),
        pipeline_lang=src.pipeline_lang,
        pipeline_text=text_b,
        pipeline_ipa=merged_ipa,
        pipeline_confidence=src.pipeline_confidence,
        notes=f"SPLIT from segment {cue_index} at {split_at:.3f}s (B)",
        split_from=src.segment_id,
        correct_speaker=second_speaker or src_speaker,
        annotated_at=datetime.utcnow(),
        annotator="split-tool",
    )
    db.session.add(ann_a)
    db.session.add(ann_b)

    src.split_into = f"{seg_a_id},{seg_b_id}"
    src.annotated_at = src.annotated_at or datetime.utcnow()
    src.notes = (src.notes + " | " if src.notes else "") + f"SPLIT into {seg_a_id} and {seg_b_id}"

    db.session.commit()
    return jsonify({
        "success": True,
        "source_segment": src.segment_id,
        "split_segments": [seg_a_id, seg_b_id],
    })


@app.route("/api/segments/merge", methods=["POST"])
def api_merge_segments():
    """Merge two segments into one synthetic merged annotation."""
    if not CURRENT_MEDIA["path"]:
        return jsonify({"error": "No media loaded"}), 400

    data = request.json or {}
    segment_ids = data.get("segment_ids") or []
    direction = (data.get("direction") or "").strip().lower()

    if len(segment_ids) != 2:
        return jsonify({"error": "segment_ids must contain exactly 2 cue indices"}), 400
    if direction not in {"prev", "next"}:
        return jsonify({"error": "direction must be 'prev' or 'next'"}), 400

    try:
        first_idx, second_idx = sorted([int(segment_ids[0]), int(segment_ids[1])])
    except (ValueError, TypeError):
        return jsonify({"error": "segment_ids must be integers"}), 400

    if first_idx <= 0 or second_idx <= 0:
        return jsonify({"error": "Only source SRT segments (positive indices) can be merged"}), 400

    media_name = Path(CURRENT_MEDIA["path"]).name
    cue_lookup = _build_cue_lookup()
    ann_a = _ensure_annotation_row(media_name, first_idx, cue_lookup)
    ann_b = _ensure_annotation_row(media_name, second_idx, cue_lookup)

    cue_a = cue_lookup.get(first_idx, {})
    cue_b = cue_lookup.get(second_idx, {})

    text_a = ann_a.pipeline_text or cue_a.get("text") or ""
    text_b = ann_b.pipeline_text or cue_b.get("text") or ""
    ipa_a = ann_a.pipeline_ipa or cue_a.get("ipa") or ""
    ipa_b = ann_b.pipeline_ipa or cue_b.get("ipa") or ""

    merged_text = f"{text_a} | {text_b}".strip(" |")
    merged_ipa = " ".join([x for x in [ipa_a.strip(), ipa_b.strip()] if x]).strip()

    merged_cue_index = _next_merged_cue_index(media_name)
    merged_segment_id = f"{media_name.replace('.', '_')}_MERGE_{first_idx:04d}_{second_idx:04d}"

    existing_merged = Annotation.query.filter_by(media_file=media_name, segment_id=merged_segment_id).first()
    if existing_merged:
        db.session.delete(existing_merged)
        db.session.flush()

    merged = Annotation(
        media_file=media_name,
        cue_index=merged_cue_index,
        segment_id=merged_segment_id,
        start_s=min(ann_a.start_s, ann_b.start_s),
        end_s=max(ann_a.end_s, ann_b.end_s),
        start_time_ms=sec_to_ms(min(ann_a.start_s, ann_b.start_s)),
        end_time_ms=sec_to_ms(max(ann_a.end_s, ann_b.end_s)),
        pipeline_lang=ann_a.pipeline_lang if ann_a.pipeline_lang == ann_b.pipeline_lang else "OTH",
        pipeline_text=merged_text,
        pipeline_ipa=merged_ipa,
        notes=f"MERGED from segments {first_idx} and {second_idx}",
        annotated_at=datetime.utcnow(),
        annotator="merge-tool",
    )
    db.session.add(merged)

    ann_a.merged_into = merged_segment_id
    ann_b.merged_into = merged_segment_id
    ann_a.annotated_at = ann_a.annotated_at or datetime.utcnow()
    ann_b.annotated_at = ann_b.annotated_at or datetime.utcnow()

    db.session.commit()

    return jsonify({
        "success": True,
        "segment_id": merged_segment_id,
        "cue_index": merged_cue_index,
        "merged_indices": [first_idx, second_idx],
        "direction": direction,
    })


@app.route("/api/segments/<int:cue_index>", methods=["DELETE"])
def api_reset_segment(cue_index: int):
    """Reset annotation fields for a segment back to unannotated state."""
    if not CURRENT_MEDIA["path"]:
        return jsonify({"error": "No media loaded"}), 400

    media_name = Path(CURRENT_MEDIA["path"]).name
    ann = Annotation.query.filter_by(media_file=media_name, cue_index=cue_index).first()
    if not ann:
        cue_lookup = _build_cue_lookup()
        cue = cue_lookup.get(cue_index)
        if cue:
            ann = Annotation.query.filter_by(
                media_file=media_name,
                start_time_ms=sec_to_ms(cue.get("start_s")),
                end_time_ms=sec_to_ms(cue.get("end_s")),
            ).first()
    if not ann:
        return jsonify({"success": True, "segment_id": None, "message": "Already unannotated"})

    ann.correct_lang = None
    ann.correct_speaker = None
    ann.human_transcription = None
    ann.boundary_suspect = False
    ann.start_adjust_ms = 0
    ann.end_adjust_ms = 0
    ann.boundary_note = None
    ann.notes = None
    ann.overlap = False
    ann.secondary_speaker = None
    ann.annotated_at = None

    db.session.commit()
    return jsonify({"success": True, "segment_id": ann.segment_id})


@app.route("/api/discuss", methods=["POST"])
def api_discuss():
    """Send segment to agent for discussion."""
    data = request.json
    agent = data.get("agent")
    segment_id = data.get("segment_id")
    question = data.get("question", "")
    cue_index = data.get("cue_index")
    include_ipa = data.get("include_ipa", True)
    
    if not agent:
        return jsonify({"error": "No agent specified"}), 400
    
    # Build message
    media_name = Path(CURRENT_MEDIA["path"]).name if CURRENT_MEDIA["path"] else "unknown"
    
    # Get segment data
    pipeline = data.get("pipeline", {})
    annotation = data.get("annotation", {})
    
    msg_parts = [
        f"ANNOTATION REVIEW: Segment {segment_id}",
        f"Media: {media_name}",
        f"Time: {data.get('start_s', 0):.2f}s - {data.get('end_s', 0):.2f}s",
        f"",
        f"Pipeline says: [{pipeline.get('lang', '?')}] {pipeline.get('text', '')}",
    ]
    
    if include_ipa and pipeline.get("ipa"):
        msg_parts.append(f"IPA: {pipeline.get('ipa')}")
    
    if annotation.get("correct_lang"):
        msg_parts.append(f"Human annotation: {annotation.get('correct_lang')}")
    
    if annotation.get("boundary_suspect"):
        msg_parts.append(f"⚠ Boundary suspect: {annotation.get('boundary_note', 'unspecified')}")
    
    if question:
        msg_parts.append(f"")
        msg_parts.append(f"Question: {question}")
    
    message = "\n".join(msg_parts)
    
    # Send to agent
    result = send_to_mezcalmux(agent, message)
    
    if result["success"]:
        # Save discussion record
        disc = Discussion(
            segment_id=segment_id,
            agent=agent,
            question=question,
            response_file=str(DISCUSSIONS_DIR / f"{segment_id}_{agent}.md")
        )
        db.session.add(disc)
        db.session.commit()
    
    return jsonify(result)


@app.route("/api/agents")
def api_agents():
    """Get available agents."""
    return jsonify(get_available_agents())


@app.route("/api/speakers")
def api_speakers():
    """Get speaker naming table for current episode, auto-populating unknown speakers."""
    if not CURRENT_MEDIA.get("path"):
        return jsonify([])

    episode = current_episode_key()
    pipeline_ids = collect_pipeline_speaker_ids()

    # Auto-create rows for all discovered pipeline speakers.
    existing = {
        r.speaker_id: r
        for r in SpeakerName.query.filter_by(episode=episode).all()
    }
    created = False
    for speaker_id in pipeline_ids:
        if speaker_id in existing:
            continue
        db.session.add(
            SpeakerName(
                episode=episode,
                speaker_id=speaker_id,
                display_name="",
                notes="",
                updated_at=datetime.utcnow(),
            )
        )
        created = True
    if created:
        db.session.commit()

    char_map = character_display_name_map()
    rows = SpeakerName.query.filter_by(episode=episode).all()
    def sort_key(row: SpeakerName):
        m = re.search(r"(\d+)", row.speaker_id or "")
        return (int(m.group(1)) if m else 10_000, row.speaker_id or "")
    rows.sort(key=sort_key)
    payload = []
    for r in rows:
        d = r.to_dict()
        # Resolve display_name from character if available
        if r.character_slug and r.character_slug in char_map:
            d["display_name"] = char_map[r.character_slug]
        payload.append(d)
    return jsonify(payload)


@app.route("/api/speakers/<speaker_id>", methods=["POST"])
def api_update_speaker(speaker_id: str):
    """Upsert speaker → character mapping for current episode."""
    if not CURRENT_MEDIA.get("path"):
        return jsonify({"error": "No media loaded"}), 400

    data = request.json or {}
    character_slug = (data.get("character_slug") or "").strip()
    display_name = (data.get("display_name") or "").strip()
    notes = (data.get("notes") or "").strip()
    episode = current_episode_key()

    # If display_name given but no slug, find or create character
    if display_name and not character_slug:
        character_slug = _slugify(display_name)
        existing_char = Character.query.get(character_slug)
        if not existing_char:
            db.session.add(Character(
                slug=character_slug,
                display_name=display_name,
                scope=current_character_scope(),
                category="named",
            ))

    row = SpeakerName.query.filter_by(episode=episode, speaker_id=speaker_id).first()
    if not row:
        row = SpeakerName(
            episode=episode,
            speaker_id=speaker_id,
            character_slug=character_slug or None,
            display_name=display_name,
            notes=notes,
            updated_at=datetime.utcnow(),
        )
        db.session.add(row)
    else:
        if character_slug:
            row.character_slug = character_slug
        row.display_name = display_name
        row.notes = notes
        row.updated_at = datetime.utcnow()

    db.session.commit()
    return jsonify({"success": True, "speaker": row.to_dict()})


@app.route("/api/characters", methods=["GET", "POST"])
def api_characters():
    """GET: list all characters. POST: create new character."""
    if request.method == "POST":
        data = request.json or {}
        display_name = (data.get("display_name") or "").strip()
        if not display_name:
            return jsonify({"error": "display_name required"}), 400
        category = (data.get("category") or "named").strip()
        notes = (data.get("notes") or "").strip()
        slug = _slugify(display_name)
        if not slug:
            return jsonify({"error": "invalid name"}), 400
        existing = Character.query.get(slug)
        if existing:
            return jsonify({"error": f"Character '{slug}' exists", "slug": slug}), 409
        char = Character(
            slug=slug,
            display_name=display_name,
            scope=current_character_scope(),
            category=category,
            notes=notes,
        )
        db.session.add(char)
        db.session.commit()
        return jsonify({"success": True, "slug": slug, "character": char.to_dict()})

    episode = current_episode_key()
    scope = current_character_scope()
    chars = [c.to_dict() for c in characters_for_scope(scope)]

    # Build speaker_lookup: pipeline SPEAKER_XX → character slug for current episode
    speaker_lookup: dict[str, dict] = {}
    for sn in SpeakerName.query.filter_by(episode=episode).all():
        if sn.character_slug:
            speaker_lookup[sn.speaker_id] = {
                "slug": sn.character_slug,
                "character": character_display_name_map().get(sn.character_slug, sn.character_slug),
            }

    return jsonify({
        "episode": episode,
        "scope": scope,
        "characters": chars,
        "speaker_lookup": speaker_lookup,
    })


@app.route("/api/characters/<slug>", methods=["PATCH"])
def api_update_character(slug: str):
    """Update character display_name or category."""
    char = Character.query.get(slug)
    if not char:
        return jsonify({"error": "not found"}), 404
    data = request.json or {}
    if "display_name" in data:
        new_name = (data["display_name"] or "").strip()
        if new_name:
            char.display_name = new_name
    if "category" in data:
        char.category = (data["category"] or "named").strip()
    if "notes" in data:
        char.notes = (data["notes"] or "").strip()
    db.session.commit()
    return jsonify({"success": True, "character": char.to_dict()})


@app.route("/api/export_srt")
def api_export_srt():
    """Export current segments as SRT using speaker display names if available."""
    if not CURRENT_MEDIA.get("srt_path"):
        return jsonify({"error": "No SRT loaded"}), 400

    srt_path = Path(CURRENT_MEDIA["srt_path"])
    if not srt_path.exists():
        return jsonify({"error": f"SRT not found: {srt_path}"}), 404

    media_name = Path(CURRENT_MEDIA["path"]).name if CURRENT_MEDIA.get("path") else ""
    cues = parse_srt(srt_path)
    source_annotations = Annotation.query.filter(
        Annotation.media_file == media_name,
        Annotation.cue_index >= 0,
        Annotation.annotated_at.isnot(None),
    ).all()
    ann_by_cue = _match_annotations_to_cues_by_time(cues, source_annotations, min_score=0.5)
    name_map = speaker_display_name_map(current_episode_key())
    _, character_lookup = character_maps_for_episode(current_episode_key())

    def fmt_srt_time(sec: float) -> str:
        total_ms = int(round(max(0.0, sec) * 1000.0))
        h = total_ms // 3_600_000
        rem = total_ms % 3_600_000
        m = rem // 60_000
        rem %= 60_000
        s = rem // 1000
        ms = rem % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    for out_idx, cue in enumerate(cues, start=1):
        ann = ann_by_cue.get(cue["index"])
        lang = (ann.correct_lang if ann and ann.correct_lang else cue.get("lang")) or "UNK"
        speaker_id = (
            (ann.correct_speaker if ann and ann.correct_speaker else None)
            or cue.get("speaker")
            or extract_speaker_tag(cue.get("text"))
            or extract_speaker_tag(cue.get("raw"))
            or "UNKNOWN"
        )
        speaker_label = (
            name_map.get(speaker_id)
            or character_lookup.get(speaker_id, {}).get("character")
            or speaker_id
        )

        if ann and ann.human_transcription:
            content_text = ann.human_transcription.strip()
        else:
            content_text = (cue.get("text") or "").strip()
            content_text = re.sub(r"^\[[^\]]+\]\s*", "", content_text).strip()

        tag = f"[{lang}|{speaker_label}]"
        payload = f"{tag} {content_text}".strip()

        lines.extend([
            str(out_idx),
            f"{fmt_srt_time(cue['start_s'])} --> {fmt_srt_time(cue['end_s'])}",
            payload,
            "",
        ])

    out = "\n".join(lines).rstrip() + "\n"
    default_name = f"{Path(media_name).stem or 'annotations'}_named.srt"
    return Response(
        out,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={default_name}"},
    )


@app.route("/media/<path:filename>")
def serve_media(filename):
    """Serve media file with range support for seeking."""
    # Look in multiple locations
    for base in [PROJECT_DIR / "validation_video", PROJECT_DIR, Path(CURRENT_MEDIA.get("path", "")).parent]:
        filepath = base / filename
        if filepath.exists():
            break
    else:
        return "File not found", 404
    
    # Detect media type
    suffix = filepath.suffix.lower()
    if suffix in [".mp4", ".mkv", ".webm", ".mov"]:
        mimetype = f"video/{suffix[1:]}"
    elif suffix in [".mp3", ".wav", ".ogg", ".flac"]:
        mimetype = f"audio/{suffix[1:]}"
    else:
        mimetype = "application/octet-stream"
    
    # Support range requests for seeking
    range_header = request.headers.get("Range")
    
    if range_header:
        # Parse range
        size = filepath.stat().st_size
        byte1, byte2 = 0, None
        m = re.search(r"(\d+)-(\d*)", range_header)
        if m:
            byte1 = int(m.group(1))
            if m.group(2):
                byte2 = int(m.group(2))
        
        if byte2 is None:
            byte2 = size - 1
        
        length = byte2 - byte1 + 1
        
        with open(filepath, "rb") as f:
            f.seek(byte1)
            data = f.read(length)
        
        resp = Response(data, 206, mimetype=mimetype)
        resp.headers["Content-Range"] = f"bytes {byte1}-{byte2}/{size}"
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = length
        return resp
    
    return send_file(filepath, mimetype=mimetype)


@app.route("/api/export")
def api_export():
    """Export annotations as JSONL."""
    annotations = Annotation.query.filter(Annotation.annotated_at.isnot(None)).all()
    
    lines = []
    for ann in annotations:
        # Get discussions
        discussions = Discussion.query.filter_by(segment_id=ann.segment_id).all()
        
        record = ann.to_dict()
        record["discussions"] = [d.to_dict() for d in discussions]
        lines.append(json.dumps(record, ensure_ascii=False))
    
    return Response(
        "\n".join(lines),
        mimetype="application/x-jsonlines",
        headers={"Content-Disposition": "attachment; filename=annotations.jsonl"}
    )


@app.route("/api/dashboard/allo_full_track")
def api_allo_full_track():
    """Return Allosaurus full-track phones mapped to SRT segments."""
    srt_rel = request.args.get("srt", "")
    ft_path = PROJECT_DIR / "eq_comparison_results" / "allo_full_track.json"

    if not ft_path.exists():
        return jsonify({"error": "No full-track data. Run Allosaurus with timestamp=True first."}), 404
    if not srt_rel:
        return jsonify({"error": "No SRT specified"}), 400

    srt_path = PROJECT_DIR / srt_rel
    if not srt_path.exists():
        return jsonify({"error": f"SRT not found: {srt_rel}"}), 404

    # Load full-track phones
    ft_data = json.loads(ft_path.read_text())  # [{phone, start, dur}, ...]

    # Build (phone, start, end) list
    phones = []
    for i, p in enumerate(ft_data):
        start = p["start"]
        end = ft_data[i + 1]["start"] if i + 1 < len(ft_data) else start + p["dur"]
        phones.append((p["phone"], start, end))

    # Parse SRT for segment boundaries
    cues = parse_srt_rich(srt_path)

    # NAH markers for highlighting
    NAH_FT_MARKERS = {"ts", "tɬ", "kʷ", "tʃʼ", "kʼ", "tɕ", "ɬ", "ʔ", "x"}
    EJECTIVE_FT = {"tʼ", "kʼ", "pʼ", "tsʼ", "tʃʼ"}

    # Map phones to segments by midpoint overlap
    result = []
    for seg in cues:
        seg_phones = []
        for phone, ps, pe in phones:
            mid = (ps + pe) / 2
            if mid >= seg["start_s"] and mid <= seg["end_s"]:
                seg_phones.append(phone)
        ipa_str = " ".join(seg_phones) if seg_phones else ""
        phone_set = set(seg_phones)
        nah_markers = sorted(phone_set & NAH_FT_MARKERS)
        ejectives = sorted(phone_set & EJECTIVE_FT)
        result.append({
            "cue": seg["cue"],
            "ipa_full_track": ipa_str,
            "ft_phone_count": len(seg_phones),
            "ft_nah_markers": nah_markers,
            "ft_ejectives": ejectives,
        })

    return jsonify({
        "total_phones": len(phones),
        "segments": result,
    })


@app.route("/comparison")
def comparison_page():
    """Standalone comparison matrix page."""
    return render_template("comparison.html")


@app.route("/api/f005_predictions")
def api_f005_predictions():
    """Return F005 wav2vec2 embedding predictions keyed by cue_index."""
    media_name = request.args.get("media", "")
    if not media_name and CURRENT_MEDIA.get("path"):
        media_name = Path(CURRENT_MEDIA["path"]).name
    if not media_name:
        media_name = "Hern\u00e1n-1-3.mp4"

    anns = Annotation.query.filter(
        Annotation.media_file == media_name,
        Annotation.f005_pred_lang.isnot(None),
    ).all()

    result = {}
    for a in anns:
        result[str(a.cue_index)] = {
            "pred_lang": a.f005_pred_lang,
            "confidence": a.f005_confidence,
            "correct_lang": a.correct_lang,
            "pipeline_lang": a.pipeline_lang,
        }
    return jsonify(result)


@app.route("/api/dashboard/comparison")
def api_dashboard_comparison():
    """Evaluate all EQ configs against ground truth annotations. Returns accuracy table."""
    eq_dir = PROJECT_DIR / "eq_comparison_results"
    if not eq_dir.exists():
        return jsonify({"error": "No eq_comparison_results directory"}), 404

    # Load ground truth from DB — NAH+SPA only, Hernán-1-3
    media_name = Path(CURRENT_MEDIA["path"]).name if CURRENT_MEDIA.get("path") else "Hernán-1-3.mp4"
    anns = Annotation.query.filter(
        Annotation.media_file == media_name,
        Annotation.correct_lang.in_(["NAH", "SPA"]),
    ).all()
    gt = [(a.start_time_ms, a.end_time_ms, a.correct_lang) for a in anns if a.start_time_ms is not None]
    if not gt:
        return jsonify({"error": "No NAH/SPA annotations found"}), 404

    def eval_srt(srt_path):
        cues = parse_srt_rich(srt_path)
        correct = total = nah_c = nah_t = spa_c = spa_t = s2n = n2s = 0
        for start_ms, end_ms, gt_lang in gt:
            gt_mid = (start_ms + end_ms) / 2
            # Match by SRT midpoint closest to GT midpoint
            best, best_d = None, 9999999
            for seg in cues:
                seg_mid = (seg["start_s"] + seg["end_s"]) * 500  # midpoint in ms
                d = abs(seg_mid - gt_mid)
                if d < 3000 and d < best_d:
                    best_d = d
                    best = seg["lang"].upper() if seg["lang"] else None
            if not best:
                continue
            total += 1
            if gt_lang == best:
                correct += 1
            if gt_lang == "NAH":
                nah_t += 1
                if best == "NAH": nah_c += 1
                elif best == "SPA": n2s += 1
            elif gt_lang == "SPA":
                spa_t += 1
                if best == "SPA": spa_c += 1
                elif best == "NAH": s2n += 1
        return {
            "accuracy": round(100 * correct / total, 1) if total else 0,
            "correct": correct, "total": total,
            "nah_recall": round(100 * nah_c / nah_t, 1) if nah_t else 0,
            "spa_recall": round(100 * spa_c / spa_t, 1) if spa_t else 0,
            "spa_as_nah": s2n, "nah_as_spa": n2s,
            "nah_total": nah_t, "spa_total": spa_t,
        }

    results = []
    for f in sorted(eq_dir.glob("*.srt")):
        if f.stem.startswith("vibevoice"):
            continue
        try:
            r = eval_srt(f)
            r["name"] = f.stem
            results.append(r)
        except Exception:
            pass

    results.sort(key=lambda x: -x["accuracy"])
    return jsonify({"configs": results, "gt_count": len(gt)})


@app.route("/api/stats")
def api_stats():
    """Get annotation statistics."""
    total = Annotation.query.count()
    annotated = Annotation.query.filter(Annotation.annotated_at.isnot(None)).count()
    corrected = Annotation.query.filter(
        Annotation.annotated_at.isnot(None),
        Annotation.correct_lang.isnot(None),
    ).count()
    confirmed = Annotation.query.filter(
        Annotation.annotated_at.isnot(None),
        Annotation.correct_lang.is_(None),
    ).count()
    boundary_issues = Annotation.query.filter_by(boundary_suspect=True).count()
    
    # Language breakdown
    lang_counts = db.session.query(
        Annotation.correct_lang, db.func.count(Annotation.id)
    ).filter(
        Annotation.annotated_at.isnot(None)
    ).group_by(Annotation.correct_lang).all()
    
    return jsonify({
        "total": total,
        "annotated": annotated,
        "corrected": corrected,
        "confirmed_correct": confirmed,
        "boundary_issues": boundary_issues,
        "by_language": {lang: count for lang, count in lang_counts if lang}
    })


def _migrate_speakers_to_character_slugs():
    """One-time migration: SPEAKER_XX → character slugs in annotations + speaker_names."""
    # Check if migration already done: if any correct_speaker doesn't match SPEAKER_\d+
    sample = db.session.execute(sa_text(
        "SELECT correct_speaker FROM annotations "
        "WHERE correct_speaker IS NOT NULL AND correct_speaker != '' LIMIT 1"
    )).fetchone()
    if sample and not re.match(r"^SPEAKER_\d+$", sample[0]):
        return  # Already migrated

    # Build per-episode slug map from speaker_names display_name values.
    # Key: (episode, speaker_id) → slug — because SPEAKER_01 means different
    # characters in different episodes.
    sn_rows = db.session.execute(sa_text(
        "SELECT speaker_id, display_name, episode FROM speaker_names "
        "WHERE display_name IS NOT NULL AND display_name != ''"
    )).fetchall()

    ep_speaker_to_slug: dict[tuple[str, str], str] = {}
    slug_to_display: dict[str, str] = {}
    slug_to_category: dict[str, str] = {}

    for speaker_id, display_name, episode in sn_rows:
        name = display_name.strip()
        if not name:
            continue
        slug = _slugify(name)
        if not slug:
            continue

        lower = name.lower()
        if lower in ("irrelevant",):
            category = "irrelevant"
        elif lower in ("unknown",):
            category = "background"
        else:
            category = "named"

        ep_speaker_to_slug[(episode, speaker_id)] = slug
        slug_to_display[slug] = name
        slug_to_category[slug] = category

    if not ep_speaker_to_slug:
        return

    # Create characters rows
    for slug, display_name in slug_to_display.items():
        existing = db.session.execute(
            sa_text("SELECT slug FROM characters WHERE slug = :s"), {"s": slug}
        ).fetchone()
        if not existing:
            db.session.execute(sa_text(
                "INSERT INTO characters (slug, display_name, category) VALUES (:s, :d, :c)"
            ), {"s": slug, "d": display_name, "c": slug_to_category.get(slug, "named")})

    # Add "background" catch-all
    existing_bg = db.session.execute(
        sa_text("SELECT slug FROM characters WHERE slug = 'background'")
    ).fetchone()
    if not existing_bg:
        db.session.execute(sa_text(
            "INSERT INTO characters (slug, display_name, category) VALUES "
            "('background', 'Background', 'background')"
        ))

    # Update speaker_names with character_slug (episode-specific)
    for (episode, speaker_id), slug in ep_speaker_to_slug.items():
        db.session.execute(sa_text(
            "UPDATE speaker_names SET character_slug = :slug "
            "WHERE episode = :ep AND speaker_id = :sid "
            "AND (character_slug IS NULL OR character_slug = '')"
        ), {"slug": slug, "ep": episode, "sid": speaker_id})

    # Migrate annotations.correct_speaker: SPEAKER_XX → slug.
    # Must resolve per media_file (episode). Build media → episode mapping.
    media_rows = db.session.execute(sa_text(
        "SELECT DISTINCT media_file FROM annotations "
        "WHERE correct_speaker IS NOT NULL AND correct_speaker != ''"
    )).fetchall()

    for (media_file,) in media_rows:
        # Episode key = stem of media file
        from pathlib import Path as _Path
        episode_key = _Path(media_file).stem

        # Find speaker_names for this episode
        ep_mappings = {
            sid: slug for (ep, sid), slug in ep_speaker_to_slug.items()
            if ep == episode_key
        }
        for speaker_id, slug in ep_mappings.items():
            db.session.execute(sa_text(
                "UPDATE annotations SET correct_speaker = :slug "
                "WHERE media_file = :mf AND correct_speaker = :sid"
            ), {"slug": slug, "mf": media_file, "sid": speaker_id})
            db.session.execute(sa_text(
                "UPDATE annotations SET secondary_speaker = :slug "
                "WHERE media_file = :mf AND secondary_speaker = :sid"
            ), {"slug": slug, "mf": media_file, "sid": speaker_id})

    db.session.commit()
    migrated = len(ep_speaker_to_slug)
    chars = len(slug_to_display)
    print(f"Migrated {migrated} speaker mappings → {chars} character slugs")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tenepal Annotator")
    parser.add_argument("--media", "-m", help="Video or audio file to annotate")
    parser.add_argument("--srt", "-s", help="SRT file with segments")
    parser.add_argument("--port", "-p", type=int, default=5050, help="Port (default 5050)")
    parser.add_argument("--export", "-e", help="Export annotations to JSONL file")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()
    
    def ensure_annotation_schema():
        """Best-effort additive schema migration for existing SQLite DBs."""
        try:
            cols = db.session.execute(sa_text("PRAGMA table_info(annotations)")).fetchall()
            col_names = {row[1] for row in cols}  # row[1] = column name
            if "human_transcription" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN human_transcription TEXT")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.human_transcription")
            if "merged_into" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN merged_into VARCHAR(100)")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.merged_into")
            if "correct_speaker" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN correct_speaker VARCHAR(20)")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.correct_speaker")
            if "overlap" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN overlap BOOLEAN DEFAULT 0")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.overlap")
            if "secondary_speaker" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN secondary_speaker VARCHAR(20)")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.secondary_speaker")
            if "split_into" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN split_into VARCHAR(200)")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.split_into")
            if "split_from" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN split_from VARCHAR(100)")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.split_from")
            if "start_time_ms" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN start_time_ms INTEGER")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.start_time_ms")
            if "end_time_ms" not in col_names:
                db.session.execute(
                    sa_text("ALTER TABLE annotations ADD COLUMN end_time_ms INTEGER")
                )
                db.session.commit()
                print("Updated DB schema: added annotations.end_time_ms")

            # Backfill timestamp columns for legacy rows.
            db.session.execute(
                sa_text(
                    """
                    UPDATE annotations
                    SET start_time_ms = CAST(ROUND(start_s * 1000.0) AS INTEGER)
                    WHERE start_time_ms IS NULL
                    """
                )
            )
            db.session.execute(
                sa_text(
                    """
                    UPDATE annotations
                    SET end_time_ms = CAST(ROUND(end_s * 1000.0) AS INTEGER)
                    WHERE end_time_ms IS NULL
                    """
                )
            )
            db.session.commit()

            # Timestamp index to speed overlap/matching lookups.
            db.session.execute(
                sa_text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_annotations_media_time
                    ON annotations (media_file, start_time_ms, end_time_ms)
                    """
                )
            )
            db.session.commit()

            # --- Character slug migration ---
            # Add character_slug column to speaker_names if missing.
            sn_cols = db.session.execute(sa_text("PRAGMA table_info(speaker_names)")).fetchall()
            sn_col_names = {row[1] for row in sn_cols}
            if "character_slug" not in sn_col_names:
                db.session.execute(
                    sa_text("ALTER TABLE speaker_names ADD COLUMN character_slug VARCHAR(50)")
                )
                db.session.commit()
                print("Updated DB schema: added speaker_names.character_slug")

            # One-time migration: convert SPEAKER_XX in correct_speaker/secondary_speaker
            # to character slugs, and populate characters table.
            _migrate_speakers_to_character_slugs()
            migrate_character_scopes()

        except Exception as e:
            print(f"Warning: schema check/migration failed: {e}")

    with app.app_context():
        db.create_all()
        ensure_annotation_schema()
    
    if args.export:
        with app.app_context():
            annotations = Annotation.query.filter(Annotation.annotated_at.isnot(None)).all()
            with open(args.export, "w") as f:
                for ann in annotations:
                    discussions = Discussion.query.filter_by(segment_id=ann.segment_id).all()
                    record = ann.to_dict()
                    record["discussions"] = [d.to_dict() for d in discussions]
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Exported {len(annotations)} annotations to {args.export}")
        return
    
    if args.media:
        media_path = Path(args.media).resolve()
        if not media_path.exists():
            print(f"Media file not found: {media_path}")
            return
        set_current_media(media_path)
        # Optional explicit override
        if args.srt:
            srt_path = Path(args.srt).resolve()
            CURRENT_MEDIA["srt_path"] = str(srt_path) if srt_path.exists() else None
            if CURRENT_MEDIA["srt_path"]:
                print(f"Loaded SRT: {srt_path}")
            else:
                print(f"Warning: SRT not found at {srt_path}")
    
    print(f"\n🎬 Tenepal Annotator")
    print(f"   Media: {CURRENT_MEDIA.get('path', 'None')}")
    print(f"   SRT:   {CURRENT_MEDIA.get('srt_path', 'None')}")
    print(f"   DB:    {DB_PATH}")
    print(f"\n   Open http://localhost:{args.port}\n")
    
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()

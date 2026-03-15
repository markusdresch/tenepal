#!/usr/bin/env python3
"""
Parselmouth Acoustic Transliteration vs Allosaurus vs wav2vec2

Compares ML phoneme backends with Parselmouth acoustic ground truth.
For each SRT segment, produces:
  - Allosaurus IPA
  - wav2vec2 IPA
  - Fused IPA
  - Parselmouth "transliteration": frame-by-frame acoustic description
    mapped to articulatory features

Parselmouth doesn't recognize phonemes - it measures physics:
  - Voicing (periodic signal = voiced)
  - Formants (F1/F2 → vowel quality)
  - Spectral centroid (fricative type)
  - Intensity contour (syllable boundaries)
  - Pitch contour (intonation)
  - Special: tɬ detection (unvoiced lateral affricate)

Usage:
    python scripts/parselmouth_comparison.py \
        --audio validation_video/Hernán-1-3.vocals.wav \
        --srt validation_video/Hernán-1-3.srt \
        [--segments 10-20] [--lang NAH] [--output comparison.md]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    print("ERROR: pip install praat-parselmouth")
    sys.exit(1)


# ── IPA Vowel Mapping from Formants ────────────────────────────────

# F1 = openness (low F1 = close, high F1 = open)
# F2 = frontness (low F2 = back, high F2 = front)
# These are approximate ranges for adult male speakers (typical in Hernán)

VOWEL_MAP = {
    # (F1_range, F2_range) → IPA
    # Close front
    "i": (250, 350, 2000, 2800),  # F1 250-350, F2 2000-2800
    # Close-mid front
    "e": (350, 500, 1800, 2400),
    # Open front
    "a": (600, 900, 1200, 1800),  # open central-front
    # Close back
    "u": (250, 400, 600, 1200),
    # Close-mid back
    "o": (400, 600, 700, 1200),
}

def formant_to_vowel(f1: float, f2: float) -> str:
    """Map F1/F2 formant values to approximate IPA vowel."""
    best = "?"
    best_dist = float("inf")
    
    for vowel, (f1_lo, f1_hi, f2_lo, f2_hi) in VOWEL_MAP.items():
        f1_mid = (f1_lo + f1_hi) / 2
        f2_mid = (f2_lo + f2_hi) / 2
        # Normalized distance
        dist = ((f1 - f1_mid) / 200) ** 2 + ((f2 - f2_mid) / 400) ** 2
        if dist < best_dist:
            best_dist = dist
            best = vowel
    
    # Confidence threshold - if too far from any vowel, mark uncertain
    if best_dist > 4.0:
        return f"({best}?)"
    return best


# ── Consonant Manner Classification ────────────────────────────────

def classify_consonant(voicing_ratio: float, friction_ratio: float, 
                       centroid: float, duration_ms: float,
                       intensity_drop: float) -> str:
    """Classify consonant manner from acoustic features."""
    
    is_voiced = voicing_ratio > 0.5
    has_friction = friction_ratio > 0.2
    voice_tag = "V" if is_voiced else "U"
    
    if intensity_drop > 15:  # Big intensity dip = stop closure
        if has_friction and duration_ms > 40:
            # Affricate: stop + friction
            if 2500 < centroid < 5500:
                return f"{voice_tag}:tɬ" if not is_voiced else f"{voice_tag}:dɮ"
            elif centroid > 5000:
                return f"{voice_tag}:tʃ" if not is_voiced else f"{voice_tag}:dʒ"
            else:
                return f"{voice_tag}:ts" if not is_voiced else f"{voice_tag}:dz"
        else:
            # Pure stop
            if centroid < 2000:
                return f"{voice_tag}:stop-velar"   # k/g
            elif centroid < 3500:
                return f"{voice_tag}:stop-alveolar" # t/d
            else:
                return f"{voice_tag}:stop-labial"   # p/b
    
    elif has_friction:
        # Fricative
        if centroid > 6000:
            return f"{voice_tag}:s/ʃ"
        elif 3000 < centroid < 6000:
            return f"{voice_tag}:ɬ/x" if not is_voiced else f"{voice_tag}:ɮ/ɣ"
        else:
            return f"{voice_tag}:h/ɦ"
    
    elif is_voiced and friction_ratio < 0.1:
        # Sonorant
        return "V:sonorant"  # could be nasal, lateral, or approximant
    
    return f"{voice_tag}:?"


# ── Frame-by-frame Acoustic Analysis ──────────────────────────────

def analyze_segment_acoustic(snd: parselmouth.Sound, t_start: float, 
                              t_end: float) -> dict:
    """
    Full Parselmouth acoustic analysis of a segment.
    Returns structured data with frame-by-frame transliteration.
    """
    duration = t_end - t_start
    if duration < 0.03 or duration > 30.0:
        return {"error": "segment too short/long", "frames": []}
    
    try:
        window = snd.extract_part(t_start, t_end, 
                                   parselmouth.WindowShape.RECTANGULAR, 1.0, False)
    except Exception as exc:
        return {"error": str(exc), "frames": []}
    
    if window.get_total_duration() < 0.03:
        return {"error": "extracted window too short", "frames": []}
    
    # ── Extract all features ──
    
    # Pitch (voicing)
    try:
        pitch = window.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    except Exception:
        pitch = None
    
    # Intensity
    try:
        intensity = window.to_intensity(minimum_pitch=100, time_step=0.01)
    except Exception:
        intensity = None
    
    # Formants (for vowel ID)
    try:
        formant = window.to_formant_burg(time_step=0.01, max_number_of_formants=5)
    except Exception:
        formant = None
    
    # Spectrum for each sub-window (for consonant classification)
    # We'll do this per-frame below
    
    # ── Build frame sequence ──
    
    frame_step = 0.02  # 20ms frames (good for phone-level)
    n_frames = max(1, int(duration / frame_step))
    
    frames = []
    prev_intensity_val = None
    
    for fi in range(n_frames):
        t = fi * frame_step + window.get_start_time()
        t_abs = t_start + fi * frame_step
        
        frame = {"t": round(t_abs, 3)}
        
        # Voicing
        if pitch is not None:
            try:
                f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
                frame["f0"] = round(f0, 1) if f0 and f0 > 0 else 0
                frame["voiced"] = f0 is not None and f0 > 0
            except Exception:
                frame["f0"] = 0
                frame["voiced"] = False
        else:
            frame["f0"] = 0
            frame["voiced"] = False
        
        # Intensity
        if intensity is not None:
            try:
                int_val = call(intensity, "Get value at time", t, "Cubic")
                frame["intensity"] = round(int_val, 1) if int_val else 0
                
                # Intensity drop from previous frame
                if prev_intensity_val is not None and int_val:
                    frame["int_drop"] = round(prev_intensity_val - int_val, 1)
                else:
                    frame["int_drop"] = 0
                prev_intensity_val = int_val
            except Exception:
                frame["intensity"] = 0
                frame["int_drop"] = 0
        
        # Formants (only meaningful for voiced frames)
        if formant is not None and frame.get("voiced"):
            try:
                f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                if f1 and f2 and f1 > 100 and f2 > 200:
                    frame["f1"] = round(f1, 0)
                    frame["f2"] = round(f2, 0)
                    frame["vowel"] = formant_to_vowel(f1, f2)
                else:
                    frame["f1"] = 0
                    frame["f2"] = 0
                    frame["vowel"] = "-"
            except Exception:
                frame["f1"] = 0
                frame["f2"] = 0
                frame["vowel"] = "-"
        else:
            frame["vowel"] = "-"
        
        # Spectral analysis for unvoiced frames (consonant classification)
        if not frame.get("voiced"):
            frame_start = t
            frame_end = min(t + frame_step, window.get_end_time())
            if frame_end - frame_start > 0.005:
                try:
                    sub = window.extract_part(frame_start, frame_end,
                                              parselmouth.WindowShape.HAMMING, 1.0, False)
                    spectrum = sub.to_spectrum()
                    total_e = call(spectrum, "Get band energy", 0, 0)
                    if total_e > 0:
                        friction_e = call(spectrum, "Get band energy", 3000, 8000)
                        centroid = call(spectrum, "Get centre of gravity...", 1)
                        frame["friction_ratio"] = round(friction_e / total_e, 3)
                        frame["centroid"] = round(centroid, 0)
                    else:
                        frame["friction_ratio"] = 0
                        frame["centroid"] = 0
                except Exception:
                    frame["friction_ratio"] = 0
                    frame["centroid"] = 0
            else:
                frame["friction_ratio"] = 0
                frame["centroid"] = 0
        
        frames.append(frame)
    
    # ── Build transliteration string ──
    transliteration = build_transliteration(frames)
    
    # ── Summary features ──
    voiced_frames = [f for f in frames if f.get("voiced")]
    unvoiced_frames = [f for f in frames if not f.get("voiced")]
    
    summary = {
        "duration_ms": round(duration * 1000, 0),
        "n_frames": len(frames),
        "voiced_ratio": round(len(voiced_frames) / max(1, len(frames)), 2),
        "mean_f0": round(np.mean([f["f0"] for f in voiced_frames]), 1) if voiced_frames else 0,
        "f0_range": round(
            max([f["f0"] for f in voiced_frames]) - min([f["f0"] for f in voiced_frames]), 1
        ) if len(voiced_frames) > 1 else 0,
    }
    
    return {
        "summary": summary,
        "frames": frames,
        "transliteration": transliteration,
    }


def build_transliteration(frames: list[dict]) -> str:
    """
    Build a pseudo-phonemic transliteration from acoustic frames.
    
    Groups consecutive frames with similar features into "phones":
    - Voiced + formants → vowel symbol
    - Unvoiced + high friction → fricative/affricate
    - Intensity dip → stop boundary
    - Voiced + low formants → sonorant
    
    Output format: space-separated acoustic phones with annotations.
    Example: "V:a V:i U:tɬ V:a U:s V:o"
    """
    if not frames:
        return "(empty)"
    
    phones = []
    current_phone = None
    current_count = 0
    
    for frame in frames:
        if frame.get("voiced"):
            vowel = frame.get("vowel", "-")
            if vowel and vowel != "-":
                phone = f"V:{vowel}"
            else:
                phone = "V:~"  # voiced but no clear vowel → sonorant
        else:
            fr = frame.get("friction_ratio", 0)
            centroid = frame.get("centroid", 0)
            int_drop = frame.get("int_drop", 0)
            
            if int_drop > 10 and fr > 0.2 and 2500 < centroid < 5500:
                phone = "U:tɬ!"  # strong tɬ candidate
            elif int_drop > 10 and fr > 0.2 and centroid > 5000:
                phone = "U:tʃ"
            elif int_drop > 10:
                phone = "U:stop"
            elif fr > 0.3:
                if centroid > 5500:
                    phone = "U:s/ʃ"
                elif centroid > 3000:
                    phone = "U:ɬ/x"
                else:
                    phone = "U:h"
            elif fr > 0.1:
                phone = "U:~"  # weak friction → aspiration or transition
            else:
                phone = "_"  # silence/pause
        
        # Group consecutive identical phones
        if phone == current_phone:
            current_count += 1
        else:
            if current_phone is not None:
                if current_count > 1:
                    phones.append(f"{current_phone}×{current_count}")
                else:
                    phones.append(current_phone)
            current_phone = phone
            current_count = 1
    
    # Last phone
    if current_phone is not None:
        if current_count > 1:
            phones.append(f"{current_phone}×{current_count}")
        else:
            phones.append(current_phone)
    
    return " ".join(phones)


# ── SRT Parsing ───────────────────────────────────────────────────

def parse_srt_timestamp(ts_str: str) -> float:
    match = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)', ts_str.strip())
    if not match:
        return 0.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt_segments(srt_path: str) -> list[dict]:
    """Parse SRT with allo/w2v2/fused lines."""
    text = Path(srt_path).read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")
    segments = []
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        
        cue = lines[0].strip()
        
        ts_match = re.match(r'(.+?)\s*-->\s*(.+?)$', lines[1])
        if not ts_match:
            continue
        
        t_start = parse_srt_timestamp(ts_match.group(1))
        t_end = parse_srt_timestamp(ts_match.group(2))
        
        content = "\n".join(lines[2:])
        
        # Language tag
        lang_match = re.search(r'\[(\w+)\|', content)
        lang = lang_match.group(1) if lang_match else "UNK"
        
        # Speaker
        spk_match = re.search(r'\|(\w+)\]', content)
        speaker = spk_match.group(1) if spk_match else ""
        
        # LLM text
        llm_match = re.search(r'\[LLM\]\s*(.+?)(?:\n|$)', content)
        llm_text = llm_match.group(1).strip() if llm_match else ""
        # Also catch non-LLM text (SPA segments often have plain text)
        if not llm_text:
            first_line = lines[2] if len(lines) > 2 else ""
            # Strip lang tag
            plain = re.sub(r'\[\w+\|\w+\]\s*', '', first_line).strip()
            if plain and not plain.startswith("♫"):
                llm_text = plain
        
        # Backend outputs
        allo = ""
        w2v2 = ""
        fused = ""
        for line in lines[2:]:
            if line.startswith("♫allo:"):
                allo = line[6:].strip()
            elif line.startswith("♫w2v2:"):
                w2v2 = line[6:].strip()
            elif line.startswith("♫fused:"):
                fused = line[7:].strip()
        
        segments.append({
            "cue": cue,
            "t_start": t_start,
            "t_end": t_end,
            "lang": lang,
            "speaker": speaker,
            "llm_text": llm_text,
            "allo": allo,
            "w2v2": w2v2,
            "fused": fused,
        })
    
    return segments


# ── Agreement Analysis ────────────────────────────────────────────

def compute_agreement(seg: dict, acoustic: dict) -> dict:
    """
    Compare backends against Parselmouth acoustic evidence.
    Returns agreement metrics and conflict flags.
    """
    trans = acoustic.get("transliteration", "")
    summary = acoustic.get("summary", {})
    
    allo = seg.get("allo", "")
    w2v2 = seg.get("w2v2", "")
    fused = seg.get("fused", "")
    
    # Check for key acoustic markers
    has_acoustic_tl = "U:tɬ!" in trans
    has_ipa_tl = "tɬ" in fused or "tɬ" in allo
    
    # Voicing agreement
    voiced_ratio = summary.get("voiced_ratio", 0)
    
    # Count voiced phones in backends
    voiced_phones = set("aeioubdgɡʒɮmɱnɴŋɲɾrlwjʋ")
    
    def count_voiced_ratio(ipa_str):
        phones = ipa_str.split()
        if not phones:
            return 0
        voiced = sum(1 for p in phones if any(c in voiced_phones for c in p))
        return voiced / len(phones)
    
    allo_voiced = count_voiced_ratio(allo)
    w2v2_voiced = count_voiced_ratio(w2v2)
    
    # Agreement scores
    allo_voice_agree = 1.0 - abs(voiced_ratio - allo_voiced)
    w2v2_voice_agree = 1.0 - abs(voiced_ratio - w2v2_voiced)
    
    conflicts = []
    
    if has_acoustic_tl and not has_ipa_tl:
        conflicts.append("ACOUSTIC_TL_MISSED: Parselmouth hears tɬ, backends don't")
    if has_ipa_tl and not has_acoustic_tl:
        conflicts.append("IPA_TL_UNCONFIRMED: Backends claim tɬ, no acoustic evidence")
    
    if voiced_ratio < 0.3 and (allo_voiced > 0.7 or w2v2_voiced > 0.7):
        conflicts.append("VOICING_MISMATCH: Segment mostly unvoiced but backends claim voiced phones")
    
    # tɬ in SPA = misclassification
    if (has_acoustic_tl or has_ipa_tl) and seg.get("lang") == "SPA":
        conflicts.append("⚠️ TL_IN_SPA: tɬ impossible in Spanish → misclassified NAH")
    
    return {
        "allo_voice_agree": round(allo_voice_agree, 2),
        "w2v2_voice_agree": round(w2v2_voice_agree, 2),
        "has_acoustic_tl": has_acoustic_tl,
        "has_ipa_tl": has_ipa_tl,
        "voiced_ratio": voiced_ratio,
        "conflicts": conflicts,
    }


# ── Output Formatters ─────────────────────────────────────────────

def format_comparison_md(seg: dict, acoustic: dict, agreement: dict) -> str:
    """Format one segment as markdown comparison block."""
    lines = []
    
    lang = seg["lang"]
    conflict_marker = " ⚠️" if agreement.get("conflicts") else ""
    
    lines.append(f"### Cue {seg['cue']} [{lang}] {seg['t_start']:.1f}–{seg['t_end']:.1f}s{conflict_marker}")
    lines.append("")
    
    if seg.get("llm_text"):
        lines.append(f"**LLM:** {seg['llm_text']}")
    
    lines.append("")
    lines.append("| Layer | Output |")
    lines.append("|-------|--------|")
    lines.append(f"| **Allosaurus** | `{seg['allo']}` |")
    lines.append(f"| **wav2vec2** | `{seg['w2v2']}` |")
    lines.append(f"| **Fused** | `{seg['fused']}` |")
    lines.append(f"| **Parselmouth** | `{acoustic['transliteration']}` |")
    
    lines.append("")
    
    # Summary
    summ = acoustic.get("summary", {})
    lines.append(f"**Acoustics:** {summ.get('duration_ms', 0):.0f}ms, "
                 f"voiced={summ.get('voiced_ratio', 0):.0%}, "
                 f"F0={summ.get('mean_f0', 0):.0f}Hz "
                 f"(range {summ.get('f0_range', 0):.0f}Hz)")
    
    # Agreement
    lines.append(f"**Voicing-Agreement:** Allo={agreement['allo_voice_agree']:.0%}, "
                 f"w2v2={agreement['w2v2_voice_agree']:.0%}")
    
    # tɬ status
    if agreement.get("has_acoustic_tl") or agreement.get("has_ipa_tl"):
        tl_status = []
        if agreement["has_acoustic_tl"]:
            tl_status.append("✅ acoustically confirmed")
        if agreement["has_ipa_tl"]:
            tl_status.append("present in IPA")
        lines.append(f"**tɬ-Signal:** {', '.join(tl_status)}")
    
    # Conflicts
    if agreement.get("conflicts"):
        lines.append("")
        for c in agreement["conflicts"]:
            lines.append(f"> **{c}**")
    
    lines.append("")
    
    # Parselmouth notation explanation
    return "\n".join(lines)


def format_legend() -> str:
    """Parselmouth transliteration legend."""
    return """## Parselmouth Transliteration Legend

The Parselmouth row is **not phoneme recognition** — it describes the
**acoustic physics** of the signal in 20ms frames:

| Symbol | Meaning | Acoustic Evidence |
|--------|---------|-------------------|
| `V:a/e/i/o/u` | Voiced + formant position → vowel | F0 > 0, F1/F2 mapped |
| `V:~` | Voiced, but no clear vowel | F0 > 0, formants unclear → sonorant (n/m/l/r/w/j) |
| `U:tɬ!` | **Unvoiced lateral affricate** | Stop burst + friction 3-5kHz, no voicing |
| `U:tʃ` | Unvoiced postalveolar affricate | Stop burst + friction > 5kHz |
| `U:stop` | Unvoiced stop (p/t/k) | Intensity dip > 10dB |
| `U:s/ʃ` | Unvoiced sibilant | Friction ratio > 0.3, centroid > 5.5kHz |
| `U:ɬ/x` | Unvoiced lateral/velar fricative | Friction 3-6kHz (lower than s/ʃ) |
| `U:h` | Aspiration/glottal fricative | Weak friction < 3kHz |
| `U:~` | Weak friction (transition) | Friction 0.1-0.3 |
| `_` | Silence/pause | No energy |
| `×N` | N consecutive identical frames | e.g. `V:a×3` = 60ms /a/ |

### Interpretation

- **Where Parselmouth says V:a and Allo says `a` → confirmed** ✅
- **Where Parselmouth says U:tɬ! → hard NAH marker**, regardless of what backends say
- **Where Parselmouth says V:~ → sonorant**: Allo/w2v2 must determine if n/m/l/r
- **Voiced ratio** measures the proportion of voiced frames — should roughly match
  the proportion of voiced phones in the backends
- **Conflicts** show where backends and acoustics diverge
"""


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parselmouth acoustic transliteration vs phoneme backends")
    parser.add_argument("--audio", required=True, help="Audio file (wav)")
    parser.add_argument("--srt", required=True, help="SRT with allo/w2v2/fused")
    parser.add_argument("--output", default=None, help="Output .md file")
    parser.add_argument("--segments", default=None, 
                       help="Cue range, e.g. '10-20' or '12,45,78'")
    parser.add_argument("--lang", default=None,
                       help="Filter by language tag: NAH, SPA, MAY, OTH")
    parser.add_argument("--limit", type=int, default=30,
                       help="Max segments to analyze (default 30)")
    parser.add_argument("--json-out", default=None,
                       help="Also write raw JSON data")
    args = parser.parse_args()
    
    print(f"Loading audio: {args.audio}")
    snd = parselmouth.Sound(args.audio)
    print(f"  Duration: {snd.get_total_duration():.1f}s, SR: {snd.sampling_frequency}Hz")
    
    segments = parse_srt_segments(args.srt)
    print(f"Parsed {len(segments)} segments")
    
    # Filter
    if args.lang:
        segments = [s for s in segments if s["lang"] == args.lang.upper()]
        print(f"  Filtered to {len(segments)} {args.lang.upper()} segments")
    
    if args.segments:
        if "-" in args.segments:
            lo, hi = args.segments.split("-")
            cue_range = set(str(i) for i in range(int(lo), int(hi) + 1))
        else:
            cue_range = set(args.segments.split(","))
        segments = [s for s in segments if s["cue"] in cue_range]
        print(f"  Selected {len(segments)} cues")
    
    if len(segments) > args.limit:
        print(f"  Limiting to {args.limit} segments")
        segments = segments[:args.limit]
    
    # Analyze
    results = []
    md_blocks = []
    
    md_blocks.append("# Parselmouth vs Allosaurus vs wav2vec2\n")
    md_blocks.append(f"**Audio:** `{args.audio}`  ")
    md_blocks.append(f"**SRT:** `{args.srt}`  ")
    if args.lang:
        md_blocks.append(f"**Filter:** {args.lang}")
    md_blocks.append(f"**Segments:** {len(segments)}\n")
    md_blocks.append(format_legend())
    md_blocks.append("---\n")
    
    for i, seg in enumerate(segments):
        print(f"\r  Analyzing {i+1}/{len(segments)}: cue {seg['cue']} [{seg['lang']}]  ", 
              end="", flush=True)
        
        acoustic = analyze_segment_acoustic(snd, seg["t_start"], seg["t_end"])
        agreement = compute_agreement(seg, acoustic)
        
        md_blocks.append(format_comparison_md(seg, acoustic, agreement))
        
        results.append({
            "segment": seg,
            "acoustic": {
                "summary": acoustic.get("summary"),
                "transliteration": acoustic.get("transliteration"),
            },
            "agreement": agreement,
        })
    
    print(f"\r  Done. {len(results)} segments analyzed.                    ")
    
    # Summary stats
    conflicts_total = sum(len(r["agreement"]["conflicts"]) for r in results)
    tl_acoustic = sum(1 for r in results if r["agreement"]["has_acoustic_tl"])
    tl_ipa = sum(1 for r in results if r["agreement"]["has_ipa_tl"])
    
    avg_allo_agree = np.mean([r["agreement"]["allo_voice_agree"] for r in results])
    avg_w2v2_agree = np.mean([r["agreement"]["w2v2_voice_agree"] for r in results])
    
    md_blocks.append("---\n")
    md_blocks.append("## Summary\n")
    md_blocks.append(f"| Metric | Value |")
    md_blocks.append(f"|--------|-------|")
    md_blocks.append(f"| Segments analyzed | {len(results)} |")
    md_blocks.append(f"| Avg voicing agreement Allo | **{avg_allo_agree:.0%}** |")
    md_blocks.append(f"| Avg voicing agreement w2v2 | **{avg_w2v2_agree:.0%}** |")
    md_blocks.append(f"| tɬ acoustically confirmed | {tl_acoustic} |")
    md_blocks.append(f"| tɬ only in IPA | {tl_ipa - tl_acoustic if tl_ipa > tl_acoustic else 0} |")
    md_blocks.append(f"| Total conflicts | {conflicts_total} |")

    # Winner analysis
    md_blocks.append("")
    md_blocks.append("### Which backend is closer?")
    md_blocks.append("")
    if avg_allo_agree > avg_w2v2_agree + 0.05:
        md_blocks.append(f"**→ Allosaurus** is closer to the acoustics for voicing "
                        f"({avg_allo_agree:.0%} vs {avg_w2v2_agree:.0%})")
    elif avg_w2v2_agree > avg_allo_agree + 0.05:
        md_blocks.append(f"**→ wav2vec2** is closer to the acoustics for voicing "
                        f"({avg_w2v2_agree:.0%} vs {avg_allo_agree:.0%})")
    else:
        md_blocks.append(f"**→ Tie** for voicing "
                        f"(Allo {avg_allo_agree:.0%}, w2v2 {avg_w2v2_agree:.0%})")

    md_blocks.append("")
    md_blocks.append("### Recommendation")
    md_blocks.append("")
    md_blocks.append("Parselmouth provides **acoustic ground truth** for:")
    md_blocks.append("1. **tɬ detection**: Hard NAH marker, independent of ML models")
    md_blocks.append("2. **Voicing validation**: Where voiced/unvoiced disagrees with backend → backend error")
    md_blocks.append("3. **Vowel approximation**: F1/F2-based, but coarser than trained models")
    md_blocks.append("4. **Fricative classification**: Centroid distinguishes s/ʃ from ɬ/x")
    md_blocks.append("")
    md_blocks.append("Allosaurus and w2v2 are better at **consonant identification** — ")
    md_blocks.append("Parselmouth can detect manner (stop/fricative/affricate), but not place.")
    md_blocks.append("**Fusion** should use Parselmouth as validator, not replacement.")
    
    # Output
    md_text = "\n".join(md_blocks)
    
    if args.output:
        Path(args.output).write_text(md_text, encoding="utf-8")
        print(f"\nSaved to {args.output}")
    else:
        print(md_text)
    
    if args.json_out:
        json_data = {
            "audio": args.audio,
            "srt": args.srt,
            "summary": {
                "n_segments": len(results),
                "avg_allo_voice_agree": round(avg_allo_agree, 3),
                "avg_w2v2_voice_agree": round(avg_w2v2_agree, 3),
                "tl_acoustic": tl_acoustic,
                "tl_ipa_only": tl_ipa - tl_acoustic if tl_ipa > tl_acoustic else 0,
                "conflicts": conflicts_total,
            },
            "segments": results,
        }
        Path(args.json_out).write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON saved to {args.json_out}")


if __name__ == "__main__":
    main()

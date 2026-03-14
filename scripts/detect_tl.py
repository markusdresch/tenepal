#!/usr/bin/env python3
"""
tɬ (voiceless lateral affricate) detector using Parselmouth.

Acoustic signature:
- Stop burst (t) followed by voiceless lateral friction (ɬ)
- Unvoiced: no periodic signal in friction phase
- Friction: broadband noise, energy peak 3-5 kHz (lower than tʃ which peaks 5-8 kHz)
- Duration: friction phase 60-120ms (much longer than tɾ tap at 20-40ms)

Usage:
    python scripts/detect_tl.py --audio vocals.wav --srt file.srt [--visualize]
    
Reads SRT timestamps, extracts audio windows, checks for tɬ presence.
Reports which segments contain tɬ regardless of language tag.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    print("ERROR: pip install praat-parselmouth")
    sys.exit(1)


def parse_srt_timestamp(ts_str: str) -> float:
    """Parse SRT timestamp to seconds."""
    match = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)', ts_str.strip())
    if not match:
        return 0.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def detect_tl_in_window(snd: parselmouth.Sound, t_start: float, t_end: float,
                         min_friction_dur: float = 0.04,
                         voicing_threshold: float = 0.3,
                         friction_ratio_threshold: float = 0.25) -> list[dict]:
    """
    Detect tɬ candidates in an audio window.
    
    Strategy:
    1. Find intensity dips (potential stop closures)
    2. After each dip, measure:
       - Voicing ratio (should be low for tɬ)
       - High-frequency friction energy (should be high for tɬ)
       - Duration of unvoiced friction phase
    3. Distinguish from tʃ via spectral centroid (tɬ lower than tʃ)
    
    Returns list of detected tɬ with timestamps and confidence.
    """
    duration = t_end - t_start
    if duration < 0.05 or duration > 30.0:
        return []
    
    try:
        window = snd.extract_part(t_start, t_end, parselmouth.WindowShape.RECTANGULAR, 1.0, False)
    except Exception:
        return []
    
    if window.get_total_duration() < 0.05:
        return []
    
    detections = []
    
    # Get intensity contour to find stop bursts
    try:
        intensity = window.to_intensity(minimum_pitch=100, time_step=0.005)
    except Exception:
        return []
    
    times = intensity.xs()
    values = [intensity.get_value(t) for t in times]
    
    if not values or all(v is None for v in values):
        return []
    
    # Replace None with 0
    values = [v if v is not None else 0.0 for v in values]
    mean_int = np.mean(values)
    
    # Find intensity dips (potential stop closures) followed by rises
    for i in range(1, len(values) - 2):
        # Look for dip: value drops below mean - 6dB, then rises
        if values[i] < mean_int - 6 and values[i+1] > values[i]:
            burst_time = times[i] + t_start  # absolute time
            
            # Analyze the 100ms AFTER the burst
            analysis_start = times[i] + window.get_start_time()
            analysis_end = min(analysis_start + 0.12, window.get_end_time())
            
            if analysis_end - analysis_start < min_friction_dur:
                continue
            
            try:
                post_burst = window.extract_part(
                    analysis_start, analysis_end,
                    parselmouth.WindowShape.HAMMING, 1.0, False
                )
            except Exception:
                continue
            
            if post_burst.get_total_duration() < min_friction_dur:
                continue
            
            # === MEASURE 1: Voicing ratio ===
            try:
                pitch = post_burst.to_pitch(time_step=0.005, pitch_floor=75, pitch_ceiling=500)
                n_frames = call(pitch, "Get number of frames")
                if n_frames > 0:
                    voiced = sum(1 for fi in range(1, n_frames + 1) 
                                if call(pitch, "Get value in frame", fi, "Hertz") > 0)
                    voicing_ratio = voiced / n_frames
                else:
                    voicing_ratio = 0.5  # uncertain
            except Exception:
                voicing_ratio = 0.5
            
            # === MEASURE 2: Friction energy (high freq ratio) ===
            try:
                spectrum = post_burst.to_spectrum()
                # Total energy
                total_e = call(spectrum, "Get band energy", 0, 0)  # 0,0 = full range
                # Friction band: 3-8 kHz
                friction_e = call(spectrum, "Get band energy", 3000, 8000)
                # Low band: 0-2 kHz (voiced formants live here)
                low_e = call(spectrum, "Get band energy", 0, 2000)
                
                if total_e > 0:
                    friction_ratio = friction_e / total_e
                    low_ratio = low_e / total_e
                else:
                    continue
            except Exception:
                continue
            
            # === MEASURE 3: Spectral centroid (tɬ ~4kHz, tʃ ~6kHz) ===
            try:
                centroid = call(spectrum, "Get centre of gravity...", 1)
            except Exception:
                centroid = 5000  # default middle
            
            # === DECISION ===
            is_unvoiced = voicing_ratio < voicing_threshold
            has_friction = friction_ratio > friction_ratio_threshold
            is_lateral = 2500 < centroid < 5500  # tɬ range (tʃ would be 5000-8000)
            is_not_sibilant = centroid < 6000  # /s/ and tʃ are higher
            
            # Duration of unvoiced phase
            unvoiced_dur = analysis_end - analysis_start
            long_enough = unvoiced_dur > min_friction_dur
            
            if is_unvoiced and has_friction and long_enough:
                # Distinguish tɬ from tʃ
                if is_lateral:
                    phone = "tɬ"
                    confidence = min(1.0, (1 - voicing_ratio) * friction_ratio * 2)
                elif centroid > 5500:
                    phone = "tʃ"  # report but mark differently
                    confidence = min(1.0, (1 - voicing_ratio) * friction_ratio * 1.5)
                else:
                    phone = "tɬ?"  # uncertain
                    confidence = min(1.0, (1 - voicing_ratio) * friction_ratio)
                
                detections.append({
                    "time": round(burst_time, 3),
                    "phone": phone,
                    "confidence": round(confidence, 3),
                    "voicing_ratio": round(voicing_ratio, 3),
                    "friction_ratio": round(friction_ratio, 3),
                    "centroid_hz": round(centroid, 0),
                    "duration_ms": round(unvoiced_dur * 1000, 1),
                })
    
    return detections


def parse_srt(srt_path: str) -> list[dict]:
    """Parse SRT file, return segments with timestamps and metadata."""
    text = Path(srt_path).read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")
    segments = []
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        
        # Timestamp line
        ts_match = re.match(r'(.+?)\s*-->\s*(.+?)$', lines[1])
        if not ts_match:
            continue
        
        t_start = parse_srt_timestamp(ts_match.group(1))
        t_end = parse_srt_timestamp(ts_match.group(2))
        
        content = "\n".join(lines[2:])
        
        # Language tag
        lang_match = re.search(r'\[(\w+)\|', content)
        lang = lang_match.group(1) if lang_match else "UNK"
        
        # Fused IPA
        fused_match = re.search(r'♫fused:\s*(.+)', content)
        fused = fused_match.group(1).strip() if fused_match else ""
        
        # IPA-based tɬ (from backends)
        has_tl_ipa = "tɬ" in fused or ("tl" in fused and "t l" in fused)
        
        segments.append({
            "cue": lines[0].strip(),
            "t_start": t_start,
            "t_end": t_end,
            "lang": lang,
            "fused": fused,
            "has_tl_ipa": has_tl_ipa,
        })
    
    return segments


def main():
    parser = argparse.ArgumentParser(description="Detect tɬ in audio segments")
    parser.add_argument("--audio", required=True, help="Audio file (wav/mp3)")
    parser.add_argument("--srt", required=True, help="SRT file with IPA")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--visualize", action="store_true", help="Print detailed table")
    parser.add_argument("--all-segments", action="store_true", 
                       help="Scan all segments (default: only those with tɬ in IPA)")
    args = parser.parse_args()
    
    print(f"Loading audio: {args.audio}")
    snd = parselmouth.Sound(args.audio)
    print(f"  Duration: {snd.get_total_duration():.1f}s, SR: {snd.sampling_frequency}Hz")
    
    segments = parse_srt(args.srt)
    print(f"Parsed {len(segments)} segments from SRT")
    
    results = []
    tl_by_lang = {}
    
    for seg in segments:
        # Skip unless we want all or IPA says tɬ
        if not args.all_segments and not seg["has_tl_ipa"]:
            continue
        
        detections = detect_tl_in_window(snd, seg["t_start"], seg["t_end"])
        
        tl_detections = [d for d in detections if d["phone"] in ("tɬ", "tɬ?")]
        
        if tl_detections or seg["has_tl_ipa"]:
            result = {
                "cue": seg["cue"],
                "t_start": seg["t_start"],
                "t_end": seg["t_end"],
                "lang": seg["lang"],
                "tl_in_ipa": seg["has_tl_ipa"],
                "tl_acoustic": len(tl_detections),
                "detections": detections,
                "fused": seg["fused"][:60],
            }
            results.append(result)
            
            lang = seg["lang"]
            if lang not in tl_by_lang:
                tl_by_lang[lang] = {"ipa_only": 0, "acoustic_only": 0, "both": 0, "total": 0}
            tl_by_lang[lang]["total"] += 1
            
            if seg["has_tl_ipa"] and tl_detections:
                tl_by_lang[lang]["both"] += 1
            elif seg["has_tl_ipa"]:
                tl_by_lang[lang]["ipa_only"] += 1
            elif tl_detections:
                tl_by_lang[lang]["acoustic_only"] += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"tɬ DETECTION RESULTS")
    print(f"{'='*80}")
    print(f"\nSegments with tɬ signal: {len(results)}")
    print(f"\nBy language tag:")
    print(f"  {'Lang':<6} {'IPA only':<12} {'Acoustic':<12} {'Both':<8} {'Total'}")
    print(f"  {'-'*50}")
    for lang in sorted(tl_by_lang.keys()):
        d = tl_by_lang[lang]
        print(f"  {lang:<6} {d['ipa_only']:<12} {d['acoustic_only']:<12} {d['both']:<8} {d['total']}")
    
    if args.visualize:
        print(f"\n{'='*80}")
        print(f"DETAILED RESULTS")
        print(f"{'='*80}")
        for r in results:
            marker = "⚠️" if r["lang"] == "SPA" else "  "
            print(f"\n{marker} Cue {r['cue']} [{r['lang']}] {r['t_start']:.1f}-{r['t_end']:.1f}s")
            print(f"   IPA tɬ: {'YES' if r['tl_in_ipa'] else 'no'}  |  Acoustic tɬ: {r['tl_acoustic']}")
            print(f"   Fused: {r['fused']}")
            for d in r["detections"]:
                print(f"   → {d['phone']} @{d['time']:.3f}s  conf={d['confidence']:.2f}  "
                      f"voicing={d['voicing_ratio']:.2f}  friction={d['friction_ratio']:.2f}  "
                      f"centroid={d['centroid_hz']:.0f}Hz  dur={d['duration_ms']:.0f}ms")
    
    # Save
    if args.output:
        out = {
            "audio": args.audio,
            "srt": args.srt,
            "summary": tl_by_lang,
            "segments": results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nSaved to {args.output}")
    
    # Key insight
    spa_tl = tl_by_lang.get("SPA", {})
    if spa_tl.get("total", 0) > 0:
        print(f"\n⚠️  {spa_tl['total']} SPA-tagged segments contain tɬ signal!")
        print(f"   These are almost certainly misclassified Nahuatl.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test overlap-aware voice separation pipeline.

Flow:
1. Pyannote diarization identifies speaker segments + overlaps
2. For each overlap region: SepFormer splits into 2 sources
3. Match separated sources to known speakers via embedding similarity
4. Re-run language detection on cleaned audio

Usage:
    python scripts/test_overlap_separation.py
"""

import subprocess
import json
import re
from pathlib import Path
import tempfile


def parse_srt_segments(srt_path: Path, start_s: float = 0, end_s: float = float('inf')):
    """Extract segments from SRT file."""
    segments = []
    current_time = None

    with open(srt_path) as f:
        for line in f:
            line = line.strip()
            if '-->' in line:
                m = re.match(r'(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)', line)
                if m:
                    h1,m1,s1,ms1,h2,m2,s2,ms2 = m.groups()
                    start = int(h1)*3600 + int(m1)*60 + int(s1) + int(ms1)/1000
                    end = int(h2)*3600 + int(m2)*60 + int(s2) + int(ms2)/1000
                    current_time = (start, end)
            elif line.startswith('[') and current_time:
                m = re.match(r'\[(\w+)\|(\w+)\]', line)
                if m:
                    lang, speaker = m.groups()
                    if current_time[0] >= start_s and current_time[1] <= end_s:
                        segments.append({
                            'start': current_time[0],
                            'end': current_time[1],
                            'speaker': speaker,
                            'lang': lang
                        })
                current_time = None

    return segments


def find_overlaps(segments: list[dict]) -> list[dict]:
    """Find speaker overlaps in segments."""
    overlaps = []
    for i, s1 in enumerate(segments):
        for s2 in segments[i+1:]:
            if s1['speaker'] != s2['speaker']:
                ov_start = max(s1['start'], s2['start'])
                ov_end = min(s1['end'], s2['end'])
                if ov_end > ov_start:
                    overlaps.append({
                        'start': ov_start,
                        'end': ov_end,
                        'duration': ov_end - ov_start,
                        'speakers': sorted([s1['speaker'], s2['speaker']]),
                        'langs': [s1['lang'], s2['lang']],
                    })

    # Dedupe and merge nearby overlaps
    overlaps.sort(key=lambda x: x['start'])
    merged = []
    for ov in overlaps:
        if merged and ov['start'] - merged[-1]['end'] < 0.5:
            # Extend previous
            merged[-1]['end'] = max(merged[-1]['end'], ov['end'])
            merged[-1]['duration'] = merged[-1]['end'] - merged[-1]['start']
            merged[-1]['speakers'] = sorted(set(merged[-1]['speakers'] + ov['speakers']))
        else:
            merged.append(ov)

    return merged


def extract_clip(audio_path: Path, start_s: float, end_s: float, out_path: Path):
    """Extract audio clip using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(audio_path),
        '-ss', f'{start_s:.3f}',
        '-to', f'{end_s:.3f}',
        '-ar', '16000', '-ac', '1',
        str(out_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def main():
    # Config
    srt_path = Path("validation_video/Hernán-1-3.srt")
    audio_path = Path("validation_video/Hernán-1-3.vocals.wav")
    scene_start = 1220.0  # 20:20
    scene_end = 1240.0    # 20:40

    print(f"=== Overlap-Aware Voice Separation Test ===")
    print(f"Scene: {scene_start:.0f}s - {scene_end:.0f}s")
    print()

    # Step 1: Parse existing diarization
    segments = parse_srt_segments(srt_path, scene_start, scene_end)
    print(f"Step 1: {len(segments)} segments in scene")

    # Step 2: Find overlaps
    overlaps = find_overlaps(segments)
    print(f"Step 2: {len(overlaps)} overlap regions")

    for i, ov in enumerate(overlaps):
        print(f"  [{i+1}] {ov['start']:.1f}-{ov['end']:.1f}s ({ov['duration']:.2f}s)")
        print(f"      Speakers: {ov['speakers']}")
        print(f"      Languages: {ov['langs']}")

    if not overlaps:
        print("No overlaps to process.")
        return

    print()
    print("Step 3: For each overlap, would run SepFormer...")
    print("  (SepFormer already tested - produces 2 sources)")
    print()

    # Step 4: Proposed integration
    print("=== Proposed Pipeline Integration ===")
    print("""
1. In process_film(), after pyannote diarization:
   - Detect speaker overlaps from turn list

2. For each overlap > 0.3s:
   - Extract overlap audio segment
   - Call separate_voices_sepformer()
   - Get 2 separated sources

3. Match sources to known speakers:
   - Extract speaker embeddings from non-overlap regions
   - Compare separated source embeddings
   - Assign each source to closest known speaker

4. Replace overlap segment with separated segments:
   - Each source gets its own segment
   - Language detection runs on clean audio

5. Output: Clean per-speaker segments, no overlaps
""")

    # Summary stats
    total_overlap_time = sum(ov['duration'] for ov in overlaps)
    scene_duration = scene_end - scene_start
    print(f"Summary:")
    print(f"  Scene duration: {scene_duration:.0f}s")
    print(f"  Overlap time: {total_overlap_time:.1f}s ({100*total_overlap_time/scene_duration:.1f}%)")
    print(f"  Unique speaker pairs: {len(set(tuple(ov['speakers']) for ov in overlaps))}")


if __name__ == "__main__":
    main()

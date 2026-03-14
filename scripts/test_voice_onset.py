#!/usr/bin/env python3
"""Test voice onset trimming on E01 first 5 minutes.

Tests the trim_to_voice function on known problem segments:
- "Listos bien" (~34s) - known to have silence padding
- "¡Capitán!" (~37-39s)
- "¡Arcabuceros!" (~33s)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import soundfile as sf

from tenepal.voice_onset import trim_to_voice, compute_trim_stats


def test_trimming():
    """Test voice onset trimming on E01 vocals."""
    audio_path = PROJECT_ROOT / "validation_video/Hernán-1-1.vocals.wav"

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert to mono
    print(f"  Duration: {len(audio)/sr:.1f}s, SR: {sr}")

    # Test segments (approximate timestamps from E01)
    test_segments = [
        {"name": "Segment ~26s (short)", "start": 26.19, "end": 26.76},
        {"name": "Segment ~29s (OTH)", "start": 29.04, "end": 29.56},
        {"name": "Segment ~30s (OTH)", "start": 30.32, "end": 31.08},
        {"name": "Listos bien ~34s", "start": 34.34, "end": 35.05},
        {"name": "Segment ~37s", "start": 37.00, "end": 38.00},
        {"name": "Segment ~39s", "start": 39.00, "end": 40.00},
    ]

    print(f"\n{'='*70}")
    print(f"Testing {len(test_segments)} segments")
    print(f"{'='*70}")

    total_trimmed = 0
    total_start_trim_ms = 0
    total_end_trim_ms = 0

    for seg in test_segments:
        start = seg["start"]
        end = seg["end"]

        trimmed_start, trimmed_end, stats = trim_to_voice(
            audio, sr, start, end
        )

        if stats.get("trimmed"):
            total_trimmed += 1
            total_start_trim_ms += stats.get("trim_start_ms", 0)
            total_end_trim_ms += stats.get("trim_end_ms", 0)

        orig_dur = (end - start) * 1000
        trim_dur = (trimmed_end - trimmed_start) * 1000

        print(f"\n{seg['name']}")
        print(f"  Original:  {start:.3f}s - {end:.3f}s ({orig_dur:.0f}ms)")
        print(f"  Trimmed:   {trimmed_start:.3f}s - {trimmed_end:.3f}s ({trim_dur:.0f}ms)")

        if stats.get("trimmed"):
            print(f"  Trim:      {stats['trim_start_ms']:+d}ms start, {stats['trim_end_ms']:+d}ms end")
        else:
            print(f"  Status:    NOT TRIMMED ({stats.get('reason', 'unknown')})")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Segments trimmed: {total_trimmed}/{len(test_segments)}")
    if total_trimmed > 0:
        print(f"  Avg start trim:   {total_start_trim_ms/total_trimmed:.0f}ms")
        print(f"  Avg end trim:     {total_end_trim_ms/total_trimmed:.0f}ms")


def test_first_5_minutes():
    """Run trimming on all segments in first 5 minutes of E01."""
    import re

    srt_path = PROJECT_ROOT / "validation_video/Hernán-1-1.srt"
    audio_path = PROJECT_ROOT / "validation_video/Hernán-1-1.vocals.wav"

    if not srt_path.exists() or not audio_path.exists():
        print("Required files not found")
        return

    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Parse SRT for first 5 minutes
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r'\n\s*\n', text)

    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        ts_match = re.match(r'(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)', lines[1])
        if not ts_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, ts_match.groups())
        start = h1*3600 + m1*60 + s1 + ms1/1000
        end = h2*3600 + m2*60 + s2 + ms2/1000

        if start > 300:  # First 5 minutes only
            break

        segments.append({"start": start, "end": end, "line": lines[2] if len(lines) > 2 else ""})

    print(f"\nTesting {len(segments)} segments in first 5 minutes")

    trimmed_count = 0
    non_count = 0
    total_start_trim = 0
    total_end_trim = 0

    for seg in segments:
        trimmed_start, trimmed_end, stats = trim_to_voice(audio, sr, seg["start"], seg["end"])

        if stats.get("trimmed"):
            trimmed_count += 1
            total_start_trim += stats.get("trim_start_ms", 0)
            total_end_trim += stats.get("trim_end_ms", 0)

            # Check if trimmed to < 100ms (would become NON)
            if (trimmed_end - trimmed_start) < 0.1:
                non_count += 1

    print(f"\n{'='*70}")
    print(f"FIRST 5 MINUTES SUMMARY")
    print(f"{'='*70}")
    print(f"  Total segments:    {len(segments)}")
    print(f"  Segments trimmed:  {trimmed_count} ({100*trimmed_count/len(segments):.1f}%)")
    print(f"  Would become NON:  {non_count}")
    if trimmed_count > 0:
        print(f"  Avg start trim:    {total_start_trim/trimmed_count:.0f}ms")
        print(f"  Avg end trim:      {total_end_trim/trimmed_count:.0f}ms")


if __name__ == "__main__":
    test_trimming()
    print("\n" + "="*70 + "\n")
    test_first_5_minutes()

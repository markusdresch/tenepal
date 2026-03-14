#!/usr/bin/env python3
"""
clip_extractor.py - Interactive segment extraction for Tenepal debugging

Usage:
    ./clip_extractor.py                     # Interactive mode
    ./clip_extractor.py --srt file.srt      # Load SRT and pick segments
    ./clip_extractor.py --video file.mkv    # Set default video
    ./clip_extractor.py --output clips/     # Set output directory
"""

import subprocess
import sys
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Segment:
    index: int
    start: str
    end: str
    text: str
    lang: str = "UNK"
    speaker: str = ""

def parse_srt(srt_path: str) -> List[Segment]:
    """Parse SRT file into segments."""
    segments = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newline (segment separator)
    blocks = re.split(r'\n\n+', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        try:
            index = int(lines[0])
        except ValueError:
            continue
        
        # Parse timestamp: 00:01:21,064 --> 00:01:24,642
        time_match = re.match(r'(\d{2}:\d{2}:\d{2}),\d+ --> (\d{2}:\d{2}:\d{2}),\d+', lines[1])
        if not time_match:
            continue
        
        start = time_match.group(1)
        end = time_match.group(2)
        
        # Get text (remaining lines)
        text = ' '.join(lines[2:])
        
        # Extract [LANG|SPEAKER] tag if present
        lang = "UNK"
        speaker = ""
        tag_match = re.match(r'\[(\w+)\|?(\w*)\]', text)
        if tag_match:
            lang = tag_match.group(1)
            speaker = tag_match.group(2) or ""
        
        segments.append(Segment(index, start, end, text, lang, speaker))
    
    return segments

def extract_clip(video: str, start: str, end: str, output: str, padding: float = 0.5):
    """Extract clip with optional padding."""
    # Convert HH:MM:SS to seconds for padding calculation
    def to_seconds(ts: str) -> float:
        parts = ts.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2].replace(',', '.'))
    
    def from_seconds(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}"
    
    start_sec = max(0, to_seconds(start) - padding)
    end_sec = to_seconds(end) + padding
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video,
        "-ss", from_seconds(start_sec),
        "-to", from_seconds(end_sec),
        "-c:a", "pcm_s16le",
        "-ar", "16000",  # 16kHz for ASR
        "-ac", "1",      # Mono
        output
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Extracted: {output}")
        return True
    else:
        print(f"✗ Error: {result.stderr}")
        return False

def display_segments(segments: List[Segment], page: int = 0, per_page: int = 20):
    """Display segment list with pagination."""
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, len(segments))
    
    print(f"\n{'='*80}")
    print(f"Segments {start_idx+1}-{end_idx} of {len(segments)}  (Page {page+1}/{(len(segments)-1)//per_page + 1})")
    print(f"{'='*80}")
    
    for seg in segments[start_idx:end_idx]:
        # Truncate text for display
        text = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        lang_color = {
            'NAH': '\033[93m',  # Yellow
            'SPA': '\033[92m',  # Green
            'MAY': '\033[94m',  # Blue
            'LAT': '\033[95m',  # Magenta
            'OTH': '\033[91m',  # Red
        }.get(seg.lang, '\033[0m')
        reset = '\033[0m'
        
        print(f"{seg.index:4d} [{lang_color}{seg.lang:3s}{reset}] {seg.start}-{seg.end} | {text}")
    
    print(f"{'='*80}")

def interactive_mode(video: Optional[str] = None, srt: Optional[str] = None, output_dir: str = "clips"):
    """Main interactive loop."""
    segments: List[Segment] = []
    current_page = 0
    
    Path(output_dir).mkdir(exist_ok=True)
    
    if srt:
        segments = parse_srt(srt)
        print(f"Loaded {len(segments)} segments from {srt}")
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Tenepal Clip Extractor                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    v <path>     - Set video file                             ║
║    s <path>     - Load SRT file                              ║
║    o <path>     - Set output directory                       ║
║    l            - List segments (current page)               ║
║    n / p        - Next / Previous page                       ║
║    f <text>     - Filter segments containing text            ║
║    fl <lang>    - Filter by language (NAH/SPA/MAY/OTH)       ║
║    x <num>      - Extract segment by number                  ║
║    x <n1> <n2>  - Extract range of segments                  ║
║    m <start> <end> <name>  - Manual extract (HH:MM:SS)       ║
║    q            - Quit                                       ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    if video:
        print(f"Video: {video}")
    if srt:
        print(f"SRT: {srt}")
    print(f"Output: {output_dir}/")
    
    filtered_segments = segments
    
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not cmd:
            continue
        
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if action == 'q':
            print("Bye!")
            break
        
        elif action == 'v':
            video = args
            if os.path.exists(video):
                print(f"✓ Video set: {video}")
            else:
                print(f"✗ File not found: {video}")
        
        elif action == 's':
            srt = args
            if os.path.exists(srt):
                segments = parse_srt(srt)
                filtered_segments = segments
                current_page = 0
                print(f"✓ Loaded {len(segments)} segments")
            else:
                print(f"✗ File not found: {srt}")
        
        elif action == 'o':
            output_dir = args
            Path(output_dir).mkdir(exist_ok=True)
            print(f"✓ Output directory: {output_dir}")
        
        elif action == 'l':
            if filtered_segments:
                display_segments(filtered_segments, current_page)
            else:
                print("No segments loaded. Use 's <file.srt>' to load.")
        
        elif action == 'n':
            max_page = (len(filtered_segments) - 1) // 20
            current_page = min(current_page + 1, max_page)
            display_segments(filtered_segments, current_page)
        
        elif action == 'p':
            current_page = max(0, current_page - 1)
            display_segments(filtered_segments, current_page)
        
        elif action == 'f':
            if not args:
                filtered_segments = segments
                print(f"Filter cleared. {len(segments)} segments.")
            else:
                filtered_segments = [s for s in segments if args.lower() in s.text.lower()]
                current_page = 0
                print(f"Found {len(filtered_segments)} segments containing '{args}'")
                if filtered_segments:
                    display_segments(filtered_segments, 0)
        
        elif action == 'fl':
            lang = args.upper()
            if lang in ['NAH', 'SPA', 'MAY', 'LAT', 'OTH', 'MIX', 'UNK']:
                filtered_segments = [s for s in segments if s.lang == lang]
                current_page = 0
                print(f"Found {len(filtered_segments)} {lang} segments")
                if filtered_segments:
                    display_segments(filtered_segments, 0)
            else:
                print("Languages: NAH, SPA, MAY, LAT, OTH, MIX")
        
        elif action == 'x':
            if not video:
                print("✗ Set video first with 'v <path>'")
                continue
            
            nums = args.split()
            if len(nums) == 1:
                # Single segment
                try:
                    idx = int(nums[0])
                    seg = next((s for s in segments if s.index == idx), None)
                    if seg:
                        name = f"seg_{idx:04d}_{seg.lang}.wav"
                        extract_clip(video, seg.start, seg.end, f"{output_dir}/{name}")
                    else:
                        print(f"✗ Segment {idx} not found")
                except ValueError:
                    print("✗ Invalid number")
            
            elif len(nums) == 2:
                # Range
                try:
                    start_idx, end_idx = int(nums[0]), int(nums[1])
                    extracted = 0
                    for seg in segments:
                        if start_idx <= seg.index <= end_idx:
                            name = f"seg_{seg.index:04d}_{seg.lang}.wav"
                            if extract_clip(video, seg.start, seg.end, f"{output_dir}/{name}"):
                                extracted += 1
                    print(f"Extracted {extracted} clips")
                except ValueError:
                    print("✗ Invalid range")
        
        elif action == 'm':
            # Manual: m 00:01:21 00:01:25 yes_sir
            if not video:
                print("✗ Set video first with 'v <path>'")
                continue
            
            manual_parts = args.split()
            if len(manual_parts) >= 3:
                start, end, name = manual_parts[0], manual_parts[1], manual_parts[2]
                if not name.endswith('.wav'):
                    name += '.wav'
                extract_clip(video, start, end, f"{output_dir}/{name}")
            else:
                print("Usage: m <start> <end> <name>")
                print("Example: m 00:01:21 00:01:25 yes_sir")
        
        else:
            print(f"Unknown command: {action}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive clip extractor")
    parser.add_argument('--video', '-v', help='Video/audio file')
    parser.add_argument('--srt', '-s', help='SRT file to load')
    parser.add_argument('--output', '-o', default='clips', help='Output directory')
    
    args = parser.parse_args()
    interactive_mode(video=args.video, srt=args.srt, output_dir=args.output)

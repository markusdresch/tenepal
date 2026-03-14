#!/usr/bin/env python3
"""
Extract Malinche/Marina dialogue segments from Hernán S01 SRT files.
Outputs JSON with episode, timestamp, context, and dialogue analysis.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Segment:
    episode: int
    index: int
    start_time: str
    end_time: str
    text: str
    context: str  # SPEAKS_TO (someone calls her), SPOKEN_BY (she speaks), MENTIONED
    speaker_hint: Optional[str] = None


def parse_srt(filepath: Path) -> list[dict]:
    """Parse SRT file into list of subtitle entries."""
    content = filepath.read_text(encoding='utf-8', errors='replace')
    entries = []

    # Split by blank lines (SRT format)
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                idx = int(lines[0])
                timestamp = lines[1]
                text = '\n'.join(lines[2:])

                # Parse timestamp
                match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', timestamp)
                if match:
                    entries.append({
                        'index': idx,
                        'start': match.group(1),
                        'end': match.group(2),
                        'text': text
                    })
            except (ValueError, IndexError):
                continue

    return entries


def classify_marina_segment(text: str, prev_text: str = "", next_text: str = "") -> tuple[str, Optional[str]]:
    """
    Classify whether Marina is being spoken to, speaking, or mentioned.
    Returns (context, speaker_hint).
    """
    text_lower = text.lower()

    # Patterns where Marina is being addressed
    address_patterns = [
        r'^marina[.,!?]?\s*$',  # Just her name being called
        r'marina[,!]?\s*(come|go|tell|translate|help|lend|no)',
        r'(talk to|translate,|don\'t be afraid,)\s*marina',
        r'what about marina',
        r'where is marina',
    ]

    # Patterns where Marina is mentioned in third person
    mention_patterns = [
        r'marina is (coming|here|the most)',
        r'(was that|about) marina',
        r'accompanies marina',
        r'name will be marina',
        r'malinche',  # The title/honorific
        r'malintzin',
    ]

    for pattern in address_patterns:
        if re.search(pattern, text_lower):
            return 'SPEAKS_TO', None

    for pattern in mention_patterns:
        if re.search(pattern, text_lower):
            return 'MENTIONED', None

    # Check dialogue format "- Marina: text" or context
    if text.startswith('- ') and 'marina' in text_lower[:20]:
        return 'SPOKEN_BY', 'Marina'

    # Default: likely speaking if contains Marina but no address pattern
    if 'marina' in text_lower and not any(x in text_lower for x in ['marina!', 'marina?', 'marina,']):
        return 'MENTIONED', None

    return 'SPEAKS_TO', None  # Default for standalone name calls


def extract_malinche_segments(srt_dir: Path) -> list[Segment]:
    """Extract all Marina/Malinche related segments from Hernán S01."""
    segments = []

    # Find all episode files
    episode_files = sorted(srt_dir.glob('Hernn.S01E*.WEB.4FIRE.en.srt'))

    for filepath in episode_files:
        # Extract episode number
        match = re.search(r'S01E(\d+)', filepath.name)
        if not match:
            continue
        episode = int(match.group(1))

        entries = parse_srt(filepath)

        # Find Marina-related entries
        for i, entry in enumerate(entries):
            text = entry['text']

            # Check for Marina/Malinche/Malintzin
            if not re.search(r'marina|malinche|malintzin', text, re.IGNORECASE):
                continue

            prev_text = entries[i-1]['text'] if i > 0 else ""
            next_text = entries[i+1]['text'] if i < len(entries)-1 else ""

            context, speaker = classify_marina_segment(text, prev_text, next_text)

            segments.append(Segment(
                episode=episode,
                index=entry['index'],
                start_time=entry['start'],
                end_time=entry['end'],
                text=text,
                context=context,
                speaker_hint=speaker
            ))

    return segments


def generate_episode_stats(segments: list[Segment]) -> dict:
    """Generate statistics per episode."""
    stats = {}

    for seg in segments:
        ep = seg.episode
        if ep not in stats:
            stats[ep] = {
                'total_mentions': 0,
                'speaks_to': 0,
                'spoken_by': 0,
                'mentioned': 0,
                'first_appearance': seg.start_time,
                'last_appearance': seg.start_time,
                'timestamps': []
            }

        stats[ep]['total_mentions'] += 1
        stats[ep][seg.context.lower()] += 1
        stats[ep]['last_appearance'] = seg.start_time
        stats[ep]['timestamps'].append(seg.start_time)

    return stats


def main():
    srt_dir = Path(__file__).parent.parent / 'reference_srt'
    output_dir = Path(__file__).parent.parent / '.planning' / 'context'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting Malinche/Marina segments from Hernán S01...")
    segments = extract_malinche_segments(srt_dir)

    print(f"Found {len(segments)} segments across {len(set(s.episode for s in segments))} episodes")

    # Generate stats
    stats = generate_episode_stats(segments)

    # Output results
    results = {
        'total_segments': len(segments),
        'episodes_covered': sorted(set(s.episode for s in segments)),
        'episode_stats': stats,
        'segments': [asdict(s) for s in segments]
    }

    output_file = output_dir / 'malinche_segments.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Output written to {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("MALINCHE PRESENCE IN HERNÁN S01")
    print("="*70)
    print(f"{'Episode':<10} {'Mentions':<10} {'Spoken To':<12} {'Mentioned':<12} {'First':<15}")
    print("-"*70)

    for ep in sorted(stats.keys()):
        s = stats[ep]
        print(f"E{ep:02d}       {s['total_mentions']:<10} {s['speaks_to']:<12} {s['mentioned']:<12} {s['first_appearance'][:8]}")

    print("-"*70)
    print(f"{'TOTAL':<10} {len(segments):<10}")

    # Key moments
    print("\n" + "="*70)
    print("KEY NARRATIVE MOMENTS")
    print("="*70)

    key_texts = [
        "your name will be Marina",
        "Malinche",
        "Malintzin",
        "accompanies Marina",
        "Translate"
    ]

    for seg in segments:
        for key in key_texts:
            if key.lower() in seg.text.lower():
                print(f"E{seg.episode:02d} {seg.start_time[:8]}: {seg.text[:60]}...")
                break

    return results


if __name__ == '__main__':
    main()

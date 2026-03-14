#!/usr/bin/env python3
"""Parselmouth to IPA Translator - Rule-based acoustic phoneme recognition.

Classical phonetics approach using formant analysis.

Usage:
    modal run scripts/parselmouth_to_ipa.py
"""

import modal
import json
from pathlib import Path

app = modal.App("parselmouth-to-ipa")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "soundfile",
        "numpy",
        "praat-parselmouth",
    )
)


# Spanish 5-vowel system - target F1/F2 centroids (Hz)
SPANISH_VOWELS = {
    'a': (750, 1300),   # open central - high F1, mid F2
    'e': (500, 1900),   # mid front - mid F1, high F2
    'i': (300, 2400),   # close front - low F1, very high F2
    'o': (500, 900),    # mid back - mid F1, low F2
    'u': (320, 750),    # close back - low F1, very low F2
}


def classify_vowel_nearest(f1: float, f2: float) -> str:
    """Classify vowel using nearest-neighbor in F1/F2 space."""
    if f1 is None or f2 is None:
        return '?'

    best_vowel = 'a'
    best_dist = float('inf')

    for vowel, (target_f1, target_f2) in SPANISH_VOWELS.items():
        # Weighted distance - F2 range is larger so normalize
        dist = ((f1 - target_f1) / 200) ** 2 + ((f2 - target_f2) / 400) ** 2
        if dist < best_dist:
            best_dist = dist
            best_vowel = vowel

    return best_vowel


def classify_consonant(cog: float, duration_ms: float, is_voiced: bool,
                       prev_vowel: str, next_vowel: str) -> str:
    """Classify consonant from spectral and contextual features."""

    if is_voiced:
        # Voiced consonants - nasals or approximants
        if duration_ms > 30:
            # Likely nasal
            if cog < 1200:
                return 'm'
            else:
                return 'n'
        else:
            # Short voiced - approximant (Spanish b/d/g intervocalic)
            if cog < 1500:
                return 'β'  # approximant /b/
            elif cog > 2500:
                return 'ð'  # approximant /d/
            else:
                return 'ɣ'  # approximant /g/
    else:
        # Unvoiced consonants
        if duration_ms < 30:
            # Short burst - plosive
            if cog > 3000:
                return 't'
            elif cog < 1500:
                return 'p'
            else:
                return 'k'
        elif duration_ms < 80:
            # Medium - could be affricate or fricative
            if cog > 4000:
                return 's'
            elif cog > 2500:
                return 'tʃ'  # affricate
            else:
                return 'f'
        else:
            # Long noise - fricative
            if cog > 3500:
                return 's'
            elif cog > 2000:
                return 'ʃ'
            else:
                return 'x'  # Spanish /j/ sound


def segment_acoustic_events(frames: list) -> list:
    """Segment frames into acoustic events (SILENCE/VOICED/UNVOICED)."""

    INTENSITY_THRESHOLD = 50  # dB

    events = []
    current_event = None
    event_start = 0
    event_frames = []

    for frame in frames:
        t_ms = frame.get('t_ms', 0)
        intensity = frame.get('intensity_db') or 0
        voiced = frame.get('voiced', False)

        if intensity < INTENSITY_THRESHOLD:
            event_type = 'SILENCE'
        elif voiced:
            event_type = 'VOICED'
        else:
            event_type = 'UNVOICED'

        if event_type != current_event:
            # Save previous event
            if current_event is not None and event_frames:
                events.append({
                    'type': current_event,
                    'start_ms': event_start,
                    'end_ms': t_ms,
                    'duration_ms': t_ms - event_start,
                    'frames': event_frames,
                })
            # Start new event
            current_event = event_type
            event_start = t_ms
            event_frames = [frame]
        else:
            event_frames.append(frame)

    # Don't forget last event
    if current_event is not None and event_frames:
        events.append({
            'type': current_event,
            'start_ms': event_start,
            'end_ms': frames[-1].get('t_ms', 0) + 10,
            'duration_ms': frames[-1].get('t_ms', 0) + 10 - event_start,
            'frames': event_frames,
        })

    return events


def events_to_ipa(events: list) -> tuple[str, list]:
    """Convert acoustic events to IPA string."""

    phonemes = []
    details = []

    for i, event in enumerate(events):
        if event['type'] == 'SILENCE':
            continue

        # Get average features across event
        f1_vals = [f.get('f1') for f in event['frames'] if f.get('f1')]
        f2_vals = [f.get('f2') for f in event['frames'] if f.get('f2')]
        cog_vals = [f.get('cog_hz') for f in event['frames'] if f.get('cog_hz')]

        avg_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else None
        avg_f2 = sum(f2_vals) / len(f2_vals) if f2_vals else None
        avg_cog = sum(cog_vals) / len(cog_vals) if cog_vals else 1500

        duration = event['duration_ms']

        if event['type'] == 'VOICED':
            # Vowel or voiced consonant
            if duration > 40 and avg_f1 and avg_f2:
                # Likely vowel
                vowel = classify_vowel_nearest(avg_f1, avg_f2)
                phonemes.append(vowel)
                details.append({
                    'type': 'vowel',
                    'phoneme': vowel,
                    'start_ms': event['start_ms'],
                    'duration_ms': duration,
                    'f1': round(avg_f1, 0) if avg_f1 else None,
                    'f2': round(avg_f2, 0) if avg_f2 else None,
                })
            elif duration > 20:
                # Could be nasal or approximant
                is_voiced = True
                prev_v = phonemes[-1] if phonemes else ''
                next_v = ''  # Would need lookahead
                consonant = classify_consonant(avg_cog, duration, is_voiced, prev_v, next_v)
                phonemes.append(consonant)
                details.append({
                    'type': 'consonant',
                    'phoneme': consonant,
                    'start_ms': event['start_ms'],
                    'duration_ms': duration,
                    'cog': round(avg_cog, 0),
                })
            else:
                # Very short voiced - tap /ɾ/
                phonemes.append('ɾ')
                details.append({
                    'type': 'tap',
                    'phoneme': 'ɾ',
                    'start_ms': event['start_ms'],
                    'duration_ms': duration,
                })

        elif event['type'] == 'UNVOICED':
            # Consonant
            is_voiced = False
            prev_v = phonemes[-1] if phonemes else ''
            next_v = ''
            consonant = classify_consonant(avg_cog, duration, is_voiced, prev_v, next_v)
            phonemes.append(consonant)
            details.append({
                'type': 'consonant',
                'phoneme': consonant,
                'start_ms': event['start_ms'],
                'duration_ms': duration,
                'cog': round(avg_cog, 0),
            })

    # Collapse consecutive identical phonemes (CTC-style)
    collapsed = []
    for p in phonemes:
        if not collapsed or collapsed[-1] != p:
            collapsed.append(p)

    return ' '.join(collapsed), details


@app.function(image=image, timeout=300)
def process_segments(segments: list) -> list:
    """Process segments and generate IPA from Parselmouth features."""

    results = []

    for seg in segments:
        cue = seg.get('cue', 0)
        whisper_text = seg.get('whisper_text', '')
        frames = seg.get('frames', [])

        if not frames:
            results.append({
                'cue': cue,
                'whisper_text': whisper_text,
                'pm_ipa': '',
                'events': [],
                'details': [],
                'error': 'No frames'
            })
            continue

        # Segment into acoustic events
        events = segment_acoustic_events(frames)

        # Convert to IPA
        pm_ipa, details = events_to_ipa(events)

        # Event summary
        event_summary = []
        for e in events:
            if e['type'] != 'SILENCE':
                event_summary.append(f"{e['type']}({e['start_ms']:.0f}-{e['end_ms']:.0f})")

        results.append({
            'cue': cue,
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'whisper_text': whisper_text,
            'pm_ipa': pm_ipa,
            'allo_ipa': seg.get('allo_ipa', ''),
            'w2v2_ipa': seg.get('w2v2_ipa', ''),
            'events': event_summary,
            'n_events': len([e for e in events if e['type'] != 'SILENCE']),
            'details': details,
        })

    return results


@app.local_entrypoint()
def main():
    """Run Parselmouth→IPA translation on all segments."""

    # Load the Parselmouth data from the previous analysis
    pm_file = Path("validation_video/analysis/parselmouth_5min.jsonl")
    if not pm_file.exists():
        print(f"ERROR: {pm_file} not found. Run full_debug_5min.py first.")
        return

    print(f"Loading Parselmouth data from {pm_file}...")
    segments = []
    with open(pm_file) as f:
        for line in f:
            segments.append(json.loads(line))

    print(f"Processing {len(segments)} segments...")

    # Process on Modal
    results = process_segments.remote(segments)

    # Save results
    output_dir = Path("validation_video/analysis")
    output_file = output_dir / "parselmouth_ipa_5min.txt"

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PARSELMOUTH → IPA TRANSLATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"Cue {r['cue']}: \"{r['whisper_text'][:50]}...\"\n" if len(r.get('whisper_text', '')) > 50
                    else f"Cue {r['cue']}: \"{r.get('whisper_text', '')}\"\n")
            f.write(f"  Time: {r.get('start', 0):.3f}-{r.get('end', 0):.3f}s\n")
            f.write(f"  Events: {' → '.join(r['events'][:10])}\n")
            f.write(f"  PM-IPA:  {r['pm_ipa']}\n")
            f.write(f"  Allo:    {r['allo_ipa']}\n")
            f.write(f"  w2v2:    {r['w2v2_ipa']}\n")
            f.write("\n")

            # Detail for first few segments
            if r['cue'] <= 10:
                f.write("  Details:\n")
                for d in r['details'][:15]:
                    if d['type'] == 'vowel':
                        f.write(f"    {d['start_ms']:>5.0f}ms: /{d['phoneme']}/ (F1={d['f1']}, F2={d['f2']}, {d['duration_ms']:.0f}ms)\n")
                    else:
                        f.write(f"    {d['start_ms']:>5.0f}ms: /{d['phoneme']}/ ({d.get('cog', '?')}Hz CoG, {d['duration_ms']:.0f}ms)\n")
                f.write("\n")

        # Summary statistics
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")

        total_events = sum(r['n_events'] for r in results)
        f.write(f"Total segments: {len(results)}\n")
        f.write(f"Total acoustic events: {total_events}\n")
        f.write(f"Avg events per segment: {total_events / len(results):.1f}\n")

        # Count phoneme types
        all_phonemes = []
        for r in results:
            all_phonemes.extend(r['pm_ipa'].split())

        from collections import Counter
        phoneme_counts = Counter(all_phonemes)
        f.write(f"\nTop phonemes:\n")
        for p, count in phoneme_counts.most_common(15):
            f.write(f"  {p}: {count}\n")

    print(f"\nSaved: {output_file}")

    # Also save as JSONL for further analysis
    jsonl_file = output_dir / "parselmouth_ipa_5min.jsonl"
    with open(jsonl_file, "w") as f:
        for r in results:
            # Remove details for compact output
            compact = {k: v for k, v in r.items() if k != 'details'}
            f.write(json.dumps(compact) + "\n")

    print(f"Saved: {jsonl_file}")

    # Print sample results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)
    for r in results[:5]:
        print(f"\nCue {r['cue']}: {r['whisper_text'][:40]}...")
        print(f"  PM-IPA: {r['pm_ipa']}")
        print(f"  Allo:   {r['allo_ipa']}")
        print(f"  w2v2:   {r['w2v2_ipa']}")


if __name__ == "__main__":
    main()

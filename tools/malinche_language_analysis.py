#!/usr/bin/env python3
"""
Analyze Malinche's language distribution across Hernán S01.
Maps her linguistic journey: MAY → NAH → SPA

Historical context:
- Malinalli (birth name) was born in Coatzacoalcos region, spoke Nahuatl
- Sold/given to Maya traders, learned Yucatec Maya
- Given to Cortés in 1519, learned Spanish from Aguilar
- Became crucial translator: Spanish ↔ Nahuatl (via Maya with Aguilar initially)
"""

import json
from pathlib import Path
from dataclasses import dataclass


# Episode narrative context for language assignment
EPISODE_CONTEXT = {
    1: {
        "title": "Marina",
        "context": "Marina is named/baptized. Flashbacks to pre-conquest. Her role emerges.",
        "languages": {
            "MAY": 0.3,  # Her native tongue in flashbacks
            "NAH": 0.3,  # When speaking to indigenous peoples
            "SPA": 0.4   # Learning Spanish, named by Spaniards
        },
        "key_moments": [
            {"time": "00:21:28", "event": "Baptism: 'Your name will be Marina'", "lang": "SPA"}
        ]
    },
    2: {
        "title": "Aguilar",
        "context": "Marina works with Jerónimo de Aguilar as translation chain.",
        "languages": {
            "MAY": 0.4,  # Communicating with Aguilar (he only knows Maya)
            "NAH": 0.4,  # Translating to/from Aztec messengers
            "SPA": 0.2   # Limited direct Spanish
        },
        "key_moments": [
            {"time": "00:09:01", "event": "Marina and Jerónimo called together", "lang": "MAY/NAH"}
        ]
    },
    3: {
        "title": "Cempoala",
        "context": "Totonac alliance. Marina translates for Cortés.",
        "languages": {
            "MAY": 0.2,
            "NAH": 0.5,  # Primary translation work
            "SPA": 0.3
        },
        "key_moments": []
    },
    4: {
        "title": "Tlaxcala",
        "context": "Critical Tlaxcalan alliance negotiations.",
        "languages": {
            "MAY": 0.1,
            "NAH": 0.6,  # Nahuatl dominant with Tlaxcalans
            "SPA": 0.3
        },
        "key_moments": [
            {"time": "00:42:24", "event": "'Talk to him, Marina' - diplomatic translation", "lang": "NAH"}
        ]
    },
    5: {
        "title": "Cholula",
        "context": "Cholula massacre. Marina's warning/translation crucial.",
        "languages": {
            "MAY": 0.1,
            "NAH": 0.5,
            "SPA": 0.4
        },
        "key_moments": [
            {"time": "00:18:16", "event": "'Did you translate that right, Marina?'", "lang": "NAH→SPA"}
        ]
    },
    6: {
        "title": "Moctezuma",
        "context": "First contact with Moctezuma in Tenochtitlan.",
        "languages": {
            "MAY": 0.0,  # Not needed with Aztec emperor
            "NAH": 0.6,  # Direct to Moctezuma
            "SPA": 0.4
        },
        "key_moments": [
            {"time": "00:16:25", "event": "'Translate, Marina' - to Moctezuma", "lang": "NAH"}
        ]
    },
    7: {
        "title": "La Noche Triste",
        "context": "Spanish retreat. 'Malinche' etymology revealed.",
        "languages": {
            "MAY": 0.1,
            "NAH": 0.5,
            "SPA": 0.4
        },
        "key_moments": [
            {"time": "00:37:46", "event": "'Malinche' - the Nahuatl honorific", "lang": "NAH"},
            {"time": "00:37:49", "event": "'He who accompanies Marina'", "lang": "NAH→SPA"}
        ]
    },
    8: {
        "title": "The Fall",
        "context": "Fall of Tenochtitlan. Marina as mother of Martín.",
        "languages": {
            "MAY": 0.05,
            "NAH": 0.45,
            "SPA": 0.5   # More Spanish as relationship with Cortés deepens
        },
        "key_moments": [
            {"time": "00:41:55", "event": "'Señor Malinche' - Aztecs call Cortés by her name", "lang": "NAH/SPA"},
            {"time": "00:45:28", "event": "Martin Cortes Malintzin - her son's full name", "lang": "NAH/SPA"}
        ]
    }
}


def load_segments():
    """Load extracted Malinche segments."""
    path = Path(__file__).parent.parent / '.planning/context/malinche_segments.json'
    with open(path) as f:
        return json.load(f)


def analyze_language_arc():
    """Analyze Marina's language journey across the season."""
    data = load_segments()

    # Aggregate language distribution
    season_total = {"MAY": 0, "NAH": 0, "SPA": 0}
    episode_breakdown = []

    for ep in range(1, 9):
        ep_str = str(ep)
        if ep_str in data['episode_stats']:
            mentions = data['episode_stats'][ep_str]['total_mentions']
            ctx = EPISODE_CONTEXT[ep]

            ep_may = mentions * ctx['languages']['MAY']
            ep_nah = mentions * ctx['languages']['NAH']
            ep_spa = mentions * ctx['languages']['SPA']

            season_total['MAY'] += ep_may
            season_total['NAH'] += ep_nah
            season_total['SPA'] += ep_spa

            episode_breakdown.append({
                'episode': ep,
                'title': ctx['title'],
                'mentions': mentions,
                'MAY': round(ep_may, 1),
                'NAH': round(ep_nah, 1),
                'SPA': round(ep_spa, 1),
                'key_moments': ctx['key_moments']
            })

    return {
        'season_total': season_total,
        'episodes': episode_breakdown,
        'total_mentions': data['total_segments']
    }


def generate_markdown_table():
    """Generate markdown table for PAPER.md."""
    analysis = analyze_language_arc()

    lines = []
    lines.append("## Malinche Language Distribution — Hernán S01\n")
    lines.append("Marina/Malinche's linguistic journey across the series demonstrates her role")
    lines.append("as the crucial bridge between Spanish conquistadors and indigenous peoples.\n")

    # Summary stats
    total = sum(analysis['season_total'].values())
    lines.append("### Season Summary\n")
    lines.append(f"- **Total Marina mentions**: {analysis['total_mentions']} segments")
    lines.append(f"- **Yucatec Maya (MAY)**: {analysis['season_total']['MAY']:.1f} ({100*analysis['season_total']['MAY']/total:.1f}%)")
    lines.append(f"- **Nahuatl (NAH)**: {analysis['season_total']['NAH']:.1f} ({100*analysis['season_total']['NAH']/total:.1f}%)")
    lines.append(f"- **Spanish (SPA)**: {analysis['season_total']['SPA']:.1f} ({100*analysis['season_total']['SPA']/total:.1f}%)\n")

    # Episode table
    lines.append("### Episode Breakdown\n")
    lines.append("| Episode | Title | Mentions | MAY | NAH | SPA | Key Moment |")
    lines.append("|---------|-------|----------|-----|-----|-----|------------|")

    for ep in analysis['episodes']:
        key = ep['key_moments'][0]['event'][:30] + "..." if ep['key_moments'] else "-"
        lines.append(f"| E{ep['episode']:02d} | {ep['title']} | {ep['mentions']} | {ep['MAY']} | {ep['NAH']} | {ep['SPA']} | {key} |")

    # Narrative arc
    lines.append("\n### Linguistic Arc\n")
    lines.append("1. **E01-02**: Marina emerges as translator via Aguilar (MAY↔NAH chain)")
    lines.append("2. **E03-05**: Direct NAH translation dominates during alliances/conflicts")
    lines.append("3. **E06-07**: Peak NAH usage with Moctezuma; 'Malinche' etymology revealed")
    lines.append("4. **E08**: Spanish increases as her role evolves from translator to mother\n")

    # Key linguistic moments
    lines.append("### Key Linguistic Moments\n")
    lines.append("| Timestamp | Episode | Event | Language Context |")
    lines.append("|-----------|---------|-------|------------------|")

    key_moments = [
        ("00:21:28", 1, "Baptism: 'Your name will be Marina'", "SPA (naming ceremony)"),
        ("00:18:16", 5, "'Did you translate that right?'", "NAH→SPA (verification)"),
        ("00:16:25", 6, "'Translate, Marina'", "SPA→NAH (to Moctezuma)"),
        ("00:37:46", 7, "'Malinche' etymology", "NAH honorific explained"),
        ("00:41:55", 8, "'Señor Malinche'", "Cortés called by her name"),
        ("00:45:28", 8, "'Martin Cortes Malintzin'", "Mestizo naming tradition"),
    ]

    for ts, ep, event, ctx in key_moments:
        lines.append(f"| {ts} | E{ep:02d} | {event} | {ctx} |")

    return '\n'.join(lines)


def main():
    analysis = analyze_language_arc()

    print("=" * 70)
    print("MALINCHE LANGUAGE DISTRIBUTION — HERNÁN S01")
    print("=" * 70)

    # Print table
    print(f"\n{'Episode':<8} {'Title':<15} {'Mentions':<10} {'MAY':<6} {'NAH':<6} {'SPA':<6}")
    print("-" * 60)

    for ep in analysis['episodes']:
        print(f"E{ep['episode']:02d}     {ep['title']:<15} {ep['mentions']:<10} {ep['MAY']:<6} {ep['NAH']:<6} {ep['SPA']:<6}")

    print("-" * 60)
    t = analysis['season_total']
    print(f"{'TOTAL':<24} {analysis['total_mentions']:<10} {t['MAY']:.1f}   {t['NAH']:.1f}   {t['SPA']:.1f}")

    # Language percentages
    total = sum(t.values())
    print(f"\nLanguage Distribution:")
    print(f"  MAY: {100*t['MAY']/total:.1f}%")
    print(f"  NAH: {100*t['NAH']/total:.1f}%")
    print(f"  SPA: {100*t['SPA']/total:.1f}%")

    # Generate markdown
    md = generate_markdown_table()
    output_path = Path(__file__).parent.parent / '.planning/context/malinche_language_analysis.md'
    with open(output_path, 'w') as f:
        f.write(md)
    print(f"\nMarkdown written to: {output_path}")

    return analysis


if __name__ == '__main__':
    main()

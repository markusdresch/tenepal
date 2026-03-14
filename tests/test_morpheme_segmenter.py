#!/usr/bin/env python3
"""Test the Nahuatl morpheme segmenter with real examples.

Run: PYTHONPATH=src python tests/test_morpheme_segmenter.py
"""
import sys
sys.path.insert(0, "src")

from tenepal.morphology import analyze_word, analyze_text, lid_score


def test_training_examples():
    """Test against the NAH→SPA training set examples."""
    print("=" * 70)
    print("MORPHEME SEGMENTER TEST — Training Set Examples")
    print("=" * 70)

    examples = [
        ("amo", "no"),
        ("tlazocamati", "gracias"),
        ("nimitznotza", "te llamo"),
        ("kilit", "quelite"),  # "In kilit" → greens
        ("monamaka", "se vende"),  # "Amo monamaka" → no se vende
        ("kualtia", "usos"),  # "Kualtia:" → uses
        ("tlacatl", "hombre"),  # from "Inin tlacatl cemi tlatquihua"
        ("tlatquihua", "es rico (posee cosas)"),
        ("mexikaj", "los mexicas"),  # mexikaj → mexica + pl
        ("niltze", "hola (saludo)"),
        ("nimitznotza", "te llamo"),
    ]

    for nah, expected_spa in examples:
        result = analyze_word(nah)
        print(f"\n  {nah:25s} → {result.interlinear}")
        print(f"  {'':25s}   hint: {result.translation_hint}")
        print(f"  {'':25s}   expected: {expected_spa}")
        print(f"  {'':25s}   LID score: {result.lid_score:.2f}  coverage: {result.coverage:.0%}  is_nah: {result.is_nahuatl}")
        if result.morphemes:
            for m in result.morphemes:
                print(f"  {'':25s}   [{m.type.value:6s}] {m.form:12s} = {m.gloss}")


def test_sentence_analysis():
    """Test multi-word analysis."""
    print("\n" + "=" * 70)
    print("SENTENCE ANALYSIS")
    print("=" * 70)

    sentences = [
        "Inin tlacatl cemi tlatquihua",  # Este hombre es muy rico
        "Amo monamaka",                  # No se vende
        "nimitznotza",                   # Te llamo
        "In kilit",                      # El quelite (the greens)
    ]

    for sentence in sentences:
        analyses = analyze_text(sentence)
        score = lid_score(sentence)
        print(f"\n  Input: {sentence}")
        print(f"  NAH LID score: {score:.2f}")
        for a in analyses:
            morphemes_str = " | ".join(f"{m.form}={m.gloss}" for m in a.morphemes)
            print(f"    {a.input_form:20s} → [{morphemes_str}]")
            print(f"    {'':20s}   hint: {a.translation_hint}  (coverage: {a.coverage:.0%})")


def test_lid_comparison():
    """Compare LID scores for NAH vs SPA vs random text."""
    print("\n" + "=" * 70)
    print("LID COMPARISON — NAH vs SPA vs random")
    print("=" * 70)

    test_items = [
        ("nimitznotza", "NAH — te llamo"),
        ("tlazocamati", "NAH — gracias"),
        ("mochihua", "NAH — se hace"),
        ("tlatquihua", "NAH — posee cosas"),
        ("nochi", "NAH — todo"),
        ("axkan", "NAH — ahora"),
        ("los soldados", "SPA"),
        ("buenas noches", "SPA"),
        ("arcabuceros", "SPA"),
        ("the soldiers", "ENG"),
        ("hello world", "ENG"),
    ]

    for word, label in test_items:
        score = lid_score(word)
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        marker = "✓ NAH" if score >= 0.3 else "✗ ---"
        print(f"  {word:25s} {label:20s} [{bar}] {score:.2f}  {marker}")


def test_ipa_mode():
    """Test IPA-mode analysis (from Allosaurus/wav2vec2 output)."""
    print("\n" + "=" * 70)
    print("IPA MODE — Direct phoneme sequence analysis")
    print("=" * 70)

    ipa_examples = [
        ("n i m i ts n o ts a", "nimitznotza (from IPA)"),
        ("tɬ a k a tɬ", "tlacatl (from IPA)"),
        ("a m o", "amo (from IPA)"),
        ("n o tʃ i", "nochi (from IPA)"),
        ("k a l i", "kalli (from IPA)"),
    ]

    for ipa, label in ipa_examples:
        result = analyze_word(ipa, as_ipa=True)
        print(f"\n  IPA: {ipa}")
        print(f"  Label: {label}")
        print(f"  Interlinear: {result.interlinear}")
        print(f"  LID score: {result.lid_score:.2f}  is_nah: {result.is_nahuatl}")


if __name__ == "__main__":
    test_training_examples()
    test_sentence_analysis()
    test_lid_comparison()
    test_ipa_mode()
    print("\n" + "=" * 70)
    print("Done!")

# Nahuatl Patterns from Allosaurus Output

Documented patterns from real Allosaurus phoneme recognition on Nahuatl audio.
Source: "Hernan" series, Moctezuma speaking Classical Nahuatl.
Test file: `test.wav` (external, not in repo)

## Verified Words

Manually identified by comparing Allosaurus IPA output against known Nahuatl vocabulary.

| Allosaurus IPA | Nahuatl | Meaning | Notes |
|----------------|---------|---------|-------|
| m o t̪ e k u s o m a | Motekusoma | Moctezuma (name) | Allosaurus renders /t/ as /t̪/ (dental) |
| a m o | amo | no/not | Negation particle |
| k uə e ts a l | quetzal | Quetzal (bird) | /kʷ/ split into k + uə, /ts/ correct |
| k a l a k̟ʲ iː | kalaki | to enter | /kʷ/ → k̟ʲ (palatalized velar) |
| m a t iː s̪ k e | matizke | they will know (future) | Future suffix -izke |
| m i k̟ʲ iː s̪ k̟ʲ ɪ | mikizke | they will die (future) | Future suffix -izke, /kʷ/ → k̟ʲ |
| t e pʲ i lʲ | tepilli | child/noble | Origin of "Pipil" people name |
| t lː a t i | -tlati | Classical Nahuatl suffix | /tɬ/ → t + lː (length mark) |

## Allosaurus-to-Nahuatl Phoneme Mapping

How Allosaurus (universal IPA model) represents Nahuatl phonemes:

| Nahuatl Phoneme | Allosaurus Output | Frequency | Notes |
|-----------------|-------------------|-----------|-------|
| /tɬ/ (lateral affricate) | t͡ɕ, tɕ, tʂ, t + l, t + lː | High | Most distinctive Nahuatl sound, never output as "tɬ" |
| /kʷ/ (labialized velar) | k̟ʲ, k + w, k + uə | High | k̟ʲ is most common variant |
| /ts/ (alveolar affricate) | ts | High | Correctly recognized (-tzin, -tza) |
| /tʃ/ (postalveolar affricate) | t͡ʃʲ, tɕʰ | Medium | Often palatalized |
| /ɬ/ (lateral fricative) | ɕ, ʂ, lʲ | Medium | lʲ most common |
| /t/ before front vowels | tʲ | High | Palatalized, especially in -tli suffix |
| /ʔ/ (glottal stop) | (not detected) | — | Allosaurus does not output glottal stops from this audio |

## Morphological Patterns

Recurring morphological structures visible in the phoneme stream. Useful for future word segmentation (v2+).

### Future Tense: -izke

The Nahuatl future suffix appears multiple times as `iː s̪ k e` or `iː s k e`:
- `m a t iː s̪ k e` → mat-izke (they will know)
- `m i k̟ʲ iː s̪ k̟ʲ ɪ` → mik-izke (they will die)

The pattern `-Vs̪ke` or `-Vske` (where V is a high vowel) is a strong word boundary signal.

### Absolutive Suffix: -tli / -tl

The Classical Nahuatl absolutive suffix appears as:
- `tʲ iː` or `tʲ ɪ` → -tli
- `t lː a` or `t l a` → -tla- (with -tl-)
- `tʂ l a` → -tla- variant

### Negation: amo

Short particle `a m o`, consistently recognized. Good anchor for segmentation.

### Directional Prefix: kal-

`k a l a k̟ʲ iː` shows the root *kalaki* (to enter), with /k/ + /a/ + /l/ consistently at word start.

## Implications for v2+

1. **Word segmentation**: The -izke future suffix and -tli/-tl absolutive suffix are reliable word boundary signals that could drive automatic segmentation
2. **Morpheme analysis**: Polysynthetic structure visible even in raw IPA — prefix-root-suffix patterns are detectable
3. **Lexicon building**: Known words (amo, kalaki, quetzal) could seed a lookup table for improved identification confidence
4. **Model fine-tuning**: The systematic mapping (e.g., /kʷ/ → k̟ʲ) suggests Allosaurus could be post-processed with a Nahuatl-specific normalization layer

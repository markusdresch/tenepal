# Parselmouth vs Allosaurus vs wav2vec2 — Comparison

**Source:** `validation_video/analysis/` (42 segments, first 5min Hernán E03)

---

## TL;DR

| Backend | Strength | Weakness | Role |
|---------|----------|----------|------|
| **Parselmouth** | Acoustic ground truth: voiced/unvoiced, formants, tɬ detection | Can only detect manner, not place — 5 pseudo-phones (`e f x a p`) dominate | **Validator**, not recognizer |
| **Allosaurus** | 67 unique phones, recognizes exotic sounds (ejectives, affricates) | 58% blank rate, often uncertain (many close calls), hallucinates phones | **Good with more confidence** (bias 2-3) |
| **wav2vec2** | Closer to expected text, fewer hallucinations, more consistent | Chinese tone markers (`5`, `ai5`), smaller phone inventory | **Better baseline decoder** |

---

## The Parselmouth "Transliteration" Explained

Parselmouth does **not recognize phonemes**. It measures physics and maps to 5 categories:

| PM Symbol | What it actually measures | Example |
|-----------|-------------------------|---------|
| `/e/` | Voiced + F1 350-500, F2 1800-2400 → close-mid front | Anything vaguely vocalic often becomes /e/ |
| `/a/` | Voiced + F1 600-900, F2 1200-1800 → open | Clearly open vowel |
| `/o/` | Voiced + F1 400-600, F2 700-1200 → mid back | Rarely detected (only 4×) |
| `/f/` | Unvoiced + weak friction (CoG < 1kHz) | Any voiceless transition |
| `/x/` | Unvoiced + stronger friction (CoG > 1kHz) | Any stronger fricative |
| `/p/` | Unvoiced + intensity dip → stop | Any stop burst |
| `/ʃ/` | Unvoiced + high CoG (>3kHz) | Sibilant detected (4×) |
| `/tʃ/` | Stop + friction with high CoG | Affricate (3×) |
| `/β/` | Voiced + friction | Voiced fricative (14×) |
| `/ɾ/` | Voiced, very short (<20ms) | Tap/flap (15×) |
| `/m/` | Voiced + low friction | Sonorant (4×) |

### Why PM hears almost everything as /e/

The F1/F2 formant ranges are **calibrated for isolated vowels**. In running speech (coarticulated, with background music, series production) formants shift substantially. PM's vowel mapping is too narrow → /e/ becomes the "default vowel".

**This is not a weakness — it is a feature.** PM tells you: "There is a voiced segment with these formants." Whether it is /i/ or /e/ is for Allo/w2v2 to decide.

---

## Segment-by-Segment Comparison (Selected Highlights)

### Cue 7: "¡Capitán!" — w2v2 wins clearly

```
Expected:  k a p i t a n
PM-IPA:    ɾ p ɾ f e f a f ɾ       ← "some voiced-unvoiced-voiced pattern"
Allo:      ʁ b ɒ                    ← 80% blank, 3 phones for 7 expected
w2v2:      w i t a                  ← closer, missing onset
```
**Allo at bias=3:** `k͡p̚ ʁ b uə i t ɒ ә t͡ɕ ә` → better, but hallucinates k͡p̚ and t͡ɕ
**PM says:** 9 acoustic events in 455ms → there is MUCH more than Allo's 3 phones

### Cue 8: "¿Dónde está?" — w2v2 again clearly

```
Expected:  d o n d e e s t a
PM-IPA:    p o x a f               ← Stop, back vowel, fricative, open vowel, fricative ✓
Allo:      ɴ o uə lʲ ɹ̩ t͡ɕ ɒ      ← ɴ instead of d, lʲ instead of n, t͡ɕ instead of t
w2v2:      d o n e s t a           ← nearly perfect
```
**PM confirms:** First stop (=d), then back vowel (=o), then fricative zone (=nde), then open vowel (=a), then fricative (=coda)

### Cue 9: "No quiere salir" — Allo's best segment (26% blank)

```
Expected:  n o k j e r e s a l i r
PM-IPA:    ɾ p e f e f e f         ← 8 events
Allo:      ɴ o ʁ k j e ɴ ɒ e ʂ ɒ l iː s  ← 14 phones, good!
w2v2:      n o k i ð e s a l i     ← 10 phones, also good
```
**Interpretation:** When Allo has low blank rate (26%), it is **more detailed than w2v2**. The problem is the default state (58% blank).

### Cue 35: "[LLM] d ɔ l a s t' a s p' a c a c a ch a l ɛ d" — NAH segment, Allo fails

```
PM-IPA:    ɾ p a x a x a f a f a f a x e f ɾ p a x ɾ  ← 21 events in 4.1s
Allo:      b o ŋ̟ m ʌ t͡ɕ a         ← 95% blank! 7 phones for 4.1 seconds!
w2v2:      d ɔ l a s t a s p a o a k a k ɔ m a l ɛ d   ← 21 phones, consistent
```
**PM says:** At least 21 acoustic events. Allo ignores 95% of them.
**w2v2 says:** Many stops and open vowels — **this looks like Nahuatl phonotactics** (CV-CV-CV pattern).
**Allo bias problem:** This segment is the most extreme example. Allo needs at least bias=5 here to produce anything useful.

### Cue 33: "[LLM] p a ʁ ɴ ŋ" — Allo uncertain, PM clear

```
PM-IPA:    p a a a a a a x          ← Stop, then long open vowel, then fricative
Allo:      t a ʁ ɴ ɨ               ← 72% blank
w2v2:      p a5 ŋ onɡ5             ← Chinese tone markers (!)
```
**PM confirms:** Long /a/ vowel (6 frames = 120ms), flanked by consonants. A clear CV pattern.
**w2v2 problem:** The `5` tone markers show w2v2 is falling back to its Chinese training here.

### Cue 41: "[LLM] ɒ a p ɒ ŋ p i y u" — NAH, PM hears only vowels

```
PM-IPA:    a a a a a C a            ← Almost entirely voiced + open
Allo:      ɒ p ɒ n pʲ iː ʁ uː     ← 64% blank, but detected stops+nasals
w2v2:      t əɜ n p j              ← short, little info
```
**PM confirms:** Predominantly open vowels with brief consonant islands. Consistent with NAH phonotactics (vowel-heavy).
**Here Allo is better** — it detects the consonant islands that PM only sees as brief "C" blips.

---

## Allo Blank-Bias Sweep — Summary

| Bias | Avg Blank% | Effect |
|------|-----------|--------|
| 0.0 | 58-80% | Too conservative, swallows phones |
| 1.0 | 47-71% | Slight improvement |
| **2.0** | **20-52%** | **Sweet spot for SPA** |
| **3.0** | **6-22%** | **Sweet spot for NAH** (needs more recall) |
| 5.0 | 0-11% | Noisy, but usable for NAH detection |
| 10.0+ | 0% | Hallucinates (ʔ spam, ghost phones) |

**Key insight:** NAH segments need **higher bias** than SPA because:
1. Allo was not trained on NAH → default is "silence"
2. NAH has sounds (tɬ, ʔ, long vowels) that Allo does not know → blank
3. SPA segments are already usable at bias=0 (26% blank at Cue 9)

→ **Recommendation: Adaptive bias** — SPA at 0-2, NAH/MAY at 3-5

---

## CTC Close-Call Analysis

Allosaurus CTC frames show "margin" = distance between blank logit and best phone logit:
- **Margin < 0**: Phone wins → decoded
- **Margin 0-2**: Close call → becomes a phone at higher bias
- **Margin > 2**: Blank wins clearly → probably genuine silence

| Cue | Close Calls | Decoded Phones | Potential (bias=3) |
|-----|------------|----------------|-------------------|
| 7 (Capitán) | 12 | 3 | 10 (+233%) |
| 9 (No quiere) | 18 | 14 | 16 (+14%) |
| 11 (Dialasutih) | 12 | 2 | 9 (+350%) |
| 35 (NAH long) | 13 | 7 | 14 (+100%) |

**Close calls = hidden phones that Allo sees but does not output.** At bias=3, most of them surface.

---

## Voicing Agreement: Which backend is closer to the acoustics?

PM's voiced/unvoiced ratio is ground truth (F0 > 0 = voiced, period).

| Cue | PM voiced% | Allo voiced% | w2v2 voiced% | Winner |
|-----|-----------|-------------|-------------|----------|
| 7 (Capitán) | ~40% | 33% (ʁ b ɒ) | 75% (w i t a) | **Allo** |
| 8 (Dónde) | ~55% | 57% | 71% | **Allo** |
| 9 (No quiere) | ~60% | 64% | 70% | **Allo** |
| 35 (NAH) | ~75% | 57% | 71% | **w2v2** |
| 41 (NAH) | ~90% | 75% | 60% | **Allo** |

**Result:** For voicing ratio, Allo is **slightly better** than w2v2, especially for SPA.
w2v2 tends to produce too many voiced phones.

---

## Conclusions: Direction Forward

### 1. Three-Layer Architecture Confirmed

```
Layer 3: Parselmouth  → Acoustic ground truth (voicing, formants, tɬ, rhythm)
Layer 2: Allo + w2v2  → Phoneme hypotheses (with bias tuning)
Layer 1: Fusion       → Decision based on agreement + PM validation
```

### 2. Allosaurus: More confidence = better, BUT adaptive

- **SPA segments:** bias=0-2 is sufficient, Allo is already good at low blank
- **NAH segments:** bias=3-5 needed, otherwise Allo swallows half
- **Implementation:** First pass at bias=0, then re-run segments with >70% blank at bias=3

### 3. wav2vec2: Better text decoder, worse phone decoder

- w2v2 is closer to "what was said" (word level)
- w2v2 is worse at "how it was pronounced" (phone level)
- The Chinese tone markers (`5`, `ai5`, `onɡ5`) are a problem for NAH
- **Role:** w2v2 as "word anchor", Allo as "phone detail"

### 4. Parselmouth: Do not replace, use as validator

PM cannot say "that is a /k/" (place), but it can say:
- ✅ "There is a voiceless stop here" (manner)
- ✅ "There is tɬ here" (specific enough for NAH marker)
- ✅ "There is a vowel with F1=700, F2=1400" (→ /a/)
- ✅ "There are 21 acoustic events, not 7" (Allo blank detector)
- ❌ "That is a /t/ not a /k/" (cannot do this)

### 5. Concrete Next Steps

| Priority | Action | Reason |
|----------|--------|--------|
| 🔴 | **Adaptive Allo bias** | +100-350% phone recovery for NAH |
| 🔴 | **PM as blank detector** | PM events > Allo phones → increase bias |
| 🟡 | **w2v2 tone marker filter** | Strip `5`, `ai5` etc. before fusion |
| 🟡 | **Voicing cross-check** in fusion | PM voiced% as weight for Allo vs w2v2 |
| 🟢 | **PM vowel mapping** improvement | Expand F1/F2 ranges for coarticulated speech |

### 6. The Insight in One Sentence

> **Allosaurus sees more than it says (58% blank = hidden phones), wav2vec2 says more than it sees (hallucinates words), and Parselmouth sees exactly what is there — but cannot name it. Fusion must surface Allo's hidden phones, use w2v2's word structure, and let PM arbitrate.**

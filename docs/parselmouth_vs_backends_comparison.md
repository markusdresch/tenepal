# Parselmouth vs Allosaurus vs wav2vec2 — Gegenüberstellung

**Quelle:** `validation_video/analysis/` (42 Segmente, erste 5min Hernán E03)

---

## TL;DR

| Backend | Stärke | Schwäche | Vertrauen |
|---------|--------|----------|-----------|
| **Parselmouth** | Akustische Wahrheit: Voiced/Unvoiced, Formanten, tɬ-Detektion | Kann nur Manner, nicht Place — 5 Pseudo-Phone (`e f x a p`) dominieren | **Validator**, nicht Recognizer |
| **Allosaurus** | 67 unique Phones, erkennt exotische Laute (Ejektive, Affrikate) | 58% blank rate, oft unsicher (viele close calls), halluziniert Phones | **Gut mit mehr Selbstbewusstsein** (bias 2-3) |
| **wav2vec2** | Näher am erwarteten Text, weniger Halluzinationen, konsistenter | Chinesische Tonmarker (`5`, `ai5`), weniger Phone-Inventar | **Besserer Baseline-Decoder** |

---

## Die Parselmouth-"Transliteration" erklärt

Parselmouth erkennt **keine Phoneme**. Es misst Physik und mappt auf 5 Kategorien:

| PM-Symbol | Was es wirklich misst | Beispiel |
|-----------|----------------------|----------|
| `/e/` | Voiced + F1 350-500, F2 1800-2400 → close-mid front | Alles was "irgendwie Vokal" ist wird oft /e/ |
| `/a/` | Voiced + F1 600-900, F2 1200-1800 → open | Klar offener Vokal |
| `/o/` | Voiced + F1 400-600, F2 700-1200 → mid back | Selten erkannt (nur 4×) |
| `/f/` | Unvoiced + schwache Friktion (CoG < 1kHz) | Jeder stimmlose Übergang |
| `/x/` | Unvoiced + stärkere Friktion (CoG > 1kHz) | Jeder kräftigere Frikativ |
| `/p/` | Unvoiced + Intensitätsdip → Stop | Jeder Stop-Burst |
| `/ʃ/` | Unvoiced + hoher CoG (>3kHz) | Sibilant erkannt (4×) |
| `/tʃ/` | Stop+Friktion mit hohem CoG | Affricate (3×) |
| `/β/` | Voiced + Friktion | Stimmhafter Frikativ (14×) |
| `/ɾ/` | Voiced, sehr kurz (<20ms) | Tap/Flap (15×) |
| `/m/` | Voiced + niedrige Friktion | Sonorant (4×) |

### Warum PM fast alles als /e/ hört

Die F1/F2-Formant-Ranges sind **auf isolierte Vokale kalibriert**. In fließender Rede (koartikuliert, mit Hintergrundmusik, Serienproduktion) verschieben sich Formanten massiv. PM's Vokal-Mapping ist zu eng → /e/ wird zum "Default-Vokal".

**Das ist keine Schwäche — es ist ein Feature.** PM sagt dir: "Hier ist ein stimmhafter Abschnitt mit diesen Formanten." Ob das ein /i/ oder /e/ ist, müssen Allo/w2v2 entscheiden.

---

## Segment-für-Segment Vergleich (ausgewählte Highlights)

### Cue 7: "¡Capitán!" — w2v2 gewinnt klar

```
Erwartet:  k a p i t a n
PM-IPA:    ɾ p ɾ f e f a f ɾ       ← "irgendwas Voiced-Unvoiced-Voiced"
Allo:      ʁ b ɒ                    ← 80% blank, 3 Phones für 7 erwartete
w2v2:      w i t a                  ← näher, fehlt Anfang
```
**Allo bei bias=3:** `k͡p̚ ʁ b uə i t ɒ ә t͡ɕ ә` → besser, aber halluziniert k͡p̚ und t͡ɕ
**PM sagt:** 9 akustische Events in 455ms → da ist VIEL mehr drin als Allos 3 Phone

### Cue 8: "¿Dónde está?" — w2v2 wieder klar

```
Erwartet:  d o n d e e s t a
PM-IPA:    p o x a f               ← Stop, Back-Vokal, Frikativ, Open-Vokal, Frikativ ✓
Allo:      ɴ o uə lʲ ɹ̩ t͡ɕ ɒ      ← ɴ statt d, lʲ statt n, t͡ɕ statt t
w2v2:      d o n e s t a           ← fast perfekt
```
**PM bestätigt:** Erst Stop (=d), dann Back-Vokal (=o), dann Frikativ-Zone (=nde), dann Open-Vokal (=a), dann Frikativ (=Schluss)

### Cue 9: "No quiere salir" — Allos bestes Segment (26% blank)

```
Erwartet:  n o k j e r e s a l i r
PM-IPA:    ɾ p e f e f e f         ← 8 Events
Allo:      ɴ o ʁ k j e ɴ ɒ e ʂ ɒ l iː s  ← 14 Phones, gut!
w2v2:      n o k i ð e s a l i     ← 10 Phones, auch gut
```
**Interpretation:** Wenn Allo wenig blankt (26%), ist es **detaillierter als w2v2**. Das Problem ist der Default-Zustand (58% blank).

### Cue 35: "[LLM] d ɔ l a s t' a s p' a c a c a ch a l ɛ d" — NAH-Segment, Allo versagt

```
PM-IPA:    ɾ p a x a x a f a f a f a x e f ɾ p a x ɾ  ← 21 Events in 4.1s
Allo:      b o ŋ̟ m ʌ t͡ɕ a         ← 95% blank! 7 Phones für 4.1 Sekunden!
w2v2:      d ɔ l a s t a s p a o a k a k ɔ m a l ɛ d   ← 21 Phones, konsistent
```
**PM sagt:** Da sind mindestens 21 akustische Events. Allo ignoriert 95% davon.
**w2v2 sagt:** Viele Stops und offene Vokale — **das klingt nach Nahuatl-Phonotaktik** (CV-CV-CV Muster).
**Allo-Bias-Problem:** Dieses Segment ist das krasseste Beispiel. Allo braucht hier mindestens bias=5 um überhaupt was zu liefern.

### Cue 33: "[LLM] p a ʁ ɴ ŋ" — Allo unsicher, PM klar

```
PM-IPA:    p a a a a a a x          ← Stop, dann langer offener Vokal, dann Frikativ
Allo:      t a ʁ ɴ ɨ               ← 72% blank
w2v2:      p a5 ŋ onɡ5             ← chinesische Tonmarker (!)
```
**PM bestätigt:** Langer /a/-Vokal (6 Frames = 120ms), umrahmt von Konsonanten. Das ist ein klares CV-Muster.
**w2v2-Problem:** Die `5`-Tonmarker zeigen, dass w2v2 hier auf sein chinesisches Training zurückfällt.

### Cue 41: "[LLM] ɒ a p ɒ ŋ p i y u" — NAH, PM hört nur Vokale

```
PM-IPA:    a a a a a C a            ← Fast durchgehend voiced + open
Allo:      ɒ p ɒ n pʲ iː ʁ uː     ← 64% blank, aber erkannte Stops+Nasale
w2v2:      t əɜ n p j              ← kurz, wenig Info
```
**PM bestätigt:** Überwiegend offene Vokale mit kurzen Konsonant-Inseln. Das passt zu NAH-Phonotaktik (vokallastig).
**Hier ist Allo besser** — es erkennt die Konsonanten-Inseln die PM nur als kurze "C"-Blips sieht.

---

## Allo Blank-Bias Sweep — Zusammenfassung

| Bias | Ø Blank% | Effekt |
|------|----------|--------|
| 0.0 | 58-80% | Zu konservativ, verschluckt Phones |
| 1.0 | 47-71% | Leichte Verbesserung |
| **2.0** | **20-52%** | **Sweet Spot für SPA** |
| **3.0** | **6-22%** | **Sweet Spot für NAH** (braucht mehr Recall) |
| 5.0 | 0-11% | Noisy, aber für NAH-Detektion brauchbar |
| 10.0+ | 0% | Halluziniert (ʔ-Spam, Geister-Phones) |

**Key Insight:** NAH-Segmente brauchen **höheren Bias** als SPA, weil:
1. Allo wurde nicht auf NAH trainiert → Default ist "Stille"
2. NAH hat Laute (tɬ, ʔ, lange Vokale) die Allo nicht kennt → blank
3. SPA-Segmente sind bei bias=0 schon brauchbar (26% blank bei Cue 9)

→ **Empfehlung: Adaptiver Bias** — SPA bei 0-2, NAH/MAY bei 3-5

---

## CTC Close-Call Analyse

Allosaurus CTC-Frames zeigen "Margin" = Abstand zwischen blank-Logit und bestem Phone-Logit:
- **Margin < 0**: Phone gewinnt → decoded
- **Margin 0-2**: Close call → bei höherem Bias wird's ein Phone
- **Margin > 2**: Blank gewinnt klar → wahrscheinlich echt Stille

| Cue | Close Calls | Decoded Phones | Potential (bias=3) |
|-----|------------|----------------|-------------------|
| 7 (Capitán) | 12 | 3 | 10 (+233%) |
| 9 (No quiere) | 18 | 14 | 16 (+14%) |
| 11 (Dialasutih) | 12 | 2 | 9 (+350%) |
| 35 (NAH long) | 13 | 7 | 14 (+100%) |

**Close Calls = versteckte Phones die Allo sieht aber nicht outputtet.** Bei bias=3 kommen die meisten raus.

---

## Voicing-Agreement: Wer liegt näher an der Akustik?

PM's Voiced/Unvoiced-Ratio ist Ground Truth (F0 > 0 = voiced, Punkt).

| Cue | PM voiced% | Allo voiced% | w2v2 voiced% | Gewinner |
|-----|-----------|-------------|-------------|----------|
| 7 (Capitán) | ~40% | 33% (ʁ b ɒ) | 75% (w i t a) | **Allo** |
| 8 (Dónde) | ~55% | 57% | 71% | **Allo** |
| 9 (No quiere) | ~60% | 64% | 70% | **Allo** |
| 35 (NAH) | ~75% | 57% | 71% | **w2v2** |
| 41 (NAH) | ~90% | 75% | 60% | **Allo** |

**Ergebnis:** Bei Voicing-Ratio ist Allo **leicht besser** als w2v2, besonders bei SPA.
w2v2 tendiert dazu, zu viele Voiced-Phones zu produzieren.

---

## Schlussfolgerungen: Wo soll die Richtung hin?

### 1. Drei-Schichten-Architektur bestätigt

```
Layer 3: Parselmouth  → Akustische Wahrheit (Voicing, Formanten, tɬ, Rhythmus)
Layer 2: Allo + w2v2  → Phonem-Hypothesen (mit Bias-Tuning)
Layer 1: Fusion       → Entscheidung basierend auf Agreement + PM-Validation
```

### 2. Allosaurus: Mehr Selbstbewusstsein = besser, ABER adaptiv

- **SPA-Segmente:** bias=0-2 reicht, Allo ist bei niedrigem Blank schon gut
- **NAH-Segmente:** bias=3-5 nötig, sonst verschluckt Allo die Hälfte
- **Implementierung:** Erst-Pass mit bias=0, dann Segmente mit >70% blank nochmal mit bias=3

### 3. wav2vec2: Besserer Text-Decoder, schlechterer Phone-Decoder

- w2v2 ist näher am "was wurde gesagt" (Wort-Ebene)
- w2v2 ist schlechter beim "wie wurde es ausgesprochen" (Phone-Ebene)
- Die chinesischen Tonmarker (`5`, `ai5`, `onɡ5`) sind ein Problem bei NAH
- **Rolle:** w2v2 als "Wort-Anker", Allo als "Phone-Detail"

### 4. Parselmouth: Nicht ersetzen, sondern als Validator

PM kann nicht sagen "das ist ein /k/" (Place), aber es kann sagen:
- ✅ "Hier ist ein stimmloser Stop" (Manner)
- ✅ "Hier ist tɬ" (spezifisch genug für NAH-Marker)
- ✅ "Hier ist ein Vokal mit F1=700, F2=1400" (→ /a/)
- ✅ "Hier sind 21 akustische Events, nicht 7" (Allo-Blank-Detektor)
- ❌ "Das ist ein /t/ und kein /k/" (kann es nicht)

### 5. Konkrete nächste Schritte

| Prio | Aktion | Warum |
|------|--------|-------|
| 🔴 | **Adaptiver Allo-Bias** implementieren | +100-350% Phone-Recovery bei NAH |
| 🔴 | **PM als Blank-Detektor** nutzen | PM-Events > Allo-Phones → Bias hochdrehen |
| 🟡 | **w2v2 Tonmarker-Filter** | `5`, `ai5` etc. strippen vor Fusion |
| 🟡 | **Voicing-Cross-Check** in Fusion | PM voiced% als Gewicht für Allo vs w2v2 |
| 🟢 | **PM Vokal-Mapping** verbessern | F1/F2-Ranges erweitern für koartikulierte Rede |

### 6. Die Erkenntnis in einem Satz

> **Allosaurus sieht mehr als es sagt (58% blank = versteckte Phones), wav2vec2 sagt mehr als es sieht (halluziniert Wörter), und Parselmouth sieht genau was da ist — aber kann es nicht benennen. Die Fusion muss Allos versteckte Phones rausholen, w2v2's Wort-Struktur nutzen, und PM als Schiedsrichter einsetzen.**

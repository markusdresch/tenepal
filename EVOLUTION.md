# Tenepal Evolution

Chronological documentation of experiments, failures, and resulting improvements.
Format: What we tried → What happened → What we learned → How we solved it.

**Scope:** Hernan Season 1 (8 chapters), Apocalypto samples, La Otra Conquista

---

## Phase 1: Foundations (12 Feb 2026)

### E001: Allosaurus as Primary Phoneme Backend
**Hypothesis:** Allosaurus (2000+ languages) should work universally well.

**Result:**
- Spanish: 58% blank frames, many phonemes swallowed
- Nahuatl: 80-95% blank frames — almost completely lost
- Hallucinates East Asian phonemes (tɕh, ɨ, ɤ) when uncertain

**Takeaway:** Universalist = mediocre at everything. Unusable as standalone for low-resource languages.

**Solution:** Dual backend with wav2vec2 as fusion partner.

---

### E002: wav2vec2-phoneme as Alternative
**Hypothesis:** Facebook's wav2vec2 fine-tuned on CommonVoice is more stable.

**Result:**
- Spanish: Significantly better, reproducible ("Arcabuceros" → `a l k a β u f e ɾ o s`)
- Nahuatl: Also poor, but different errors (Chinese tone markers: 5, ai5)

**Takeaway:** Each backend has different strengths. Allo better for exotic phonemes (tɬ, ejectives), w2v2 better for Romance languages.

**Solution:** IPA fusion via Needleman-Wunsch alignment.

---

## Phase 2: Phonotactics Experiments (13-15 Feb)

### E003: Bigram-based Language Classification
**Hypothesis:** Phoneme bigrams are language-specific enough for classification.

**Result:**
- 44% bigram overlap between NAH and SPA
- Even higher due to pipeline noise (garbage phonemes)
- Works for NAH vs DEU, but not NAH vs SPA

**Takeaway:** Phonotactics is ONE signal, not THE signal.

**Solution:** Combination with marker-based approaches.

---

### E004: Trigram instead of Bigram
**Hypothesis:** Longer n-grams are more discriminative.

**Result:** FAIL — MAY inflation from 12% to 43%

**Analysis:** Sparsity problem. Smallest corpus (MAY: 1122 tokens) wins on OOV tokens because it says "no" with less certainty.

**Takeaway:** More context doesn't help when data is too sparse.

**Solution:** Abandoned. Implemented language prior instead (commit ffd9902).

---

### E005: Language Prior for Corpus Bias
**Hypothesis:** Explicit prior probabilities correct corpus size bias.

**Setup:** `score += log(prior)` with SPA=0.55, NAH=0.28, MAY=0.12

**Result:** MAY inflation from 43% to 4.3% — SUCCESS

**Takeaway:** Bayesian thinking saves statistical models.

---

## Phase 3: Whisper Integration (14-16 Feb)

### E006: Whisper for Language Detection
**Hypothesis:** Whisper's built-in language detection helps with LID.

**Result:** CATASTROPHE
- Nahuatl: 100% detected as Spanish/English
- Whisper's confidence is poorly calibrated — gives 95% confidence on complete garbage

**Takeaway:** Whisper doesn't know indigenous languages. Auto-detection is useless.

**Solution:** Always force language parameter based on Tenepal's decision.

---

### E007: Whisper Hallucination as Signal
**Hypothesis:** When Whisper hallucinates Danish/Dutch/Norwegian on NAH audio, that's a positive NAH signal.

**Result:** BREAKTHROUGH
- NAH audio → Whisper says "Det var jeg stoppen" (Danish) = strong NAH signal
- Phonotactically similar (Germanic + unknown phonemes)
- Very reliable as rejection filter

**Takeaway:** Errors can be features. Whisper's hallucinations are predictable.

**Implementation:** `da/nl/no` detection in Whisper output = NAH flag

---

### E008: Spanish Leak Detector
**Hypothesis:** Whisper sometimes produces Spanish text on NAH audio (false positive).

**Setup:** Detector for Spanish words in Whisper output with NAH tags.

**Result:** +17% SPA recall through leak correction

**Takeaway:** Two-way validation: Hallucination↔Leak as complementary signals.

---

## Phase 4: Segmentation & VAD (16-18 Feb)

### E009: Root Cause Analysis: Why is IPA so Unstable?
**Observation:** Old SRTs (13 Feb) vs New (16 Feb) — completely different IPA for the same dialogues.

**Analysis:**
- The backends didn't change
- VAD/diarization cuts differently → different audio chunks → different IPA
- ±100ms at the boundary = next speaker bleeds in OR silence → hallucination

**Takeaway:** Garbage in, garbage out. Upstream fix needed.

**Solution:** Voice onset trimming (Parselmouth-based).

---

### E010: Voice Onset Trimming
**Hypothesis:** Signal-based trimming of segment boundaries improves backend output.

**Setup:**
- Parselmouth intensity analysis
- Two windows: Whisper (wide) vs IPA (tightly trimmed)

**Result:** Significant improvement in IPA quality on tested segments.

**Takeaway:** First real upstream fix. Everything downstream benefits automatically.

---

## Phase 5: Acoustic Markers (19-21 Feb)

### E011: Parselmouth as Third Detection Layer
**Hypothesis:** Acoustic features (voicing, formants, CoG) as ground truth for backend validation.

**Result:**
- Parselmouth reliably detects manner (voiced/unvoiced, stop/fricative)
- Cannot detect place (t vs k)
- Perfect as quality metric: PM events > 2x backend phones = problem

**Takeaway:** Stethoscope (measuring features) vs doctor (making diagnoses) — both needed.

---

### E012: tɬ Acoustic Detection
**Hypothesis:** tɬ (voiceless lateral affricate) has a unique acoustic profile, detectable without ML.

**Setup:**
- voicing_ratio < 0.3
- friction_ratio > 0.25
- spectral centroid 2.5-5.5 kHz (tɬ) vs 5-8 kHz (tʃ)
- friction duration > 40ms

**Result:**
- 57 segments with tɬ signal in Chapter 3 (9.4%)
- **11 SPA segments with tɬ = definitely misclassified NAH**
- tɬ does NOT exist in Spanish

**Takeaway:** tɬ is a hard NAH marker. Deterministic, no ML bias. **Proof that confusion exists.**

**Implementation:** `scripts/detect_tl.py`, integration as hard override in pipeline in progress.

---

### E013: Adaptive CTC Blank Bias
**Hypothesis:** Allosaurus' blank token absorbs too many frames. Bias reduction recovers lost phonemes.

**Setup:** Subtract constant from blank logit before decoding.

**Result:**

| Segment | bias=0 blank% | bias=3 blank% | Recovery |
|---------|---------------|---------------|----------|
| "Capitán" (SPA) | 80% | 7% | **+233%** |
| NAH segment (4.1s) | 95% | ~50% | **+100%** |

**Trigger:** PM_events > 2x Allo_phones → re-run with bias=3.0

**Takeaway:** CTC decoding is tunable. Allo SEES the phonemes, it just doesn't emit them.

**Implementation:** Adaptive blank bias in Modal pipeline (commit bff9024).

---

## Phase 6: Lexicon & Corpus (20-21 Feb)

### E014: Corpus-based NAH Lexicon
**Hypothesis:** Larger lexicon (2791 vs 20 entries) improves NAH detection.

**Setup:** Jonathan Amith Zacatlan Corpus → IPA lexicon with frequencies.

**Result:**

| min_freq | entries | segs matched | improvement |
|----------|---------|--------------|-------------|
| OLD | 20 | 28.2% | baseline |
| 50 | 140 | 57.6% | **+29.4%** |
| 3 | 2,024 | 76.5% | **+48.3%** |

**Problem:** 56% of SPA segments also match NAH words — too many false positives.

**Analysis:** Short morphemes (se, i:n, ka) randomly match in any language.

**Solution:** min_phonemes=5 for cross-language matching, OR only hard markers (tɬ, kʷ, ʔ).

---

### E015: match_subsequence() Performance
**Problem:** O(n²) algorithm, 17 minutes for 8 chapters.

**Solution:** Pre-computation + bounded edit distance + first-phoneme index → O(n)

**Result:** 56x speedup (17min → 71s)

**Takeaway:** Optimize algorithms before scaling infrastructure.

---

## Insights & Principles

### What works:
1. **Multi-signal fusion** — No single signal suffices, combination is robust
2. **Errors as features** — Whisper hallucinations, Allo bias, all usable
3. **Acoustic ground truth** — Parselmouth as validator for ML backends
4. **Hard markers** — tɬ, ejectives: deterministic, no model bias
5. **Upstream fixes** — Voice onset trimming helps everything downstream

### What doesn't work:
1. **Universalists** — Allosaurus (2000 languages) is good at none
2. **More context with sparse data** — Trigrams worsen sparsity
3. **Whisper auto-detection** — Doesn't know indigenous languages
4. **Naive score fusion** — Different scales = dominant signal wins

### Honest evaluation:
- Ground truth: only 57/259 segments manually annotated
- Accuracy on sample: ~80% (60/75)
- **Known errors:** 11 SPA segments with tɬ = misclassified NAH
- "0% confusion" in the paper = marketing (based on confidence threshold, not real accuracy)

### Open questions:
- Voice onset trimming: How many ms is optimal?
- tɬ vs tʃ distinction: fine-tune centroid threshold
- Ejective detection: robustness across speakers?
- MAY-specific markers: ʔ alone isn't enough

---

## Phase 7: Voice Separation (22 Feb 2026)

### E016: SepFormer for Overlap Scenes
**Problem:** Trilingual dialogue with overlapping voices. Pipeline sees the mix, classifies incorrectly.

**Crown jewel use case:** Hernan E03 20:20-20:40 — Malinche (NAH) translates between Cortes (SPA) and Maya chief (MAY). Three voices, two overlapping constantly.

**Setup:** 3-Way Comparison
- Option A: SepFormer → Pyannote (separation first)
- Option B: Pyannote → SepFormer (diarization first)
- Option C: Cascaded 4-stems (2x SepFormer)

**Result:** Option A clearly best separation

| Stem | F0 Median | Interpretation |
|------|-----------|----------------|
| SRC_00 | 126 Hz | Male voice (Cortes/Chief) |
| SRC_01 | 231 Hz | Female voice (Malinche) |

**Breakthrough #1:** SepFormer separates acoustically cleanly by pitch, not by identity. Male vs Female = perfect separation.

**Breakthrough #2:** Malinche stem (SRC_01) shows code-switching:
- 4 SPA segments
- 3 NAH segments
- = Interpreter actively translating back and forth!

**Takeaway:** Voice separation makes overlap scenes analyzable. 5 original segments → 26 after separation.

---

### E017: 8kHz Degradation
**Problem:** SepFormer outputs 8kHz audio. Pipeline expects 16kHz.

**Result:** Language detection degrades at 8kHz:
- More OTH classifications
- Less clear NAH/SPA separation
- Whisper lang detection unreliable

**Solution:** 8kHz → 16kHz upsampling before pipeline (TODO)

**Takeaway:** Sample rate matters. Transformer backends are trained on 16kHz.

---

### E018: Merged STEM SRT Format
**Hypothesis:** Separate SRTs per stem lose temporal overlap information.

**Solution:** One SRT with STEM tags:
```
[NAH|SPEAKER_01|STEM_A] ʔɒ a t ә i n i
[SPA|SPEAKER_00|STEM_B] Quiero que vayamos a...
```

**Result:**
- 26 cues with overlapping timestamps
- Both stems visually distinguishable
- Annotator can compare simultaneous utterances

**Takeaway:** Data format must reflect overlap reality. Single-track SRT = information loss.

---

## Insights & Principles

### What works:
1. **Multi-signal fusion** — No single signal suffices, combination is robust
2. **Errors as features** — Whisper hallucinations, Allo bias, all usable
3. **Acoustic ground truth** — Parselmouth as validator for ML backends
4. **Hard markers** — tɬ, ejectives: deterministic, no model bias
5. **Upstream fixes** — Voice onset trimming helps everything downstream
6. **Voice separation** — SepFormer for overlap scenes: 5 → 26 segments, code-switching visible

### What doesn't work:
1. **Universalists** — Allosaurus (2000 languages) is good at none
2. **More context with sparse data** — Trigrams worsen sparsity
3. **Whisper auto-detection** — Doesn't know indigenous languages
4. **Naive score fusion** — Different scales = dominant signal wins
5. **8kHz audio** — Degrades language detection significantly

### Honest evaluation:
- Ground truth: only 57/259 segments manually annotated
- Accuracy on sample: ~80% (60/75)
- **Known errors:** 11 SPA segments with tɬ = misclassified NAH
- "0% confusion" in the paper = marketing (based on confidence threshold, not real accuracy)
- **Voice separation:** Overlap scenes now analyzable, but 8kHz upsampling needed

### Open questions:
- Voice onset trimming: How many ms is optimal?
- tɬ vs tʃ distinction: fine-tune centroid threshold
- Ejective detection: robustness across speakers?
- MAY-specific markers: ʔ alone isn't enough

---

## Phase 8: Voice Separation Comparison (22 Feb 2026)

### E019: 4-Way Separation Method Comparison
**Problem:** SepFormer has too much bleeding between stems. Alternative methods needed.

**Setup:** Comparison on E03 20:10-20:42 (32s, Male 113Hz + Female 235Hz)
- SepFormer (speechbrain/sepformer-whamr)
- ConvTasNet (asteroid/ConvTasNet_Libri2Mix)
- PYIN + Wiener (librosa pitch detection + spectral filtering)
- Hybrid (adaptive harmonic extraction with bandpass bank)

**Result:**

| Method | Male F0 Range | Female F0 Range | Purity | Score |
|---------|---------------|-----------------|--------|-------|
| **Pitch** | 86-288Hz | 120-318Hz | 45.8% | **23.0** |
| SepFormer | 61-493Hz | 59-285Hz | 44.3% | 22.1 |
| ConvTasNet | 58-497Hz | 52-499Hz | 41.6% | 20.8 |
| Hybrid | 58-499Hz | 62-499Hz | 29.0% | 14.5 |

**Takeaway:**
1. **Pitch-based wins** with distinct F0s (Male ~120Hz vs Female ~230Hz)
2. ML models (SepFormer, ConvTasNet) have significant bleeding
3. All methods at 8kHz → Pitch-based at 16kHz = better LID quality
4. Hybrid approach (adaptive bandpass) introduces artifacts instead of improvement

**Limitations:**
- Pitch-based fails when F0s overlap (two male voices)
- All methods have >200Hz range = noticeable bleeding
- "Clean separation" (other voice inaudible) not achieved with any method

**Recommendation:** Pitch-based as default for male/female pairs. SepFormer as fallback for similar voices.

**Status:** Voice separation parked. CSS (Continuous Speech Separation) noted as candidate for Phase 2.

---

## Phase 9: NAH Detection Improvements (22 Feb 2026)

### E020: P0 — NAH Lexicon Expansion
**Problem:** Modal pipeline had only ~120 lexicon entries vs 45k in local codebase.

**Setup:** `min_freq` lowered from 50 to 3.

**Result:**

| Lexicon | Entries | NAH Detection |
|---------|---------|---------------|
| OLD (min_freq=50) | ~120 | Baseline |
| NEW (min_freq=3) | **2,004** | +13 NAH in 32s test |

**Takeaway:** Lexicon expansion is low-hanging fruit. No ML needed, immediate improvement.

**Implementation:** commit cce997d

---

### E021: P2 — Speaker Prior Implementation
**Problem:** NAH speakers are misclassified per-segment. SPEAKER_12/09/07/14 are 100% NAH in annotations, but pipeline shows 47% OTH.

**Hypothesis:** Speakers rarely switch languages. Majority vote per speaker → propagation to all their segments.

**Setup:**
```python
def apply_speaker_prior(results, min_segments=3, min_ratio=0.6):
    # 1. Count languages per speaker (excluding OTH)
    # 2. If speaker has 60%+ in one language → lock
    # 3. Override their OTH segments with majority language
```

**Result on E03 (45min):**

| Metric | Before | After |
|--------|--------|-------|
| **OTH%** | ~47% | **12.7%** |
| **NAH Segments** | ~150 | **333** |
| **Speakers Locked** | 0 | **15** (13 NAH, 2 SPA) |
| **Segments Upgraded** | 0 | **76** |

**Language Distribution:**
- NAH: 333 (55.6%)
- SPA: 169 (28.2%)
- OTH: 76 (12.7%)
- MAY: 21 (3.5%)

**Takeaway:** Speaker prior is dramatically effective for films with consistent speakers. +80pp NAH accuracy.

**Implementation:** commit fa7e673

---

### E022: P1 — Lexicon Override before Marker Scoring
**Problem:** `check_nah_lexicon()` only returns a match at threshold=0.75. Partial matches (0.6-0.74) are overridden by marker scoring.

**Hypothesis:** Lexicon signal should preempt marker scoring, not only on perfect match.

**Setup:**
- `check_nah_lexicon()` now returns `(word, score)` tuple
- P1 logic: if `lex_score >= 0.6` → NAH, even without full word match
- Applied to primary AND compare backends

**Implementation:** commit 1fb5ebd

**Status:** Test running, results pending.

---

## Insights & Principles (Updated)

### What works:
1. **Multi-signal fusion** — No single signal suffices, combination is robust
2. **Errors as features** — Whisper hallucinations, Allo bias, all usable
3. **Acoustic ground truth** — Parselmouth as validator for ML backends
4. **Hard markers** — tɬ, ejectives: deterministic, no model bias
5. **Upstream fixes** — Voice onset trimming helps everything downstream
6. **Voice separation** — SepFormer for overlap scenes: 5 → 26 segments, code-switching visible
7. **Speaker prior** — Majority vote per speaker: OTH 47% → 12.7% (**+80pp NAH accuracy**)
8. **Lexicon expansion** — 120 → 2004 entries: immediate impact without ML

### What doesn't work:
1. **Universalists** — Allosaurus (2000 languages) is good at none
2. **More context with sparse data** — Trigrams worsen sparsity
3. **Whisper auto-detection** — Doesn't know indigenous languages
4. **Naive score fusion** — Different scales = dominant signal wins
5. **8kHz audio** — Degrades language detection significantly
6. **Voice separation with same F0** — All methods have bleeding with overlapping pitch ranges

### Honest evaluation (E03):
- **Old pipeline accuracy:** ~14% (against ground truth)
- **New pipeline (P0+P1+P2):** ~55% NAH correct (estimated)
- **OTH rate:** 47% → 12.7%
- Voice separation parked (CSS for later)

### What's still missing for better NAH detection:
1. ~~P0: Lexicon expansion~~ ✓ DONE
2. ~~P1: Lexicon override before marker scoring~~ ✓ DONE
3. ~~P2: Speaker prior~~ ✓ DONE
4. **P3:** Multi-segment confidence (context window)
5. **P4:** Diarization-informed language (speaker consistency across scenes)
6. **P5:** Propagate tɬ hard override to all segments of the same speaker

---

*Last updated: 2026-02-22*
*Next review: P3-P5 prioritization, annotation verification*

---

## Phase F: Whisper Finetuning (NAH)

### F001: LoRA Finetuning on OpenSLR-92 (Puebla Nahuatl)

**Setup:**
- Model: whisper-large-v3, LoRA (r=32, α=64, target: q_proj/v_proj)
- Corpus: OpenSLR 92 — 135,544 segments, 150.1h training, 20.6h dev, 19.0h test
- Hardware: A100 80GB, fp16, batch 16 (8 + grad_accum 2), lr=1e-5
- Proxy language: Spanish (Nahuatl not in Whisper vocabulary)
- Training: Modal ephemeral app, ~$3.50-4/h

**Checkpoints saved:**
- checkpoint-1500
- checkpoint-3000
- checkpoint-4500
- checkpoint-6000
- checkpoint-6738 (1 full epoch)

**Baseline (500 samples, seed=42):**

| Metric | Baseline | 3K Steps (0.45ep) | Improvement |
|--------|----------|-------------------|--------------|
| WER | 166% | 137% | -17.7% rel. |
| CER | 108% | 70% | -35.3% rel. |

WER >100% = baseline hallucinates more tokens than reference contains.
CER is the relevant metric — model produces recognizable Nahuatl characters.

**Pipeline integration (Hernan E03, N=343 NAH segments):**

| Metric | Before FT (LLM fallback) | After FT (3K Steps) |
|--------|----------------------|--------------------|
| Readable NAH transcription | 0 | 343 (100%) |
| LLM gibberish | 337 | 0 |
| Nahuatl morphology | No | Yes |
| SPA code-switch correct | No | Yes |

Example before: *"Jenkatani kamlam bktchazn"*
Example after: *"kapitan n' itsmotlatlawtilia, mah ximokkixti"*

---

### F002: Overfitting Analysis (1 Full Epoch)

**Finding:** 288/343 FT lines differ between 3K and 6738 steps (84%).

**Three overfitting signals at checkpoint-6738:**

1. **SPA degradation:** Spanish transcribed with Nahuatl phonology
   - 3K: *"Yo puedo acompañarles si creéis"*
   - 1ep: *"Yo poro kompañarli si kreey se"*

2. **Corpus memorization:** Speaker names from OpenSLR-92 hallucinated
   - *"Anastacio Nicolás Damián"*, *"José Ernesto Vázquez Chanico"*
   - Appears in 16 different segments across all speakers
   - Model memorized speaker metadata instead of learning phonology

3. **NAH word boundaries:** Mixed — partially better, partially artifacts
   - Better: *"Amo n'mech wika"* vs. *"a moh anmeh xwikas"*
   - Worse: *"Tlok 5"* instead of *"tloksi"*

**Hypothesis:** Sweet spot lies between step 3000 and 4500.

---

### F003: Checkpoint Comparison via Edge Case Test Set

**Method:** 20 clips (~8.5 min) from Hernan E03 in 6 categories:
NAH clean (4), NAH divergent (4), SPA code-switch (4), Short <1s (3), MAY (3), Hallucination (2).
Eval: only Whisper inference per checkpoint (~2 min/checkpoint), no full pipeline overhead.

**Results (4 checkpoints, 20 clips):**

| Signal | checkpoint-3000 | checkpoint-4500 | checkpoint-6000 | checkpoint-6738 |
|--------|:-:|:-:|:-:|:-:|
| **Hallucination (corpus names)** | 0/2 clean | 0/2 clean | **2/2 HALLUC** | 1/2 |
| **NAH clean stability** | 4/4 (ref) | 1/4 identical | 0/4 | 1/4 |
| **SPA degradation** | 0/4 | 0/4 | 0/4 | 0/4 |
| **Repetitions** | 0/20 | 0/20 | 0/20 | 0/20 |

**1. Hallucination onset: checkpoint-6000**
- `hallucination_01`: 3K/4500 = *"Altepiltsi, owelih xe n' tonalpika..."* → 6000/6738 = **"Anastacio Nicolás Damián"**
- `hallucination_02`: 3K = *"Jajaja, hijos de perra"*, 4500 = *"Jajaja, ijóros de perra"* → 6000 = **"José Ernesto Vázquez Chanico"**, 6738 = recovers to *"Jajaja. Hijos de perra"*
- Surprise: checkpoint-4500 does **not** hallucinate — but on NAH-divergent clips it does!
- `nah_divergent_01` + `nah_divergent_03`: From 4500 = "Anastacio Nicolás Damián"

**2. SPA degradation: NOT reproduced**
- All 4 SPA clips remain clean across all checkpoints!
- *"Yo puedo acompañarles si creéis"* — identical 3K through 6738
- *"Pero sobre todo no quiero ningún enfrentamiento"* — identical
- The SPA degradation observed in F002 was apparently **segment-specific**, not systematic
- Exception: `spa_codeswitch_04` hallucinates "Anastacio" on ALL checkpoints (incl. 3K!)

**3. NAH clean: 3K→4500 shows slight drift**
- `nah_clean_01`: 3K = *"Ihkwicho, kawi moyo law"* → 4500+ = syllabic decomposition *"Ih..., ii..., cho..."*
- `nah_clean_02/03/04`: Minimal differences (word boundary shifts), content preserved
- 4500 improves some segments: *"Altepiltsi"* → better word separation on hallucination_01

**4. 3K vs 4500 on NAH:**
- 4500 produces slightly better Nahuatl morphology on divergent clips
- BUT: 4500 already hallucinates "Anastacio" on 2 of 20 clips (nah_divergent_01, nah_divergent_03)
- 3K: 1/20 hallucinates (spa_codeswitch_04 — but that's a pre-existing problem)
- **3K is the safe checkpoint, 4500 is marginally better at NAH but already unstable**

**5. Unexpected finding: spa_codeswitch_04**
- *"Yo los prometo muerte"* → "Anastacio Nicolás Damián" on ALL 4 checkpoints
- = Pre-existing hallucination, not caused by overfitting
- This segment may have acoustic similarity to a corpus speaker

**Conclusion:** Sweet spot = **checkpoint-3000**.
- 4500 brings marginal NAH improvements, but hallucinations begin
- 6000 is clearly overfit (2/2 dedicated hallucination clips + 2 divergent)
- 6738 shows partial recovery (hallucination_02), but remains unstable
- SPA remains surprisingly stable — overfitting manifests as **corpus memorization**, not SPA phonology degradation

**Next steps:**
- F004 Multi-dialect training building on checkpoint-3000
- Investigate `spa_codeswitch_04` hallucination (acoustic fingerprint?)
- Use test set as regression guard for future training runs

**Status:** DONE — Test set under `clips/edge_cases/`, tooling under `tenepal_whisper_train.py::run_eval_checkpoint`.

---

### F004: Multi-Dialect Training — SLR 92 + Mozilla Corpora (2026-04-01)

**Hypothesis:** Overfitting on speaker names arises from too little speaker diversity.
With 220h+ across four regions the model can no longer overfit to individual speakers.

**Discovery:** SLR 147 (Orizaba) and SLR 148 (Tepetzintla) have **no transcriptions** —
only audio. Instead found three transcribed corpora on Mozilla Data Collective:

| Corpus | Region | Segments | Hours | Speakers | Source |
|--------|--------|----------|-------|----------|--------|
| SLR 92 | Highland Puebla | 135,544 | 189.7h | ~800+ | OpenSLR (existing) |
| Zacatlán ASR | Zacatlán-Tepetzintla | 9,272 | 14.2h | 37 | Mozilla DC (Kaltepetlahtol) |
| Tetelancingo | Sierra Oeste Puebla | 2,671 | 3.4h | 5 | Mozilla DC (Kaltepetlahtol) |
| CV Orizaba | Orizaba/Veracruz | 7,722 | 13.4h | 16 | Mozilla DC (Common Voice) |
| **Total** | **4 regions** | **155,209** | **220.7h** | **58+ new** | |

**Setup:** Continual learning from single-dialect LoRA adapter (checkpoint-3000).
Temperature-sampled dataloader (T=2.0) upsamples minority dialects from ~14% to ~40%:
- SLR 92: 59.9% effective weight (down from 86%)
- CV Orizaba: 15.5%, Zacatlán: 15.6%, Tetelancingo: 8.9%

LR=5e-6 (half of single-dialect), warmup=50, 3000 steps on A100.

**Training results:**
- Train loss: 2.075
- Eval loss: 0.861
- Runtime: 2h 53min, ~$15
- Epoch: 0.38 (38% of combined data seen)
- Saved: `lora-adapter-multidialect`, `model-multidialect`

**Status:** Training complete. Eval pending — need WER/CER comparison vs single-dialect
baseline on SLR 92 test set (regression check) and Mozilla test sets (transfer check).
Corpus memorization check (speaker name hallucination) also pending.

---

### F005: wav2vec2 Embedding-based LangID Baseline (2026-04-01)

**Hypothesis:** wav2vec2 embeddings encode language identity even when ASR fails.
Segments where Whisper hallucinates AND Allosaurus is noisy still have acoustic
information. Inspired by Pugh et al. "wav2pos" — wav2vec2 contains syntactic info
without transcription; hypothesis: also contains LangID signal for NAH vs SPA.

**Setup:** Extract wav2vec2-base mean-pooled embeddings (768-dim) for all 1,006
annotated segments (Hernán-1-3 + La Otra Conquista). Train LogisticRegression
(balanced, 5-fold stratified CV). Cost: ~$0.20 on Modal CPU.

**Results:**

| Metric | Value |
|--------|-------|
| Binary NAH vs OTHER accuracy | 85.1% |
| Binary balanced accuracy | 85.0% |
| Multi-class accuracy | 78.3% |
| NAH precision / recall | 83% / 84% |
| SPA precision / recall | 85% / 80% |

**Cross-comparison with pipeline (Hernán-1-3, 588 segments):**

| Combination | Accuracy |
|-------------|----------|
| Pipeline alone (old DB run) | 53.7% |
| F005 alone | 77.7% |
| **Oracle (best of both)** | **86.9%** |

Key finding: **195 pipeline errors could be rescued** by F005, only 54 would be
introduced. The errors are complementary — pipeline uses linguistic signals
(IPA, Whisper, morphology), F005 uses raw acoustic embeddings. They fail on
different segments.

Top rescue categories:
- 52× NAH→SPA pipeline errors rescued (F005 detects NAH acoustically)
- 45× SPA→NAH pipeline errors rescued (F005 rejects false NAH)
- 42× NAH→OTH pipeline errors rescued

**Next:** Add `f005_pred_lang` + `f005_confidence` to annotations DB.
Implement confidence-gated fallback: when pipeline confidence low AND F005
confidence high → use F005 prediction. Test on LOC 24m-34m (best rescue:break
ratio at 4:1).

---

### Benchmark Methodology Update (2026-04-01)

The Hernán accuracy numbers reported in experiments E053-E055 and the "Accuracy Progression"
section (85.7% on 551 segments) used **midpoint matching** between SRT timestamps and GT
timestamps, and a GT snapshot that no longer exists as a standalone artifact. Those numbers
are not reproducible from current artifacts.

**What changed:**

1. **Matching method:** Midpoint matching → **cue-index matching**. Midpoint matching was
   fragile (a 50ms timestamp drift could swap a match to the wrong GT segment). Cue-index
   matching aligns by segment position in the SRT, which is deterministic and robust.

2. **Ground truth source:** The old GT was an exported JSON snapshot (`eq_comparison_gt.json`)
   that diverged from the annotator DB over time as corrections were made. The annotator DB
   is now the **canonical GT source** (v2 snapshot), and `evaluate.py` reads from it directly.

3. **Primary metric:** Segment accuracy → **duration-weighted accuracy**. Equal segment
   counting gives a 0.3s "uh" the same weight as a 15s Nahuatl monologue. Duration weighting
   measures what actually matters: how much film time is correctly classified.

**Canonical numbers going forward (config 13_v7_morphology_expansion, NAH+SPA only):**

| Metric | Value |
|--------|-------|
| Segment accuracy | 71.6% (394/550 segments) |
| Duration-weighted accuracy | **73.7%** (568s/770s of film time) |
| NAH precision | 75.5% |
| NAH recall | 76.1% |

**Why the drop from 85.7% to 71.6%:** The difference is methodological, not a regression.
Midpoint matching was more lenient (could match segments across small timestamp gaps),
and the old GT snapshot had fewer corrections than the current DB. The pipeline itself
has not changed.

**Historical numbers in EVOLUTION.md remain as-is.** They document what was measured at the
time with the methodology available. They are not claims about current reproducible accuracy.

---

### E051: tɬ Override Spanish False Positive Fix (2026-03-05)

**Hypothesis:** tɬ acoustic hard override is too aggressive — forces NAH on every segment where Allosaurus detects tɬ, even when Whisper has clearly transcribed Spanish.

**Problem:** Spanish has "tl" clusters (atlantico, extremauncion, Tlaxcala) that Allosaurus misinterprets as tɬ. In the film "La Otra Conquista" (10min test clip, 04:15-14:15), scenes with purely Spanish dialogue were falsely tagged as NAH.

**Before:** 50 NAH / 22 SPA / 7 OTH / 1 MAY — far too much NAH

**Fix:** Spanish context guard — tɬ override is now only triggered when ALL conditions are met:
1. Allosaurus detects tɬ (existing check)
2. Whisper text does NOT contain 3+ Spanish function words (la, el, de, que, en, los, las, un, una, por, con, del, al, se, su, más, como, hay, pero, ya, no, es, le, lo, y, a)
3. OR: Whisper is not confident-Spanish (conf < 0.85 or lang != es) with < 1 function word

New backend tag: `tl-acoustic-conditional-override` (instead of `tl-acoustic-hard-override`).
Suppressed overrides logged as `[tl-suppressed]`.

**After:** 37 NAH / 39 SPA / 3 OTH / 1 MAY — 14 segments correctly rescued to SPA

**Remaining FPs:** ~7-12 NAH segments in the SPA range where Whisper delivers no usable text (whisper_valid=False, no function words detectable). These go through the IPA-only path where the tɬ override still applies — acceptable residual risk.

**Takeaway:** Acoustic markers alone aren't enough — Whisper text as plausibility check is essential. The guard costs virtually nothing (string matching) and prevents the most common false positives.

---

### E052: Checkpoint 3000 vs 6738 Comparison — La Otra Conquista (2026-03-05)

**Hypothesis:** Full-epoch checkpoint (6738 steps) should deliver better NAH transcription than half-epoch (3000 steps).

**Setup:** La Otra Conquista 10min test clip (04:15-14:15), A10G, both checkpoints loaded as LoRA adapters (manual merging, 192 layers). 28 manually annotated segments as ground truth.

**Result — Real NAH segments (SPEAKER_07, 01, 04, 02):**

| Aspect | 3K | 6738 | Winner |
|--------|-----|------|----------|
| "tonanzini" | toh n' atsineh | tohna n' tsini | 6738 |
| "nikankatotlizi ntotlakatzi" | Nika n' ka totliitsi n' totlapaltsi | Nika n' ka totliitsi n' totlapaaltsi | 6738 |
| Morphology endings | 4/10 correct | 6/10 correct | 6738 |

**Result — False positive NAH (actually Spanish, 10 segments):**

| Aspect | 3K | 6738 | Winner |
|--------|-----|------|----------|
| "donde van todos los mortales" | ¿Dónde van todos los mortales? | Sonde van todos los mortales | 3K |
| "no nos preocupeis" | No os preocupéis | No os precopeis | 3K |
| "se podria tomar como blasfemia" | se podría tomar como plazo | se podría tomar como plasmo | 3K |
| SPA transparency overall | 4/5 clean | 1/5 clean | 3K |

**Hallucinations (both checkpoints):**
- "Ernesto Vázquez Chanico" and "Anastacio Nicolás Damián" — speaker names from OpenSLR-92 Puebla Nahuatl corpus
- Appear on silence/unclear audio → corpus contamination
- Both "nahuatlize" Spanish when pipeline falsely says NAH: "a saber" → "Asa weh"

**Root cause:** All 10 NAH→SPA errors come from the tɬ override, NOT from the FT model. The FT model blindly follows the pipeline label — when the pipeline says "NAH", it compulsively produces Nahuatl.

**Takeaway:** tɬ fix is the blocking issue. FT evaluation is meaningless as long as language detection classifies 10 Spanish segments as NAH. Re-evaluation of both checkpoints AFTER full tɬ fix needed.

**Trade-off:** 3K = better SPA transparency, 6738 = better NAH morphology. Recommendation: 3K as default until multi-dialect training (F004) solves the overfitting problem.

---

---

### E053: SPA Detection — Diagnosis and Fix (2026-03-06)

**Problem:** Spanish is often classified as OTH. Across all 4 baseline episodes: 104 of 511 OTH segments (20.4%) have IPA that is clearly Spanish.

**Baseline measurement (Hernan Ep1-4):**

| Episode | Segs | SPA | OTH | NAH | MAY | OTH-as-SPA |
|---------|------|-----|-----|-----|-----|------------|
| Ep1 | 627 | 400 (64%) | 175 (28%) | 30 (5%) | 22 (4%) | 32/175 |
| Ep2 | 633 | 430 (68%) | 165 (26%) | 22 (4%) | 16 (3%) | 35/165 |
| Ep3(3k) | 598 | 168 (28%) | 69 (12%) | 343 (57%) | 18 (3%) | 14/69 |
| Ep4 | 486 | 328 (68%) | 102 (21%) | 37 (8%) | 19 (4%) | 23/102 |
| **Total** | **2344** | **1326 (57%)** | **511 (22%)** | **432 (18%)** | **75 (3%)** | **104/511** |

**Root cause analysis — Two bottlenecks:**

1. **`validate_whisper` too strict (line 2091):** Hard 35% threshold for known words. "Los arcabuceros prepararon sus mosquetes" has only 20% known words → `whisper_valid=False` → IPA path.

2. **IPA path can barely detect SPA:** SPA profile has only 5 markers with low weights (b:0.20, d:0.05, ɡ:0.20, ɲ:0.80, ɾ:0.40), threshold 2.25, no bigrams. Short segments never reach threshold. Confidence gate (0.6) kicks the rest to OTH.

**Cascade:** Whisper "Arcabuceros" → validate_whisper: REJECTED → IPA: SPA score 1.4 < 2.25 → OTH → Whisper text deleted → LLM gibberish "Kabuufes"

**Fixes implemented:**

**Fix 1: Sliding scale for `validate_whisper`**
- Instead of hard 35%: threshold varies with Whisper confidence
- avg_log_prob -0.2 (high) → 15% threshold
- avg_log_prob -0.5 (medium) → 25% threshold
- avg_log_prob -1.0 (low) → 35% threshold (as before)
- Formula: `threshold = 0.35 - 0.20 * confidence`

**Fix 2: SPA Text Rescue in IPA path (FIX 8)**
- After confidence gate (OTH), BEFORE Spanish leak detection
- Checks `guess_language_from_text_markers()` (uses large SPA_COMMON set)
- Only when Whisper text looks like SPA (conf ≥ 0.5)
- Guard: No strong indigenous IPA markers (tɬ, kʷ, ɬ, kʼ, tʼ, tsʼ, pʼ)
- Result: OTH → SPA with text confidence

**Result (La Otra Conquista 10min test clip):**

| Version | SPA | NAH | OTH | MAY | Description |
|---------|-----|-----|-----|-----|-------------|
| Baseline (3k) | 39 (49%) | 37 (46%) | 3 (4%) | 1 (1%) | Before fixes |
| spa-fix-v1 | 51 (64%) | 25 (31%) | 3 (4%) | 1 (1%) | All 3 fixes, `whisper_lang="es"` for reclaim |
| spa-fix-v2 | 42 (53%) | 34 (43%) | 3 (4%) | 1 (1%) | All 3 fixes, `whisper_lang="auto"` + NAH text guard |

**v1 analysis (12 changed segments):**
- 7 correct: "tal pareciese que busca morir" → SPA ✓, "¿Dónde van todos los mortales?" → SPA ✓, etc.
- 2 wrong: "Ini xki chintopa no mochi" (real NAH) falsely to SPA, "Kein esmok, kawil" likewise
- 3 unclear: short/ambiguous segments

**v2 analysis (3 changed segments):**
- 0 false positives from SPA reclaim (guard works)
- 3 changes come from SPA text rescue (Fix 2) and/or pipeline nondeterminism
- Too conservative: misses good reclaims like "os convierto en cristian" (only 1 SPA word), "se podria tomar como blasfemia" (needs ≥2 SPA words with `auto`)

**Open items for next iteration:**
- Reclaim should check BOTH texts: FT text AND original Whisper text
- `guess_language_from_text_markers` with `"auto"` too conservative — SPA_COMMON lacks verbs (convierto, preocupeis, pareciese)
- Alternative: Own reclaim logic instead of `guess_language_from_text_markers` — direct SPA_COMMON check with lower threshold

**Measurement script:** `scripts/measure_lang_distribution.py` for SRT-based A/B comparisons.

**Fix 3: SPA Reclaim after NAH FT pass (STEP 6.55)**
- After FT: If FT text (or original Whisper text) looks like Spanish → NAH→SPA
- `guess_language_from_text_markers(ft_text, "es")` with threshold 0.55
- Guard: Keeps NAH when IPA has strong NAH-exclusive markers (tɬ, kʷ, ɬ)
- Clears nah_ft_text → shows original Whisper text in SRT
- Goal: Rescue ~10 Spanish segments in La Otra Conquista that were falsely tagged as NAH through tɬ cascade or speaker prior

**Risk:** Fix could falsely rescue NAH/MAY segments to SPA when Whisper hallucinates Spanish text over indigenous audio. Guard (indigenous IPA markers) should prevent this.

**Annotation comparison (La Otra Conquista, 16 annotated segments):**

| Version | Correct | Accuracy | Description |
|---------|---------|----------|-------------|
| Baseline (6738) | 1/16 | 6% | No reclaim |
| v1 | 7/16 | 44% | `guess_language_from_text_markers(ft_text, "es")` — too aggressive |
| v3 | 6/16 | 38% | Dual text (FT + orig Whisper), SPA_COMMON ≥2/30% — Whisper hallucinates SPA over NAH |
| **v4** | **9/16** | **56%** | NAH morphology guard + expanded SPA_COMMON |

**v4 details:**
- **NAH morphology guard** (new): Regex recognizes Nahuatl text patterns (tl[aeiou], xk[aeiou], kw[aeiou], chih, chiw, moch, nik[aeiou], tik[aeiou], ntik, ntek). If FT text contains Nahuatl morphology → NO reclaim. Eliminates all FPs on real NAH.
- **SPA_COMMON expanded** (+15 words): y, os, todos, van, santo, espiritu, salvar, morir, viaje, ultimo, tomar, podria, supone, fray, diego, cristiano, mortales, blasfemia. Rescues 36.21 "os convierto en cristian" and 585.96 "No os precopeis" that v1 caught but v3 missed.
- **Dual text** retained: Checks BOTH FT text and original Whisper text (higher score wins), but NAH morphology guard blocks FPs.

**Per-segment detail:**

| Timestamp | GT | Baseline | v1 | v4 | Transcription |
|-----------|-----|----------|-----|-----|--------------|
| 19.39 | SPA | NAH | OK | OK | al parecer se busca morir |
| 31.23 | SPA | NAH | NAH | **OK** | y del espiritu santo |
| 36.21 | SPA | NAH | OK | OK | os convierto cristiano |
| 50.99 | OTH | NAH | SPA | SPA | (silence/unclear audio) |
| 56.85 | SPA | NAH | OK | OK | el ultimo viaje |
| 92.17 | SPA | NAH | OK | OK | donde van todos los mortales |
| 304.03 | NAH | SPA | SPA | SPA | enesbokawili |
| 354.88 | NAH | OTH | OTH | OTH | (NAH not detected) |
| 366.03 | NAH | OK | SPA | SPA | (short NAH, speaker prior FP) |
| 369.93 | NAH | MAY | MAY | MAY | (ejective FP → MAY instead of NAH) |
| 378.37 | NAH | OTH | OTH | OTH | (NAH not detected) |
| 484.76 | SPA | NAH | OK | OK | se podria tomar como blasfemia |
| 502.31 | SPA | NAH | OK | NAH | (FT too garbled for detection) |
| 512.14 | SPA | NAH | OK | OK | se supone que compartimos nada |
| 534.65 | SPA | NAH | NAH | **OK** | a saber |
| 582.79 | SPA | NAH | NAH | **OK** | a salvar |

**Error analysis v4 (7/16 errors):**

| Category | Segments | Cause |
|-----------|----------|---------|
| SPA reclaim FP | 50.99, 366.03 | "Yo yo" ∈ SPA_COMMON; short NAH via speaker prior |
| Whisper hallucination | 304.03 | Whisper hears "Callate de smoke a Willy" instead of Nahuatl |
| NAH recall (OTH/MAY) | 354.88, 369.93, 378.37 | IPA path doesn't detect NAH despite Nahuatl phonemes |
| Garbled FT | 502.31 | "Asemejans a," — 1 SPA word < threshold 2 |

**Overall distribution La Otra Conquista 10min:**

| Version | SPA | NAH | OTH | MAY | Accuracy |
|---------|-----|-----|-----|-----|----------|
| Baseline (6738) | 40 (50%) | 36 (45%) | 3 (4%) | 1 (1%) | 6% |
| v1 | 51 (64%) | 25 (31%) | 3 (4%) | 1 (1%) | 44% |
| v3 | 51 (64%) | 25 (31%) | 3 (4%) | 1 (1%) | 38% |
| **v4** | **52 (65%)** | **24 (30%)** | **3 (4%)** | **1 (1%)** | **56%** |

**Takeaway:** SPA reclaim itself works precisely (9/11 NAH→SPA correct). The remaining errors are upstream: Whisper hallucination (1), NAH recall in IPA path (3), speaker prior FP (1), unavoidably short segment (1), garbled FT (1).

---

### E054: Voter Improvements — Ejectives, Speaker Prior, FT Gate (2026-03-06)

**4 changes implemented simultaneously:**

**1. Ejective differentiation NAH vs MAY:**
- tɬ confirmed → strong NAH (+2)
- Ejective + tɬ → very strong NAH (+3)
- Ejective + implosive-ɓ → strong MAY (+2)
- Ejective alone (no tɬ, no ɓ) → weak NAH (+0.5), NOT sufficient alone for classification
- Implosive-ɓ alone → MAY (+1)
- Log tags: `ejective-nah` / `ejective-may` / `ejective-ambiguous`

**2. Inline speaker prior in voter:**
- `_speaker_lang_history` counter updated per segment
- With >3 segments and >70% of one language: +0.1 confidence boost
- NAH speaker with OTH segment → OTH→NAH rescue
- NAH speaker (>80%) with MAY segment (no ɓ) → MAY→NAH correction
- SPA speaker with NAH segment (no tɬ) → NAH confidence -0.1
- Log tags: `speaker-prior-nah` / `speaker-prior-spa`

**3. Minimum segment length before FT model:**
- Segment < 0.8s → skip FT, keep stock Whisper text
- Segment < 0.8s AND IPA < 4 phonemes → classify as OTH
- **Effect:** Eliminates FT hallucination "Anastacio Nicolas Damian" on Seg 32 (353.04s, 0.78s)
- Log tag: `ft-skipped-too-short`

**4. tɬ override remains soft** (no change, from previous fix)

**Result (La Otra Conquista 10min, 16 annotated segments):**

| Version | Correct | Accuracy | SPA | NAH | OTH | MAY |
|---------|---------|----------|-----|-----|-----|-----|
| v4 (before) | 9/16 | 56% | 52 | 24 | 3 | 1 |
| v5 (voter) | 9/16 | 56% | 52 | 24 | 3 | 1 |

Accuracy the same, but qualitative improvement:
- **Seg 32:** FT hallucination eliminated ("Anastacio Nicolas Damian" → LLM "Tontsin")
- **Seg 33/37:** Not yet fixed (inline speaker prior needs more context; SPEAKER_00/04 have too few segments at segment creation)
- No regressions — all other segments identical

**Limitation of inline speaker prior:** Only takes effect from segment 4+ of the same speaker. For early segments, the post-processing speaker prior (STEP 6.4) remains relevant. The two systems complement each other: inline = real-time voter signal, post-processing = global correction.

---

## Phase 9: Whisper Language Misdetection Fix (2026-03-06)

### E030: Whisper Detects Spanish as English (La Otra Conquista)

**Problem:** Whisper's file-level language detection says `lang=en` for LOC clips 2-5. Consequence: All valid Whisper segments are classified as ENG, even when the text is clearly Spanish ("¿Por que no le matamos aqui?" → ENG).

**Analysis (Clip 3, 24:15-34:15, 39 annotated segments):**

| Error type | Count | Example |
|-----------|-------|----------|
| ENG→SPA | 21 | "¡Calla, perro!" as ENG |
| NAH→SPA | 6 | tɬ override on Spanish |
| ENG→NAH | 5 | Nahuatl as English |

Baseline accuracy: **5%** (2/39 correct)

### E031: 12K Spanish Dictionary + Text-based Override

**Approach:** Three-stage fix:
1. `SPA_COMMON`: 200 hand-curated words → 12,317 from frequency corpus (hermitdave/FrequencyWords Top 10K + existing spa_wordlist.txt)
2. `spa-text-override`: When Whisper `lang≠es` but text has SPA markers (SPA_COMMON hits ≥2, ratio ≥0.25), override to SPA
3. `detect_spanish_text_leak`: Extended from {nah,may,other} to {nah,may,other,eng}
4. `_has_spanish_orthography`: ¿¡áéíóúñ as strong SPA signal even with few word hits

**Critical bug:** `add_local_dir` was missing — `spa_wordlist.txt` was not available on Modal. Locally 12K words, on Modal only 28-word fallback. Fix: `add_local_dir("src/tenepal/data", remote_path="/data/tenepal")` + `_resolve_data_path()` for Modal/local resolution.

**Iterations:**

| Version | SPA | NAH | ENG | MAY | Accuracy |
|---------|-----|-----|-----|-----|----------|
| v1 (original) | 6 | 44 | 26 | 1 | 5% |
| v2 (+spa-override) | 21 | 45 | 8 | 1 | 41% |
| v3 (+12K local only) | 13 | 51 | 9 | 1 | 41% |
| v4 (+eng leak) | 13 | 51 | 9 | 1 | 41% |
| **v5 (+Modal mount)** | **40** | **28** | **8** | **1** | **61%** |

### E032: Remaining Errors (v5)

| Type | Count | Cause |
|-----|-------|---------|
| NAH→SPA | 3 | tɬ override on Spanish |
| ENG→NAH | 3 | Nahuatl text, Whisper says EN |
| SPA→NAH | 3 | NAH text falsely as SPA (spa-override too aggressive) |
| ENG→SPA | 3 | Single words ("potente", "Done!") |
| MAY→NAH | 1 | Ejective false alarm |
| →OTH | 2 | Silence/noise as language |

**Takeaways:**
- Whisper's language code is often wrong at segment level, but the transcribed text is usually correct → text-based correction works
- The SPA dictionary must be available on Modal (not just locally)
- Short segments (1-2 words) are hard to classify
- tɬ override remains the most aggressive error driver for SPA scenes

### E033: La Otra Conquista Overall Result (4 clips, 40 minutes)

| Clip | Time range | SPA | NAH | ENG | MAY | Segs |
|------|----------|-----|-----|-----|-----|------|
| 2 | 14:15-24:15 | 31 | 7 | 0 | 1 | 39 |
| 3 | 24:15-34:15 | 40 | 28 | 8 | 1 | 77 |
| 4 | 34:15-44:25 | 111 | 22 | 0 | 1 | 134 |
| 5 | 44:25-55:25 | 81 | 23 | 0 | 0 | 104 |
| **Total** | | **263** | **80** | **8** | **3** | **354** |

MAY: 70 → 3 (Phase 1 fix + ejective guard)
SPA dominant in all clips. NAH detection active in all scenes.

---

### E034: Hernan-1-3 Annotations Ground Truth (2026-03-11)

**Data basis:** 598 segments from Hernan Episode 1-3 manually annotated (tools/annotator).

| Annotated language | Count |
|---------------------|-------|
| NAH | 267 |
| SPA | 233 |
| OTH | 64 |
| UNK | 29 |
| MAY | 3 |
| LAT | 2 |

**Pipeline baseline accuracy:** 299/598 = **50%** (258 corrections needed)

**Error distribution (top 6):**

| Error type | Count | Explanation |
|-----------|-------|-----------|
| SPA→NAH | 63 | Whisper hallucinates NAH text on Spanish |
| NAH→SPA | 63 | FT over-tags: real SPA marked as NAH |
| OTH→NAH | 44 | Silence/music transcribed as NAH |
| OTH→SPA | 27 | Silence/music transcribed as SPA |
| →UNK | 29 | Indeterminable segments (short, noise) |
| MAY→NAH | 16 | Ejective false alarms → wrong MAY |

**Speaker concentration:** Errors NOT evenly distributed:
- SPEAKER_17: 72 errors (28% of all errors) — mainly SPA→NAH
- SPEAKER_15: 60 errors (23%) — mixed SPA↔NAH
- Remaining 12 speakers: <20 errors each

### E035: v7 Accuracy Fixes — Experiment (2026-03-11)

**5 fixes implemented** (toggleable via CLI flags in `tenepal_modal.py`):

| Fix | Flag | Description |
|-----|------|--------------|
| FT SPA Guard | `--ft-spa-guard` | When FT Whisper outputs SPA text → demote NAH→SPA |
| Speaker Prior Strong | `--speaker-prior-strong` | Majority language per speaker overrides uncertain segments |
| Ejective Strict | `--ejective-strict` | ≥3 ejectives needed + NAH phonemes (tɬ/kʷ/ɬ) block MAY |
| Noise Gate | `--noise-gate` | Segments ≤0.4s without text → OTH |
| All Fixes | all 4 combined | |

**Test methodology:**
- 3 clips of 2 minutes each extracted from Hernan-1-3 (ffmpeg):
  - Clip A (08:00-10:00): 36 cues, 15 errors — all 3 main error types
  - Clip B (04:00-06:00): 21 cues, 12 errors — SPA→NAH dominant
  - Clip C (26:00-28:00): 27 cues, 15 errors — OTH→NAH + MAY→NAH
- Ground truth: JSONL from annotator DB, timestamp matching (IoU ≥0.3)
- 3 clips × 6 variants = 18 Modal runs (parallel, 1 GPU each)

**Results:**

| Variant | Clip A | Clip B | Clip C | Delta vs baseline |
|----------|--------|--------|--------|-------------------|
| baseline | 22/36 | 12/21 | 14/27 | — |
| ft_spa_guard | 21/36 | 11/21 | 13/27 | **-3** |
| speaker_strong | 22/36 | 12/21 | 15/27 | +1 |
| ejective_strict | 22/36 | 13/21 | 14/27 | +1 |
| noise_gate | 22/36 | 12/21 | 14/27 | 0 |
| all_fixes | 21/36 | 11/21 | 13/27 | **-3** |

**Findings:**
1. **ft_spa_guard is harmful** (-3): Demotes real NAH segments to SPA because Nahuatl text contains Spanish loanwords
2. **speaker_strong marginal** (+1): 2-minute clips too short for meaningful speaker history
3. **ejective_strict marginal** (+1): Few MAY errors in test clips
4. **noise_gate neutral** (0): Few short segments in test clips

**Core problem identified:** 39/46 remaining errors across all clips are SPA→NAH (Whisper hallucinates Nahuatl text on Spanish audio). This is a Whisper hallucination problem — post-processing alone cannot solve it.

**Next approach:** IPA-NAH-override — if IPA contains NAH-exclusive phonemes (tɬ, kʷ, ɬ) but Whisper says SPA → NAH wins. Conversely: if IPA has no NAH phonemes and Whisper says NAH → SPA suspect.

### E036: IPA-NAH-Override — Failed (2026-03-11)

**Implementation:** `--ipa-nah-override` flag. When pipeline says NAH but IPA contains no tɬ/kʷ/ɬ AND no tɬ acoustically detected AND no Nahuatl text pattern → demote NAH→SPA.

**Result on test clips:** -2, -3, -3 (net -8). 9 correct fixes, 17 regressions.

**Result on full film (598 segments):** 84 NAH segments demoted, including real Nahuatl like "Ini n' cholaskeh", "Akatsitsiwé!", "Ona ya kimkowa". Combined with the other fixes: 337/597 = 56.4% (only +6.4% over baseline).

**Why it failed:** tɬ/kʷ/ɬ are NAH-exclusive when PRESENT, but their ABSENCE doesn't mean "not NAH". Many Nahuatl words contain none of these phonemes. Allosaurus/Wav2Vec2 also don't detect them reliably.

**Conclusion:** Both ft_spa_guard and ipa-nah-override are harmful. Post-processing approaches for SPA↔NAH differentiation fail because the two languages share too many phonemes. Only POSITIVE NAH markers (tɬ) are reliable, NEGATIVE evidence (absence) is not.

### E037: v7 Full Film Result — Speaker Prior as Main Driver (2026-03-11)

**Best run:** `--speaker-prior-strong --ejective-strict --noise-gate` (without ft_spa_guard, without ipa-nah-override)

| Variant | Accuracy | Delta vs baseline |
|----------|----------|-------------------|
| Baseline (original) | 299/598 = 50.0% | — |
| v7 + ipa-nah-override (all 4) | 337/597 = 56.4% | +6.4% |
| **v7 without ipa-nah-override (3 fixes)** | **397/597 = 66.5%** | **+16.5%** |

**Statistics:** 146 fixed, 68 broken, net +78

**Remaining errors (200):**

| Error type | Count | Explanation |
|-----------|-------|-----------|
| SPA→NAH | 119 | Speaker prior overrides NAH→SPA for bilingual speakers |
| OTH→NAH | 20 | Silence/music transcribed as NAH |
| OTH→UNK | 13 | Indeterminable segments not recognized as OTH |
| NAH→UNK | 13 | Real NAH annotated as UNK (borderline) |
| NAH→SPA | 11 | FT over-tags: real SPA as NAH |
| Rest | 24 | Various confusions |

**Takeaways:**
1. **Speaker-prior-strong is the most important fix** — effective on full film with enough speaker history
2. **Marginal on 2-min clips** — too few segments per speaker
3. **SPA→NAH (119) is the next bottleneck** — bilingual speakers (Malinche) switch between SPA and NAH, speaker prior cannot model this
4. **Ejective-strict and noise-gate** — small but positive contributions

**Recommended flags for production:** `--nah-finetuned --speaker-prior-strong --ejective-strict --noise-gate`

### E038: Prosody Analysis NAH vs SPA (2026-03-11)

**Method:** Parselmouth features extracted from 540 annotated segments (370 NAH, 170 SPA).

**Top 5 features by separability (Cohen's d):**

| Feature | d | Direction | Meaning |
|---------|---|----------|-----------|
| HNR | 0.84 (LARGE) | NAH > SPA | NAH has clearer voice quality |
| int_range | 0.63 (MEDIUM) | NAH > SPA | NAH has stronger stress contrasts |
| nPVI | 0.60 (MEDIUM) | NAH > SPA | NAH rhythmically more variable |
| int_std | 0.57 (MEDIUM) | NAH > SPA | Stronger intensity fluctuations |
| shimmer | 0.53 (MEDIUM) | SPA > NAH | SPA vocally "rougher" |

F0 (pitch) barely separable — both languages overlap almost completely.

**Classifier results (5-fold CV):**

| Classifier | Accuracy | F1-NAH | F1-SPA |
|-----------|----------|--------|--------|
| LogReg | 73.3% | 81.7% | 50.1% |
| GBM | 73.0% | 81.1% | 51.6% |
| SVM-RBF | 72.6% | 81.3% | 47.8% |

SPA recall only 43% — prosody alone cannot detect SPA well.

### E039: EQ Config System + Smoke Tests (2026-03-11)

**EQ system:** 19 tunable parameters as JSON file, loadable via `--eq config.json`.
Categories: prosody voter, speaker prior, ejective, tɬ, SPA reclaim, noise gate, confidence.

**Prosody voter:** Hardcoded LogReg coefficients (no sklearn needed on Modal), `--prosody` flag.
Score = P(NAH) per segment. Overrides when confidence blend falls below threshold.

**6 EQ configs tested on clip_A (36 segments):**

| Config | Description | Accuracy |
|--------|-------------|----------|
| 01_v7_best | Speaker-prior + ejective + noise-gate | 21/36 (58%) |
| 02_v7_prosody | + Prosody (w=0.5, t=0.65) | 21/36 (58%) |
| 03_v7_prosody_gentle | + Prosody (w=0.25, t=0.70) | 21/36 (58%) |
| 04_v7_speaker_tight | Speaker 80%/5seg minimum | 21/36 (58%) |
| 05_v7_spa_reclaim_loose | SPA reclaim 1 hit/20% | 21/36 (58%) |
| 06_v7_prosody_spa_reclaim | Kitchen sink | 21/36 (58%) |

All identical — **2-min clips too short for EQ differences**. Speaker prior, SPA reclaim, and prosody voter need full-film context.
Prosody voter was correctly activated (34 segments scored, 0 overrides — confidence blend never below flip threshold).

**Next step:** Full-film test with EQ configs when Modal credits available.

**Recommended production:** `--nah-finetuned --eq eq_v7_best.json` (= `--speaker-prior-strong --ejective-strict --noise-gate`)

### E040: Full Film EQ Comparison — 6 Configs (2026-03-11)

**Setup:** 6 EQ configs in parallel on Modal (Hernan-1-3 complete, 598 annotated segments).
All runs with `--nah-finetuned --eq <config>`. Evaluator bug fixed: `[LANG|SPEAKER]` format.

**Results:**

| Config | Accuracy | Delta | Description |
|--------|----------|-------|-------------|
| **01_v7_best** | **66.7%** (398/597) | baseline | Speaker-prior + ejective-strict + noise-gate |
| 02_v7_prosody | 60.8% (363/597) | **-35** | + Prosody (w=0.5, t=0.65) |
| 03_v7_prosody_gentle | 60.6% (362/597) | **-36** | + Prosody (w=0.25, t=0.70) |
| 04_v7_speaker_tight | 59.6% (356/597) | **-42** | Speaker 80%/5seg minimum |
| 05_v7_spa_reclaim_loose | 60.1% (359/597) | **-39** | SPA reclaim 1 hit/20% |
| 06_v7_prosody_spa_reclaim | 58.0% (346/597) | **-52** | Kitchen sink |

**Conclusion:** 01_v7_best remains optimal. All variants worsen the result.

- **Prosody voter** (-5 to -7): Overrides NAH→SPA on correct segments. No SPA gain.
- **Speaker prior tight** (-44): Catastrophic. Bilingual speakers (Malinche) never reach 80% — prior drops out.
- **SPA reclaim loose** (-38): Too aggressive. NAH segments with Spanish loanwords → false reclaim.
- **Kitchen sink** (-53): Cumulative losses from all three harmful adjustments.

**Error distribution (01_v7_best):** NAH→SPA 116 (bilingual code-switches), UNK 32, NAH→OTH 19, SPA→NAH 11, MAY 8.

**Recommendation:** `--nah-finetuned --eq eq_configs/01_v7_best.json` remains production default.
Next improvement must work at segment level (e.g., Whisper token confidence, bilingual speaker prior).

### E041: Tier-1 Paper Hardening — 4 Items (2026-03-11)

**#5 Reject Option (UNK Gate):** New EQ parameter `unk_gate_enabled` + `unk_gate_threshold` (default 0.5).
Segments with `lang_conf < threshold` → `lang = "unknown"` → SRT tag `[UNK]`.
Positioned after speaker prior + noise gate, before NAH-FT (no FT inference on garbage).
CLI: `--unk-gate --unk-gate-threshold 0.5`. EQ config: `07_v7_unk_gate.json`.
Concept: Honest abstention instead of coin flip. Makes precision numbers clean.

**#13 Asymmetric Error Cost Matrix (evaluate.py):**
- NAH→SPA = 2.0x (indigenous language lost)
- SPA→NAH = 1.5x (indigenous language hallucinated)
- X→UNK = 0.5x (honest abstention)
- UNK→X = 0x (unknowable GT = free pass)
- New reporting: Raw Accuracy + excl-UNK Accuracy + Weighted Score
- 01_v7_best: **66.7%** raw → **70.4%** excl-UNK → **73.1%** weighted (updated GT with 32 UNK)

**#7 Adversarial Spanish Stress Test Set:** `adversarial_spa/` with manifest + check script.
Categories: tl orthography (Atlantico, Tlaxcala), liturgical Latin (Ave Maria),
prosodic edges (whisper, choir), code-switch boundaries.
Every NAH/MAY tag = error. Check: `python adversarial_spa/check.py *.srt`.

**#12 Paper Framing "Hallucination as Sensor":** PAPER.md completely rewritten.
- Abstract: "hallucination distributions encode distance-to-training-support"
- Intro: From anecdote level to principle (3 observations: invariant rate, structured distribution, acoustic correlation)
- Discussion 6.1: "Hallucination as a Sensor" — general principle for OOD detection
- Conclusion: "The principle extends beyond our application"
- New Section 4.4: Adversarial Spanish Stress Test
- Section 4.2: Asymmetric evaluation metrics documented
- New contribution: "asymmetric evaluation framework"

**Next step:** Modal run with `07_v7_unk_gate.json` for quantitative UNK gate impact.
Extract adversarial audio clips from Hernan (Tlaxcala/Tenochtitlan scenes).

### E042: UNK Gate Result + GT Re-Annotation + Whisper "Thanks for watching" (2026-03-11)

**GT re-annotation:** 32 segments annotated as UNK (previously 29). Short interjections ("ah", "hmm")
and ambiguous fragments that cannot be assigned to any language, honestly labeled as UNK.
New GT distribution: NAH 370, SPA 179, UNK 32, OTH 8, MAY 8, LAT 1.

**Results (updated GT, 598 segments):**

| Config | Raw Acc | excl-UNK | Weighted | Delta |
|--------|---------|----------|----------|-------|
| **01_v7_best** | 66.7% (398/597) | **70.4%** | 73.1% | baseline |
| **07_v7_unk_gate** | 65.8% (393/597) | 68.0% | **73.3%** | +0.2 weighted |
| 02_v7_prosody | 60.8% | 64.2% | 67.9% | -5.2 |
| 03_v7_prosody_gentle | 60.6% | 64.1% | 67.8% | -5.3 |
| 05_spa_reclaim_loose | 60.1% | 63.5% | 66.2% | -6.9 |
| 04_speaker_tight | 59.6% | 63.0% | 66.2% | -6.9 |
| 06_prosody+reclaim | 58.0% | 61.2% | 64.1% | -9.0 |

**Conclusion:** UNK gate wins on weighted score (73.3% vs 73.1%). The asymmetric
cost matrix rewards honest abstention: 30 segments → UNK instead of coin flip. Raw accuracy
slightly lower because UNK predictions on correct GT labels don't count as "correct."

**Whisper "Thanks for watching" hallucination:**
In the UNK gate SRT, YouTube-typical Whisper hallucinations appear:
`[UNK|SPEAKER_22] ? Thanks for watching!` (segments 58, 287, 3834).
These segments contain silence/background noise where Whisper
hallucinates YouTube boilerplate — a known phenomenon (Whisper trained on
YouTube subtitles, "Thanks for watching" is the most common closer).
Interesting: The UNK gate catches these correctly (low confidence → UNK),
but the hallucinated text remains visible in the SRT. In other configs, the
IPA override hides these hallucinations (Whisper text replaced by IPA).

**New principle:** "Thanks for watching" is further evidence for hallucination-as-sensor:
Whisper projects not only onto languages but also onto genres. Silence/noise →
YouTube closing phrase is a genre hallucination, not a language hallucination.
Both are structured, not random.

**Bug: Noise gate trusts hallucinated text.**
Segments ~300ms (under `noise_gate_max_s=0.4s`) are only gated as OTH
when `len(_words) < 2`. But "Thanks for watching" has 3 words → gate doesn't trigger.
The noise gate logic should check IPA length AND Whisper text plausibility, not
just word count of the (possibly hallucinated) text.

### E043: Bilingual Lexicon Overlap NAH↔SPA — "amo", "si", "no" (2026-03-11)

**Problem segment:** Cue 13, 00:01:20.760–00:01:21.604 (844ms)
- Pipeline: `[SPA|SPEAKER_15] Amo a ti, quizás que.`
- IPA: `a m o t i tʃ k i ts s a`
- GT: NAH

**Analysis:**
- "amo" = SPA "I love" / NAH "no/not" — lexically ambiguous
- "quizás que" = Whisper hallucination of NAH `/tsáske/` onto nearest Spanish equivalent
- `/ts/` in IPA is NOT a Spanish phoneme — that's the smoking gun
- Similarly: "si" (SPA "if/yes" / NAH morpheme), "no" (SPA "no" / NAH morpheme)

**Why pipeline fails:**
- Whisper hears short NAH utterances and maps to Spanish cognates
- Speaker prior sets SPEAKER_15 as SPA (majority of their segments)
- IPA-based detection finds no NAH-exclusive markers (tɬ missing)
- `/ts/` is not used as NAH discriminator

**Possible fixes:**
1. **ts phoneme as NAH indicator:** `/ts/` doesn't exist in the Spanish phoneme system.
   If IPA contains `/ts/` AND segment short + speaker bilingual → NAH boost
2. **Whisper hallucination detector:** If Whisper text produces Spanish cognates
   but IPA contains non-Spanish phonemes (`ts`, `tɬ`, `kʷ`) → flag as mismatch
3. **Bilingual speaker prior:** SPEAKER_15 (Malinche) has mixed profile.
   Instead of "80% SPA → everything SPA", per-segment confidence-weighted

**Impact:** 116 NAH→SPA errors are the largest error block. Many of them are exactly
this pattern: short NAH phrases with Spanish-sounding words from bilingual speakers.

### E044: SPEAKER_15 = Xicotencatl — NAH-only Speaker, 40% of all NAH→SPA Errors (2026-03-11)

**Critical insight:** SPEAKER_15 is Xicotencatl, a NAH-only speaker (GT: 94% NAH, 4% SPA).
Pipeline tags him 55% SPA / 38% NAH — **47 of 116 NAH→SPA errors (40%) come from him alone.**

**Per-speaker error analysis:**

| Speaker | Errors | Main error | GT profile |
|---------|--------|-------------|-----------|
| SPEAKER_15 (Xicotencatl) | 53 | 47 nah→spa | 94% NAH |
| SPEAKER_18 | 20 | 7 nah→spa, 5 unk→oth | mixed |
| SPEAKER_17 | 16 | 4 may→spa, 4 spa→nah | mixed |
| SPEAKER_06 | 14 | 10 nah→spa | NAH-heavy |
| SPEAKER_16 | 14 | 10 nah→spa | NAH-heavy |
| SPEAKER_11 | 6 | 6 nah→spa | NAH-heavy |
| SPEAKER_14 | 5 | 5 nah→spa | NAH-heavy |

**Root cause:** Chicken-and-egg in speaker prior:
1. Whisper maps Xicotencatl's NAH onto SPA cognates ("amo" → "Amo a ti", "tsáske" → "quizás que")
2. First segments are classified as SPA
3. Speaker prior accumulates wrong profile (55% SPA)
4. Prior reinforces SPA bias for all subsequent segments
5. Loop: More SPA → stronger prior → more SPA

**Possible fixes:**
1. **Character-locked speaker prior:** Annotator has speaker→character mapping.
   Xicotencatl = NAH-only → set prior to 100% NAH (no accumulation needed)
2. **IPA-based prior override:** If IPA contains non-Spanish phonemes (`ts`, `tɬ`, `kʷ`)
   AND speaker prior says SPA → override prior
3. **Bootstrapped prior:** Two passes: 1) Classify without prior, 2) Derive prior from pass-1 profiles
   and classify again
4. **Nahuatl FT as prior signal:** FT output for SPEAKER_15 is almost always recognizable NAH
   → FT confidence as additional prior signal

**Estimated impact:** Fix for SPEAKER_15 alone → ~47 fewer errors → accuracy from 66.7% to ~74.6%.
Fix for all NAH-heavy speakers (06, 11, 14, 16) → ~80 errors → ~80%+ accuracy.

### E045: Two-Pass Speaker Prior — IPA Phoneme Evidence (2026-03-11)

**Implementation:** New EQ parameter `two_pass_prior` (STEP 6.39, before STEP 6.4 speaker prior).
Checks `ipa_fused` of each segment for non-Spanish phonemes (`ts`, `tɬ`, `kʷ`, `tʃʼ`, `kʼ`, `tɕ`).
Speakers with ≥15% IPA NAH evidence and ≥3 marker segments: all SPA segments with
`lang_conf < 0.7` are overridden to NAH.

**Result (Hernan-1-3, 598 annotated segments):**

| Config | Raw Acc | excl-UNK | Weighted | NAH→SPA | SPA→NAH |
|--------|---------|----------|----------|---------|---------|
| 01_v7_best (baseline) | 66.7% | 70.4% | 73.1% | 116 | 11 |
| **08_v7_two_pass** | **68.7%** | **72.6%** | **75.4%** | **101** | **11** |
| 07_v7_unk_gate | 65.8% | 68.0% | 73.3% | — | — |
| 09_v7_two_pass_unk | 62.1% | 64.6% | 71.9% | — | — |

**+2.0% raw, +2.2% excl-UNK, +2.3% weighted.** NAH→SPA from 116 to 101 (-15 errors).
0 new SPA→NAH false positives — the `conf < 0.7` threshold protects correct SPA segments.

**Affected speakers (64 overrides total):**

| Speaker | IPA-NAH Evidence | Overrides |
|---------|-----------------|-----------|
| SPEAKER_15 (Xicotencatl) | 15/93 (16%) | 41 SPA available |
| SPEAKER_06 | 13/54 (24%) | 16 SPA available |
| SPEAKER_11 | 8/26 (31%) | 9 SPA available |
| SPEAKER_10 | 8/23 (35%) | 8 SPA available |
| SPEAKER_13 | 4/22 (18%) | 6 SPA available |
| SPEAKER_21 | 4/18 (22%) | 3 SPA available |

**Why not all 47 Xicotencatl errors fixed?**
Many of his SPA segments have `lang_conf ≥ 0.7` (already pushed up by the inline speaker prior
during classification). The `< 0.7` threshold is conservative —
more aggressive would be risky because of the 4 real SPA segments (loanwords like "capitan").

**09 (two-pass + UNK gate) worse:** UNK gate catches the two-pass overrides
because their `lang_conf` still has the old low value. The combination needs
a confidence recalculation after the override.

**Next steps:**
- Set confidence after two-pass override to ≥0.6 (so UNK gate doesn't trigger)
- Test more aggressive threshold (conf < 0.8 instead of 0.7)
- FT output as second signal: if FT produces recognizable NAH → boost NAH confidence

### E046: FT-First — NAH Finetuned Whisper before Speaker Prior (2026-03-11)

**Implementation:** New EQ parameter `ft_first` (STEP 6.38, BEFORE STEP 6.39 two-pass prior).
Loads NAH finetuned Whisper and processes ALL segments, not just those already classified as NAH.
FT output with NAH morphology regex (`-tl`, `hua`, `nahu`, etc.) overrides SPA→NAH.

**Result (Hernan-1-3, 598 annotated segments):**

| Config | Raw Acc | excl-UNK | Weighted | NAH→SPA | SPA→NAH |
|--------|---------|----------|----------|---------|---------|
| 08_v7_two_pass | 68.7% | 72.6% | 75.4% | 101 | 11 |
| **10_v7_ft_first** | **70.9%** | **74.6%** | **78.0%** | **84** | **33** |

**+2.2% raw, +2.0% excl-UNK, +2.6% weighted.** NAH→SPA from 101 to 84 (-17), but
SPA→NAH from 11 to 33 (+22). FT-first is more aggressive: finds more NAH, but also
more false positives. Net still better because NAH→SPA is double-weighted.

### E047: Three Accuracy Levers (2026-03-12)

**Three new EQ parameters built on 10_v7_ft_first:**

1. **`whisper_uncertain_ipa`** (STEP 6.5): Rescue pass updates text, but KEEPS
   IPA-based language classification instead of overriding it.
2. **`prior_reset`** (STEP 6.58): Run speaker prior again AFTER all corrections
   (two-pass, FT-first, SPA-reclaim, IPA-NAH-override, dj-tc). Produces cleaner profiles.
3. **`dj_tc_marker`** (STEP 6.57): w2v2 dʒ + Allosaurus tɕ agreement → NAH marker
   (when text has <2 SPA function words).

**Result:**

| Config | Raw Acc | excl-UNK | Weighted | NAH→SPA | SPA→NAH |
|--------|---------|----------|----------|---------|---------|
| 10_v7_ft_first | 70.9% | 74.6% | 78.0% | 84 | 33 |
| **11_v7_three_levers** | **72.7%** | **76.6%** | **79.3%** | **80** | **34** |

**+1.8% raw, +2.0% excl-UNK, +1.3% weighted.** 11 new correct segments (+8 NAH, +3 SPA),
only 1 new SPA→NAH error. `prior_reset` was the main driver (16 overrides), `dj_tc_marker`
contributed 3.

### E048: Allosaurus Full-Track Experiment (2026-03-12)

**Hypothesis:** Per-segment Allo (600x, 0.2-5s each) loses ~100ms to CTC warmup.
Full-track (1x, entire vocals) should yield more phonemes with better context.

**Implementation:** `AllosaurusBackend.recognize_full_track(wav, emit=2.0)` with
`timestamp=True`, mapping to segments via midpoint overlap. `emit=1.0` (default)
produces only 18 phones on 45 min — CTC blank dominates. `emit=2.0` → 22,725 phones.

**Result: Hypothesis disproved.**

- **No time offset** — optimal offset = 0.0s
- **Jaccard overlap = 0.099** — PS and FT recognize completely different things
- **FT inventory: 41 unique phones** (PS: 109). FT focuses on vowels + ʔ
- **NAH markers lost:** FT finds no tɕ (82 in PS), no kʼ (10), barely ts (12 vs 25)
- **ʔ doesn't discriminate:** Cohen's d = -0.242 (wrong direction: SPA has MORE ʔ)
- **Phones/sec by duration:** FT worse at <2s (0.68-0.87x), equal at 2-5s, better at >5s

**Conclusion:** Per-segment FORCES CTC into more specific consonants. Full-track with
emit=2.0 takes the path of least resistance (vowels + glottal stop). Feature parked —
`allo_full_track` EQ parameter remains for future experiments.

### E049: Reverse G2P Coverage Gap (2026-03-12)

**Problem:** LLM (Qwen2-1.5B) copies IPA symbols 1:1 into text. Downstream
lexicon checks (`_NAH_TEXT_RX`, `NAH_TEXT_MARKERS`, `guess_language_from_text_markers`)
need clean Nahuatl orthography — IPA garbage breaks the entire NAH detection chain.

**Root cause:** `_NAH_IPA_TO_TEXT` had only 20 entries (63% coverage). 56 allophones
from the fused Allo+W2V2 pipeline unknown (ɒ, ә, ʁ, ɴ, ŋ, ɾ, ɪ, dʒ...). Additionally
G2P output was space-separated (`i t l a c a m`), but regex searches for `itlacam`.

**Solution:**
1. `_NAH_IPA_TO_TEXT` 20→90, `_SPA_IPA_TO_TEXT` 24→75 (all allophones mapped)
2. G2P output joined (no spaces) for morphology checks
3. `_NAH_TEXT_RX` 11→28 patterns (+`tz[V]`, `hu[V]`, `cu[V]`, `xic`, `atl` etc.)

**Result on 332 NAH segments:**
- IPA garbage in text: 62→0 segments (100% cleaned)
- G2P coverage: 63%→99%, 476/476 pass 30% threshold
- NAH morphology recognized: 49%→58% (+28 segments)
- Qwen LLM is effectively no longer needed for NAH/MAY

**Insight:** All three lexicon checks are exact string/regex matches — no
fuzzy matching. Contiguous orthography is mandatory. The remaining 42% unrecognized
segments have IPA without recognizable NAH morphemes — upstream problem (Allo/W2V2).

---

### E050: Fuzzy NAH Text Matching (2026-03-12)

**Problem:** `_NAH_TEXT_RX` recognizes only 58% of NAH segments (191/332). The 42%
have plausible Nahuatl orthography (from expanded G2P), but no exact
regex hits like `tla`, `tz[V]`, `hu[V]`.

**Approaches tested:**
1. **Trigram LLR (NAH vs SPA):** 1155 NAH trigrams from lexicon. Does NOT discriminate —
   SPA false positive rate 55-85% at all thresholds. NAH/SPA share too many phonemes.
2. **Lexicon substring + morphology:** Works! 4+ char lexicon words (1943) as
   substrings + NAH-exclusive morphemes (`neci`, `mati`, `ichan`, `ipan`, `pil`, `-lis`, `-isti`)
   + SPA cluster penalty (`bl`, `br`, `str`).

**Result:**
- +23 NAH recognized (59%→66%), only +4 SPA false positives (12%→15%)
- Plausible hits: `nectimatiicimacustanilichan` (mati+ichan), `matipiinnpiipilteca` (mati+pil)
- 4 SPA FP through random 4-char substring matches (`taman` in `estamanespirant`)

**Implementation:** `fuzzy_nah_text_check()` as fallback in all 3 `_NAH_TEXT_RX` locations
(ft-first SPA→NAH override, SPA-reclaim guard, IPA-NAH-override guard).

---

## E051: Systematic NAH Morphology Expansion (2026-03-12)

**Problem:** 41 SPA→NAH and 29 OTH→NAH segments are not recognized by existing morphology patterns.
Many contain clearly identifiable NAH morphemes that aren't in the regex.

**Method:** All error segments extracted from annotations DB, G2P recovery on IPA,
then 25+ candidate patterns validated against ground truth (403 NAH, 240 SPA).
Only patterns with high NAH-hit/SPA-FP ratio accepted.

**New patterns (all 0 new SPA false positives):**

| Pattern | NAH Hits | Meaning |
|---------|----------|-----------|
| `otl\|utl` | 13+7 | Absolutive suffixes (ostotl, mazatl variants) |
| `tict\|ticm\|ticn` | 8 | tic- prefix combinations (subject+object) |
| `nicn\|nicm\|nict` | 4 | nic- prefix combinations |
| `mict[aeiou]` | 3 | mictia — to kill |
| `tequi` | 1 | tequitl — work/to cut |
| `chiua` | 1 | chihua — to make (without h variant) |
| `cipac` | 2 | cipactli — crocodile (calendar) |
| `mani` | 4 | mani — to spread |
| `pach[aeiou]` | 2 | pachoa — to press |
| `quin[aeiou]` | 3 | quin- — 3pl object prefix |
| `nican\|ican` | 3+13 | Locatives (here / at) |
| `tlaco` | 3 | Middle |
| `tlam[aeiou]` | 4 | tlamia — to finish |
| `ilis\|ilia` | 19 | Applicative suffix (1 SPA FP, ratio 19:1) |
| `cali` | 1 | calli — house |
| `tonan\|acal\|pach` | 1+4+2 | Exclusive: mother/boat/to press |

**Result:**
- `_NAH_TEXT_RX`: SPA→NAH 46→51 (+5), OTH→NAH 31→35 (+4)
- `_NAH_EXCLUSIVE_RX` + fuzzy: SPA→NAH 46→54 (+8), OTH→NAH 31→37 (+6)
- SPA false positives: 76→76 (+0 new!)
- Theoretical accuracy gain: +7-14 segments (48.3%→49-50%)
- Still 33+23 hard cases — too short or without recognizable morphology

**Insight:** NAH prefix combinations (`tict`, `nicm`, etc.) are extremely specific
(0 SPA FP with 12 NAH hits). Absolutive suffixes (`-otl`, `-utl`) were completely missing and
are the largest single gain with 20 hits.

**Modal run result (EQ: v7_full_track + morphology expansion):**
- **72.0% overall accuracy** (407/565, +5.5pp over 66.5% v7_three_levers)
- NAH+SPA only: **71.9%** (395/549)
- Without overlap: **71.6%** (385/538)
- NAH recall: **95.8%** (346/361) — nearly perfect
- NAH→SPA errors: only 15 (previously significantly more)
- SPA→NAH errors: 138 — main problem, speaker prior too aggressive for bilingual speakers
- Overlap segments: 90.9% (10/11)
- Baseline saved: `eq_comparison_results/13_v7_morphology_expansion.srt`

## Accuracy Progression (Hernan-1-3, NAH+SPA, 549 annotated)

| Config | Accuracy | NAH Recall | SPA Recall | SPA→NAH | N→S | Date |
|--------|----------|------------|------------|---------|-----|-------|
| Baseline (no EQ) | 50.0% | — | — | — | — | 2026-03-11 |
| speaker_tight | 67.8% | 69.8% | 63.5% | 64 | 110 | 2026-03-12 |
| ft_first | 68.1% | 87.1% | 28.7% | 127 | 47 | 2026-03-12 |
| three_levers | 68.3% | 87.6% | 28.1% | 128 | 46 | 2026-03-12 |
| **ipa_dict_only** | **69.0%** | **76.3%** | **53.9%** | 82 | 88 | 2026-03-13 |
| spa_reclaim_loose | 69.2% | 70.9% | 65.7% | 60 | 105 | 2026-03-12 |
| morphology_expansion | 71.9% | 96.0% | 21.9% | 139 | 15 | 2026-03-13 |
| prosody_gentle | 72.9% | 76.8% | 64.6% | 62 | 83 | 2026-03-12 |
| v7_best | 73.6% | 77.4% | 65.7% | 60 | 81 | 2026-03-12 |
| **two_pass** | **76.1%** | **81.7%** | **64.6%** | 62 | 65 | 2026-03-12 |

---

## E052: IPA-First — The "Drunken Insight" (2026-03-13)

**Starting question:** What do the IPA/lexicon features alone achieve, without speaker prior and
FT override?

**Experiment:** `14_ipa_dict_only.json` — only ejective-strict, noise-gate, whisper_uncertain_ipa,
allo_full_track. No speaker_prior_strong, no ft_first, no two_pass_prior.

**Result: 69.0%** — purely from phoneme analysis, without any contextual information.

**The insight (aka "the drunken idea"):**

> When I hear someone talking, I roughly know where they're from. "That could be Romanian"
> or "that sounds like India." You recognize languages by their SOUNDS, not by words
> or grammar.

That's exactly what the pipeline does: Allosaurus/Wav2Vec2 hear /tɬ/, /kʷ/, /ts/ — and the
morphology patterns say "this sound combination only exists in Nahuatl." Like a person
recognizing retroflex Indian consonants or Romanian palatals without understanding a word.

**Architecture consequence:**

```
IPA analysis (Allosaurus + Wav2Vec2 + G2P + morphology)     = 69% ← foundation
  + Speaker context (two_pass: IPA-informed prior)           = 76% ← refinement
  + Aggressive override (ft_first + strong prior)            = 72% ← degradation!
```

- IPA is the **backbone**. Two thirds of all decisions are phonetically derivable.
- Speaker prior is the equivalent of "they've been speaking for 5 minutes, they don't
  suddenly switch languages" — contextual knowledge, not linguistic knowledge.
- `two_pass` works best because it feeds IPA evidence INTO the prior (informed).
- `ft_first` + `strong_prior` harm because they OVERRIDE IPA evidence (blind).

**Design rule for all future features:**
New features should USE IPA evidence, not override it.

**Next step:** Combine two_pass + morphology_expansion — should yield >76%.
Then: finetuning.

---

## E053: Complete EQ Ablation — 13 Configs x 551 Segments (2026-03-13)

**Experiment:** All 13 EQ configurations evaluated on the same Hernan-1-3 ground truth
(551 NAH+SPA segments, after UNK/OTH/MAY filtering). Midpoint matching between GT and SRT.

**Result — Layer decomposition:**

| Layer | Config | Accuracy | Delta | NAH→SPA | SPA→NAH |
|-------|--------|----------|---|---------|---------|
| IPA-only | 14_ipa_dict_only | 65.7% | — | 156 | 20 |
| + Speaker-Prior | 01_v7_best | 72.6% | +6.9pp | 117 | 9 |
| + Two-Pass IPA | 08_two_pass | 74.6% | +2.0pp | 103 | 9 |
| + FT-First | 10_ft_first | 80.6% | +6.0pp | 86 | 14 |
| + 3 Levers | 11_three_levers | 82.4% | +1.8pp | 83 | 14 |
| + Morphology Exp. | **13_morphology_expansion** | **85.7%** | +3.3pp | 56 | 23 |

> **Note (2026-04-01):** These numbers used midpoint matching and a GT snapshot that no longer exists. The canonical reproducible benchmark uses cue-index matching against the annotator DB (v2 snapshot): 73.7% duration-weighted / 71.6% segment accuracy on 550 segments. See "Benchmark Methodology Update" below F005 for details. Historical numbers here remain as measured.

**Oracle ceiling:** 500/551 = 90.7% — 51 segments correctly classified by NO config.

**Toggle impact (cumulative):**
- IPA-only: 65.7% — Phonemes carry 2/3 of the signal
- +speaker_prior: +6.9pp — Contextual knowledge ("they were speaking NAH before")
- +two_pass: +2.0pp — IPA evidence informs speaker profiles
- +ft_first: +6.0pp — Largest single lever, FT provides real morphology
- +whisper_uncertain_ipa + prior_reset + dj_tc_marker: +1.8pp
- +morphology_regex: +3.3pp — 28 patterns, 0 new SPA FP

**22 Unique wins:** Segments that ONLY morphology_expansion correctly solves.
**35 hardest:** NAH segments without FT output (FT didn't run).

**Insight:** The architecture is additive — each layer brings monotonic improvement.
FT-first is the largest lever because it provides *real NAH morphology* where LLM only produces garbage.

---

## E054: ft_first strict/lenient — Failed Experiment (2026-03-13)

**Hypothesis:** IPA check on ft_first override reduces SPA→NAH false positives.
Two variants: strict (IPA must confirm NAH) and lenient (IPA OR morphology suffices).
Additionally: extend ft_first to OTH→NAH.

**Result: WORSE** — 84.0% strict, 84.2% lenient (vs 85.7% baseline).

**Root cause:** OTH→NAH extension (`old_lang in ("spa", "other")`) caused +7 new
SPA→NAH errors. OTH segments that were actually SPA were falsely overridden to NAH.

**Lesson:** OTH is not "probably NAH" — OTH is ambiguous. ft_first may only override
SPA→NAH, not OTH→NAH. Code reverted to original logic.

---

## E055: Cross-Validation on La Otra Conquista (2026-03-13)

**Question:** Does the 85.7% accuracy from Hernan generalize, or is it overfitting to one film?

**Experiment:** morphology_expansion (EQ: 12_v7_full_track.json + expanded patterns) on
La Otra Conquista clip 24m15-34m15 (10 min, ritual scene with lots of Nahuatl). Ground truth:
74 annotated NAH+SPA segments (42 NAH, 32 SPA).

**Result: 81.1%** (60/74)

| | Hernan-1-3 | LOC 24m-34m |
|--|-----------|-------------|
| **Accuracy** | 85.7% (551 seg) | 81.1% (74 seg) |
| NAH→SPA/OTH | 56 (10.2%) | 9 (12.2%) |
| SPA→NAH | 23 (4.2%) | 5 (6.8%) |

**Error analysis LOC:**
- NAH→SPA (9): Whisper hallucinates English/French on NAH audio
  ("C'est mon sport wally", "I'll take my good time with you")
- SPA→NAH (5): LLM produces NAH-looking text ("Majonchitojua"),
  triggers morphology false positives
- SPEAKER_05 dominates the errors (analogous to young_xicontencatl in Hernan)

**Conclusion:** -4.6pp is expected with smaller sample, different film (different acoustics,
different speakers, different NAH dialect). **No overfitting.** The architecture generalizes.

Further LOC clips are being annotated for more robust statistics.

---

## Accuracy Progression — Definitive Numbers (2026-03-13)

> **Note (2026-04-01):** Numbers in this section used midpoint matching and a GT snapshot that no longer exists. See "Benchmark Methodology Update" below F005 for canonical reproducible numbers. Historical numbers here remain as measured at the time.

### Hernan-1-3 (551 NAH+SPA segments)

| Config | Acc | NAH% | SPA% | N→S | S→N |
|--------|-----|------|------|-----|-----|
| **13_morphology_expansion** | **85.7** | 85.0 | 87.1 | 56 | 23 |
| 11_three_levers | 82.4 | 77.7 | 92.1 | 83 | 14 |
| 10_ft_first | 80.6 | 75.6 | 91.0 | 86 | 14 |
| 08_two_pass | 74.6 | 66.5 | 91.6 | 103 | 9 |
| 01_v7_best | 72.6 | 63.3 | 92.1 | 117 | 9 |
| 14_ipa_dict_only | 65.7 | 56.0 | 86.0 | 156 | 20 |

### Cross-Validation: La Otra Conquista 24m-34m (74 NAH+SPA segments)

| Config | Acc | NAH→SPA | SPA→NAH |
|--------|-----|---------|---------|
| 13_morphology_expansion | 81.1% | 9 | 5 |

### Cross-Validation: La Otra Conquista 14m-24m (35 NAH+SPA segments)

| Config | Acc | NAH→SPA | SPA→NAH |
|--------|-----|---------|---------|
| 13_morphology_expansion | 80.0% | 4 | 3 |

### Cross-Validation: La Otra Conquista 34m-44m (133 NAH+SPA segments)

| Config | Acc | NAH Acc | SPA Acc | NAH→X | SPA→NAH | NAH Prec |
|--------|-----|---------|---------|-------|---------|----------|
| 13_morphology_expansion | 88.7% | 80% | 89.1% | 1 | 14 | 22% |
| 17_overlap_gate | 89.5% | 80% | 89.8% | 1 | 13 | 24% |

**Problem:** 5 NAH vs 128 SPA — class-imbalanced. 14 SPA→NAH errors = overlap.
Tecuichpo translates NAH simultaneously in background → tɬ/NAH phonemes bleed into SPA segments.

### LOC Combined (3 clips, 244 NAH+SPA segments)

| | NAH Acc | SPA Acc | Overall | NAH→X | SPA→NAH | NAH Prec |
|--|---------|---------|---------|-------|---------|----------|
| morphology_expansion | 76% | 87% | 84.4% | 15 | 23 | 67.6% |

---

## E056: Overlap Detection Gate — First Attempt (2026-03-13)

**Hypothesis:** Bimodal F0 distribution (Male 80-180Hz + Female 180-350Hz) detects simultaneous
speakers. Overlap flag blocks tɬ override and ft_first SPA→NAH override.

**Implementation:** `detect_overlap()` in tenepal_modal.py:
- Parselmouth F0 on segment window
- Bimodal = >20% voiced frames in BOTH frequency bands
- Additionally: HNR<10 (degraded harmonicity), intensity CV>0.15
- Score = bimodal×0.5 + low_hnr×0.3 + high_int_var×0.2
- Gate triggers only on bimodal=True (HNR alone too noisy)

**Result v1:** 101/136 flagged (score≥0.5 without bimodal requirement) — far too aggressive,
HNR<10 triggers on every noisy single-speaker segment.

**Result v2 (bimodal required):** 35/136 flagged, 10 tl-overrides blocked.
- 34m-44m: 88.7% → **89.5%** (+0.8pp), SPA→NAH 14→13, NAH Prec 22%→24%
- Small gain, no harm.

**Why only +1 error fixed:**
The gate sits in the WRONG place. Current pipeline order:

```
Audio → Diarization → IPA Extraction → Language Scoring → tɬ-Check → [Overlap Gate]
```

The 13 remaining SPA→NAH errors are classified as NAH DURING IPA extraction and language scoring
— BEFORE the overlap gate even runs. The gate only catches
tɬ overrides and ft_first overrides, not the initial misclassification.

**Correction needed — Pipeline order must be:**

```
Audio → Diarization → Overlap Detection → IPA (dampened on overlap) →
    Language Scoring (dampened on overlap) → tɬ-Check (disabled on overlap)
```

Overlap detection must come BEFORE IPA extraction, not after. The F0 analysis only needs
the vocals waveform (Parselmouth), not the IPA features — so it can be moved upstream.

**Next step:** Move overlap detection into the per-turn loop, BEFORE
`recognize_phonemes()` and `guess_language_from_text_markers()`. On overlap:
- Skip tɬ check completely
- Reduce IPA scoring weights (overlap_morphology_weight)
- Cap language confidence at max 0.5 (= uncertain, goes to OTH)

---

---

## E018: MossFormer2 Voice Separation on Overlap Turns (LOC 34m-44m)

**Date:** 2026-03-13
**Config:** `eq_configs/18_v7_overlap_separation.json`
**Hypothesis:** When Parselmouth detects overlap, use MossFormer2 for voice separation
and extract IPA separately per voice → cleaner language classification in overlap regions.

### Approach

1. Parselmouth `detect_overlap()` marks turns with bimodal F0 + low HNR
2. MossFormer2 (ClearVoice, `MossFormer2_SS_16K`) separates the audio into 2 sources
3. IPA extraction + language classification on each source separately
4. Source selection: prefer the non-NAH source (since SPA→NAH FP is the main problem)

### Pipeline Changes

- New Modal function `separate_overlap_segments()` on `mossformer_image` (T4 GPU)
- Batch processing: all 35 overlap segments in one remote call
- Per-turn: `_run_backend()` with `chunk_override` for separated audio
- Source selection v1 (RMS-based) → v2 (non-NAH preference)

### Results

| Config | Accuracy | SPA→NAH | NAH→SPA | NAH Prec | NAH Recall |
|--------|----------|---------|---------|----------|------------|
| 17 baseline (overlap_gate only) | 88.8% | 13 | 1 | 22.2% | 80.0% |
| 18 v1 (RMS source selection) | 88.8% | 13 | 1 | 22.2% | 80.0% |
| 18 v2 (non-NAH source selection) | 88.8% | 13 | 1 | 22.2% | 80.0% |

**Result: No effect.** All three variants identical.

### Analysis: Why MossFormer2 Doesn't Help Here

**1. RMS normalization:** Both separated sources have identical RMS values
(e.g., 0.0366 / 0.0366). MossFormer2 normalizes both outputs — no
energy-based differentiation possible.

**2. Separation ineffective:** In 20 of 35 overlap segments, BOTH
sources classify to the same language:
- Both NAH: turns 2, 3, 8, 96, 110, 122 (6 turns)
- Both OTHER: turns 6, 7, 45, 57, 60, 92, 95, 101, 108, 116 (10 turns)
- Both SPA: 0 turns

**3. "Non-NAH" source is usually OTHER, not SPA:** When sources
differ (15 turns), non-NAH selects "other" — the IPA on
the separated source doesn't look like Spanish. This is because
MossFormer2 is trained on English speech separation and doesn't
cleanly separate NAH/SPA.

**4. Root cause:** MossFormer2 is optimized for 2-speaker separation in English.
For short overlap regions (1-5s) with typologically similar
voices (NAH/SPA, similar phonology), no clean separation occurs.

---

## E018b: Full Voice Separation Comparison (LOC 34m-44m)

**Date:** 2026-03-13
**Context:** After MossFormer2 failure, three additional methods tested.

### Parselmouth Overlap Detection: Works

Parselmouth `detect_overlap()` flags **12 of 13** error cues correctly as overlap:

| Cue | Time | Score | F0 Range | HNR | Overlap? |
|-----|------|-------|----------|-----|----------|
| 56 | 191.0s | 1.00 | 86-129Hz | 7.2 | ✓ |
| 65 | 217.7s | 1.00 | 81-334Hz | 4.0 | ✓ |
| 71 | 234.7s | 1.00 | 77-378Hz | 1.2 | ✓ |
| 74 | 242.3s | 1.00 | 69-194Hz | 9.1 | ✓ |
| 76 | 275.8s | 0.80 | 148-200Hz | 7.9 | ✓ |
| 81 | 298.7s | 1.00 | 141-157Hz | 8.9 | ✓ |
| 82 | 302.5s | 0.80 | 94-153Hz | 6.7 | ✓ |
| 83 | 306.0s | 0.80 | 99-178Hz | 6.2 | ✓ |
| 93 | 336.6s | 1.00 | 114-222Hz | 6.8 | ✓ |
| 95 | 345.9s | 0.80 | 88-168Hz | 8.1 | ✓ |
| 111 | 412.1s | 0.00 | — | — | ✗ |
| 116 | 513.8s | 1.00 | 338-367Hz | 0.2 | ✓ |
| 132 | 594.6s | 1.00 | 224-414Hz | 2.8 | ✓ |

**The trigger is not the problem.**

### The Real Problem: F0 Gap

| Film | Speakers | F0 Gap | Separation |
|------|----------|--------|----------|
| **Hernan E03** | M ~116Hz vs F ~234Hz | **118Hz** | ✓ works |
| **LOC 34-44** | M ~100-160Hz vs M ~140-180Hz | **<30Hz** | ✗ impossible |

LOC 34-44 has two male speakers with nearly identical fundamental frequency.
Harmonics overlap completely — no spectral filtering possible.

### SepFormer (SpeechBrain, ML)

30s chunks around the error regions, parallel on Modal T4.

| Cue | SPK1 F0 | SPK2 F0 | SPK1 Lang | SPK2 Lang |
|-----|---------|---------|-----------|-----------|
| 56 | 101Hz | 104Hz | SPA | SPA |
| 74 | 137Hz | 137Hz | SPA | SPA |
| 76 | 157Hz | 159Hz | ? | SPA |
| 81 | 152Hz | 152Hz | ? | ? |
| 82 | 132Hz | 130Hz | SPA | SPA |
| 83 | 118Hz | 118Hz | NAH | SPA |
| 95 | 124Hz | 124Hz | SPA | SPA |

**Result:** Stems have identical F0 — SepFormer doesn't separate.
Both stems show SPA-like IPA. 8kHz output degrades additionally.

### PYIN+Wiener (Physics-based)

`scripts/pitch_based_separation.py` + `tools/pyin_wiener_sep.py`:
5 parameter variations tested (Harmonics 8-12, Width 15-35Hz, Male/Female ranges).

**Result:** Acoustic separation (different RMS), but Allosaurus IPA
on both streams → SPA. Not a single one of the 13 error cues shows NAH on one
side and SPA on the other.

### VibeVoice (Microsoft, ASR+Diarization)

Not a separator — delivers text + speaker ID. On the error region (180-360s):

- 17 segments, 2 speakers detected
- **Everything transcribed as Spanish** (correct!)
- Even recognizes NAH names: "Tecuichpo. Coaxtato acto quince" (cue 76 region)
- Confirms: the acoustic signal IS Spanish, the pipeline hallucinates NAH

### Conclusion: Voice Separation Doesn't Solve This Problem

All four methods (MossFormer2, SepFormer, PYIN+Wiener, VibeVoice) fail
at the same physical limit: **two male voices with <30Hz F0 distance
are spectrally inseparable.**

Voice separation only works with:
- Male/female pairs (>80Hz F0 gap) → Hernan ✓
- Distinctly different pitch ranges → not the case in LOC 34-44

The 13 SPA→NAH errors arise from:
1. **Overlap** (12/13 Parselmouth-confirmed) → IPA on mixed signal is ambiguous
2. **LLM fallback** hallucinates NAH-like text ("Koomenar", "Rohuyopulldtchetse")
3. **Pipeline** classifies these hallucinations as NAH markers

**Next approach: Overlap damping instead of separation:**
- On detected overlap: cap language confidence (max 0.5)
- Skip tɬ check on overlap completely
- Reduce IPA scoring weights
- Goal: more OTH/UNK instead of false NAH classification

---

---

*Last updated: 2026-04-01*
*Next step: Implement overlap damping and evaluate on LOC 34m-44m*

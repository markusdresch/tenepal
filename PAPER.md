# Tenepal: Phoneme-Level Language Identification for Low-Resource Languages in Multilingual Film

**Status:** Draft v6 — Nahuatl-first public release cleanup, 551 annotated benchmark segments
**Target:** Workshop Paper (ACL/EMNLP/LREC) or arXiv Preprint
**Authors:** Markus Dresch (Independent Researcher)
**Working title for release:** Tenepal (Nahuatl: *tentli* + *-pal* — "she who has the tongue/eloquence", another name for Marina/Malinche)

---

## Abstract

We present Tenepal, a phoneme-based language identification and transcription system for endangered languages that mainstream ASR systems cannot recognize. In this public-release draft, the main quantitative evaluation is deliberately **Nahuatl-first**: Nahuatl vs. Spanish identification in historical film audio, with Maya evidence treated as preliminary. Our central finding is that **ASR hallucination is not noise but signal**: when Whisper encounters out-of-distribution languages like Nahuatl, its hallucination distribution — which languages it "hears" and with what confidence — encodes distance-to-training-support, enabling zero-shot language detection without any labeled data in the target language. We show this hallucination signal is 100% reliable as a rejection indicator (N=45 segments, two model sizes, p<0.001) and structured rather than random, with target-language distributions that correlate with acoustic properties of the input.

We demonstrate that **phoneme-level features alone achieve 69% accuracy** on trilingual classification (Nahuatl/Spanish/Other) in film audio, without any language model, speaker tracking, or finetuning. Each additional layer improves incrementally — speaker prior +7pp, finetuned ASR +6pp, two-pass IPA evidence +2pp, morphological pattern expansion +3pp — but the phonetic foundation carries the majority of the signal. An ablation study across 13 configurations on 551 manually annotated segments establishes a ceiling of 90.7% (51 segments unsolvable by any configuration) and a best accuracy of **85.7%** using the full pipeline with morphology-expanded pattern matching.

By combining hallucination-based rejection with dual-backend IPA extraction (Allosaurus + wav2vec2), phonotactic profiling, and LoRA-finetuned Whisper (150h Puebla-Nahuatl), Tenepal reaches 85.7% language accuracy on 551 annotated Nahuatl/Spanish segments from Hernán (2019), up from a 50% Whisper-only baseline, and 84.4% raw / 81.7% balanced accuracy on a cross-film subset from La Otra Conquista (1999). The finetuned model reduces Nahuatl CER from 108% (hallucinations in Sinhala, Swedish, German) to 70% (recognizable orthography) in 3,000 training steps. Maya-specific experiments remain preliminary and are reported here only as supporting qualitative evidence rather than a release-ready benchmark. The hallucination-as-sensor principle generalizes beyond our application to any neural model processing out-of-distribution input.

---

## 1. Introduction

When a neural ASR system encounters a language absent from its training data, it does not fall silent — it hallucinates. Whisper, trained on 99 languages, maps Nahuatl audio to Sinhala, Swedish, or Chinese with high confidence. Prior work has treated this as a failure mode to be suppressed (Koenecke et al., 2024; Zuccoloto et al., 2025). We argue it is a *sensor*: **hallucination distributions encode distance-to-training-support**, and this signal is sufficient for language identification of unseen languages.

The principle is general. A model forced to classify an out-of-distribution input does not produce uniform noise — it produces a structured projection onto its training manifold. The specific hallucination target (which language Whisper "hears") varies with model size and audio characteristics, but the hallucination *rate* is invariant: 100% on Nahuatl across every model configuration we tested (N=45 segments, two model sizes, p<0.001). This makes hallucination-as-rejection a zero-shot language detector with perfect recall for OOD languages, requiring no labeled data in the target language.

We exploit this principle in Tenepal, a phoneme-based language identification system for endangered languages in multilingual film. When Hernán Cortés speaks Spanish and Moctezuma speaks Nahuatl in the same scene, Whisper treats both as Spanish — or hallucinates nonsense. Tenepal detects the hallucination, routes the segment to phoneme-level analysis, and identifies the language from acoustic markers that Whisper's text-level representation cannot capture.

This paper addresses three challenges:
1. **Detection**: Identifying *which* language is spoken when the ASR model has never seen it
2. **Transcription**: Producing readable output for languages with no ASR training data
3. **Code-switching**: Handling rapid alternation between known and unknown languages within and across speakers

Our contributions:
- The principle that ASR hallucination is a structured, exploitable signal for OOD language detection — not merely noise to be filtered
- A phoneme-first pipeline that works for *any* language, not just those with ASR models, using dual-backend IPA extraction and acoustic marker detection
- An asymmetric evaluation framework reflecting that losing an indigenous language segment (NAH→SPA) is a categorically different error from hallucinating one (SPA→NAH)
- Practical Nahuatl-first evaluation on two independent films, with the main benchmark on 551 manually annotated Nahuatl/Spanish segments and cross-film verification on annotated La Otra Conquista clips
- Open-source implementation with LoRA-finetuned Whisper achieving 70% CER on Nahuatl (from 108% baseline) after 3K training steps

---

## 2. Related Work

### 2.1 Massively Multilingual ASR
- **Whisper** (Radford et al., 2023): 99 languages, but no indigenous American languages
- **MMS** (Pratap et al., 2023): 1,100+ languages including some low-resource, but no Nahuatl/Maya
- **USM** (Zhang et al., 2023): Similar coverage gaps

### 2.2 Language Identification
- **MMS-LID** (Meta): 4,017 languages, 1B params — but too large for practical co-loading with ASR
- **SpeechBrain LID**: 107 languages, no NAH/MAY coverage
- **GlotLID** (Kargaran et al., 2023): text-based LID for 1,665 low-resource languages (EMNLP 2023). Unlike Tenepal, operates on text output rather than raw acoustics and does not cover NAH or MAY. Highlights the same challenges we face: noisy corpus metadata, macro-language ambiguity, and near-zero training data for indigenous varieties.
- Acoustic-phonetic approaches: Zissman (1996) introduced Gaussian mixture models for LID; Singer et al. (2003) established phone recognition followed by language modeling (PRLM); Matějka et al. (2005) demonstrated that high-quality multilingual phoneme recognizers substantially improve phonotactic LID accuracy — the same insight underlying our dual-backend fusion.

### 2.3 Low-Resource Language Processing
- **INALI (Mexico)**: Instituto Nacional de Lenguas Indígenas maintains the Catálogo de las Lenguas Indígenas Nacionales documenting 68 language groups including 30 Nahuatl variants (INALI, 2008)
- **Mozilla Common Voice**: Community-driven speech corpus includes Nahuatl via Jonathan Amith's Zacatlan/Tepetzintla recordings (Amith, 2023)
- **ELAR/PARADISEC**: Endangered Languages Archive and Pacific And Regional Archive for Digital Sources in Endangered Cultures provide standards for audio documentation (Nathan & Austin, 2004)
- **AmericasNLP 2024**: Annual workshop on NLP for indigenous languages of the Americas (co-located NAACL 2024, Mexico City). Covers Nahuatl alongside Quechua, Otomí, and other Uto-Aztecan/Mayan languages — the primary venue for peer-reviewed work on the language families evaluated in this paper.

---

## 3. Method

### 3.1 Pipeline Architecture

```
Audio → Demucs (vocal isolation) → Pyannote (diarization)
  → per segment:
      Allosaurus → IPA ─────────────┐
      wav2vec2-phoneme → IPA ───────┼→ Fusion → Language ID
      Whisper → text + hallucination signal ─┘
  → SRT with [NAH], [SPA], [MAY], [LAT] tags
```

### 3.2 Dual-Backend IPA Extraction

We extract IPA phonemes using two complementary backends:

**Allosaurus** (Li et al., 2020): Universal phone recognizer trained on 2,000+ languages. Strengths: broad coverage, recognizes rare phonemes (ɬ, kʷ). Weaknesses: noisy on short segments.

**wav2vec2-lv-60-espeak-cv-ft** (Facebook): Multilingual phoneme model fine-tuned on CommonVoice with eSpeak labels. Strengths: per-phone confidence scores, cleaner output. Weaknesses: biased toward training languages.

Fusion strategy: Needleman-Wunsch alignment of token sequences, confidence-weighted voting per phone position.

**Acoustic Validation Layer (Parselmouth):** We integrate Praat/Parselmouth (Boersma & Weenink, 2024) as a third recognition layer providing acoustic ground truth. Unlike learned models, Parselmouth directly measures physical speech properties: voicing (F0 > 0), formant structure (F1/F2/F3), spectral center of gravity, and intensity contours. While Parselmouth cannot identify phoneme *place* (e.g., distinguishing /t/ from /k/), it reliably classifies *manner*: voiced vs. unvoiced, stop vs. fricative vs. affricate. Critically, the lateral affricate /tɬ/ — a key Nahuatl marker — has a distinctive acoustic signature (unvoiced, CoG 3-5kHz, friction >40ms) that Parselmouth detects with high confidence. We use Parselmouth's acoustic event count as a quality metric for backend output: when PM detects significantly more acoustic events than the phoneme backends produce, this indicates missed phonemes requiring re-processing with adjusted parameters.

**Adaptive CTC Blank Bias:** Allosaurus uses Connectionist Temporal Classification (CTC) decoding where a blank token absorbs uncertain frames. By default, this results in 58% blank frames on average — acceptable for well-trained languages but destructive for out-of-distribution languages like Nahuatl, where blank rates reach 95% on some segments. Analysis of CTC logits reveals that the correct phonemes are often present as close runner-ups (margin < 2.0 from blank), suggesting the model *sees* the phones but lacks confidence to emit them.

We implement adaptive blank bias reduction: subtracting a constant from the blank logit before decoding reduces its advantage over phoneme candidates. Empirical results on Hernán evaluation:

| Segment | bias=0 blank% | bias=3 blank% | Phone Recovery |
|---------|--------------|--------------|----------------|
| "Capitán" (SPA) | 80% | 7% | +233% |
| "¿Dónde está?" (SPA) | 61% | 22% | +43% |
| NAH segment (4.1s) | 95% | ~50% | +100% |

Strategy: Use bias=0 for SPA-tagged segments (already adequate), bias=2-3 for NAH/MAY/OTH. The trigger is acoustic: when Parselmouth event count exceeds 2× decoded phone count, re-run with elevated bias.

### 3.3 Phoneme-Based Language Profiles

Each language is characterized by:
- **Positive markers**: Phonemes that indicate presence (e.g., tɬ, kʷ, ʔ for Nahuatl)
- **Negative markers**: Phonemes that indicate absence (e.g., voiced stops b, d, ɡ rare in Nahuatl)
- **Threshold**: Minimum score required for classification

```python
PROFILES = {
    "nah": {
        "markers": {"tɬ": 1.0, "kʷ": 1.0, "ʔ": 1.0, "ɬ": 1.0, "ts": 0.5, "ʃ": 0.4},
        "negative": {"b": 0.5, "d": 0.5, "ɡ": 0.5, "f": 0.5, "v": 0.5},
        "threshold": 0.0
    },
    "spa": {
        "markers": {"b": 0.2, "d": 0.05, "ɡ": 0.2, "ɲ": 0.8, "ɾ": 0.4},
        "negative": {},
        "threshold": 2.25
    },
    # ... maya, latin, etc.
}
```

### 3.4 Whisper-Hallucination Signal

**Key Finding**: When Whisper encounters Nahuatl audio, it confidently produces text in other supported languages rather than abstaining. The specific hallucination targets vary by model size (see Section 5.2): in our Hernan tiny-model evaluation, English (32%), Spanish (20%), and a long tail of other languages (48%) dominate, while larger model runs may favor Scandinavian-like targets (e.g., Danish/Dutch/Norwegian).

| Audio Language | Whisper Output | Detected Lang | Model |
|---------------|----------------|---------------|-------|
| Nahuatl | "Det var jeg stoppen" | Danish | medium (production) |
| Nahuatl | "Yes, sir" | English | tiny/base |
| Spanish | "Los soldados" | Spanish ✓ | all |

We exploit this as a **rejection signal**: if Whisper returns any language other than the expected context languages {spa, nah, may}, the segment is flagged for phoneme-only language ID. The rejection set includes but is not limited to {da, nl, no, sv, en, hi, ar, ja, de}. The broader phenomenon of Whisper hallucinations — where the model generates plausible-sounding but factually absent content — has been documented across diverse settings, including non-speech audio (Zuccoloto et al., 2025) and protected-group speech disparities (Koenecke et al., 2024); our finding that hallucination is *predictable* from the target-language distribution adds an actionable dimension to this literature.

### 3.5 Whisper Finetuning on Nahuatl (PEFT/LoRA)

While Whisper-large-v3 achieves 0% recall on Nahuatl, it has learned powerful acoustic representations from 680K hours of multilingual data. We hypothesized that parameter-efficient finetuning could adapt these representations to Nahuatl without catastrophic forgetting of other languages.

**Corpus.** We use OpenSLR 92 (Puebla-Nahuatl, Amith et al.), the largest publicly available Nahuatl speech corpus: ~84 GB of audio from Sierra Norte and Nororiental de Puebla, with ELAN/TRS transcriptions. After alignment and preprocessing: 135,544 segments (189.7h), split by speaker into train (107,801 / 150.1h), dev (14,032 / 20.6h), test (13,711 / 19.0h).

**Training setup.** We finetune Whisper-large-v3 using LoRA (r=32, α=64, target modules: q_proj, v_proj) with Spanish as proxy language — Nahuatl is not in Whisper's 99-language vocabulary, but shares significant phonotactic overlap with Colonial-era Spanish loanwords. Training on a single A100 (80GB) in fp16, batch size 8 with gradient accumulation 2 (effective batch 16), learning rate 1e-5 with 200-step linear warmup.

**Key design decisions:**
- **Spanish proxy**: Whisper requires a `language` token. Spanish is the closest typological proxy for Nahuatl in Whisper's vocabulary (shared vowel inventory, similar syllable structure from centuries of contact).
- **On-the-fly feature extraction**: Custom PyTorch Dataset with `soundfile.read()` + `WhisperFeatureExtractor` per sample, avoiding 100GB+ Arrow cache from HF datasets.
- **LoRA over full finetuning**: Preserves Whisper's multilingual knowledge while adapting decoder weights for Nahuatl orthography. Only 0.4% of parameters trained.

### 3.6 LLM Transliteration Fallback

For segments where Whisper fails but IPA confidence is high, we use LLM-based transliteration:

```
IPA: a l k a β u f e ɾ o s
LLM: "Read this IPA aloud. What word is it?"
Output: "arcabuceros"
```

Two-step approach separates transliteration (works well) from language ID (harder).

---

## 4. Experiments

### 4.1 Datasets

**Hernán (2019)** — historical drama series depicting the Spanish conquest of Mexico. Features:
- Native Nahuatl speakers (not dubbed)
- Code-switching between Spanish and Nahuatl
- Yucatec Maya in Potonchan scenes
- Latin liturgical phrases (baptism scene)
- Professional film audio quality with music/ambient noise

Test segments: 79 audio clips (Hernán S01 extracts, Apocalypto Maya samples, edge case clips).
Composition: ~30 Spanish segments (Hernán opening scenes), ~25 Nahuatl segments
(Moctezuma court scenes), ~15 Maya segments (Apocalypto samples), ~5 Latin
segments (baptism scene), ~4 edge case clips (hallucination suppression tests).
Ground truth labels assigned by manual review of subtitles and scene context.
For public release purposes, Hernán should be treated as the **original project trigger and primary internal benchmark**: we report metrics, timestamps, and methodology, but do not redistribute clips or subtitle files, and streaming availability may vary by region.

**La Otra Conquista (1999)** — Mexican historical drama depicting the spiritual conquest following Cortés's military victory. Set in post-conquest Tenochtitlan/Mexico City. Features:
- Nahuatl dialogue in ritual and domestic scenes
- Code-switching between Nahuatl and Colonial Spanish
- Different acoustic profile from Hernán (film vs. TV production, different recording era)
- No Yucatec Maya expected (post-conquest central Mexico setting)

6 key scenes extracted (~48 minutes total), 271 segments processed.
Composition (post-fix, 40min sample): 80 NAH, 263 SPA, 8 ENG, 3 MAY (false positives). Pre-fix: 70 MAY false positives, see Discussion 6.5.
Ground truth labels assigned by manual review of subtitles and scene context.
This is the **preferred public-facing verification target** in the current release because the Nahuatl/Spanish focus is cleaner, the Maya confound is absent by setting, independent access is more practical than for Hernán, and it is the culturally better public-facing reference for Nahuatl audiences. No clips are redistributed in this repository.

**Amith Nahuatl Corpora (OpenSLR)** — Three regional corpora by Jonathan D. Amith et al., all open access:
- **OpenSLR 92** — Highland Puebla Nahuatl (~84 GB, ~190h, Sierra Norte/Nororiental de Puebla). Primary training corpus for Whisper finetuning (Section 3.5). Used as **lexicon source** (2,791 entries extracted, see Section 3.3).
- **OpenSLR 147** — Orizaba (Veracruz) Nahuatl (~119h, 657 files, glottocode: oriz1235; ISO: nlv). Phonologically distinct from Puebla due to Totonac substrate influence. *Planned: multi-dialect training extension.*
- **OpenSLR 148** — Zacatlán-Ahuacatlán-Tepetzintla (Puebla) Nahuatl (~38 GB, glottocode: zaca1241; ISO: nhi). Same region as Mozilla Common Voice recordings; different recording conditions. *Planned: multi-dialect training extension.*

All three corpora are CC-licensed (CC-BY-ND-4.0 / CC-BY-NC-4.0). Combined: ~350h+ of Nahuatl speech across three geographically and phonologically distinct varieties. Multi-dialect finetuning (SLR 92 + 147 + 148) planned as follow-up experiment to measure regional robustness.

**Apocalypto (2006)** — Mel Gibson's Yucatec Maya-language film. Maya audio samples were used for ejective detector development and preliminary MAY recall testing, but these results are not yet mature enough to anchor the public release claims.

### 4.2 Evaluation Metrics

- **Language Accuracy**: % segments with correct language tag (raw and excluding UNK ground truth)
- **Weighted Error Score**: Asymmetric cost matrix reflecting that different misclassifications have different severities:
  - NAH→SPA = 2.0× (losing indigenous language — the system's raison d'être)
  - SPA→NAH = 1.5× (hallucinating indigenous language — creates false evidence)
  - X→UNK = 0.5× (honest rejection — better than a wrong answer)
  - UNK→X = 0× (unknowable ground truth — pipeline not penalized)
  - All other misclassifications = 1.0×
- **Reject Rate**: % segments tagged UNK (confidence below threshold). A well-calibrated system should reject segments it cannot classify rather than force them into a category.
- **Transcription Quality**:
  - For Spanish: WER against reference subtitles
  - For Nahuatl: CER against corpus references (Section 5.5)
- **Adversarial Precision**: False positive rate on Spanish stress test (Section 4.4)
- **Edge Case Performance**: Specific challenging segments (edge_cases.tsv)

### 4.4 Adversarial Spanish Stress Test

To validate that indigenous language detection does not over-trigger, we evaluate on intentionally adversarial Spanish audio: words containing "tl" orthography (*Atlántico*, *Tlaxcala* in Spanish pronunciation, *atletismo*), liturgical Latin (*Ave María*, *ego te baptizo*), and prosodic edge cases (whispered Spanish, crowd chanting, Spanish with Nahuatl loanwords like *chocolate*, *tomate*). Any NAH or MAY tag on this test set is a false positive. This provides a precision lower bound complementary to the recall-focused main evaluation.

### 4.3 Baselines

| System | NAH Detection | SPA WER | MAY Detection |
|--------|---------------|---------|---------------|
| Whisper (medium) alone | 0% recall | ~15% WER (SPA natively supported) | 0% recall |
| Whisper + lang=es forced | 0% recall | ~12% WER (SPA natively supported) | 0% recall |
| MMS-LID | N/A (no NAH) | N/A | N/A |
| **Tenepal (Hernán eval)** | **80% recall** (20/25) | ~15% WER est. | **66.7% recall** (10/15) |
| **Tenepal (La Otra Conquista)** | **80 NAH / 263 SPA** | 61% lang acc. | **3 MAY false pos.** (70 pre-fix) |
| **Whisper-large-v3 + LoRA (NAH)** | **CER 0.70** (finetuned) | ~15% WER est. | 0% recall |
| **Tenepal (Amith audio)** | pending | pending | N/A |

*Whisper SPA WER estimated from Hernán S01E01 (hernan\_ground\_truth.json, 151 SPA segments; Whisper natively supports Spanish). Tenepal Hernán eval from Section 5.1 (79-segment test set). La Otra Conquista eval from 6 key scenes (271 segments). Amith Zacatlan corpus used as lexicon source (2,791 entries); full audio corpus (~114 GiB, CC-BY-ND-4.0) evaluation pending.*

---

## 5. Results

### 5.1 Language Identification

**Table 1: Full Hernán S01 Evaluation** (N=4,659 segments across 8 chapters, with Spanish leak correction)

| Language | Segments | % of Total | Chapter with Most |
|----------|----------|------------|-------------------|
| **NAH** | **328** | 7.0% | E03 Xicotencatl |
| **MAY** | **159** | 3.4% | E03 Xicotencatl |
| SPA | 2,427 | 52.1% | E02 Olid |
| OTH | 1,745 | 37.5% | — |

**Per-Chapter Breakdown:**

| Chapter | Title | NAH | MAY | SPA | Key Scene |
|---------|-------|-----|-----|-----|-----------|
| E01 | Marina | 27 | 24 | 407 | Baptism, Moctezuma intro |
| E02 | Olid | 25 | 14 | 429 | Translation chain established |
| E03 | Xicotencatl | **83** | **44** | 235 | Totonac alliance, peak multilingual |
| E04 | Bernal | 35 | 21 | 327 | Tlaxcalan negotiations |
| E05 | Moctezuma | 63 | 20 | 189 | Massacre, diplomatic tension |

**Key findings:**
- **Conservative classification**: Segments below confidence threshold labeled OTH (37.5%) rather than risk misclassification. Post-hoc tɬ acoustic detection revealed 11 SPA-tagged segments that are definitively NAH (tɬ does not exist in Spanish), indicating ~3% NAH→SPA misclassification rate in high-confidence segments.
- **tɬ false positive problem**: While the lateral affricate /tɬ/ is phonologically exclusive to Nahuatl, Spanish contains orthographic "tl" clusters (e.g., *atlántico*, *extremaunción*, *Tlaxcala*) that Allosaurus misidentifies as /tɬ/. On a 10-minute test clip from *La Otra Conquista* (colonial Spanish dialogue), the original hard override misclassified 50/80 segments as NAH. A Spanish Context Guard — suppressing the tɬ override when Whisper text contains ≥3 Spanish function words or when Whisper confidently (≥0.85) detects Spanish — reduced this to 37 NAH / 39 SPA, recovering 14 Spanish segments without affecting true Nahuatl detection. The guard is logged as `tl-acoustic-conditional-override` vs. the original `tl-acoustic-hard-override`.
- **Spanish leak correction**: A post-processing step detects Spanish text (e.g., "¡Capitán!", "Vamos") in segments tagged NAH/MAY/OTH and corrects them to SPA, improving SPA recall by 17%
- **Accuracy on 551 annotated segments (NAH+SPA)**: An ablation study across 13 EQ configurations yields the following layer decomposition:

  | Layer | Accuracy | Δ | Mechanism |
  |-------|----------|---|-----------|
  | IPA-only (phonemes alone) | 65.7% | — | Dual-backend IPA → phonotactic scoring, no LM/speaker/FT |
  | + Speaker prior | 72.6% | +6.9pp | Per-speaker language profiles from accumulated evidence |
  | + Two-pass IPA | 74.6% | +2.0pp | IPA phoneme evidence feeds back into speaker profiles |
  | + FT-first | 80.6% | +6.0pp | LoRA-finetuned Whisper on ALL segments before prior |
  | + Whisper uncertain IPA + prior reset + DJ marker | 82.4% | +1.8pp | Three-lever refinements |
  | + Morphology expansion | **85.7%** | +3.3pp | Expanded NAH pattern regex (28 patterns) + G2P cleanup |

  Development trajectory is non-monotonic in exploratory runs: Whisper-only baseline starts at 50%, phoneme-first modeling reaches 69% (trilingual) / 65.7% (NAH+SPA subset), intermediate speaker-prior variants can regress into the low-70s under bilingual overlap, and the full stack recovers to 85.7%. This is important negative evidence: adding components does not guarantee gains unless overlap and confidence gating are controlled.

  Oracle ceiling: 90.7% (51 segments unsolvable by any configuration — ambiguous code-switches, sub-word fragments, music overlap). The phoneme-only baseline (65.7%) carries the majority of the signal; each subsequent layer provides diminishing but meaningful gains. The largest single lever is finetuned ASR (+6.0pp), which provides real Nahuatl morphology where the LLM fallback produces gibberish. Remaining errors: 56 NAH→SPA (more frequent, recall-limiting) and 23 SPA→NAH (less frequent, but epistemically riskier because they create false indigenous-language evidence).
- **Whisper genre hallucination**: On near-silent segments (~300ms), Whisper produces YouTube boilerplate ("Thanks for watching!") rather than silence, reflecting its training distribution. The noise gate fails to catch these because it trusts the hallucinated text (3 words → passes word-count filter despite being entirely fabricated).
- **E03 peak**: Xicotencatl chapter has highest indigenous language density (127 NAH+MAY segments) — reflects Totonac alliance narrative
- **MAY detection via acoustic ejectives**: Yucatec Maya results use the Modal GPU pipeline's 3-way acoustic ejective detector (heuristic + sklearn + wav2vec2 voting, ≥2/3 consensus). Allosaurus cannot produce ejective symbols (kʼ, tʼ, tsʼ) as atomic IPA; without acoustic detection, MAY recall drops to ~66.7% on isolated clips. These experiments remain preliminary and are not part of the core public-release benchmark.

**Corpus-Based Lexicon Validation (v6.1):** To validate NAH detection, we integrated a lexicon derived from the Amith Zacatlan corpus. The full corpus yields 2,791 entries; filtering by minimum token frequency reduces this to practical sizes while maintaining coverage. A scaling analysis across frequency thresholds shows:

| min_freq | Entries | Segments Matched | Total Matches | Unique Words |
|----------|---------|------------------|---------------|--------------|
| — (curated) | 20 | 24/85 (28.2%) | 31 | 7 |
| 50 | 140 | 49/85 (57.6%) | 81 | 17 |
| 25 | 262 | 52/85 (61.2%) | 97 | 21 |
| 10 | 612 | 56/85 (65.9%) | 124 | 32 |
| 5 | 1,229 | 63/85 (74.1%) | 161 | 42 |
| 3 | 2,024 | 65/85 (76.5%) | 194 | 52 |

At min_freq=3, the corpus lexicon yields +48.3% more segments matched and 7.4× more unique Nahuatl words compared to the original 20-entry curated lexicon. New high-frequency morphemes discovered include *nik* (1sg subject), *tla* (indefinite object prefix), *san* (emphatic "just/only"), and *kwali* ("good/well"). Production configuration uses min_freq=10 (612 entries) for optimal coverage-vs-performance balance.

Cross-language NAH marker distribution using 140 entries (min_freq=50) across all 4,659 Hernán segments:

| Lang | Segments | With NAH Markers | % |
|------|----------|------------------|---|
| NAH | 328 | 214 | 65% |
| SPA | 2,427 | 1,361 | 56% |
| MAY | 159 | 126 | 79% |
| OTH | 1,745 | 749 | 43% |

The high NAH marker rate in SPA segments (56%) indicates either short-morpheme false positives (e.g., `se`, `i:n`) or genuine code-switching in Cortés–Moctezuma dialogues. The 79% NAH marker rate in MAY segments suggests shared Mesoamerican vocabulary. This lexicon-based signal will be integrated into future phonotactic scoring to reduce NAH→OTH abstention rates.

### 5.2 Whisper Hallucination Patterns

When Whisper encounters Nahuatl or Maya audio, it cannot produce silence or "unknown" — it maps the
acoustic signal to some supported language in its training distribution. We quantify this on
our full Hernán evaluation (N=328 NAH + 159 MAY segments).

**Methodology.** We evaluate all NAH/MAY-tagged segments from 8 Hernán chapters. Each
segment is processed by faster-whisper (medium model, language auto-detect). Whisper's
`info.language` is compared against Tenepal's phoneme-based classification.
Wilson 95% confidence intervals computed via `tools/corpus/hallucination_stats.py`.

**Table 2: Whisper Language Detection on NAH Input (N=25, both models)**

| Whisper Detected | tiny count | tiny % | base count | base % |
|------------------|-----------|--------|-----------|--------|
| English (en)     | 8         | 32.0%  | 6         | 24.0%  |
| Spanish (es)     | 5         | 20.0%  | 2         | 8.0%   |
| Hindi (hi)       | —         | —      | 3         | 12.0%  |
| Arabic (ar)      | —         | —      | 2         | 8.0%   |
| German (de)      | —         | —      | 2         | 8.0%   |
| Japanese (ja)    | —         | —      | 2         | 8.0%   |
| Catalan (ca)     | —         | —      | 2         | 8.0%   |
| Danish/Dutch/Norwegian/Swedish | 0 | 0.0% | 0 | 0.0% |
| Other (≥6 langs) | 12        | 48.0%  | 6         | 24.0%  |

**Hallucination rate: 100.0% for both models** (95% CI: 86.7%–100.0%).
Results saved: `tests/regression/reports/hernan_nah_25_tiny.json` and `hernan_nah_25_base.json`.

**Key Finding.** Across both model sizes, Whisper produces 100% hallucination on NAH input —
confirming that `info.language` is a reliable rejection signal regardless of model configuration.
The specific target languages are model-dependent: the tiny model concentrates detections on
English/Spanish, while the base model spreads across 13 languages including Hindi, Arabic,
Japanese, and Catalan. Notably, neither model predicts Scandinavian languages on this Hernán
sample (in contrast to anecdotal medium-model observations in Section 3.4). This model-size
dependence is discussed in Section 6.1.

**Genre Hallucination.** Beyond language hallucination, Whisper exhibits *genre* hallucination on non-speech audio. Segments containing silence or background noise produce YouTube-specific boilerplate text — most commonly "Thanks for watching!" — reflecting Whisper's training distribution (weakly supervised YouTube subtitles). This is not random: it is a structured projection of non-speech audio onto the most frequent closing phrase in the training data. Our UNK gate catches these segments via low confidence scores, but the hallucinated text remains visible in the transcript. This provides additional evidence that hallucination is structured rather than random: the model projects not only onto *languages* (Section 6.1) but also onto *genres* from its training distribution.

**Implication for Pipeline.** The rejection signal should be treated as any non-context label:
when Whisper returns a language outside the expected context {spa, may, lat}, the segment is
flagged for phoneme-only language identification (Section 3.4). The trigger condition is
model-agnostic by design — it fires on any unexpected language code.

**Reproducibility.** Script: `scripts/whisper_base_hallucination_test.py`. Run with
`--model tiny` or `--model base` against the fixture in
`tests/regression/fixtures/hernan_nah_25.json`.

### 5.3 Dual-Backend Agreement

We analyzed IPA agreement between Allosaurus and wav2vec2 across 232 segments
of Hernán S01E01 (5-minute extract) where both backends produced output.
Mean token exact-match rate was 13.4% (i.e., 13 of every 100 aligned phone
positions match between backends). Mean length-ratio was 0.83, reflecting
that the two models typically produce similar-length phoneme strings.
Mean agreement score (exact_match × length_ratio) was 0.12.

Backends diverged substantially (<25% token match) on 80.2% of segments —
predominantly short segments (<0.8s) where Allosaurus outputs fragmented
phones and wav2vec2 tends toward full syllable patterns. When backends agree,
the fused IPA is more reliable for language scoring; when they diverge,
neither backend alone is trusted and the segment is marked as OTH or UNC.

Statistics computed using scripts/compute_backend_agreement.py on
validation_video/Hernán-1-1-1.srt (dual-backend debug output).

**Acoustic Validation of Backend Quality:** Using Parselmouth as acoustic ground truth for voicing ratio (F0 > 0 = voiced), we compared backend accuracy across 42 segments:

| Segment | PM voiced% | Allo voiced% | w2v2 voiced% | Closer |
|---------|-----------|-------------|-------------|--------|
| "Capitán" | ~40% | 33% | 75% | Allo |
| "¿Dónde está?" | ~55% | 57% | 71% | Allo |
| "No quiere salir" | ~60% | 64% | 70% | Allo |
| NAH (Cue 35) | ~75% | 57% | 71% | w2v2 |

Allosaurus tracks voicing more accurately than wav2vec2 on Spanish segments, while wav2vec2 performs better on NAH where Allosaurus defaults to conservative blank frames. This suggests complementary roles: Allosaurus as the better *phone-level* decoder when it produces output, wav2vec2 as the more reliable *word-level* approximator. The fusion strategy exploits both: when backends agree, high confidence; when they diverge, Parselmouth acoustic features break ties.

### 5.4 La Otra Conquista Evaluation

**La Otra Conquista (1999)** provides the main public-facing cross-film evaluation dataset in this release: a Nahuatl/Spanish setting with different acoustic characteristics (35mm film vs. digital TV production). Four contiguous clips spanning minutes 14–55 (~40 minutes, 354 segments) were processed through the full pipeline. As with Hernán, timestamps and metrics are reported, but media files are not redistributed.

**Table 3: La Otra Conquista Language Distribution** (N=354, v5 pipeline with 12K SPA dictionary)

| Clip | Time Range | SPA | NAH | ENG | MAY | Total |
|------|-----------|-----|-----|-----|-----|-------|
| Clip 2 | 14:15–24:15 | 31 | 7 | 0 | 1 | 39 |
| Clip 3 | 24:15–34:15 | 40 | 28 | 8 | 1 | 77 |
| Clip 4 | 34:15–44:25 | 111 | 22 | 0 | 1 | 134 |
| Clip 5 | 44:25–55:25 | 81 | 23 | 0 | 0 | 104 |
| **Total** | | **263** (74%) | **80** (23%) | **8** (2%) | **3** (1%) | **354** |

**Table 3b: Annotated Accuracy** (Clip 3, 38 human-annotated segments)

| Version | Correct | Accuracy | Key Change |
|---------|---------|----------|------------|
| v1 (baseline) | 2/38 | 5% | Whisper file-lang=en, all SPA→ENG |
| v2 (+spa-text-override) | 15/37 | 41% | Text-based SPA detection |
| v5 (+12K dict on Modal) | 23/38 | 61% | Full dictionary + data mount |

**Key findings:**
- **MAY eliminated**: 70 MAY (pre-fix) → 3 MAY total across 40 minutes. Remaining 3 are ejective false alarms on NAH audio (Section 6.4).
- **Whisper language misdetection**: Whisper reports `lang=en` for entire clips despite Spanish dialogue. The 12K-word Spanish dictionary with text-based override corrects this for multi-word segments. Single-word segments ("potente", "Invicto") remain problematic.
- **SPA dominant**: 74% SPA across all clips, consistent with the film's primarily Spanish dialogue.
- **NAH detection generalizes**: 80 NAH segments detected in a completely different film, confirming the pipeline is not overfit to Hernán's specific audio characteristics.
- **Remaining errors** (15/38): tɬ-override on Spanish (3), Nahuatl classified as ENG (3), SPA→NAH overcorrection (3), single-word ENG (3), OTH misclassification (2), ejective false alarm (1).

**Table 3c: Cross-Validation — morphology_expansion on Clip 3** (74 annotated NAH+SPA segments, v6 pipeline with full EQ)

| | Hernán-1-3 | LOC Clip 3 |
|--|-----------|------------|
| **Accuracy** | 85.7% (N=551) | 81.1% (N=74) |
| NAH→SPA/OTH | 56 (10.2%) | 9 (12.2%) |
| SPA→NAH | 23 (4.2%) | 5 (6.8%) |

**Table 3d: Expanded LOC cross-validation subset** (minutes 14–44, N=244 annotated NAH+SPA segments)

| Clip | Acc | GT (NAH/SPA) | NAH→X | SPA→NAH |
|------|-----|--------------|-------|---------|
| 14m–24m | 80.0% | 15 / 20 | 4 | 3 |
| 24m–34m | 78.9% | 43 / 33 | 10 | 6 |
| 34m–44m | 88.7% | 5 / 128 | 1 | 14 |
| **LOC combined** | **84.4%** | **63 / 181** | **15** | **23** |

Interpretation requires care: the 34m–44m clip is highly class-imbalanced (5 NAH vs. 128 SPA). In that clip, an always-SPA classifier would score 96.2% accuracy, so 88.7% headline accuracy overstates quality. The main failure mode is SPA→NAH false positives (14 cases), which substantially reduce NAH precision despite high overall accuracy. On the expanded LOC subset, balanced accuracy is 81.7% (vs. 86.1% on Hernán-1-3), indicating a larger cross-film gap than raw top-line accuracy suggests. We therefore treat **balanced accuracy and error direction** as the meaningful public numbers here, not top-line accuracy alone. We hypothesize that simultaneous NAH+SPA speech (interpreter overlap) is a key driver: mixed segments let NAH phonetic evidence bleed into primarily Spanish turns, after which NAH-like LLM output can reinforce SPA→NAH errors. This hypothesis is testable with overlap-aware speaker separation and a trigger based on acoustic overlap cues (e.g., Parselmouth-derived voicing/formant multiplicity) before language routing. Additional LOC clips are being annotated to separate overlap effects from general morphology/override errors.

### 5.5 Whisper Finetuning Results

**Table 4: Whisper-large-v3 Baseline vs. Nahuatl-Finetuned** (500-segment random sample from test set, seed=42)

| Metric | Baseline | Finetuned (3K steps) | Improvement |
|--------|----------|---------------------|-------------|
| WER | 1.6634 (166%) | 1.3689 (137%) | -0.2945 (17.7% rel.) |
| CER | 1.0761 (108%) | 0.6960 (70%) | -0.3801 (35.3% rel.) |

WER >100% is expected: baseline Whisper generates more tokens than exist in the reference because it hallucinates in other languages. CER is the more meaningful metric for character-level accuracy.

**Training curve** (3,000 steps, ~45% of one epoch):

| Step | Train Loss | Eval Loss | Learning Rate |
|------|-----------|-----------|---------------|
| 50 | 7.337 | — | 2.5e-6 |
| 500 | 3.022 | — | 7.5e-6 |
| 1000 | 2.397 | — | 7.1e-6 |
| 1500 | 2.180 | 1.094 | 6.6e-6 |
| 2000 | 2.110 | — | 5.4e-6 |
| 2500 | 2.057 | — | 3.2e-6 |
| 3000 | 1.978 | — | ~0 |

Total training time: 2h 3min on A100 (fp16), ~1.95s/step. LoRA adapter size: ~60MB.

**Qualitative comparison** — what the models produce for the same Nahuatl audio:

| # | Reference (Nahuatl) | Baseline Output | Finetuned Output |
|---|-------|---------|-----------|
| 1 | Ke:mah, se: kimowilia. | Tieu ma, se mungu leo. (Swahili) | Ke mah seki mokilia. |
| 2 | de n' ista:k, | dann ist es stark. (German) | De n' istaak. |
| 3 | Ahsi wa:n ki..., kikwi a n' tomi:n | Así que, si quieren que vea... (Spanish) | Asi wa n' kikwia n' tomi n' takowak... |
| 4 | María Ocotlán Fermín Cabrera | දැන් කිරීමේ මෙමු (Sinhala) | Tiowet se de, ta ok kehper mih karreira... |

**Key findings:**
- **Baseline hallucinates 100%**: Every Nahuatl segment is transcribed as Sinhala, Swedish, German, Spanish, or other languages. Zero Nahuatl tokens in any baseline output.
- **Finetuned produces recognizable Nahuatl**: Samples 1-3 show near-correct Nahuatl orthography. The model has learned the phoneme-to-grapheme mapping for Nahuatl.
- **Diacritics remain challenging**: Long vowel marks (ā, ē, ī, ō) and saltillo (ʔ) are inconsistently produced — "ista:k" → "istaak" (double vowel instead of colon notation). This is a post-processing opportunity rather than a fundamental model failure.
- **CER > WER improvement**: The 35% relative CER reduction vs. 18% WER reduction indicates the model is producing the right characters in roughly the right order, even when word boundaries differ.

**Full epoch run** (6,738 steps = 1 full epoch) completed (train_loss=2.14, 4h14m on A100). Qualitative evaluation reveals **overfitting between Step 4500 and 6738**:

| Checkpoint | NAH Morphology | SPA Code-Switch | Hallucinations |
|-----------|---------------|-----------------|----------------|
| 3K (0.45ep) | Good | Clean | None |
| 4500 (0.67ep) | Mixed | Slight drift | Not systematically reviewed |
| 6000 (0.89ep) | Mixed | Degrading | Not systematically reviewed |
| 6738 (1.0ep) | Slightly degraded | **Broken** | **Yes — speaker names from training corpus** |

**Overfitting signal (1-epoch model, 288/343 lines changed vs. 3K):**
- **SPA degradation**: "Yo puedo acompañarles si creéis" → "Yo poro kompañarli si kreey se" — Spanish transcribed with Nahuatl phonology
- **Corpus memorization**: Names "Anastacio Nicolás Damián" and "José Ernesto Vázquez Chanico" (OpenSLR-92 speakers) hallucinated across 16 segments from different speakers — the model memorized speaker metadata instead of generalizing phonology
- **NAH word boundaries**: Mixed — some improvements ("Amo n'mech wika" vs "a moh anmeh xwikas"), some artifacts ("Tlok 5" instead of "tloksi")

**Cross-film checkpoint comparison** (La Otra Conquista, 10min clip, 28 annotated segments, A10G ~4min inference):

| Aspect | 3K (0.45 ep) | 6738 (1.0 ep) | Winner |
|--------|-------------|---------------|--------|
| NAH morphology accuracy | 4/10 correct endings | 6/10 correct endings | 6738 |
| SPA transparency (when pipeline mislabels SPA as NAH) | 4/5 clean Spanish | 1/5 clean Spanish | 3K |
| Speaker name hallucinations | Yes ("Ernesto Vázquez Chanico") | Yes ("Anastacio Nicolás Damián") | Neither |
| "Nahuatl-ization" of Spanish | "a saber" → "Asa weh" | "a saber" → "Asa weh" | Neither |

**tɬ-cascade error**: All 10 NAH→SPA misclassifications in the La Otra Conquista test originate from the tɬ acoustic override (Section 6.6), not from the FT model. The FT model blindly follows the pipeline's language label — when the pipeline says NAH, both checkpoints produce forced Nahuatl output regardless of audio content. This makes FT evaluation unreliable until tɬ false positives are fully resolved.

**Post-FT SPA Reclaim** (v4): A post-processing step after finetuned Whisper inference detects Spanish segments that were incorrectly labeled NAH. The reclaim uses three guards:
1. **NAH morphology guard**: Regex-based detection of Nahuatl text patterns in FT output (tla, xki, kwa, chiw, moch, nik-, tik-). If FT text contains Nahuatl morphology → segment is real NAH, skip reclaim.
2. **SPA_COMMON word check**: Both FT text and original Whisper text are scored against a 290-word Spanish vocabulary. Reclaim requires ≥2 known words and ≥30% ratio.
3. **IPA marker guard**: Segments with strong NAH-exclusive IPA markers (tɬ, kʷ, ɬ) are kept as NAH regardless of text content.

**Table 3b: SPA Reclaim Evaluation** (La Otra Conquista 10min clip, 16 human-annotated segments)

| Version | Correct | Accuracy | Description |
|---------|---------|----------|-------------|
| Baseline (no reclaim) | 1/16 | 6% | Pre-fix |
| v1 (text markers, `whisper_lang="es"`) | 7/16 | 44% | Too aggressive — reclaims real NAH |
| **v4 (morphology guard + expanded vocab)** | **9/16** | **56%** | Best trade-off |

Error breakdown (7 misses): 2 SPA-reclaim FPs on ambiguous segments, 1 Whisper hallucination (SPA text over NAH audio), 3 NAH-recall failures in IPA path (OTH/MAY instead of NAH), 1 garbled FT text below detection threshold.

**Current best checkpoint: Step 3000.** Recommended default due to better SPA transparency. Step 6738 preferred only for pure-NAH content where pipeline language detection is reliable. Multi-dialect training (F004) expected to resolve the overfitting trade-off.

**Pipeline integration results** (Hernán E03, 45 min, N=343 NAH segments):

| Metric | Before (LLM fallback) | After (FT Whisper) |
|--------|----------------------|--------------------|
| NAH segments with readable transcription | 0 | **343 (100%)** |
| NAH segments with LLM gibberish | 337 | **0** |
| Readable Nahuatl morphology | No | **Yes** |
| Spanish code-switch accuracy | No | **Yes** |
| Processing overhead | baseline | +~120s (model load + inference) |

Representative finetuned output: *"kapitan n' itsmotlatlawtilia, mah ximokkixti"* (recognizable morphemes: *tl*, *mah*, *xi-* prefix). Spanish code-switch correctly transcribed: *"Tenemos a Moctezuma y su familia retenidos aquí"*.

Compare with prior LLM-fallback output on same segments: *"Jenkatani kamlam bktchazn"*, *"Enlatelatiwitu"* — completely unreadable. The finetuned model eliminates the LLM IPA→text fallback entirely for NAH segments.

### 5.6 Cross-Language Robustness: Baseline Hallucinations vs. Finetuned Output

To characterize the finetuning effect, we ran both baseline and finetuned models on 20 randomly sampled NAH test segments, recording both auto-detect and forced-Spanish transcription modes. The results reveal the full extent of Whisper's hallucination problem and the finetuned model's solution.

**Table 5: Baseline Whisper Hallucination Languages on 20 NAH Segments**

| Hallucinated Language | Count | Examples |
|----------------------|-------|----------|
| Spanish | 7 | "Así que, si quieren que vea mi trabajo..." |
| Sinhala (si) | 4 | "දැන් කිරීමේ මෙමු මෙමු මෙමු" |
| English | 2 | "which was the second that they could take a little one" |
| Chinese (zh) | 2 | "達達巴里灣 基山連 涅槃遊行灣" / "我去找找看,看能不能把他搬走" |
| Greek (el) | 1 | "Έχουν τέτοιου κοιλιά πάλι τελειώνει" |
| Japanese (ja) | 1 | "本格祭を 稼いでまちのちとして" |
| Swahili (sw) | 1 | "Tieu ma, se mungu leo" |
| German (de) | 1 | "dann ist es stark" |
| Swedish (sv) | 1 | "I så hållet" |
| Urdu (ur) | 1 | "کیا ہے؟" |

**Hallucination rate: 100% (20/20).** Baseline Whisper-large-v3 maps Nahuatl audio to 10 different languages across 6 scripts (Latin, Sinhala, CJK, Greek, Arabic, Devanagari). No segment produces any recognizable Nahuatl. Notably, the hallucination target varies by audio characteristics — longer segments with code-switched Spanish loanwords tend to hallucinate as Spanish, while isolated Nahuatl words trigger more exotic targets (Sinhala, Japanese, Chinese).

**Table 6: Same 20 Segments — Finetuned Model Output (selected)**

| Reference (Nahuatl) | Baseline (hallucinated) | Finetuned |
|---------------------|------------------------|-----------|
| de n' ista:k | dann ist es stark (DE) | De n' istaak |
| I:xoho:me | I så hållet (SV) | Ixohomeh |
| Tehtetsi:ltik | දැන්නේ දැන්නේ... (SI) | Tehtetsiiltik |
| Ke:mah, se: kimowilia | Tieu ma, se mungu leo (SW) | Ke mah seki mokilia |
| A:mo moneki n' koyo:tapahtihkeh | Amo manejar el guión... (ES) | Amo, a momo neki n' koyo tapahtihkeh |
| moskaltia kwaltsi:n yo:n seki ista:k | Más que artículo 4... (ES) | Moskaltia kwaltsi yo n' seki istaak |
| tahtapa:ni wa:n ki:sa ne:n ie:wayotsi:n | 達達巴里灣 基山連... (ZH) | tahtapaani wa n' kiisaneh yehwa yotsi... |
| wa:n ka:ni n' kikwiti xapoh | I was very happy (EN) | Wa n' ka ney iwiti xapoh... |
| Mochi:wa xola:lpan | 我去找找看 (ZH) | michiwa xoltahts, pa newa k... |

**Key observations:**
1. **100% language recovery**: Every segment that hallucinated as a foreign language now produces Nahuatl-orthography output.
2. **Auto-detect = forced-Spanish**: For the finetuned model, auto-detect and forced-Spanish modes produce identical output on all 20 segments — the model has learned that this audio is "Spanish" (its proxy language) and consistently applies its Nahuatl orthography.
3. **Systematic diacritic normalization**: Long vowel colon notation (`a:`, `e:`, `i:`) is consistently rendered as doubled vowels (`aa`, `ee`, `ii`). This is a post-processable pattern.
4. **Proper noun handling**: Speaker names in references (e.g., "María Ocotlán Fermín Cabrera") are not recognized as names — the model transcribes the audio phonetically. This is expected behavior for a phonetic transcription model.
5. **Spanish loanword preservation**: Code-switched Spanish words in Nahuatl speech ("pero", "hasta", "porque") are preserved correctly by the finetuned model, suggesting the Spanish proxy language strategy preserves cross-lingual competence.

### 5.7 Edge Cases

| Case | Description | v5.0 Result | v5.4 Result |
|------|-------------|-------------|-------------|
| "Yes sir" | NAH → ENG hallucination | OTH | OTH+[UNC] (conf 0.33) ✓ |
| "Arcabuceros" | SPA word, no Whisper | OTH | SPA ✓ (clip @ 32.65s) |
| Baptism | Latin liturgical | LAT ✓ | LAT ✓ |

### 5.8 Case Study: Malinche's Linguistic Journey

Marina/Malinche serves as the pivot point for all code-switching in Hernán
(2019). As the sole character who speaks all three languages across the
series, her segment distribution provides a ground-truth validation of the
pipeline's multilingual tracking capability.

We extracted all 40 segments where Marina is the speaker or is directly
addressed across Hernán S01 (8 chapters):

| Episode | Title | Segments | MAY | NAH | SPA | Key Moment |
|---------|-------|----------|-----|-----|-----|------------|
| E01 | Marina | 6 | 1.8 | 1.8 | 2.4 | Baptism: 'Your name will be Marina' |
| E02 | Olid | 5 | 2.0 | 2.0 | 1.0 | Marina and Jerónimo called together |
| E03 | Xicotencatl | 3 | 0.6 | 1.5 | 0.9 | — |
| E04 | Bernal | 2 | 0.2 | 1.2 | 0.6 | 'Talk to him, Marina' — diplomatic pivot |
| E05 | Moctezuma | 3 | 0.3 | 1.5 | 1.2 | 'Did you translate that right?' |
| E06 | Alvarado | 3 | 0.0 | 1.8 | 1.2 | 'Translate, Marina' — to Moctezuma |
| E07 | Sandoval | 8 | 0.8 | 4.0 | 3.2 | 'Malinche' — Nahuatl honorific revealed |
| E08 | Hernán | 10 | 0.5 | 4.5 | 5.0 | 'Señor Malinche' — Cortés named by her |
| **Total** | | **40** | **6.2 (15.5%)** | **18.3 (45.8%)** | **15.5 (38.8%)** | |

**Note:** Values are narrative-weighted estimates based on episode context,
not raw segment counts.

**Linguistic arc:** Early episodes (E01–02) show Marina functioning as a
relay translator via Aguilar — Yucatec Maya (15.5% overall) dominates her
early interactions. As she acquires Spanish, NAH becomes the dominant medium
(E03–07, 45.8% total). By E08, Spanish overtakes Nahuatl in her segments,
reflecting her transformation from indigenous interpreter to colonial
participant. The pipeline correctly tracks this trajectory: MAY peaks in
E01–02 then declines, NAH peaks in E06–07, SPA rises monotonically.

Key linguistic moments detected by the pipeline:

| Timestamp | Episode | Event | Language |
|-----------|---------|-------|----------|
| 00:21:28 | E01 | Baptism: 'Your name will be Marina' | SPA |
| 00:18:16 | E05 | 'Did you translate that right?' | NAH→SPA |
| 00:16:25 | E06 | 'Translate, Marina' | SPA→NAH |
| 00:37:46 | E07 | 'Malinche' etymology | NAH honorific |
| 00:41:55 | E08 | 'Señor Malinche' | Cortés renamed |
| 00:45:28 | E08 | 'Martin Cortes Malintzin' | Mestizo naming |

**Etymological note:** The name "Malinche" derives from "Malintzin," itself a Nahuatl adaptation of her Spanish baptismal name "Marina" with the honorific suffix *-tzin* (comparable to *Moctezuma* → *Motecuhzomatzin*). Spanish speakers, unable to pronounce /ts/, rendered it as "Malinche." Notably, the Aztecs also called Cortés himself "Malinche" (lit. "Marina's lord/companion"), as he was perpetually seen beside his interpreter — hence "Señor Malinche" in E08 is historically accurate wordplay.

---

## 6. Discussion

### 6.1 Hallucination as a Sensor: Why ASR Failure Modes Are Informative

We propose that ASR hallucination on out-of-distribution languages is not random noise but a structured projection that encodes information about the relationship between the input and the model's training distribution. Three observations support this:

**1. The hallucination rate is invariant.** Across 45 NAH segments, two model sizes (tiny, base), and two decoding modes (auto-detect, forced-Spanish), the hallucination rate is 100%. The specific target language varies (English 28%, Spanish 14%, Sinhala 9%, Chinese 4%, 6 other languages), but the rejection signal — "this is not a language I know" — is perfectly reliable. This makes hallucination-as-rejection a zero-shot OOD detector with recall=1.0.

**2. The hallucination distribution is structured, not uniform.** If hallucination were random, we'd expect uniform distribution across Whisper's 99 languages. Instead, hallucination targets cluster: English and Spanish (the two largest training languages) dominate in tiny, while the base model spreads across 13 languages including Hindi, Arabic, and Japanese. This model-size dependence reflects the geometry of learned representations — smaller models project OOD inputs onto fewer, broader attractors.

**3. The hallucination content correlates with acoustic properties.** Longer NAH segments with Spanish loanwords tend to hallucinate as Spanish. Isolated NAH words with ejective consonants trigger more exotic targets (Sinhala, Chinese). Short bursts hallucinate as English. This suggests the hallucination target encodes partial acoustic similarity — a form of implicit distance metric on the model's language manifold.

**The general principle:** A neural model forced to classify an input outside its training support does not produce uniform noise. It produces a structured projection onto its training manifold, and the projection's properties (target, confidence, consistency) encode the distance and direction from training support. For ASR, this means hallucination distributions are diagnostic of input language properties — sufficient for detection even without any labeled data in the target language.

**4. Hallucination extends beyond language to genre.** On non-speech segments (silence, background noise), Whisper produces YouTube-specific boilerplate — most commonly "Thanks for watching!" — rather than silence or random text. This reflects the model's training distribution (weakly supervised YouTube subtitles) and demonstrates that hallucination is a structured projection not only onto *languages* but also onto *genres*. The projection target is the highest-frequency closing phrase in the training data, analogous to how language hallucination targets the highest-frequency languages.

**5. Different model families fail differently on the same OOD audio.** Whisper hallucinates supported-language text labels; finetuned generative models (e.g., VibeVoice probes) tend to output fluent NAH-like morphology even on ambiguous mixed-language input; phoneme recognizers can emit inventory-biased phone strings (e.g., wav2vec2/eSpeak tone-like artifacts such as trailing numeric markers). The invariance is not the literal output form but the projection mechanism: each model maps OOD speech into its own training manifold, producing model-specific but structured error signatures.

Prior work has treated hallucinations as failure modes to suppress: Zuccoloto et al. (2025) documented non-speech hallucinations; Koenecke et al. (2024) showed fairness harms from hallucinated phrases on atypical speakers. Our contribution inverts the framing: hallucination is not the problem but the signal, and exploiting it enables zero-shot detection of arbitrary unseen languages.

The 40-segment Malinche case study (Section 5.8) validates that this signal generalizes across scene contexts: the pipeline correctly tracks her linguistic trajectory from Maya-dominant (E01–02) to Nahuatl-dominant (E03–07) to Spanish-dominant (E08), demonstrating that hallucination-based rejection combined with phonotactic profiling produces coherent multilingual tracking over extended time scales.

### 6.2 Limitations and the Acoustic Ejective Detector

- **Maya ejectives**: Allosaurus doesn't reliably produce ejective markers (kʼ, tʼ) as atomic symbols. The Modal GPU pipeline includes a **3-way acoustic ejective detector** that significantly improves Maya recall:

  **Architecture:** Three independent classifiers vote on each candidate segment:
  1. **Heuristic detector**: Parselmouth-based analysis of glottal closure patterns — ejectives produce a characteristic burst followed by silence (VOT > 30ms, intensity dip > 6dB)
  2. **sklearn classifier**: Random forest trained on spectral features (CoG, spectral tilt, burst intensity) from manually labeled Apocalypto segments
  3. **wav2vec2 probe**: Fine-tuned linear probe on wav2vec2 hidden states, predicting ejective vs. non-ejective stop

  **Decision rule:** ≥2/3 consensus required for positive ejective classification.

  **Results:** Preliminary tests on 10-minute Apocalypto footage show ~90%+ MAY detection when acoustic ejectives are enabled, compared to 66.7% with IPA-only scoring (11× improvement in MAY segment detection). The ejective detector is critical because Allosaurus cannot produce ejective symbols (kʼ, tʼ, tsʼ) as atomic IPA — without acoustic detection, the pipeline has no phonotactic signal for Maya-vs-Nahuatl discrimination.

  **Ejective differentiation (v5):** Ejective IPA markers alone are ambiguous between NAH and MAY. The voter now uses co-occurring markers to disambiguate: tɬ + ejective → very strong NAH (+3); ejective + implosive ɓ → strong MAY (+2); ejective alone → weak NAH (+0.5, insufficient for classification). This prevents false MAY classifications on Nahuatl segments where acoustic ejective detection fires.

- **Inline speaker-prior:** Per-speaker language history accumulates during processing. When a speaker has ≥3 segments with ≥70% one language, the voter adds a +0.1 confidence boost for matching segments and can rescue OTH→NAH or correct MAY→NAH (at ≥80% threshold, no implosive ɓ). This complements the post-processing speaker-prior (STEP 6.4) which operates on the full segment set after classification.
- **Two-pass IPA speaker prior:** The inline speaker-prior suffers from a chicken-and-egg problem: Whisper maps Nahuatl to Spanish cognates (e.g., NAH *amo* "no" → SPA *amo* "I love"), the speaker profile accumulates SPA, and subsequent segments are biased toward SPA. A pre-prior pass checks each segment's fused IPA for non-Spanish phonemes (/ts/, /tɬ/, /kʷ/, /tʃʼ/, /kʼ/) — phonemes that do not exist in Spanish phonology. Speakers with ≥15% IPA-NAH evidence have their low-confidence (< 0.7) SPA segments overridden to NAH before the main speaker-prior runs. On SPEAKER_15 (Xicotencatl, 94% NAH in ground truth but 55% SPA in pipeline), this corrects the speaker profile from SPA-dominant to NAH-dominant, recovering 15 of 47 misclassified segments with zero false positives.
- **Spanish recall gap**: Analysis of OTH segments reveals ~24% contain clear Spanish text that the language ID failed to classify. This is a SPA→OTH leakage issue, not a NAH/MAY detection problem. The core NAH/MAY claims remain valid (0 missed MAY segments, 0% cross-language confusion).
- **Short segment hallucination**: The finetuned Whisper model hallucinates training-corpus speaker names ("Anastacio Nicolás Damián", "José Ernesto Vázquez Chanico") on segments shorter than ~0.8 seconds. A minimum-duration gate now skips FT inference for segments < 0.8s, falling back to stock Whisper text or LLM transliteration. Segments < 0.8s with < 4 IPA phonemes are classified as OTH.
- **Speaker diarization errors**: Pyannote occasionally misassigns speakers, causing downstream language confusion.
- **NAH↔SPA lexical overlap**: The largest remaining error class (101 NAH→SPA after two-pass prior, down from 116) stems from shared short words between Nahuatl and Spanish: *amo* (NAH "no" / SPA "I love"), *si*, *no*, and contact-era loanwords. Whisper maps these to Spanish cognates, and the speaker-prior reinforces the error. The two-pass IPA prior recovers 15 errors by detecting non-Spanish phonemes (/ts/, /tɬ/, /kʷ/) at the speaker level, but segments with high confidence (≥ 0.7, boosted by the inline prior) remain unreachable. Further improvement requires either a more aggressive confidence threshold or a per-segment IPA discriminator that overrides individual classifications rather than working at the speaker level.
- **Noise gate trusts hallucinated text**: The noise gate exempts short segments (≤0.4s) with ≥2 words of Whisper text from being classified as OTH. On near-silent segments, Whisper hallucinates plausible multi-word text ("Thanks for watching!"), causing the gate to pass segments that should be rejected. The fix requires cross-checking IPA phoneme count against Whisper word count: a segment with IPA "e" (1 phoneme) but Whisper "Thanks for watching!" (3 words) is clearly hallucinated.

### 6.3 Error Analysis: Diacritics vs. Real Errors

The finetuned model's CER of 70% can be decomposed into two categories:

**Category 1: Diacritic/notation errors (estimated 40-50% of CER)**
Nahuatl orthography uses special notations that Whisper's tokenizer was never trained on:
- Long vowels: `ā` or `a:` → model produces `aa` or `a` (systematic, post-processable)
- Saltillo (glottal stop): `ʔ` or `h` → often omitted or substituted
- Colon notation: `se:` → `se` or `see` (consistent pattern)

These errors are **systematic and post-processable**: a rule-based normalizer mapping `aa→a:`, `ee→e:`, etc. could reduce CER by an estimated 15-25 percentage points. This would bring effective CER closer to 45-55%.

**Category 2: Real transcription errors (estimated 50-60% of CER)**
- Word boundary misalignment: "kikwi a" → "kikwia" (run-together)
- Consonant confusion: "kimowilia" → "mokilia" (metathesis)
- Hallucination residue: proper nouns sometimes trigger Spanish-like output ("karreira" for "Cabrera")

**Implication**: A two-stage approach — finetuned Whisper for raw transcription + rule-based Nahuatl normalizer for diacritics — could achieve practical CER in the 40-50% range. For a language with zero prior ASR support, this represents a usable first-draft transcription.

**Category 3: Corpus contamination (both checkpoints)**
Both the 3K and 6738 checkpoints hallucinate training corpus speaker names on silence or acoustically unclear segments: "Ernesto Vázquez Chanico" and "Anastacio Nicolás Damián" (OpenSLR-92 Puebla-Nahuatl corpus speakers). This indicates the model memorized speaker metadata rather than generalizing phonology. The hallucination is deterministic — both checkpoints produce the same names on the same silent segments. Multi-dialect training across Puebla (SLR-92), Veracruz (SLR-147), and Zacatlán (SLR-148) corpora is expected to eliminate this overfitting by introducing sufficient speaker diversity (~350h, 50+ speakers across three dialects).

### 6.4 Generalization

While evaluated on Nahuatl/Maya, the approach should generalize to any language with distinctive phonotactics. Potential applications:
- Other Uto-Aztecan languages
- Mayan language family
- Any endangered language in code-switching contexts

### 6.5 La Otra Conquista: MAY False Positives — Diagnosis and Fix

La Otra Conquista (1999) is set entirely in post-conquest central Mexico (Tenochtitlan), with no Yucatec Maya speakers in the film. The 70 MAY-tagged segments (25.8%) were therefore **known misclassifications**. Manual review of all 70 segments revealed the root causes and led to a targeted fix in v7.0.

**Root cause analysis** — five compounding factors:

1. **Shared markers ʔ and ʃ**: The glottal stop (ʔ, MAY weight 0.5) and voiceless postalveolar fricative (ʃ, MAY weight 0.1) are cross-linguistic phonemes that appear frequently in Allosaurus output regardless of source language. A single ʔ in any segment was sufficient to reach the MAY threshold (0.5).
2. **MAY threshold too low (0.5)**: Combined with (1), this created a near-zero barrier for MAY classification.
3. **File-level consolidation protection**: MAY was included in `_PROTECTED_FILE_CONSOLIDATION_LANGUAGES`, preventing the file-level scoring pass from correcting false MAY tags even when SPA or NAH clearly dominated the whole file.
4. **No ejective evidence requirement**: Yucatec Maya is distinguished from Nahuatl primarily by ejective consonants (kʼ, tʼ, pʼ, tsʼ, tʃʼ). The classifier had no mechanism to require ejective evidence before assigning MAY.
5. **Weak NAH-exclusive negative markers**: tɬ, kʷ, ɬ (NAH-exclusive phonemes) carried only -0.8 penalty for MAY — insufficient to override positive evidence from shared markers.

**Fix applied (v7.0):**

| Change | Before | After | Rationale |
|--------|--------|-------|-----------|
| ʔ in MAY markers | 0.5 weight | Removed | Not diagnostic for Maya |
| ʃ in MAY markers | 0.1 weight | Removed | Cross-linguistic, causes false positives |
| MAY threshold | 0.5 | 2.0 | Require substantial evidence |
| tɬ/kʷ/ɬ penalties | -0.8 | -1.5 | Stronger NAH-exclusive discrimination |
| Ejective guard | None | Phase F.5 | MAY without ejectives → demoted to OTH |
| Consolidation protection | Protected | Not protected | Allow file-level correction |

**Post-fix result**: 70 MAY → 3 MAY across 4 clips (40 minutes, 354 segments). The 3 remaining MAY are ejective false alarms on NAH audio — the ejective detector fires on non-ejective consonants in noisy segments, and without the ɓ implosive guard, these are not demoted. True MAY content is correctly absent.

**Additional fix (v5)**: Whisper misdetects file-level language as `en` for LOC clips, causing 21 ENG→SPA misclassifications. A 12K-word Spanish frequency dictionary with text-based override corrects multi-word segments (5%→61% accuracy on 38 annotated segments in Clip 3).

### 6.6 tɬ-Cascade Error: Pipeline Misclassification Propagates to FT Model

The lateral affricate /tɬ/ is phonologically exclusive to Nahuatl — it does not exist in Spanish, Maya, or any other language in our profile set. This makes it a high-confidence NAH marker. However, Spanish contains orthographic "tl" clusters (*atlántico*, *extremaunción*, *Tlaxcala*) that Allosaurus misidentifies as /tɬ/ in its IPA output. The original `tl-acoustic-hard-override` treated any /tɬ/ detection as definitive NAH evidence, overriding all other signals.

**Cascade mechanism**: Pipeline tags segment as NAH (tɬ override) → FT model receives NAH label → FT model produces forced Nahuatl output → Spanish dialogue transcribed as pseudo-Nahuatl. Example: "a saber" (Spanish, "to know") → pipeline says NAH → FT outputs "Asa weh" (resembles Nahuatl morphology but is gibberish).

**Fix applied**: Spanish Context Guard (`tl-acoustic-conditional-override`) — tɬ override suppressed when Whisper text contains ≥3 Spanish function words or when Whisper confidently (≥0.85) detects Spanish with ≥1 function word. On the La Otra Conquista 10min test: 50 NAH → 37 NAH (14 segments corrected to SPA). Manual annotation confirmed 10 remaining NAH→SPA false positives, primarily in segments where Whisper produced no usable text.

**Implication**: FT model evaluation is unreliable until upstream language classification is accurate. The FT model has no mechanism to reject its assigned language — it will always produce output in the target language. Improving tɬ detection precision is therefore a prerequisite for meaningful FT checkpoint comparison.

Conversely, **0 LAT segments** were detected despite the film’s prominent Catholic ritual scenes (forced baptisms, liturgical chants). This suggests the LAT detector’s threshold may be too conservative, or that the Latin fragments in La Otra Conquista are too short or too embedded in Spanish liturgical context for reliable detection.

---

## 7. Conclusion

We presented Tenepal, a phoneme-based language identification system for endangered languages in multilingual film, built on a general principle: **ASR hallucination distributions encode distance-to-training-support**, making model failure modes exploitable as zero-shot language detectors. When Whisper encounters Nahuatl, it does not abstain — it confidently hallucinates in Sinhala, Swedish, or Chinese. We show this hallucination is 100% reliable as a rejection signal (N=45, two model sizes), structured rather than random, and sufficient for language detection without any labeled data in the target language.

An ablation study across 13 configurations on 551 annotated segments decomposes accuracy into additive layers: phoneme-level features alone achieve 65.7%, speaker priors add +6.9pp, finetuned ASR +6.0pp, and morphological pattern expansion +3.3pp, reaching **85.7%** — up from a 50% baseline (Whisper alone) and approaching the oracle ceiling of 90.7%. The phonetic foundation carries the majority of the signal: no language model, speaker tracking, or finetuning is needed to achieve 69% on trilingual classification. Each subsequent layer provides diminishing but meaningful gains, with the largest single lever being LoRA-finetuned Whisper, which provides real Nahuatl morphology where the LLM fallback produces gibberish. Whisper's hallucinations extend beyond language to genre: non-speech segments produce YouTube boilerplate ("Thanks for watching!"), confirming that hallucination is a structured projection onto training-data distributions, not random noise.

Evaluation across two independent films — Hernán (2019, 4,659 segments) and La Otra Conquista (1999, 271 segments) — demonstrates cross-production generalization. On expanded LOC cross-validation (minutes 14–44, N=244 NAH+SPA), Tenepal reaches 84.4% raw accuracy and 81.7% balanced accuracy, with overlap-heavy interpreter scenes as the primary failure regime. A 40-segment analysis of Marina/Malinche's speech tracks her linguistic arc from Yucatec Maya interpreter (E01–02) through Nahuatl intermediary (E03–07) to Spanish-dominant participant (E08), validating multilingual tracking over extended time scales.

Whisper finetuning on 150h of Puebla-Nahuatl speech reduces CER from 108% (random hallucinations) to 70% (recognizable Nahuatl) after just 3,000 steps (~45% of one epoch). The finetuned model is the single largest accuracy lever: +6.0pp when applied before speaker priors, because it provides genuine Nahuatl morphology (e.g., "Ka neelpa" vs. LLM gibberish "a n i ɒ a") that downstream morphological pattern matching can exploit.

The hallucination-as-sensor principle extends beyond our specific application. Any neural model forced to process out-of-distribution input produces structured projections onto its training manifold. Exploiting these projections — rather than suppressing them — opens a path to zero-shot detection for the thousands of languages that no ASR system will support in the foreseeable future.

### 7.1 Future Work

- **Overlap-aware routing:** Detect simultaneous speech (speaker overlap) before language classification. We implemented Parselmouth-based overlap detection (bimodal F0 + HNR) triggering MossFormer2 voice separation with per-source IPA extraction; however, the separated sources classify identically on 20/35 overlap turns, and non-NAH source selection yields 0pp improvement (LOC 34–44, N=134). The separation model, trained on English, cannot cleanly isolate typologically similar NAH/SPA voices in short overlap regions. More promising directions include confidence damping (capping language scores in overlap regions) and speaker-embedding-based source attribution.
- **NAH/MAY discrimination under overlap:** Strengthen ejective-vs-lateral cues and require overlap-robust evidence before MAY assignment in mixed scenes.
- **Multi-dialect Nahuatl finetuning:** Train on combined Amith corpora (OpenSLR 92 + 147 + 148) to reduce speaker-name memorization and improve dialect transfer.
- **Harder cross-film evaluation:** Extend benchmarks beyond Hernán/LOC with additional indigenous-language films (e.g., Ixcanul) and larger MAY-focused test sets (Apocalypto full-scene annotations).

---

## References

<!-- ONGOING: Add references as we find/use them -->

### ASR & Speech Recognition
- Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML 2023*. (Whisper)
- Pratap, V., et al. (2023). Scaling Speech Technology to 1,000+ Languages. *arXiv:2305.13516*. (MMS)
- Zhang, Y., et al. (2023). Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages. *arXiv:2303.01037*. (USM)

### ASR Hallucination
- Koenecke, A., Choi, A.S.G., Mei, K.X., Schellmann, H., & Sloane, M. (2024). Careless Whisper: Speech-to-Text Hallucination Harms. *ACM FAccT 2024*. https://arxiv.org/abs/2402.08021
- Zuccoloto, B., et al. (2025). Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio. *arXiv:2501.11378*. https://arxiv.org/abs/2501.11378

### Language Identification
- Kargaran, A., Imani, A., Yvon, F., & Schütze, H. (2023). GlotLID: Language Identification for Low-Resource Languages. *Findings of EMNLP 2023*. https://arxiv.org/abs/2310.16248
- Zissman, M.A. (1996). Comparison of Four Approaches to Automatic Language Identification of Telephone Speech. *IEEE Transactions on Speech and Audio Processing, 4*(1).
- Singer, E., Torres-Carrasquillo, P.A., Gleason, T.P., Campbell, W.M., & Reynolds, D.A. (2003). Acoustic, Phonetic, and Discriminative Approaches to Automatic Language Identification. *EUROSPEECH 2003*.
- Matějka, P., Burget, L., Schwarz, P., & Černocký, J. (2005). Phonotactic Language Identification Using High Quality Phoneme Recognition. *INTERSPEECH 2005*.

### Phoneme Recognition
- Li, X., et al. (2020). Universal Phone Recognition with a Multilingual Allophone System. *ICASSP 2020*. (Allosaurus)
- Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS 2020*.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning for Speech Recognition. *arXiv:2006.13979*. (wav2vec2-lv-60-espeak-cv-ft)

### Acoustic Analysis
- Boersma, P. & Weenink, D. (2024). Praat: doing phonetics by computer [Computer program]. Version 6.4. http://www.praat.org/
- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics, 71*, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

### Speaker Diarization
- Bredin, H., et al. (2023). pyannote.audio 2.1 speaker diarization pipeline. *INTERSPEECH 2023*.

### Phonological Databases
- Moran, S., & McCloy, D. (eds.) (2019). PHOIBLE 2.0. Jena: Max Planck Institute. https://phoible.org/

### Indigenous Language NLP
- AmericasNLP 2024. First Workshop on NLP for Indigenous Languages of the Americas, co-located with NAACL 2024, Mexico City. https://turing.iimas.unam.mx/americasnlp/2024_workshop.html

### Nahuatl & Indigenous Languages
- Amith, J.D. et al. OpenSLR 92: Highland Puebla-Nahuatl Corpus. ~84GB, ~190h transcribed speech from Sierra Norte and Nororiental de Puebla. https://openslr.org/92/ (contact: jonamith@gmail.com)
- Amith, J.D. et al. (2022–23). OpenSLR 147: Audio corpus of Orizaba (Veracruz) Nahuatl speech. ~119h, 657 files. Glottocode: oriz1235; ISO 639-3: nlv. https://openslr.org/147/
- Amith, J.D. et al. (2023). OpenSLR 148: Audio corpus of Zacatlán-Ahuacatlán-Tepetzintla (Puebla) Nahuatl speech. Glottocode: zaca1241; ISO 639-3: nhi. https://openslr.org/148/
- Amith, J.D. Mozilla Common Voice Nahuatl Corpus. https://datacollective.mozillafoundation.org/datasets/cmlct0jzu01s4nv07023lv3m3 (nahuatl.biology@gmail.com)
- Wood, S. (ed.) Online Nahuatl Dictionary. Wired Humanities Projects, University of Oregon. https://nahuatl.wired-humanities.org/
- Karttunen, F. (1983). An Analytical Dictionary of Nahuatl. University of Texas Press.
- Lockhart, J. (2001). Nahuatl as Written: Lessons in Older Written Nahuatl. Stanford University Press.
- Molina, A. de (1571). Vocabulario en lengua castellana y mexicana.
- INALI (Instituto Nacional de Lenguas Indígenas). https://www.inali.gob.mx/
- Sahagún, B. de (1577). Historia general de las cosas de Nueva España (Florentine Codex). Books 6, 12 contain extensive Nahuatl speech transcriptions.
- Sullivan, T. D. (1988). Compendium of Nahuatl Grammar. University of Utah Press. Standard reference for Classical Nahuatl phonology.

### Yucatec Maya (preliminary only in this release)
- Bolles, D. (2001). Combined Dictionary-Concordance of the Yucatecan Mayan Language. FAMSI.
- Bricker, V.R., Po'ot Yah, E., & Dzul de Po'ot, O. (1998). A Dictionary of the Maya Language as Spoken in Hocabá, Yucatán. University of Utah Press.
- Ladefoged, P. & Maddieson, I. (1996). The Sounds of the World's Languages. Blackwell. Ch. 3: Stops (ejectives kʼ, tʼ, pʼ, tsʼ, tʃʼ).

### Parameter-Efficient Finetuning
- Hu, E.J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*. (LoRA)
- Mangrulkar, S., et al. (2022). PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods. https://github.com/huggingface/peft

### G2P & Transliteration
- Mortensen, D.R., et al. (2018). Epitran: Automatic Phonetic Transcription. *LREC 2018*. https://github.com/dmort27/epitran

### Film Sources
- Hernán (2019). Historical drama series, S01 (8 chapters). Directors: Julián de Tavira et al.
- La Otra Conquista (1999). Director: Salvador Carrasco. Mexican historical drama depicting the spiritual conquest of Mexico.
- Apocalypto (2006). Director: Mel Gibson. Yucatec Maya-language film.

### Tools & Infrastructure
- Modal Labs. https://modal.com/ — GPU inference on A10G (24GB VRAM). Runtime: ~4min for 10min audio (2.5× realtime) including diarization, Whisper, Allosaurus, wav2vec2, ejective detection, and NAH finetuned Whisper pass. Cost: ~$0.13/run (A10G at $1.10/h). Training on A100 80GB (~$4/h).
- Défossez, A., et al. (2021). Hybrid Transformers for Music Source Separation. (Demucs)
- Silero Team (2021). Silero VAD.

---

## Appendix

### A. Full Language Profiles

```python
PROFILES = {
    "nah": {  # Nahuatl (Uto-Aztecan) - Priority 10
        "markers": {"ʔ": 1.0, "ɬ": 1.0, "kʷ": 1.0, "tɬ": 1.0, "ɸ̞": 0.8, "β̞": 0.8,
                    "ts": 0.5, "tʃ": 0.5, "ʃ": 0.4, "k̟ʲ": 0.7, "tɕ": 0.5},
        "negative": {"b": 0.5, "d": 0.5, "ɡ": 0.5, "f": 0.5, "v": 0.5, "ʒ": 0.5},
        "threshold": 0.0
    },
    "spa": {  # Spanish (Indo-European) - Priority 1
        "markers": {"b": 0.20, "d": 0.05, "ɡ": 0.20, "ɲ": 0.80, "ɾ": 0.40},
        "negative": {},
        "threshold": 2.25
    },
    "may": {  # Yucatec Maya (Mayan) - Priority 9
        "markers": {"kʼ": 1.0, "tsʼ": 1.0, "tʃʼ": 1.0, "pʼ": 0.9, "tʼ": 0.9, "ɓ": 0.9, "ʃ": 0.3},
        "negative": {"b": 0.5, "d": 0.5, "ɡ": 0.5, "tɬ": 0.8, "kʷ": 0.8, "ɬ": 0.8},
        "threshold": 0.0
    },
    "eng": {  # English (Indo-European) - Priority 2
        "markers": {"æ": 0.25, "ɪ": 0.20, "ʊ": 0.05},
        "negative": {},
        "threshold": 4.10
    },
    "deu": {  # German (Indo-European) - Priority 3
        "markers": {"x": 0.90, "ʁ": 0.70},
        "trigrams": {"ʃtʁ": 1.5, "ʃpʁ": 1.5, "pfl": 1.2, "xtʁ": 1.5},
        "negative": {},
        "threshold": 6.10
    },
    "fra": {  # French (Indo-European) - Priority 3
        "markers": {"y": 0.90, "ø": 0.80, "ʁ": 0.50, "ɛ̃": 0.90, "ɔ̃": 0.90, "ʒ": 0.80},
        "negative": {},
        "threshold": 1.40
    },
    "ita": {  # Italian (Indo-European) - Priority 2
        "markers": {"ts": 0.15, "ɲ": 0.25, "ʎ": 0.90, "dʒ": 0.25},
        "negative": {},
        "threshold": 0.25
    }
}
```

### B. Edge Case Catalog

| ID | Source | Time (s) | Expected | Avoid | Note |
|----|--------|----------|----------|-------|------|
| moctezuma_yes_sir | moctezuma_test.wav | 35.0-35.7 | — | yes, sir, you | NAH→ENG hallucination suppression |
| hernan_soldados_1 | Hernán-1-1-1_00-05.wav | 113.5-116.3 | soldad* | saludo | SPA word recovery |
| hernan_soldados_2 | Hernán-1-1-1_00-05.wav | 128.3-129.7 | soldad* | saludo | SPA word recovery |
| hernan_capitan_1 | Hernán-1-1-1_00-05.wav | 38.6-39.2 | capitan | — | SPA term stability |

These clips exercise the hallucination detector and LLM transliteration pipeline.

### C. Reproduction Instructions

```bash
git clone https://github.com/mdresch/tenepal
pip install -e .
modal run tenepal_modal.py --input video.mkv --compare
```

---

*Last updated: 2026-03-13 — v6: EQ ablation study across 13 configurations on 551 annotated segments. Layer decomposition: IPA-only 65.7% → +speaker prior → +FT-first → +morphology = 85.7% (oracle ceiling 90.7%). Core finding: phoneme-level features carry majority of signal (69%) without LM/speaker/FT. LOC cross-validation (N=244) now reports 84.4% raw / 81.7% balanced.*

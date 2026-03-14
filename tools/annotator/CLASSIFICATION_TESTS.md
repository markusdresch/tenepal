# Classification Hypothesis Tests (annotations.db)

DB: `tools/annotator/annotations.db`  
Analyzed rows: 66 total, 59 with `correct_lang`.

## 1) Speaker-based prior / continuity

Question: if we apply a speaker continuity rule (`same speaker -> same language`), how many `OTH -> NAH` errors could be fixed?

Baseline from annotations:

- `OTH -> NAH`: 27 rows (largest error class)

Speaker evidence:

- `SPEAKER_12`: 10/10 corrected rows are `NAH`.
- Also NAH-stable in labeled rows: `SPEAKER_09` (10/10), `SPEAKER_07` (9/9), `SPEAKER_13` (8/8), `SPEAKER_14` (6/6).
- SPA-stable pocket: `SPEAKER_05` (6/6 -> `SPA`).

Test results:

- **Upper-bound (oracle per-speaker majority):** 27/27 `OTH->NAH` would be corrected.
- **Leave-one-out speaker majority (less optimistic):** 24/26 eligible `OTH->NAH` corrected (1 row had no prior evidence).
- **Sequential continuity (use last known language for same speaker):** 23/23 applicable `OTH->NAH` corrected.

Conclusion:

- Speaker continuity is high-impact and should be enabled.
- `SPEAKER_12` should get immediate NAH-prior treatment.

## 2) Lexicon expansion from `human_transcription`

Extracted `human_transcription` examples include:

- NAH-side: `mishcova`, `ompa`, (artifact: `diff-case`)
- Other seen strings: `Kemma! Kemma! comer!`, `amo`, `de trascala (de tlaxcala)`, etc.

Token association from corrected rows:

- NAH-only tokens (in corrected sample): `mishcova`, `ompa` (+ artifact `diff`, `case`)
- SPA-side tokens: `dicelo`, `capitan`, `tlaxcala`, `trascala`, etc.

Coverage test (limited by sparse transcriptions):

- `OTH->NAH` rows with non-empty `human_transcription`: 3
- Of those, rows containing NAH candidate tokens: 3/3

Conclusion:

- Lexicon expansion likely helps, but current transcription coverage is small.
- Add `mishcova`, `ompa` as soft NAH lexicon hints.
- Do not treat `diff-case` as lexical evidence.

## 3) Threshold tuning (`NAH threshold -0.5`)

Question: how many OTH would flip to NAH if NAH threshold is lowered by 0.5?

Data check:

- `pipeline_confidence` non-null rows: **0/66**.
- Therefore exact threshold simulation is **not possible** from this DB.

What we can still bound from labels:

- Labeled `OTH` rows: 35 total
- If all labeled `OTH` flipped to NAH:
  - Correct flips: 27 (`OTH->NAH`)
  - Wrong flips introduced: 8 (`OTH->SPA`)
  - Net in labeled subset: +19 correct

Conclusion:

- Add confidence logging first; threshold tuning should be deferred until scores are available.
- Without confidence, only coarse what-if bounds are possible.

## 4) Ejective disambiguation for `MAY -> NAH`

Hypothesis: `ejectives + no tɬ + NAH lexicon match => NAH (not MAY)`.

Observed `MAY->NAH` rows: 5

Feature check on these 5 rows:

- Ejective-like markers present in IPA: 5/5
- `tɬ` present: 1/5 (absent in 4/5)
- NAH lexicon match from `human_transcription`: 0/5 (no transcriptions available on those rows)

Conclusion:

- Ejectives alone are not sufficient (they appear in all MAY->NAH errors).
- The current rule is directionally useful if reframed as:
  - `MAY + ejectives + NAH speaker prior + (no strong MAY lexicon)` -> bias to NAH.
- `no tɬ` can be a weak feature, not a hard gate.

## Recommended implementation order

1. Enable speaker-prior override (start with `SPEAKER_12`, then all stable speakers).
2. Add confidence output to pipeline (required for real threshold tuning).
3. Add NAH lexicon hints (`mishcova`, `ompa`) as soft decoder bias.
4. Add MAY-vs-NAH rescoring feature bundle (ejective present, tɬ absent/present, speaker prior, lexicon evidence).

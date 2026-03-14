# Amith Corpora Access

This project uses Jonathan D. Amith's Nahuatl corpora as training and lexical resources.

## Main Sources

- **OpenSLR 92**: Highland Puebla Nahuatl
  - https://openslr.org/92/
  - Main corpus used for Whisper finetuning

- **Mozilla Data Collective / Zacatlan-Tepetzintla Nahuatl**
  - Dataset page referenced by the current download script:
  - https://datacollective.mozillafoundation.org/datasets/cmlcqxjwl01t8mm07wz7c08bz
  - Used here for corpus evaluation scaffolding and lexicon extraction

- **OpenSLR 147**: Orizaba Nahuatl
  - https://openslr.org/147/

- **OpenSLR 148**: Zacatlán-Ahuacatlán-Tepetzintla Nahuatl
  - https://openslr.org/148/

## Mozilla Data Collective Workflow

The repository includes a helper script:

```bash
python scripts/download_amith_corpus.py --dry-run
python scripts/download_amith_corpus.py --check
python scripts/download_amith_corpus.py
```

What it expects:

1. Open the Mozilla Data Collective dataset page.
2. Accept the dataset license terms.
3. Obtain the required access token.
4. Export it locally:

```bash
export MOZILLA_DC_TOKEN=<your-token>
```

5. Run the downloader:

```bash
python scripts/download_amith_corpus.py
```

The corpus audio is downloaded into:

```bash
corpus_audio/amith_nah/
```

That directory is intentionally local-only and should not be committed to the public repository.

## Notes for the Public Draft

- The public GitHub draft should explain where the corpora come from, but should not bundle restricted audio.
- The current `tools/corpus/results/baseline.json` is only a placeholder until audio is downloaded locally.
- If the dataset URL or access procedure changes, update `scripts/download_amith_corpus.py` and this document together.

# Public Release Workflow

This repository is the private working tree and should remain the source of truth for active research, annotation, and experiments.

Public GitHub drafts should be created as a **fresh export with new history**, not by pushing this repository directly.

## Recommended Local Setup

- Keep working in this private repository: `tenepal`
- Treat `tenepal` as the canonical project name everywhere going forward
- Keep the old `slangophone` symlink only as a local compatibility shim while scripts and habits catch up
- Do not mention `slangophone` in public docs, package metadata, or the exported public repo except where legacy notes are historically necessary

## Why a Fresh Export

This repo contains or has contained:

- copyrighted film clips and derivatives
- local logs and scratch outputs
- private notes and planning files
- experimental artifacts not suitable for public history

A fresh export avoids leaking that history.

## Public Draft Strategy

1. Work normally in the private repo.
2. Run `scripts/export_public_draft.sh`.
3. Review the generated sibling directory, by default `../tenepal-public`.
4. Initialize or update Git there and push that repo to GitHub.
5. Repeat whenever you want to refresh the public draft.

## What the Export Includes

- core source code in `src/`
- tests in `tests/`
- selected tools and scripts
- public docs such as `README.md`, `PAPER.md`, `DATA_ACCESS.md`, `EVOLUTION_PUBLIC.md`
- package metadata and license
- selected JSON/metrics artifacts that support claims without redistributing media
- public-safe annotation exports in `benchmarks/annotations/` generated from the local SQLite DB

## Public Narrative

- `Hernán` is the original trigger for the project and remains the main ablation benchmark.
- `La Otra Conquista` is the preferred public-facing Nahuatl reference point for external readers and future GitHub presentation.
- Malinche remains the symbolic avatar of the project as a linguistic mediator from the conquest era, even though she is not a character in `La Otra Conquista`.

## What the Export Excludes

- `.git/` history from this repo
- planning and agent files
- private notes such as `PERSONA.md`
- film clips, extracted audio, and most raw experiment outputs
- logs with local paths
- large validation/media directories
- private or mixed-language research notebooks such as `EVOLUTION.md`
- the live annotator SQLite database (`tools/annotator/annotations.db`)

## Future Cleanup Direction

If you want public SRT-based reproducibility, move the surviving public benchmark artifacts into a dedicated structure such as:

- `benchmarks/hernan/`
- `benchmarks/la-otra-conquista/`
- `benchmarks/reports/`

with a manifest and provenance note for each file. Keep only the minimum set needed to defend claims.

For annotations, prefer line-based JSON exports over shipping SQLite directly. The DB is convenient locally, but JSONL is easier to diff, review, and redact.

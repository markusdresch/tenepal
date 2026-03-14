# Open Problems

This document tracks the highest-value unresolved technical problems for the public Tenepal draft.

## Highest Priority

### Diarization and speaker discrimination

This is the main unresolved systems problem.

Current state:

- overlap-heavy scenes still degrade language identification
- speaker turns are not always cleanly separated
- similar male voices remain especially hard
- wrong speaker grouping can contaminate language priors and segment interpretation

What needs improvement:

- more reliable speaker diarization on multilingual film audio
- better discrimination between similar speakers within the same scene
- overlap-aware scoring that dampens misleading evidence earlier in the pipeline
- cleaner propagation of speaker confidence into annotation and evaluation views

Why it matters:

- it is a major source of false `SPA -> NAH` and mixed-turn confusion
- it limits trustworthy public evaluation on dialogue-heavy scenes
- it is more important right now than adding new benchmark languages

Current public evidence:

- overlap-heavy `La Otra Conquista` clips remain the clearest failure case in the current release draft
- the summarized evidence trail is in [EVOLUTION_PUBLIC.md](../EVOLUTION_PUBLIC.md) and the longer analysis is in [PAPER.md](../PAPER.md)

## Secondary Problems

- more annotated `La Otra Conquista` coverage
- stronger public packaging for reproducible subtitle and metric artifacts
- tighter validation of older baseline claims that are currently weaker than the main audited numbers
- better release-ready framing for Maya work, which is still exploratory

# Data Access and Redistribution

This repository is intended for a public code release. It does **not** include redistribution rights for copyrighted film audio, video, or subtitle material.

## What Is Not Redistributed

- No `Hernán` film clips
- No `La Otra Conquista` film clips
- No `Apocalypto` film clips
- No commercial subtitle files derived from those films
- No full extracted audio from commercial releases

If these files still exist locally in your checkout, they should be removed from the public release branch and from Git history before publishing.

## Research Clips Referenced in the Paper

These are the main film regions referenced in the current write-up.

- `Hernán-1-3`
  - 551 annotated NAH+SPA segments used for the main ablation benchmark
  - Original project trigger and main internal benchmark
  - Not redistributed here
  - Streaming/catalog availability may vary by region

- `La Otra Conquista`
  - Preferred public-facing Nahuatl-facing reference
  - Annotated subset covering minutes `14:15-44:25`
  - Paper tables also reference the contiguous `14:15-55:25` run for pipeline output distribution
  - No clips are redistributed here
  - The film is easier to locate lawfully than `Hernán`, including public web availability in some jurisdictions
  - It is also the culturally better public-facing reference point for Nahuatl audiences in the current release framing

- `Apocalypto`
  - Used only for preliminary Maya/ejective experiments
  - Not treated as a release-ready benchmark
  - Not redistributed here

## Public Release Position

The public release should be framed as:

- Nahuatl-first
- Reproducible from lawfully obtained source media
- Conservative about Maya claims
- Explicit that benchmark clips are references, not bundled assets
- Explicit that `Hernán` was the original trigger and main benchmark, while `La Otra Conquista` is the preferred public-facing cultural reference
- Explicit that Malinche is the symbolic project avatar, not a claim about who appears in `La Otra Conquista`

## Source Notes

- Malintzin / `Tenepal` naming discussion: https://en.wikipedia.org/wiki/La_Malinche
- OpenSLR 92: https://openslr.org/92/
- OpenSLR 147: https://openslr.org/147/
- OpenSLR 148: https://openslr.org/148/
- Mozilla Data Collective Nahuatl access page: https://datacollective.mozillafoundation.org/datasets/cmlcqxjwl01t8mm07wz7c08bz

Availability note: statements here about lawful public discoverability are intentionally practical and conservative. They should be read as reproduction guidance, not as a guarantee of identical regional access everywhere.

## Recommended GitHub Release Policy

- Keep private notes such as `PERSONA.md` outside the public tree, preferably in a hidden directory such as `.private/`
- Remove versioned film clips, extracted audio, subtitle dumps, and experiment logs containing copyrighted content
- Keep only manifests, annotation schemas, metrics, and code
- Replace direct media paths with clip identifiers and timestamps
- Document how another researcher can recreate the clips from their own lawful copy of the source material

## Recommended arXiv Paper Policy

- State explicitly that copyrighted film media is not redistributed
- State which exact scenes/minute ranges were evaluated
- Separate internal benchmarks (`Hernán`) from easier-to-verify public-facing checks (`La Otra Conquista`)
- Treat Maya results as preliminary unless they have a clean, independently reproducible evaluation set

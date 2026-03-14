# Tenepal Annotator

Web-based annotation tool for reviewing and correcting language identification results against video/audio with synchronized subtitles.

## Setup

```bash
pip install flask flask-sqlalchemy
```

## Usage

```bash
# Launch with media and SRT file
python tools/annotator/app.py --media validation_video/Hernán-1-3.mp4 --srt validation_video/Hernán-1-3.srt

# Export annotations
python tools/annotator/app.py --export annotations.jsonl

# Export public-safe benchmark annotations from the local SQLite DB
python scripts/export_public_annotations.py --outdir benchmarks/annotations
```

The annotator opens a web interface (default: `http://localhost:5000`) where you can:

- Play video/audio synchronized with subtitle segments
- Review language labels assigned by the pipeline
- Correct misidentified segments (ground-truth annotation)
- Compare multiple pipeline configurations side-by-side
- Export annotations as JSONL for evaluation

## Database

Annotations are stored in a local SQLite database (`annotations.db`). The database is gitignored — each annotator instance maintains its own local state.

For public release, do not publish `annotations.db` directly. Use `scripts/export_public_annotations.py` to generate redacted JSONL artifacts instead.

## Comparison Matrix

The annotator includes a comparison matrix view for evaluating different EQ configurations against the same media. See `CLASSIFICATION_TESTS.md` for details on the test framework.

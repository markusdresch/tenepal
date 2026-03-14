#!/usr/bin/env bash
set -euo pipefail

INPUT="validation_video/Hernán-1-1-1.wav"
VOCALS="validation_video/Hernán-1-1-1.vocals.wav"
OUT_DIR="validation_video"
REPORT_TSV="$OUT_DIR/backend_eval.tsv"
BASELINE_SRT="$OUT_DIR/Hernán-1-1-1.srt"

RUN_MODE="run"
if [[ "${1:-}" == "--dry-run" ]]; then
  RUN_MODE="dry"
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Missing input: $INPUT" >&2
  exit 1
fi
if [[ ! -f "$VOCALS" ]]; then
  echo "Missing vocals: $VOCALS" >&2
  exit 1
fi

backends=("allosaurus" "wav2vec2")

echo -e "backend\tsrt\tsource\tmay\tlat\tsegments\tprocessing_s\tlanguages" > "$REPORT_TSV"

for backend in "${backends[@]}"; do
  out_srt="$OUT_DIR/Hernán-1-1-1.${backend}.srt"
  out_log="$OUT_DIR/Hernán-1-1-1.${backend}.log"

  cmd=(
    modal run tenepal_modal.py
    --input "$INPUT"
    --vocals "$VOCALS"
    --no-demucs
    --phoneme-backend "$backend"
    --output "$out_srt"
  )

  echo "=== Backend: $backend ==="
  if [[ "$RUN_MODE" == "dry" ]]; then
    printf 'DRY RUN: %q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}" | tee "$out_log"
  fi

  parse_srt="$out_srt"
  source_tag="generated"
  if [[ ! -f "$parse_srt" && "$backend" == "allosaurus" && -f "$BASELINE_SRT" ]]; then
    parse_srt="$BASELINE_SRT"
    source_tag="baseline"
  fi

  if [[ -f "$parse_srt" ]]; then
    may_count=$(python - <<'PY' "$parse_srt"
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(encoding='utf-8', errors='ignore')
print(text.count('[MAY|'))
PY
)
    lat_count=$(python - <<'PY' "$parse_srt"
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(encoding='utf-8', errors='ignore')
print(text.count('[LAT|'))
PY
)
    segments=$(python - <<'PY' "$parse_srt"
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text(encoding='utf-8', errors='ignore')
print(len(re.findall(r'^\d+$', text, flags=re.M)))
PY
)
  else
    may_count=0
    lat_count=0
    segments=0
    source_tag="missing"
  fi

  processing_s="n/a"
  languages="n/a"
  if [[ -f "$out_log" ]]; then
    processing_s=$(awk '/Processing:/{print $2; exit}' "$out_log")
    languages=$(awk -F'Languages:  ' '/Languages:/{print $2; exit}' "$out_log" | tr '\t' ' ')
  fi

  echo -e "${backend}\t${out_srt}\t${source_tag}\t${may_count}\t${lat_count}\t${segments}\t${processing_s}\t${languages}" >> "$REPORT_TSV"
done

echo "Wrote report: $REPORT_TSV"
cat "$REPORT_TSV"

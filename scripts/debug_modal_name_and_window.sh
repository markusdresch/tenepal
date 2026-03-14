#!/usr/bin/env bash
set -euo pipefail

if [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
fi

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI not found in PATH" >&2
  exit 127
fi

ROOT="validation_video/edge_debug/modal_name_window"
CLIPS="$ROOT/clips"
RUNS="$ROOT/runs"
REPORT="$ROOT/report.tsv"
mkdir -p "$CLIPS" "$RUNS"
VOCALS_FULL="validation_video/Hernán-1-1-1.vocals.wav"

CTX_JSON=".planning/context/hernan-film-context.json"
prompt_extra=$(CTX_JSON="$CTX_JSON" python - <<'PY'
import json, os
from pathlib import Path
obj = json.loads(Path(os.environ['CTX_JSON']).read_text(encoding='utf-8'))
parts = []
parts.extend(obj.get('key_names', []))
parts.extend(obj.get('expected_langs', []))
for k,v in (obj.get('speaker_hints', {}) or {}).items():
    parts.append(str(k)); parts.append(str(v))
print(' '.join(parts))
PY
)

extract_text() {
  local srt="$1"
  if [[ ! -f "$srt" ]]; then
    echo "MISSING"
    return
  fi
  local txt
  txt=$(grep -E '^\[[A-Z]' "$srt" 2>/dev/null | sed -E 's/^\[[^]]+\]\s*//' | tr '\n' ' ' | tr -s ' ' || true)
  if [[ -z "${txt// }" ]]; then
    echo "ZERO_SEG"
  else
    echo "$txt"
  fi
}

# 1) seg137 window sweep around 718.365s
base_center=718.365
printf "case\tmode\tstart\tdur\ttext\n" > "$REPORT"
for offset in -2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0; do
  start=$(awk -v c="$base_center" -v o="$offset" 'BEGIN{v=c+o; if(v<0)v=0; printf "%.3f", v}')
  dur=5.0
  id="seg137_o${offset//./p}"
  clip="$CLIPS/${id}.wav"
  clip_v="$CLIPS/${id}.vocals.wav"
  out_base="$RUNS/${id}.base.srt"
  out_safe="$RUNS/${id}.safe.srt"

  ffmpeg -v error -y -ss "$start" -t "$dur" -i "validation_video/Hernán-1-1-1.wav" -ac 1 -ar 16000 "$clip" || true
  if [[ -f "$VOCALS_FULL" ]]; then
    ffmpeg -v error -y -ss "$start" -t "$dur" -i "$VOCALS_FULL" -ac 1 -ar 16000 "$clip_v" || true
  fi

  modal run tenepal_modal.py \
    --input "$clip" \
    --output "$out_base" \
    --vocals "$clip_v" \
    --no-demucs \
    --show-confidence \
    --phone-vote \
    --show-phone-conf \
    --show-nbest \
    --min-turn-s 0.05 >/tmp/modal_seg137_${id}_base.log 2>&1 || true

  modal run tenepal_modal.py \
    --input "$clip" \
    --output "$out_safe" \
    --vocals "$clip_v" \
    --no-demucs \
    --show-confidence \
    --phone-vote \
    --show-phone-conf \
    --show-nbest \
    --uncertain-text-policy suppress \
    --min-turn-s 0.05 >/tmp/modal_seg137_${id}_safe.log 2>&1 || true

  txt_base=$(extract_text "$out_base")
  txt_safe=$(extract_text "$out_safe")

  printf "seg137\tbase\t%s\t%s\t%s\n" "$start" "$dur" "$txt_base" >> "$REPORT"
  printf "seg137\tsafe\t%s\t%s\t%s\n" "$start" "$dur" "$txt_safe" >> "$REPORT"
  echo "[seg137] start=$start done"
done

# 2) name span sweep with broader windows + context
for cfg in "473.8 2.6" "472.8 4.0" "472.0 5.5"; do
  s=$(echo "$cfg" | awk '{print $1}')
  d=$(echo "$cfg" | awk '{print $2}')
  id="name_s${s//./p}_d${d//./p}"
  clip="$CLIPS/${id}.wav"
  clip_v="$CLIPS/${id}.vocals.wav"
  out_base="$RUNS/${id}.base.srt"
  out_ctx="$RUNS/${id}.context.srt"

  ffmpeg -v error -y -ss "$s" -t "$d" -i "validation_video/Hernán-1-1-1.wav" -ac 1 -ar 16000 "$clip" || true
  if [[ -f "$VOCALS_FULL" ]]; then
    ffmpeg -v error -y -ss "$s" -t "$d" -i "$VOCALS_FULL" -ac 1 -ar 16000 "$clip_v" || true
  fi

  modal run tenepal_modal.py \
    --input "$clip" \
    --output "$out_base" \
    --vocals "$clip_v" \
    --no-demucs \
    --show-confidence \
    --phone-vote \
    --min-turn-s 0.05 >/tmp/modal_name_${id}_base.log 2>&1 || true

  modal run tenepal_modal.py \
    --input "$clip" \
    --output "$out_ctx" \
    --vocals "$clip_v" \
    --no-demucs \
    --show-confidence \
    --phone-vote \
    --whisper-prompt \
    --whisper-prompt-extra "$prompt_extra" \
    --min-turn-s 0.05 >/tmp/modal_name_${id}_context.log 2>&1 || true

  txt_base=$(extract_text "$out_base")
  txt_ctx=$(extract_text "$out_ctx")

  printf "name\tbase\t%s\t%s\t%s\n" "$s" "$d" "$txt_base" >> "$REPORT"
  printf "name\tcontext\t%s\t%s\t%s\n" "$s" "$d" "$txt_ctx" >> "$REPORT"
  echo "[name] start=$s dur=$d done"
done

echo
echo "=== REPORT ==="
column -t -s $'\t' "$REPORT"

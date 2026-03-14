#!/usr/bin/env bash
set -euo pipefail

CASES="${1:-validation_video/edge_debug/modal_focus/cases.tsv}"
CTX_JSON="${2:-.planning/context/hernan-film-context.json}"
ROOT="validation_video/edge_debug/modal_focus"
CLIPS="$ROOT/clips"
RUNS="$ROOT/runs"
REPORT="$ROOT/report.tsv"

mkdir -p "$CLIPS" "$RUNS"

if [[ -f "venv/bin/activate" ]]; then
  # Ensure modal CLI from project venv is available.
  # User-facing commands remain plain `modal ...`.
  source venv/bin/activate
fi

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI not found in PATH (activate venv first)." >&2
  exit 127
fi

prompt_extra=$(CTX_JSON="$CTX_JSON" python - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["CTX_JSON"])
obj = json.loads(p.read_text(encoding='utf-8'))
parts = []
parts.extend(obj.get('key_names', []))
sh = obj.get('speaker_hints', {})
for k,v in sh.items():
    parts.append(str(k))
    parts.append(str(v))
parts.extend(obj.get('expected_langs', []))
print(' '.join(parts))
PY
)

extract() {
  while IFS=$'\t' read -r id src start end pad expect; do
    [[ -z "$id" || "${id:0:1}" == "#" ]] && continue
    ss=$(awk -v s="$start" -v p="$pad" 'BEGIN{v=s-p; if(v<0)v=0; printf "%.3f", v}')
    dur=$(awk -v s="$start" -v e="$end" -v p="$pad" 'BEGIN{v=(e-s)+(2*p); if(v<0.3)v=0.3; printf "%.3f", v}')
    out="$CLIPS/${id}.wav"
    ffmpeg -v error -y -ss "$ss" -t "$dur" -i "$src" -ac 1 -ar 16000 "$out"
    echo "[extract] $id -> $out (ss=$ss dur=$dur)"
  done < "$CASES"
}

run_case() {
  local id="$1"
  local mode="$2"
  local clip="$CLIPS/${id}.wav"
  local out="$RUNS/${id}.${mode}.srt"

  cmd=(modal run tenepal_modal.py
    --input "$clip"
    --output "$out"
    --no-demucs
    --show-confidence
    --phone-vote
    --show-phone-conf
    --show-nbest
  )

  if [[ "$mode" == "context" ]]; then
    cmd+=(--whisper-prompt --whisper-prompt-extra "$prompt_extra")
  elif [[ "$mode" == "safe" ]]; then
    cmd+=(--uncertain-text-policy suppress)
  elif [[ "$mode" == "safe_context" ]]; then
    cmd+=(--uncertain-text-policy suppress --whisper-prompt --whisper-prompt-extra "$prompt_extra")
  fi

  echo "[run] $id :: $mode"
  "${cmd[@]}" >/tmp/modal_focus_${id}_${mode}.log 2>&1
}

summarize() {
  printf "case\tmode\ttext\n" > "$REPORT"
  while IFS=$'\t' read -r id src start end pad expect; do
    [[ -z "$id" || "${id:0:1}" == "#" ]] && continue
    for mode in base context safe safe_context; do
      srt="$RUNS/${id}.${mode}.srt"
      txt="MISSING"
      if [[ -f "$srt" ]]; then
        txt=$(grep -E '^\[[A-Z]' "$srt" | sed -E 's/^\[[^]]+\]\s*//' | tr '\n' ' ' | tr -s ' ')
      fi
      printf "%s\t%s\t%s\n" "$id" "$mode" "$txt" >> "$REPORT"
    done
  done < "$CASES"

  echo
  echo "=== MODAL FOCUS REPORT ==="
  column -t -s $'\t' "$REPORT"
}

extract
while IFS=$'\t' read -r id src start end pad expect; do
  [[ -z "$id" || "${id:0:1}" == "#" ]] && continue
  run_case "$id" base
  run_case "$id" context
  run_case "$id" safe
  run_case "$id" safe_context
done < "$CASES"

summarize

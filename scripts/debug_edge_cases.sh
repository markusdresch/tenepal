#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
MANIFEST="${2:-validation_video/edge_cases.tsv}"
OUT_ROOT="validation_video/edge_debug"
CLIP_DIR="$OUT_ROOT/clips"
RUN_DIR="$OUT_ROOT/runs"
SCORE_FILE="$OUT_ROOT/scoreboard.tsv"

mkdir -p "$CLIP_DIR" "$RUN_DIR"

extract_clips() {
  while IFS=$'\t' read -r id source start_s end_s pad_s expect_present expect_absent note; do
    [[ -z "${id}" || "${id:0:1}" == "#" ]] && continue

    local_start=$(awk -v s="$start_s" -v p="$pad_s" 'BEGIN{v=s-p; if (v<0) v=0; printf "%.3f", v}')
    local_dur=$(awk -v s="$start_s" -v e="$end_s" -v p="$pad_s" 'BEGIN{v=(e-s)+(2*p); if (v<0.2) v=0.2; printf "%.3f", v}')
    clip="$CLIP_DIR/${id}.wav"

    ffmpeg -v error -y -ss "$local_start" -t "$local_dur" -i "$source" -ac 1 -ar 16000 "$clip"
    echo "[extract] $id -> $clip (start=$local_start dur=$local_dur)"
  done < "$MANIFEST"
}

run_profile() {
  local id="$1"
  local profile="$2"
  local clip="$CLIP_DIR/${id}.wav"
  local out="$RUN_DIR/${id}.${profile}.srt"

  base=(venv/bin/modal run tenepal_modal.py
    --input "$clip"
    --output "$out"
    --no-demucs
    --whisper-only
    --spanish-orthography
    --show-confidence
    --show-phone-conf
    --show-nbest
    --phone-vote
  )

  case "$profile" in
    auto)
      ;;
    auto_prompt)
      base+=(--whisper-prompt)
      ;;
    forced_es)
      base+=(--whisper-force-lang es)
      ;;
    forced_es_prompt)
      base+=(--whisper-force-lang es --whisper-prompt)
      ;;
    forced_es_prompt_float)
      base+=(--whisper-force-lang es --whisper-prompt --floating-window --floating-window-shift-s 0.25)
      ;;
    *)
      echo "Unknown profile: $profile" >&2
      return 1
      ;;
  esac

  echo "[run] $id :: $profile"
  "${base[@]}" >/tmp/edge_case_${id}_${profile}.log 2>&1 || {
    echo "[run] failed $id :: $profile (see /tmp/edge_case_${id}_${profile}.log)"
    return 1
  }
}

score_run() {
  local id="$1"
  local profile="$2"
  local expect_present="$3"
  local expect_absent="$4"
  local srt="$RUN_DIR/${id}.${profile}.srt"

  if [[ ! -f "$srt" ]]; then
    printf "%s\t%s\t-1\t0\t0\t1\tMISSING\n" "$id" "$profile"
    return
  fi

  text=$(grep -E '^\[[A-Z]' "$srt" | sed -E 's/^\[[^]]+\]\s*//' | tr '\n' ' ' | tr -s ' ')
  low_unc=0
  grep -q 'LOW UNC\| UNC' "$srt" && low_unc=1 || true

  present_hit=0
  absent_hit=0
  score=0

  [[ "$expect_present" == "-" ]] && expect_present=""
  [[ "$expect_absent" == "-" ]] && expect_absent=""

  if [[ -n "$expect_present" ]]; then
    if echo "$text" | grep -Eiq "$expect_present"; then
      present_hit=1
      score=$((score + 2))
    fi
  else
    score=$((score + 1))
  fi

  if [[ -n "$expect_absent" ]]; then
    if echo "$text" | grep -Eiq "$expect_absent"; then
      absent_hit=1
    else
      score=$((score + 2))
    fi
  else
    score=$((score + 1))
  fi

  if [[ "$low_unc" -eq 0 ]]; then
    score=$((score + 1))
  fi

  preview=$(echo "$text" | cut -c1-160)
  printf "%s\t%s\t%d\t%d\t%d\t%d\t%s\n" "$id" "$profile" "$score" "$present_hit" "$absent_hit" "$low_unc" "$preview"
}

run_all() {
  profiles=(auto auto_prompt forced_es forced_es_prompt forced_es_prompt_float)
  printf "case\tprofile\tscore\tpresent_hit\tabsent_hit\tlow_unc\tpreview\n" > "$SCORE_FILE"

  while IFS=$'\t' read -r id source start_s end_s pad_s expect_present expect_absent note; do
    [[ -z "${id}" || "${id:0:1}" == "#" ]] && continue

    for profile in "${profiles[@]}"; do
      run_profile "$id" "$profile" || true
      score_run "$id" "$profile" "$expect_present" "$expect_absent" >> "$SCORE_FILE"
    done
  done < "$MANIFEST"

  echo
  echo "=== EDGE DEBUG SCOREBOARD ==="
  column -t -s $'\t' "$SCORE_FILE"
  echo
  echo "=== BEST PER CASE ==="
  awk -F'\t' 'NR==1{next} {k=$1; if(!(k in best) || $3>best[k]) {best[k]=$3; line[k]=$0}} END{for (k in line) print line[k]}' "$SCORE_FILE" | sort | column -t -s $'\t'
}

case "$MODE" in
  extract)
    extract_clips
    ;;
  run)
    run_all
    ;;
  all)
    extract_clips
    run_all
    ;;
  *)
    echo "Usage: $0 [extract|run|all] [manifest.tsv]" >&2
    exit 1
    ;;
esac

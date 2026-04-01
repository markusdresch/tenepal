#!/usr/bin/env bash
# Full-Film EQ Comparison: 6 configs PARALLEL against Hernán-1-3
set -euo pipefail

INPUT="validation_video/hernan/Hernán-1-3.wav"
GT="benchmarks/snapshots/eq_comparison_gt_v1.json"
RESULTS_DIR="eq_comparison_results"
SUMMARY="$RESULTS_DIR/SUMMARY.md"

mkdir -p "$RESULTS_DIR"

CONFIGS=(
    "eq_configs/01_v7_best.json"
    "eq_configs/02_v7_prosody.json"
    "eq_configs/03_v7_prosody_gentle.json"
    "eq_configs/04_v7_speaker_tight.json"
    "eq_configs/05_v7_spa_reclaim_loose.json"
    "eq_configs/06_v7_prosody_spa_reclaim.json"
)

echo "=== Full-Film EQ Comparison (PARALLEL) ==="
echo "Input: $INPUT"
echo "Configs: ${#CONFIGS[@]} — launching all simultaneously"
echo "Started: $(date '+%H:%M:%S')"
echo ""

# Launch all 6 Modal runs in parallel
PIDS=()
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .json)
    srt="$RESULTS_DIR/${name}.srt"
    log="$RESULTS_DIR/${name}.log"

    echo "🚀 Launching: $name"

    (
        modal run tenepal_modal.py::main \
            --input "$INPUT" \
            --output "$srt" \
            --nah-finetuned \
            --eq "$cfg" \
            > "$log" 2>&1

        echo "✅ $name finished at $(date '+%H:%M:%S')"
    ) &
    PIDS+=($!)
done

echo ""
echo "⏳ All ${#CONFIGS[@]} runs launched. Waiting for completion..."
echo ""

# Wait for all to finish
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .json)
    if wait "$pid"; then
        echo "✅ $name — done (PID $pid)"
    else
        echo "❌ $name — FAILED (PID $pid)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "All runs complete. Failed: $FAILED/${#CONFIGS[@]}"
echo "Finished: $(date '+%H:%M:%S')"
echo ""

# Evaluate all SRTs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EVALUATING..."
echo ""

cat > "$SUMMARY" <<'EOF'
# EQ Full-Film Comparison Results

| # | Config | Accuracy | Correct | Total | Delta vs 01 |
|---|--------|----------|---------|-------|-------------|
EOF

BASELINE_CORRECT=0

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .json)
    srt="$RESULTS_DIR/${name}.srt"

    if [ ! -f "$srt" ]; then
        echo "⚠️ $name: No SRT output — skipped"
        echo "| $(printf '%02d' $((i+1))) | $name | FAILED | - | - | - |" >> "$SUMMARY"
        continue
    fi

    echo "📊 Evaluating: $name"
    eval_output=$(python3 evaluate.py "$GT" "$srt" 2>&1)
    echo "$eval_output" > "$RESULTS_DIR/${name}_eval.txt"

    # Extract accuracy
    acc_line=$(echo "$eval_output" | grep "LANGUAGE ACCURACY" || echo "N/A")
    correct=$(echo "$acc_line" | grep -oP '\d+(?=/\d+)' | head -1 || echo "0")
    total=$(echo "$acc_line" | grep -oP '(?<=/)(\d+)' | head -1 || echo "0")
    pct=$(echo "$acc_line" | grep -oP '\d+\.\d+%' || echo "N/A")

    if [ "$i" -eq 0 ]; then
        BASELINE_CORRECT="$correct"
        delta="baseline"
    else
        diff=$((correct - BASELINE_CORRECT))
        if [ "$diff" -ge 0 ]; then
            delta="+$diff"
        else
            delta="$diff"
        fi
    fi

    echo "  → $pct ($correct/$total) [delta: $delta]"
    echo "| $(printf '%02d' $((i+1))) | $name | $pct | $correct | $total | $delta |" >> "$SUMMARY"
done

echo "" >> "$SUMMARY"
echo "_Generated: $(date '+%Y-%m-%d %H:%M')_" >> "$SUMMARY"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY:"
echo ""
cat "$SUMMARY"
echo ""
echo "Detail reports: $RESULTS_DIR/*_eval.txt"
echo "Logs: $RESULTS_DIR/*.log"

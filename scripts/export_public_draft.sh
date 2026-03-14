#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/../tenepal-public}"

mkdir -p "$DEST"

rsync -a --delete \
  --exclude='.git/' \
  --exclude='.agentmux/' \
  --exclude='.agentmux_state' \
  --exclude='.ardua/' \
  --exclude='.benchmarks/' \
  --exclude='.claude/' \
  --exclude='.mezcalmux/' \
  --exclude='.planning/' \
  --exclude='.private/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='PERSONA.md' \
  --exclude='.task_codex.md' \
  --exclude='DIRECTIVE.md' \
  --exclude='SITUATION.md' \
  --exclude='AGENTS.md' \
  --exclude='AGENT_WORKFLOW.md' \
  --exclude='CLAUDE.md' \
  --exclude='GSD.md' \
  --exclude='MILESTONE_*.md' \
  --exclude='PARKING_LOT.md' \
  --exclude='ANNOTATION_ANALYSIS.md' \
  --exclude='EVOLUTION_PUBLIC.md' \
  --exclude='DATA_ACCESS.md' \
  --exclude='PUBLIC_RELEASE.md' \
  --exclude='MULTILAYER_MODEL.md' \
  --exclude='PITCH_PRESEGMENTATION.md' \
  --exclude='RESULTS.md' \
  --exclude='agentmux.log' \
  --exclude='mezcalmux.log' \
  --exclude='modal_run.log' \
  --exclude='output' \
  --exclude='validation_audio/' \
  --exclude='validation_video/' \
  --exclude='clips/' \
  --exclude='results/' \
  --exclude='reference_srt/' \
  --exclude='codices/' \
  --exclude='checkpoints/' \
  --exclude='colab-workbenches/' \
  --exclude='adversarial_spa/' \
  --exclude='*.wav' \
  --exclude='*.log' \
  --exclude='moctezuma*.txt' \
  --exclude='tuning_report*.txt' \
  --exclude='test_*.srt' \
  --exclude='hernan_annotator.html' \
  --exclude='eq_comparison_results/' \
  --exclude='src/*.egg-info/' \
  --exclude='tools/annotator/*.db' \
  --exclude='tools/annotator/.venv/' \
  --exclude='tools/segment_dashboard.html' \
  "$ROOT/" "$DEST/"

mkdir -p "$DEST/benchmarks/reports"
mkdir -p "$DEST/benchmarks/annotations"

copy_if_exists() {
  local src_rel="$1"
  local dest_rel="$2"
  if [[ -e "$ROOT/$src_rel" ]]; then
    mkdir -p "$(dirname "$DEST/$dest_rel")"
    cp "$ROOT/$src_rel" "$DEST/$dest_rel"
  fi
}

copy_if_exists "tests/regression/reports/hernan_nah_25_base.json" "benchmarks/reports/hernan_nah_25_base.json"
copy_if_exists "tests/regression/reports/hernan_nah_25_baseline.json" "benchmarks/reports/hernan_nah_25_baseline.json"
copy_if_exists "tools/corpus/results/baseline.json" "benchmarks/reports/amith_placeholder_baseline.json"
copy_if_exists "eq_comparison_gt.json" "benchmarks/reports/eq_comparison_gt.json"

python "$ROOT/scripts/export_public_annotations.py" \
  --db "$ROOT/tools/annotator/annotations.db" \
  --outdir "$DEST/benchmarks/annotations"

cat > "$DEST/benchmarks/README.md" <<'EOF'
# Benchmarks

This directory contains public, non-media benchmark artifacts copied from the private research repository.

- `reports/hernan_nah_25_base.json`: Whisper hallucination-language report on a 25-segment Nahuatl subset
- `reports/hernan_nah_25_baseline.json`: companion baseline report
- `reports/amith_placeholder_baseline.json`: placeholder corpus-evaluation manifest showing that audio was not bundled
- `reports/eq_comparison_gt.json`: comparison metadata used in evaluation workflows
- `annotations/*.jsonl`: public-safe annotation exports per media file
- `annotations/manifest.json`: summary counts for the exported annotation sets

Film clips, raw audio, and most subtitle outputs are intentionally excluded from the public draft.
EOF

cat > "$DEST/.gitignore" <<'EOF'
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.private/
venv/
EOF

echo "Public draft exported to: $DEST"
echo "Next steps:"
echo "  1. cd \"$DEST\""
echo "  2. git init"
echo "  3. git add ."
echo "  4. git commit -m 'Initial public draft'"

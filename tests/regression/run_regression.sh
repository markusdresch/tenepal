#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT="tests/regression/reports/hernan_nah_25_baseline.json"

if [[ -n "${1:-}" ]]; then
  OUT="$1"
fi

./venv/bin/python tests/regression/eval_regression.py \
  --fixture tests/regression/fixtures/hernan_nah_25.json \
  --prediction validation_video/Hernán-1-1-1.srt \
  --prediction validation_video/Hernán-1-1-2.srt \
  --output "$OUT"

echo "Done. Report: $OUT"

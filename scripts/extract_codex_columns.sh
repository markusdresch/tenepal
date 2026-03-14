#!/usr/bin/env bash
set -euo pipefail

# Extract and OCR two columns from scanned codex pages.
# Outputs per-page left/right text plus paired TSV/JSONL.
#
# Usage:
#   scripts/extract_codex_columns.sh \
#     --pdf codices/codex-florentin.pdf \
#     --start 50 --end 55 \
#     --out codices/extracted/florentine \
#     --dpi 400 \
#     --lang "spa+spa_old+lat"

PDF=""
START=""
END=""
OUT=""
DPI="400"
LANGS="spa+spa_old+lat"
PSM="6"
LEFT_PCT="50"
RIGHT_PCT="50"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdf) PDF="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --dpi) DPI="$2"; shift 2 ;;
    --lang) LANGS="$2"; shift 2 ;;
    --psm) PSM="$2"; shift 2 ;;
    --left-pct) LEFT_PCT="$2"; shift 2 ;;
    --right-pct) RIGHT_PCT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PDF" || -z "$START" || -z "$END" || -z "$OUT" ]]; then
  echo "Missing required args. Need --pdf --start --end --out" >&2
  exit 1
fi

if [[ ! -f "$PDF" ]]; then
  echo "PDF not found: $PDF" >&2
  exit 1
fi

if ! command -v pdftoppm >/dev/null 2>&1; then
  echo "pdftoppm not found." >&2
  exit 1
fi
if ! command -v tesseract >/dev/null 2>&1; then
  echo "tesseract not found." >&2
  exit 1
fi
if ! command -v convert >/dev/null 2>&1; then
  echo "ImageMagick not found." >&2
  exit 1
fi

mkdir -p "$OUT/pages" "$OUT/columns/left" "$OUT/columns/right" "$OUT/text/left" "$OUT/text/right"

TSV="$OUT/pairs.tsv"
JSONL="$OUT/pairs.jsonl"
echo -e "page\tleft_text_path\tright_text_path\tleft_image_path\tright_image_path" > "$TSV"
rm -f "$JSONL"

for ((p=START; p<=END; p++)); do
  base="p$(printf "%04d" "$p")"
  page_png="$OUT/pages/$base.png"
  page_proc_png="$OUT/pages/${base}_proc.png"
  left_png="$OUT/columns/left/${base}_left.png"
  right_png="$OUT/columns/right/${base}_right.png"
  left_txt="$OUT/text/left/${base}_left.txt"
  right_txt="$OUT/text/right/${base}_right.txt"

  # Render a single page from PDF.
  pdftoppm -f "$p" -singlefile -r "$DPI" -gray -png "$PDF" "${page_png%.png}" >/dev/null 2>&1 || true

  if [[ ! -f "$page_png" ]]; then
    echo "WARN: failed to render page $p" >&2
    continue
  fi

  # Light preprocessing to help OCR on old scans.
  if command -v magick >/dev/null 2>&1; then
    magick "$page_png" \
      -deskew 40% \
      -normalize \
      -contrast-stretch 0 \
      -sharpen 0x1 \
      "$page_proc_png"
  else
    convert "$page_png" \
      -deskew 40% \
      -normalize \
      -contrast-stretch 0 \
      -sharpen 0x1 \
      "$page_proc_png"
  fi

  # Split into two columns.
  if command -v magick >/dev/null 2>&1; then
    magick "$page_proc_png" -crop "${LEFT_PCT}%x100%+0+0" +repage "$left_png"
    magick "$page_proc_png" -crop "${RIGHT_PCT}%x100%+${LEFT_PCT}%+0" +repage "$right_png"
  else
    convert "$page_proc_png" -crop "${LEFT_PCT}%x100%+0+0" +repage "$left_png"
    convert "$page_proc_png" -crop "${RIGHT_PCT}%x100%+${LEFT_PCT}%+0" +repage "$right_png"
  fi

  # OCR each side.
  tesseract "$left_png" "${left_txt%.txt}" -l "$LANGS" --oem 1 --psm "$PSM" >/dev/null 2>&1 || true
  tesseract "$right_png" "${right_txt%.txt}" -l "$LANGS" --oem 1 --psm "$PSM" >/dev/null 2>&1 || true

  # Ensure text files exist even on OCR failure.
  [[ -f "$left_txt" ]] || : > "$left_txt"
  [[ -f "$right_txt" ]] || : > "$right_txt"

  echo -e "${p}\t${left_txt}\t${right_txt}\t${left_png}\t${right_png}" >> "$TSV"
  printf '{"page": %d, "left_text_path": "%s", "right_text_path": "%s", "left_image_path": "%s", "right_image_path": "%s"}\n' \
    "$p" "$left_txt" "$right_txt" "$left_png" "$right_png" >> "$JSONL"
done

echo "Done."
echo "TSV:   $TSV"
echo "JSONL: $JSONL"

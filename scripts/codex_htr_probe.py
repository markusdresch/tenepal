"""Modal HTR probe for old manuscript/page images.

Runs one or more free TrOCR models on local images and writes OCR outputs to file.

Examples:
    ./venv/bin/modal run codex_htr_probe.py \
      --inputs codices/extracted/florentine/columns/left/p0050_left.png

    ./venv/bin/modal run codex_htr_probe.py \
      --inputs codices/extracted/florentine/columns/left/p0050_left.png,codices/extracted/florentine/columns/right/p0050_right.png \
      --model-ids microsoft/trocr-base-printed,microsoft/trocr-base-handwritten \
      --num-beams 1 --max-new-tokens 512
"""

import io
import os
from pathlib import Path

import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "pillow",
    )
)

app = modal.App("codex-htr-probe", image=image)
model_cache = modal.Volume.from_name("tenepal-models", create_if_missing=True)
CACHE_DIR = "/tenepal-models"


@app.function(
    gpu="T4",
    timeout=900,
    volumes={CACHE_DIR: model_cache},
)
def transcribe_images(
    image_blobs: list[bytes],
    image_names: list[str],
    model_ids: list[str],
    max_new_tokens: int = 256,
    num_beams: int = 4,
    preprocess: bool = True,
):
    import torch
    from PIL import Image, ImageOps, ImageFilter
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if len(image_blobs) != len(image_names):
        raise ValueError("image_blobs and image_names must have the same length")

    cache_dir = f"{CACHE_DIR}/trocr"
    os.makedirs(cache_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {}
    for model_id in model_ids:
        try:
            print(f"Loading model: {model_id}")
            proc = TrOCRProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            mdl = VisionEncoderDecoderModel.from_pretrained(model_id, cache_dir=cache_dir)
            mdl.to(device)
            mdl.eval()
            loaded[model_id] = (proc, mdl)
        except Exception as exc:
            loaded[model_id] = (None, exc)

    def quality_score(text: str) -> int:
        # Prefer non-empty text with more alphabetic content.
        if not text:
            return -1
        alpha = sum(1 for c in text if c.isalpha())
        return alpha

    results = []
    for name, blob in zip(image_names, image_blobs):
        base_img = Image.open(io.BytesIO(blob)).convert("RGB")
        if preprocess:
            gray = ImageOps.grayscale(base_img)
            gray = ImageOps.autocontrast(gray)
            gray = gray.filter(ImageFilter.SHARPEN)
            base_img = gray.convert("RGB")

        candidates = []
        for model_id in model_ids:
            proc_or_none, mdl_or_exc = loaded[model_id]
            if proc_or_none is None:
                candidates.append(
                    {
                        "model_id": model_id,
                        "text": "",
                        "error": f"{type(mdl_or_exc).__name__}: {mdl_or_exc}",
                        "score": -1,
                    }
                )
                continue

            proc = proc_or_none
            mdl = mdl_or_exc
            text = ""
            error = ""
            try:
                pixel_values = proc(images=base_img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(mdl.device)
                with torch.inference_mode():
                    generated_ids = mdl.generate(
                        pixel_values,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
                text = proc.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            score = quality_score(text)
            candidates.append(
                {
                    "model_id": model_id,
                    "text": text,
                    "error": error,
                    "score": score,
                }
            )

        best = max(candidates, key=lambda c: c["score"]) if candidates else {
            "model_id": "",
            "text": "",
            "error": "no-candidates",
            "score": -1,
        }
        results.append(
            {
                "image": name,
                "best_model_id": best["model_id"],
                "text": best["text"],
                "candidates": candidates,
            }
        )

    model_cache.commit()
    return results


@app.local_entrypoint()
def main(
    inputs: str = "codices/extracted/florentine/columns/left/p0050_left.png",
    output: str = "codices/extracted/florentine/htr_probe_results",
    model_ids: str = "microsoft/trocr-base-printed,microsoft/trocr-base-handwritten",
    max_new_tokens: int = 256,
    num_beams: int = 4,
    preprocess: bool = True,
):
    paths = [p.strip() for p in inputs.split(",") if p.strip()]
    if not paths:
        print("No input paths provided.")
        return

    blobs: list[bytes] = []
    names: list[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Missing file: {p}")
            return
        blobs.append(path.read_bytes())
        names.append(path.name)

    selected_model_ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not selected_model_ids:
        print("No model IDs provided.")
        return

    print(f"Uploading {len(paths)} image(s) to Modal...")
    results = transcribe_images.remote(
        image_blobs=blobs,
        image_names=names,
        model_ids=selected_model_ids,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        preprocess=preprocess,
    )

    out_base = Path(output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    txt_path = out_base.with_suffix(".txt")

    import json as _json
    json_path.write_text(_json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    for row in results:
        lines.append(f"=== {row['image']} ===")
        lines.append(f"best_model: {row.get('best_model_id', '')}")
        lines.append(row["text"] if row["text"] else "(empty)")
        if row.get("candidates"):
            lines.append("candidates:")
            for c in row["candidates"]:
                model = c.get("model_id", "")
                score = c.get("score", -1)
                err = c.get("error", "")
                preview = (c.get("text", "") or "").replace("\n", " ")
                if len(preview) > 120:
                    preview = preview[:120] + "..."
                lines.append(f"- {model} | score={score} | err={err or '-'} | {preview}")
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")

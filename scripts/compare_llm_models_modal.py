#!/usr/bin/env python3
"""Modal-based LLM transliteration comparison.

Run with: modal run scripts/compare_llm_models_modal.py
"""

import modal

app = modal.App("llm-transliteration-compare")

# GPU image with all models pre-loaded
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "mistralai",
        "sentencepiece",
    )
)

# Test IPA sequences
TEST_CASES = [
    # Nahuatl
    ("tɬakatemek", "NAH", "tlacatémec"),
    ("kwaʊtli", "NAH", "cuauhtli"),
    ("tenoːtʃtitɬan", "NAH", "Tenochtitlan"),
    ("malinaltsin", "NAH", "Malintzin"),
    ("tlatoːani", "NAH", "tlatoani"),
    ("witsilopoːtʃtli", "NAH", "Huitzilopochtli"),
    # Maya
    ("kʼuːx a wol", "MAY", "k'ux a wol"),
    ("tʃakʼan putun", "MAY", "chak'an putun"),
    ("baʔal a kʼaʔ", "MAY", "ba'al a k'a'"),
    ("xukuːb", "MAY", "xukub"),
    # Spanish
    ("soldado", "SPA", "soldado"),
    ("konkeistadoɾ", "SPA", "conquistador"),
    # Mixed/difficult
    ("notlasoːmalintsine", "NAH", "notlazomalintzine"),
]

NAHUATL_FEW_SHOT = """Examples (Nahuatl):
IPA: tɬakatemek → tlacatémec
IPA: kwaʊtli → cuauhtli
IPA: tenoːtʃtitɬan → Tenochtitlan
IPA: malinaltsin → Malintzin
IPA: witsilopoːtʃtli → Huitzilopochtli"""

MAYA_FEW_SHOT = """Examples (Maya):
IPA: kʼuːx a wol → k'ux a wol
IPA: tʃakʼan putun → chak'an putun
IPA: baʔal a kʼaʔ → ba'al a k'a'
IPA: xukuːb → xukub"""


def get_system_prompt(lang: str) -> str:
    if lang == "NAH":
        few_shot = NAHUATL_FEW_SHOT
    elif lang == "MAY":
        few_shot = MAYA_FEW_SHOT
    else:
        few_shot = NAHUATL_FEW_SHOT + "\n" + MAYA_FEW_SHOT

    return f"""You transcribe IPA phonemes to readable orthography.
Output ONLY the transcribed text, nothing else.
No explanations, no quotes, no markdown.

Rules:
- ʔ = glottal stop → ' (Maya) or h (Nahuatl)
- tɬ = lateral affricate → tl (Nahuatl)
- Ejectives kʼ tʼ tsʼ tʃʼ pʼ → k' t' ts' ch' p' (Maya)
- tʃ → ch
- ʃ → sh or x

{few_shot}"""


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("mistral-secret")],
)
def test_all_models():
    import os
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    # Models to test
    models = {
        "Qwen2-1.5B": "Qwen/Qwen2-1.5B-Instruct",
        "Gemma-2B": "google/gemma-2b-it",
        "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }

    def clean_result(text: str) -> str:
        """Clean LLM output."""
        text = text.split("\n")[0].strip()
        text = text.strip('"\'*`')
        for prefix in ["Text:", "text:", "Answer:", "Output:", "Transcription:", "IPA:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        return text

    # Test Mistral first
    print("\n=== Testing Mistral ===")
    mistral_results = {}
    api_key = os.getenv("MISTRAL_API_KEY")

    if api_key:
        try:
            from mistralai import Mistral
            client = Mistral(api_key=api_key)

            for ipa, lang, expected in TEST_CASES[:5]:  # Test subset to avoid rate limits
                try:
                    response = client.chat.complete(
                        model="mistral-small-latest",
                        messages=[
                            {"role": "system", "content": get_system_prompt(lang)},
                            {"role": "user", "content": f"IPA: {ipa}"},
                        ],
                    )
                    result = clean_result(response.choices[0].message.content)
                    mistral_results[ipa] = result
                    print(f"  {ipa} -> {result}")
                    time.sleep(0.5)  # Rate limit
                except Exception as e:
                    if "429" in str(e):
                        mistral_results[ipa] = "[RATE_LIMITED]"
                        print(f"  {ipa} -> RATE LIMITED")
                        break
                    else:
                        mistral_results[ipa] = f"[ERROR]"
                        print(f"  {ipa} -> ERROR: {e}")
        except Exception as e:
            print(f"  Mistral init failed: {e}")
    else:
        print("  No MISTRAL_API_KEY")

    results["Mistral"] = mistral_results

    # Test HuggingFace models
    for name, model_id in models.items():
        print(f"\n=== Testing {name} ===")
        model_results = {}

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            for ipa, lang, expected in TEST_CASES:
                try:
                    messages = [
                        {"role": "system", "content": get_system_prompt(lang)},
                        {"role": "user", "content": f"IPA: {ipa}"},
                    ]

                    if hasattr(tokenizer, "apply_chat_template"):
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    else:
                        # Fallback for models without chat template
                        text = f"{get_system_prompt(lang)}\n\nUser: IPA: {ipa}\nAssistant:"

                    inputs = tokenizer([text], return_tensors="pt").to(model.device)

                    start = time.perf_counter()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    latency = (time.perf_counter() - start) * 1000

                    generated = outputs[0][inputs["input_ids"].shape[-1]:]
                    result = tokenizer.decode(generated, skip_special_tokens=True)
                    result = clean_result(result)

                    model_results[ipa] = result
                    print(f"  {ipa} -> {result} ({latency:.0f}ms)")

                except Exception as e:
                    model_results[ipa] = f"[ERROR]"
                    print(f"  {ipa} -> ERROR: {e}")

            # Free memory
            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Model load failed: {e}")
            for ipa, lang, expected in TEST_CASES:
                model_results[ipa] = "[LOAD_FAILED]"

        results[name] = model_results

    return results


@app.function(image=image)
def format_table(results: dict):
    """Format results as markdown table."""
    models = list(results.keys())

    # Header
    lines = ["| IPA | Lang | Expected |"]
    for m in models:
        lines[0] += f" {m} |"

    lines.append("|" + "---|" * (3 + len(models)))

    # Rows
    for ipa, lang, expected in TEST_CASES:
        row = f"| `{ipa}` | {lang} | {expected} |"
        for m in models:
            val = results.get(m, {}).get(ipa, "—")
            # Mark correct with checkmark
            if val.lower().strip() == expected.lower().strip():
                val = f"✓ {val}"
            row += f" {val} |"
        lines.append(row)

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print("Running LLM transliteration comparison on Modal GPU...")
    results = test_all_models.remote()

    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80 + "\n")

    table = format_table.remote(results)
    print(table)

    print("\n" + "=" * 80)
    print("Summary:")
    for model, r in results.items():
        correct = sum(1 for ipa, lang, exp in TEST_CASES if r.get(ipa, "").lower().strip() == exp.lower().strip())
        total = len(TEST_CASES)
        print(f"  {model}: {correct}/{total} ({100*correct/total:.0f}%)")

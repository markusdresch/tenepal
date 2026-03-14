#!/usr/bin/env python3
"""Test different LLM backends for IPA→text transliteration.

Compares: Mistral, Qwen2-1.5B, Gemma-2B, Phi-3-mini
Outputs a comparison table.
"""

import os
import time
from dataclasses import dataclass

# Test IPA sequences from real Hernán/Apocalypto processing
TEST_CASES = [
    # Nahuatl examples
    ("tɬakatemek", "NAH", "tlacatémec"),  # man-cutter (warrior)
    ("kwaʊt.li", "NAH", "cuauhtli"),  # eagle
    ("tenoːtʃtitɬan", "NAH", "Tenochtitlan"),
    ("malinaltsin", "NAH", "Malintzin"),
    ("tlatoːani", "NAH", "tlatoani"),

    # Maya examples
    ("kʼuːx a wol", "MAY", "k'ux a wol"),  # how are you
    ("tʃakʼan putun", "MAY", "chak'an putun"),
    ("baʔal a kʼaʔ", "MAY", "ba'al a k'a'"),
    ("xukuːb", "MAY", "xukub"),

    # Spanish examples (should stay close to Spanish)
    ("soldado", "SPA", "soldado"),
    ("konkeistadoɾ", "SPA", "conquistador"),
    ("seɲoɾ koɾtes", "SPA", "señor Cortés"),
]


@dataclass
class LLMResult:
    model: str
    text: str
    latency_ms: float
    error: str | None = None


def test_mistral(ipa: str, lang: str, few_shot: str) -> LLMResult:
    """Test Mistral API."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return LLMResult("mistral-small", "", 0, "NO_API_KEY")

    try:
        from mistralai import Mistral

        system_prompt = f"""You transcribe IPA to orthography. Output ONLY the transcribed text, nothing else.
No explanations, no comments, no "Here is", no quotes. Just the word(s).

{few_shot}"""

        client = Mistral(api_key=api_key)
        start = time.perf_counter()
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"IPA: {ipa}"},
            ],
        )
        latency = (time.perf_counter() - start) * 1000
        result = response.choices[0].message.content.strip()
        result = result.split("\n")[0].strip().strip('"\'').strip('*').strip('`')
        return LLMResult("mistral-small", result, latency)
    except Exception as e:
        err = str(e)
        if "429" in err:
            return LLMResult("mistral-small", "", 0, "RATE_LIMITED")
        return LLMResult("mistral-small", "", 0, str(e)[:50])


def test_hf_model(ipa: str, lang: str, few_shot: str, model_name: str) -> LLMResult:
    """Test a HuggingFace model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        system_prompt = f"""You transcribe IPA to orthography. Output ONLY the transcribed text, nothing else.
No explanations, no comments, no "Here is", no quotes. Just the word(s).

{few_shot}"""

        # Load model (will be cached)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"IPA: {ipa}"},
        ]

        start = time.perf_counter()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True).strip()
        latency = (time.perf_counter() - start) * 1000

        # Clean up
        result = result.split("\n")[0].strip().strip('"\'').strip('*').strip('`')
        short_name = model_name.split("/")[-1]
        return LLMResult(short_name, result, latency)
    except Exception as e:
        short_name = model_name.split("/")[-1]
        return LLMResult(short_name, "", 0, str(e)[:50])


# Few-shot examples
NAHUATL_FEW_SHOT = """Examples (Nahuatl):
IPA: tɬakatemek → tlacatémec
IPA: kwaʊt.li → cuauhtli
IPA: tenoːtʃtitɬan → Tenochtitlan
IPA: malinaltsin → Malintzin"""

MAYA_FEW_SHOT = """Examples (Maya):
IPA: kʼuːx a wol → k'ux a wol
IPA: tʃakʼan putun → chak'an putun
IPA: baʔal a kʼaʔ → ba'al a k'a'"""


def get_few_shot(lang: str) -> str:
    if lang == "NAH":
        return NAHUATL_FEW_SHOT
    elif lang == "MAY":
        return MAYA_FEW_SHOT
    else:
        return NAHUATL_FEW_SHOT + "\n" + MAYA_FEW_SHOT


def main():
    # Models to test
    HF_MODELS = [
        "Qwen/Qwen2-1.5B-Instruct",
        "google/gemma-2b-it",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    print("=" * 100)
    print("LLM Transliteration Comparison")
    print("=" * 100)

    # Header
    print(f"{'IPA':<25} {'Lang':<5} {'Expected':<20} ", end="")
    print(f"{'Mistral':<20} ", end="")
    for m in HF_MODELS:
        print(f"{m.split('/')[-1]:<20} ", end="")
    print()
    print("-" * 100)

    for ipa, lang, expected in TEST_CASES:
        few_shot = get_few_shot(lang)

        # Test Mistral
        mistral = test_mistral(ipa, lang, few_shot)

        # Test HF models
        hf_results = []
        for model in HF_MODELS:
            result = test_hf_model(ipa, lang, few_shot, model)
            hf_results.append(result)

        # Print row
        print(f"{ipa[:24]:<25} {lang:<5} {expected:<20} ", end="")

        if mistral.error:
            print(f"[{mistral.error}]".ljust(20), end=" ")
        else:
            print(f"{mistral.text[:19]:<20} ", end="")

        for r in hf_results:
            if r.error:
                print(f"[{r.error[:17]}]".ljust(20), end=" ")
            else:
                print(f"{r.text[:19]:<20} ", end="")
        print()

    print("=" * 100)


if __name__ == "__main__":
    main()

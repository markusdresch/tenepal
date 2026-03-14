#!/usr/bin/env python3
"""LLM probe: transliterate IPA-like lines to colonial Spanish Nahuatl orthography.

Reads an SRT with IPA lines (e.g., "♫allo:" / "♫w2v2:"), sends each IPA line to an
LLM with a Nahuatl-IPA dictionary prompt, and writes a new SRT with transliterations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


def load_dictionary(path: Path, max_entries: int) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data[:max_entries]:
        ipa = " ".join(item.get("ipa", []))
        word = item.get("word", "")
        if not ipa or not word:
            continue
        rows.append(f"- {ipa} -> {word}")
    return "\n".join(rows)


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return (msg.get("content") or "").strip()


def call_ollama_chat(
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {api_base}. Start it with: 'ollama serve'"
        ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc

    msg = data.get("message", {})
    return (msg.get("content") or "").strip()


def split_blocks(srt_text: str) -> list[str]:
    return [b for b in re.split(r"\n\s*\n", srt_text.strip()) if b.strip()]


def parse_tag_and_inline_ipa(tag_line: str) -> tuple[str, str, str, str]:
    """Parse '[TAG] text' line and capture inline IPA when text is missing.

    Returns: (prefix, original_text, inline_allo, inline_w2v2)
    """
    m = re.match(r"(\[[^\]]+\]\s*)(.*)$", tag_line.strip())
    if not m:
        return "", "", "", ""
    prefix, rest = m.groups()
    rest = rest.strip()
    inline_allo = ""
    inline_w2v2 = ""
    original = rest
    if rest.startswith("♫allo:"):
        inline_allo = rest[len("♫allo:"):].strip()
        original = ""
    elif rest.startswith("♫w2v2:"):
        inline_w2v2 = rest[len("♫w2v2:"):].strip()
        original = ""
    elif rest.startswith("♫ "):
        # Unknown inline IPA line; treat as allosaurus fallback by convention.
        inline_allo = rest[2:].strip()
        original = ""
    return prefix, original, inline_allo, inline_w2v2


def extract_ipa_line(lines: list[str], backend: str, allow_fallback: bool = True) -> str:
    tag = "♫allo:" if backend == "allo" else "♫w2v2:"
    alt_tag = "♫w2v2:" if backend == "allo" else "♫allo:"
    for ln in lines:
        if ln.startswith(tag):
            return ln[len(tag):].strip()
    if not allow_fallback:
        return ""
    # Fallback 1: use alternate backend line if requested backend is missing.
    for ln in lines:
        if ln.startswith(alt_tag):
            return ln[len(alt_tag):].strip()
    # Fallback 2: unlabeled IPA line like "♫ ...".
    for ln in lines:
        if ln.startswith("♫ "):
            return ln[2:].strip()
    return ""


def transliterate_ipa(
    ipa: str,
    system_prompt: str,
    dictionary_text: str,
    model: str,
    provider: str,
    api_base: str,
    api_key: str,
) -> str:
    if not ipa:
        return ""
    user_prompt = (
        "Nahuatl-IPA dictionary:\n"
        f"{dictionary_text}\n\n"
        f"Phonetic transcription:\n{ipa}\n"
    )
    if provider == "ollama":
        return call_ollama_chat(
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
    return call_chat_completion(
        api_base=api_base,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
    )


def build_dual_rows(
    srt_text: str,
    system_prompt: str,
    dictionary_text: str,
    model: str,
    provider: str,
    api_base: str,
    api_key: str,
) -> list[dict]:
    rows: list[dict] = []
    blocks = split_blocks(srt_text)
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        idx = lines[0].strip()
        timecode = lines[1].strip()
        prefix, original, inline_allo, inline_w2v2 = parse_tag_and_inline_ipa(lines[2])
        if not prefix:
            continue
        ipa_allo = inline_allo or extract_ipa_line(lines, "allo", allow_fallback=False)
        ipa_w2v2 = inline_w2v2 or extract_ipa_line(lines, "w2v2", allow_fallback=False)
        if not ipa_allo and not ipa_w2v2:
            # If neither explicit backend line exists, use generic fallback once.
            generic = extract_ipa_line(lines, "allo", allow_fallback=True)
            ipa_allo = generic
            ipa_w2v2 = generic

        tr_allo = transliterate_ipa(
            ipa_allo, system_prompt, dictionary_text, model, provider, api_base, api_key
        ) if ipa_allo else ""
        tr_w2v2 = transliterate_ipa(
            ipa_w2v2, system_prompt, dictionary_text, model, provider, api_base, api_key
        ) if ipa_w2v2 else ""
        tr_fused = fuse_transliterations(
            tr_allo=tr_allo,
            tr_w2v2=tr_w2v2,
            ipa_allo=ipa_allo,
            ipa_w2v2=ipa_w2v2,
            spanish_orthography=("Spanish orthography" in system_prompt),
        )

        rows.append(
            {
                "idx": idx,
                "time": timecode,
                "prefix": prefix,
                "original": original,
                "ipa_allo": ipa_allo,
                "ipa_w2v2": ipa_w2v2,
                "tr_allo": tr_allo,
                "tr_w2v2": tr_w2v2,
                "tr_fused": tr_fused,
            }
        )
    return rows


_FUSION_STOPWORDS = {
    "de", "del", "la", "el", "y", "que", "en", "por", "para", "con", "no", "es", "un", "una",
}


def _word_score(word: str) -> float:
    w = word.lower()
    score = 0.0
    if w in _FUSION_STOPWORDS:
        score -= 0.3
    for cue in ("tl", "tz", "hu", "qu", "cu", "x", "z", "k", "tlan", "meh"):
        if cue in w:
            score += 0.25
    if len(w) >= 6:
        score += 0.1
    return score


def _ipa_signals(ipa_allo: str, ipa_w2v2: str) -> set[str]:
    s = f"{ipa_allo} {ipa_w2v2}".lower()
    signals = set()
    # Likely future/plural-like ending traces in acoustic output.
    if re.search(r"(s|z|ts|ʃ)\s*k\s*i\b", s):
        signals.add("zki")
    if re.search(r"\bk\s*i\b", s):
        signals.add("ki")
    if re.search(r"\bk\s*e\b", s):
        signals.add("ke")
    return signals


def recover_suffixes(text: str, ipa_allo: str, ipa_w2v2: str, spanish_orthography: bool) -> str:
    if not text:
        return text
    signals = _ipa_signals(ipa_allo, ipa_w2v2)
    words = text.split()
    if not words:
        return text
    last = words[-1]
    low = last.lower()
    # If both backends suggest an sibilant+k+i tail, recover -zke / -izke style.
    if "zki" in signals:
        if low.endswith("ki"):
            last = re.sub(r"ki$", "izque" if spanish_orthography else "izke", last, flags=re.IGNORECASE)
        elif low.endswith("ke"):
            last = re.sub(r"ke$", "zque" if spanish_orthography else "zke", last, flags=re.IGNORECASE)
        elif low.endswith("i"):
            last = last + ("zque" if spanish_orthography else "zke")
    words[-1] = last
    return " ".join(words)


def fuse_transliterations(
    tr_allo: str,
    tr_w2v2: str,
    ipa_allo: str,
    ipa_w2v2: str,
    spanish_orthography: bool,
) -> str:
    if tr_allo and not tr_w2v2:
        return recover_suffixes(tr_allo, ipa_allo, ipa_w2v2, spanish_orthography)
    if tr_w2v2 and not tr_allo:
        return recover_suffixes(tr_w2v2, ipa_allo, ipa_w2v2, spanish_orthography)
    if not tr_allo and not tr_w2v2:
        return ""

    wa = tr_allo.split()
    ww = tr_w2v2.split()
    n = max(len(wa), len(ww))
    merged = []
    for i in range(n):
        a = wa[i] if i < len(wa) else ""
        w = ww[i] if i < len(ww) else ""
        if a and not w:
            merged.append(a)
            continue
        if w and not a:
            merged.append(w)
            continue
        # Choose token with higher Nahuatl-shape score; tie -> longer token.
        sa = _word_score(a)
        sw = _word_score(w)
        if sw > sa + 0.05:
            merged.append(w)
        elif sa > sw + 0.05:
            merged.append(a)
        else:
            merged.append(a if len(a) >= len(w) else w)

    fused = " ".join(t for t in merged if t).strip()
    return recover_suffixes(fused, ipa_allo, ipa_w2v2, spanish_orthography)


def write_table(rows: list[dict], path: Path, fmt: str) -> None:
    if fmt == "tsv":
        with path.open("w", encoding="utf-8") as f:
            f.write(
                "idx\ttime\toriginal\tipa_allo\tipa_w2v2\ttranslit_allo\ttranslit_w2v2\ttranslit_fused\n"
            )
            for r in rows:
                vals = [
                    r["idx"], r["time"], r["original"], r["ipa_allo"], r["ipa_w2v2"], r["tr_allo"], r["tr_w2v2"], r["tr_fused"]
                ]
                vals = [v.replace("\t", " ").replace("\n", " ") for v in vals]
                f.write("\t".join(vals) + "\n")
        return

    # markdown
    with path.open("w", encoding="utf-8") as f:
        f.write("| # | Time | Original | Allo IPA | W2V2 IPA | Allo Transliteration | W2V2 Transliteration | Fused |\n")
        f.write("|---|------|----------|----------|----------|-----------------------|-----------------------|-------|\n")
        for r in rows:
            vals = [
                r["idx"], r["time"], r["original"], r["ipa_allo"], r["ipa_w2v2"], r["tr_allo"], r["tr_w2v2"], r["tr_fused"]
            ]
            vals = [v.replace("|", "\\|").replace("\n", " ") for v in vals]
            f.write("| " + " | ".join(vals) + " |\n")


def transliterate_srt(
    srt_text: str,
    backend: str,
    system_prompt: str,
    dictionary_text: str,
    model: str,
    provider: str,
    api_base: str,
    api_key: str,
) -> tuple[str, int]:
    blocks = split_blocks(srt_text)
    out_blocks: list[str] = []
    changed = 0

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            out_blocks.append(block)
            continue

        _prefix, _original, inline_allo, inline_w2v2 = parse_tag_and_inline_ipa(lines[2])
        inline = inline_allo if backend == "allo" else inline_w2v2
        ipa = inline or extract_ipa_line(lines, backend, allow_fallback=True)
        if not ipa:
            out_blocks.append(block)
            continue

        out = transliterate_ipa(
            ipa=ipa,
            system_prompt=system_prompt,
            dictionary_text=dictionary_text,
            model=model,
            provider=provider,
            api_base=api_base,
            api_key=api_key,
        )
        if out:
            # Replace text after speaker tag line, keep timing/speaker/IPA intact.
            m = re.match(r"(\[[^\]]+\]\s*)(.*)$", lines[2])
            if m:
                lines[2] = m.group(1) + out
                changed += 1
        out_blocks.append("\n".join(lines))

    return "\n\n".join(out_blocks) + "\n", changed


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM IPA->Nahuatl transliteration probe.")
    parser.add_argument("--input-srt", required=True, help="Input SRT path")
    parser.add_argument("--output-srt", default="", help="Output SRT path")
    parser.add_argument("--backend", choices=["allo", "w2v2"], default="allo")
    parser.add_argument("--dict", default="src/tenepal/data/nah_lexicon.json")
    parser.add_argument("--dict-max", type=int, default=120, help="Max dictionary entries in prompt")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--spanish-orthography",
        action="store_true",
        help="Enforce colonial/Spanish orthography normalization in output.",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Transliterate both allo and w2v2 IPA and output comparison table.",
    )
    parser.add_argument(
        "--table-out",
        default="",
        help="Output path for dual comparison table (markdown/tsv).",
    )
    parser.add_argument(
        "--table-format",
        choices=["markdown", "tsv"],
        default="markdown",
        help="Format for --table-out.",
    )
    parser.add_argument(
        "--dual-in-srt",
        action="store_true",
        help="In dual mode, also write SRT text as: [allo](...) [w2v2](...).",
    )
    args = parser.parse_args()

    in_path = Path(args.input_srt)
    if not in_path.exists():
        raise SystemExit(f"Input SRT not found: {in_path}")

    out_path = Path(args.output_srt) if args.output_srt else in_path.with_suffix(".llm_translit.srt")
    dict_path = Path(args.dict)
    if not dict_path.exists():
        raise SystemExit(f"Dictionary not found: {dict_path}")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if args.provider == "openai" and not api_key:
        raise SystemExit("OPENAI_API_KEY is required for --provider openai")

    dictionary_text = load_dictionary(dict_path, args.dict_max)
    system_prompt = (
        "You are a linguistic assistant. I will provide a phonetic transcription. "
        "Use the provided Nahuatl-IPA dictionary to correct it into the colonial Spanish "
        "transliteration. Do not translate. Only output the transliterated Nahuatl."
    )
    if args.spanish_orthography:
        system_prompt += (
            " Enforce Spanish orthography conventions where applicable "
            "(e.g., hu/w, c/qu/k normalization by context, tz for affricates). "
            "Prefer a single readable Nahuatl line in Spanish-style orthography. "
            "Do not explain decisions."
        )
    system_prompt += (
        " Write like a Spanish conquistador or fray would have transliterated it."
    )

    srt_text = in_path.read_text(encoding="utf-8", errors="replace")
    if args.dual:
        rows = build_dual_rows(
            srt_text=srt_text,
            system_prompt=system_prompt,
            dictionary_text=dictionary_text,
            model=args.model,
            provider=args.provider,
            api_base=args.api_base,
            api_key=api_key,
        )
        table_path = Path(args.table_out) if args.table_out else in_path.with_suffix(".llm_dual.md")
        write_table(rows, table_path, args.table_format)
        print(f"Wrote: {table_path} ({len(rows)} rows)")
        if args.dual_in_srt:
            blocks = split_blocks(srt_text)
            out_blocks = []
            row_by_idx = {r["idx"]: r for r in rows}
            changed = 0
            for block in blocks:
                lines = block.splitlines()
                if len(lines) < 3:
                    out_blocks.append(block)
                    continue
                idx = lines[0].strip()
                r = row_by_idx.get(idx)
                if not r:
                    out_blocks.append(block)
                    continue
                m = re.match(r"(\[[^\]]+\]\s*)(.*)$", lines[2])
                if not m:
                    out_blocks.append(block)
                    continue
                prefix, _old = m.groups()
                new_text = f"[allo]({r['tr_allo']}) [w2v2]({r['tr_w2v2']}) [fused]({r['tr_fused']})".strip()
                lines[2] = prefix + new_text
                changed += 1
                out_blocks.append("\n".join(lines))
            out_srt = "\n\n".join(out_blocks) + "\n"
            out_path.write_text(out_srt, encoding="utf-8")
            print(f"Wrote: {out_path} (updated {changed} subtitle lines)")
        return

    output_srt, changed = transliterate_srt(
        srt_text=srt_text,
        backend=args.backend,
        system_prompt=system_prompt,
        dictionary_text=dictionary_text,
        model=args.model,
        provider=args.provider,
        api_base=args.api_base,
        api_key=api_key,
    )
    out_path.write_text(output_srt, encoding="utf-8")
    print(f"Wrote: {out_path} (updated {changed} subtitle lines)")


if __name__ == "__main__":
    main()

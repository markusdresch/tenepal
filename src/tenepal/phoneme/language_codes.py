"""Language code mapping between Tenepal and ASR backends."""

from __future__ import annotations

LANGUAGE_CODE_MAP: dict[str, dict[str, list[str]]] = {
    "allosaurus": {
        "nah": ["ncj", "ncl", "nch", "ncx", "nco", "ncu"],
        "spa": ["spa"],
        "eng": ["eng"],
        "deu": ["deu"],
        "ipa": ["ipa"],
    },
    "omnilingual": {
        "nah": ["ncj", "ncl", "nch", "ncx", "nco", "ncu"],
        "spa": ["spa"],
        "eng": ["eng"],
        "deu": ["deu"],
    },
}


def resolve_language_code(lang: str, backend: str = "allosaurus") -> str:
    """Resolve a Tenepal language code to a backend-specific code."""
    backend_map = LANGUAGE_CODE_MAP.get(backend, {})
    codes = backend_map.get(lang)
    if codes:
        return codes[0]
    return lang

"""Tenepal on Modal — GPU-powered Whisper-first pipeline.

Usage:
    # First time setup:
    pip install modal
    modal setup  # authenticates via browser

    # Test run with local audio file:
    modal run tenepal_modal.py --input hernan_s01e01.wav

    # Deploy as persistent endpoint:
    modal deploy tenepal_modal.py
    # Then POST audio to the URL it gives you

Costs: ~$0.10-0.15 per 20min episode on T4 ($0.59/h)
"""

import modal
import json
import os
import re
import unicodedata
import inspect
from collections import Counter
from abc import ABC, abstractmethod
from pathlib import Path

# ---------------------------------------------------------------------------
# Container image — all deps baked in, models cached in Volume
# ---------------------------------------------------------------------------
tenepal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install(
        "faster-whisper>=1.0.0",
        "allosaurus",
        "demucs",
        "speechbrain>=1.0.0",  # 1.0+ supports torchaudio 2.x
        "huggingface_hub>=0.23,<0.25",  # compat with speechbrain token API
        "pyannote.audio",
        "silero-vad",
        "soundfile",
        "numpy",
        "torch==2.5.1",
        "torchaudio==2.5.1",  # pinned to match torch
        "transformers",
        "phonemizer",
        "epitran",
        "mistralai",
        "praat-parselmouth",
        "scikit-learn",
        "accelerate",  # For HuggingFace LLM fallback
        "asteroid",  # ConvTasNet for voice separation
        "librosa",  # For pitch-based separation
    )
    .add_local_dir("src/tenepal/data", remote_path="/data/tenepal")
)

# Separate image for Omnilingual (needs older transformers)
omnilingual_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install(
        "omnilingual-asr",
        "soundfile",
        "numpy",
        "torch",
        "epitran",
    )
)

# Separate image for VibeVoice-ASR-7B (Microsoft)
vibevoice_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "packaging",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers>=4.51.3",
        "accelerate",
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "numba>=0.57.0",
        "llvmlite>=0.40.0",
        "pydub",
        "tqdm",
    )
    .run_commands("pip install git+https://github.com/microsoft/VibeVoice.git")
)

app = modal.App("tenepal", image=tenepal_image)

# Persistent volume for model weights (survives container restarts)
model_cache = modal.Volume.from_name("tenepal-models", create_if_missing=True)
CACHE_DIR = "/tenepal-models"
MODAL_TIMEOUT_S = int(os.getenv("TENEPAL_MODAL_TIMEOUT_S", "5400"))

# Finetuned NAH model volume (from tenepal_whisper_train.py training pipeline)
nah_model_vol = modal.Volume.from_name("tenepal-data", create_if_missing=True)
NAH_MODEL_DIR = "/nah-data"

# LLM for IPA→text fallback with few-shot examples from Nahuatl codices
LLM_FALLBACK_ENABLED = True
# Skip Mistral API (rate limited) — use HuggingFace Qwen2-1.5B directly
SKIP_MISTRAL_API = True

# Few-shot examples from Zacatlan-Tepetzintla Nahuatl transcriptions
NAHUATL_FEW_SHOT = """Examples of IPA to Nahuatl orthography:

IPA: t a k a t i
Text: Tahkahtih

IPA: k e m a
Text: Kemah

IPA: n o tʃ i k i w a w a ts a
Text: Nochi kiwahwatsa

IPA: tɬ a a l p a s i tɬ
Text: Tlaalpasitl

IPA: t i tɬ a o k o j a
Text: Titlaokoyah

IPA: k s i w i tɬ
Text: Xiwitl

IPA: t o tɬ a k w a l
Text: Totlakwal

IPA: tɬ a k p a k
Text: Tlakpak

IPA: k o s i tɬ
Text: Kohsitl

IPA: n i k a n t i m a t i
Text: Nikan timatih

IPA: i k w a k p e w a
Text: Ihkwak pewah

IPA: ʔ a tɬ
Text: Atl (water)

IPA: k a ʎ i
Text: Calli (house)

IPA: t e o tɬ
Text: Teotl (god)
"""

# Few-shot examples for Yucatec Maya (from linguistic sources)
MAYA_FEW_SHOT = """Examples of IPA to Yucatec Maya orthography:

IPA: k i n w a ʔ a l i k
Text: Kin wa'alik (I say)

IPA: b a ʔ a x
Text: Ba'ax (what)

IPA: t e ʔ e l o ʔ
Text: Te'elo' (there)

IPA: m a ʔ a l o b
Text: Ma'alob (good)

IPA: k ʼ i n
Text: K'in (sun/day)

IPA: ts ʼ o k
Text: Ts'ok (finished)

IPA: p ʼ a a t
Text: P'aat (wait)

IPA: tʃ ʼ u p
Text: Ch'up (woman)

IPA: h u n t u l
Text: Huntul (one person)

IPA: k a ʔ a n a l
Text: Ka'anal (high)
"""


# Global HuggingFace model cache for LLM fallback
_hf_llm_model = None
_hf_llm_tokenizer = None


def _init_hf_llm():
    """Lazy-init HuggingFace LLM for fallback when Mistral rate-limited."""
    global _hf_llm_model, _hf_llm_tokenizer
    if _hf_llm_model is not None:
        return True
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2-1.5B-Instruct"
        print(f"  [llm] Loading HuggingFace fallback: {model_name}")
        _hf_llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _hf_llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"  [llm] HuggingFace fallback ready")
        return True
    except Exception as e:
        print(f"  [llm] HuggingFace init failed: {e}")
        return False


def _hf_transcribe_ipa(ipa_sequence: str, system_prompt: str) -> str | None:
    """Use local HuggingFace model to transcribe IPA."""
    global _hf_llm_model, _hf_llm_tokenizer
    if _hf_llm_model is None:
        if not _init_hf_llm():
            return None
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"IPA: {ipa_sequence}"},
        ]
        text = _hf_llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _hf_llm_tokenizer([text], return_tensors="pt").to(_hf_llm_model.device)
        outputs = _hf_llm_model.generate(
            **inputs, max_new_tokens=64, do_sample=False, pad_token_id=_hf_llm_tokenizer.eos_token_id
        )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        result = _hf_llm_tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Clean up result
        result = result.split("\n")[0].strip().strip('"\'').strip('*').strip('`')
        for prefix in ["Text:", "text:", "Answer:", "Output:", "Transcription:"]:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()
        if result == "?" or not result or len(result) > 100:
            return None
        return result
    except Exception as e:
        print(f"  [llm-hf] error: {e}")
        return None


def llm_transcribe_ipa(ipa_sequence: str, lang_hint: str = "unknown") -> str | None:
    """Use Mistral to transcribe IPA phonemes to readable text.

    Uses few-shot examples from Nahuatl/Maya codices for better accuracy.
    Falls back to HuggingFace model on rate limit (429).
    Returns transcribed text or None if LLM unavailable/fails.
    """
    if not LLM_FALLBACK_ENABLED:
        return None

    # Select few-shot examples based on language hint
    if lang_hint in ("nah", "nahuatl"):
        few_shot = NAHUATL_FEW_SHOT
    elif lang_hint in ("may", "maya", "yucatec"):
        few_shot = MAYA_FEW_SHOT
    else:
        # Use both for unknown languages
        few_shot = NAHUATL_FEW_SHOT + "\n" + MAYA_FEW_SHOT

    system_prompt = f"""You transcribe IPA to orthography. Output ONLY the transcribed text, nothing else.
No explanations, no comments, no "Here is", no quotes. Just the word(s).

Rules:
- ʔ = glottal stop → ' (Maya) or h (Nahuatl)
- tɬ = lateral affricate → tl (Nahuatl)
- Ejectives kʼ tʼ tsʼ tʃʼ pʼ → k' t' ts' ch' p' (Maya)
- tʃ → ch
- ʃ → sh or x
- ŋ → ng or n

{few_shot}"""

    # Try Mistral first (unless skipped due to rate limits)
    api_key = os.getenv("MISTRAL_API_KEY")
    rate_limited = False

    if api_key and not SKIP_MISTRAL_API:
        try:
            from mistralai import Mistral

            client = Mistral(api_key=api_key)
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"IPA: {ipa_sequence}"},
                ],
            )
            result = response.choices[0].message.content.strip()
            # Take only first line, remove quotes, markdown, and extra whitespace
            result = result.split("\n")[0].strip().strip('"\'').strip('*').strip('`')
            # Remove common prefixes the model might add
            for prefix in ["Text:", "text:", "Answer:", "Output:", "Transcription:"]:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].strip()
            if result == "?" or not result or len(result) > 100:
                return None
            return result
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                rate_limited = True
                print(f"  [llm] Mistral rate limited, trying HuggingFace fallback")
            else:
                print(f"  [llm] error: {e}")
                return None

    # Fallback to HuggingFace if no API key, rate limited, or Mistral skipped
    if not api_key or rate_limited or SKIP_MISTRAL_API:
        return _hf_transcribe_ipa(ipa_sequence, system_prompt)

    return None

# ---------------------------------------------------------------------------
# Embedded Tenepal engine (no git clone needed)
# ---------------------------------------------------------------------------

SPACING_MODIFIERS = frozenset("ːʲʰˤʼ")

ALLOPHONE_MAP = {
    "ð": "d", "β": "b", "ɣ": "ɡ", "g": "ɡ",  # ASCII g -> IPA ɡ
    "ɸ": "f", "ɻ": "ɾ", "r": "ɾ",
}

PROFILES = {
    "nah": {
        "name": "Nahuatl", "priority": 10, "threshold": 0.0,
        "markers": {
            "ʔ": 1.0, "ɬ": 1.0, "kʷ": 1.0, "tɬ": 1.0,
            "tɕ": 0.5, "tɕʰ": 0.5, "tʂ": 0.5, "ts": 0.5,
            "t͡ɕ": 0.5, "t͡ʃʲ": 0.5, "tʃ": 0.5,
            "ɕ": 0.5, "ʂ": 0.5, "ʃ": 0.4,
            "k̟ʲ": 0.7, "lʲ": 0.4, "tʲ": 0.4,
        },
        "negative": {"b": 0.5, "d": 0.5, "ɡ": 0.5, "f": 0.5, "v": 0.5, "ʒ": 0.5},
    },
    "spa": {
        "name": "Spanish", "priority": 1, "threshold": 2.25,
        "markers": {"b": 0.20, "d": 0.05, "ɡ": 0.20, "ɲ": 0.80, "ɾ": 0.40},
        "negative": {},
    },
    "eng": {
        "name": "English", "priority": 1, "threshold": 4.10,
        "markers": {"ɹ": 1.0, "æ": 0.8, "ʌ": 0.7, "dʒ": 0.6, "w": 0.3},
        "negative": {},
    },
    "may": {
        "name": "Yucatec Maya", "priority": 9, "threshold": 2.0,
        "markers": {
            "kʼ": 1.0, "tsʼ": 1.0, "tʃʼ": 1.0, "pʼ": 0.9, "tʼ": 0.9,
            "ɓ": 0.9,
        },
        "negative": {
            "b": 0.5, "d": 0.5, "ɡ": 0.5, "f": 0.5, "v": 0.5, "ʒ": 0.5,
            "tɬ": 1.5, "kʷ": 1.5, "ɬ": 1.5,
            "ɹ": 0.6, "æ": 0.5, "ʌ": 0.4, "ð": 0.4, "θ": 0.3,
        },
    },
}

def _load_spa_wordlist() -> set[str]:
    """Load Spanish wordlist from data file (12K+ words from frequency corpus)."""
    import unicodedata
    # Try Modal path first, then local development path
    for candidate in [Path("/data/tenepal/spa_wordlist.txt"),
                      Path(__file__).parent / "src/tenepal/data/spa_wordlist.txt"]:
        if candidate.exists():
            wordlist_path = candidate
            break
    else:
        wordlist_path = None
    words = set()
    if wordlist_path:
        for line in wordlist_path.read_text().splitlines():
            w = line.strip().lower()
            if w and len(w) >= 2:
                words.add(w)
    if not words:
        # Minimal fallback if file not found (e.g. bare Modal without mount)
        words = {"de", "que", "no", "la", "el", "en", "es", "por", "con", "para",
                 "una", "los", "del", "las", "pero", "como", "más", "todo", "ya",
                 "su", "le", "si", "ser", "este", "ha", "era", "muy", "también"}
    return words

SPA_COMMON = _load_spa_wordlist()

NAH_LEXICON = [
    {"word": "koali",  "ipa": ["k","o","a","l","i"]},
    {"word": "tlein",  "ipa": ["tɬ","e","i","n"]},
    {"word": "ken",    "ipa": ["k","e","n"]},
    {"word": "amo",    "ipa": ["a","m","o"]},
    {"word": "kena",   "ipa": ["k","e","n","a"]},
    {"word": "nikan",  "ipa": ["n","i","k","a","n"]},
    {"word": "axkan",  "ipa": ["a","ʃ","k","a","n"]},
    {"word": "teotl",  "ipa": ["t","e","o","t","l"]},
    {"word": "atl",    "ipa": ["a","t","l"]},
    {"word": "calli",  "ipa": ["k","a","l","i"]},
]

LANG_MAP = {
    "es": "spa", "en": "eng", "de": "deu", "fr": "fra", "it": "ita",
    "pt": "spa", "ca": "spa", "gl": "spa",
}

LANG_NAMES = {
    "nah": "Nahuatl", "spa": "Español", "eng": "English", "may": "Maya", "lat": "Latin",
    "deu": "Deutsch", "fra": "Français", "ita": "Italiano",
    "other": "???",
}

DEFAULT_LATIN_KEYWORDS = {
    "ego", "te", "baptizo", "nomine", "patris", "filii", "spiritus", "sancti",
    "amen", "dominus", "deus", "pater", "noster", "maria", "gratia",
}

DEFAULT_MAYA_HINT_WORDS = {
    "ba", "baal", "kes", "kes'", "uchala", "takin", "k'an", "k'an", "lo'",
}

NAH_TEXT_MARKERS = {
    "tl", "tz", "hu", "cuauh", "quauh", "teotl", "tlatoani", "tenochtitlan",
    "mexica", "motecuhzoma", "xicotencatl", "tzintli", "itz", "xoch", "coatl",
}

MAY_TEXT_MARKERS = {
    "k'", "ts'", "ch'", "halach", "ajaw", "kukul", "k'uk", "yuum", "uay",
    "kaax", "balam", "chan", "k'in", "kab",
}

# NAH-exclusive morpheme patterns for fuzzy text matching
# These patterns are highly specific to Nahuatl and rarely appear in Spanish
_NAH_EXCLUSIVE_RX = re.compile(
    r"nik[aeiou]|tik[aeiou]|mik[aeiou]"  # verb subject+object prefixes
    r"|tict|ticm|ticn|nicn|nicm|nict"     # prefix combos (8+4 NAH, 0 SPA)
    r"|ichi|pil|neci|mati|caci|pano"       # common NAH roots
    r"|mict|tequi|chiua|cipac|mani"        # to kill / work / make / crocodile / extend
    r"|tonan|nican|tlaco|acal|cali"        # mother / here / middle / boat / house
    r"|ichan|ipan|ican"                    # locative morphemes
    r"|isti|lis|ilia|ilis"                 # nominalization + applicative suffixes
    r"|teca|tlaca|xica|naca|meca|titec"    # ethnic/gentilicio suffixes
    r"|teka|tlaka|xika|naka|meka|titek"    # k-variants (modern orthography)
    r"|iske|isce|monec|monek|oneki|oneci"  # verb plural + monequi (to need)
    r"|otl|utl|pach"                       # absolutive suffixes + pachoa
)


# ---------------------------------------------------------------------------
# EQ Config — all tunable parameters in one place
# ---------------------------------------------------------------------------
DEFAULT_EQ = {
    # --- Prosody voter (NAH vs SPA) ---
    "prosody_enabled": False,        # Enable prosody-based language voting
    "prosody_weight": 0.5,           # How much prosody score shifts lang confidence (0-1)
    "prosody_threshold": 0.65,       # Min prosody P(NAH) to vote NAH (else SPA)
    "prosody_min_dur": 0.3,          # Min segment duration for prosody analysis
    # --- Speaker prior ---
    "speaker_prior_min_segments": 3, # Min segments before speaker lock
    "speaker_prior_min_ratio": 0.6,  # Min majority ratio for speaker lock
    "speaker_prior_strong": False,   # Also override SPA↔NAH (not just OTH)
    # --- Ejective detection ---
    "ejective_min_count": 2,         # Min consensus ejectives for Maya
    "ejective_maya_boost": 0.3,      # Score boost per ejective
    "ejective_strict": False,        # Require ≥3 + block when NAH phonemes
    # --- tɬ detection ---
    "tl_confidence_threshold": 0.0,  # Min tɬ confidence to trigger NAH override
    # --- SPA reclaim ---
    "spa_reclaim_min_hits": 2,       # Min Spanish word hits
    "spa_reclaim_min_ratio": 0.30,   # Min Spanish word ratio
    # --- Noise gate ---
    "noise_gate_enabled": False,     # Tag short segments as OTH
    "noise_gate_max_s": 0.4,        # Max duration for noise gate
    # --- IPA NAH override (EXPERIMENTAL — harmful, off by default) ---
    "ipa_nah_override": False,       # Demote NAH when IPA lacks NAH phonemes
    # --- Language confidence ---
    "lang_conf_threshold": 0.6,      # OTH fallback if lang confidence below this
    "low_conf_threshold": 0.35,      # LOW tag threshold
    # --- FT SPA guard (EXPERIMENTAL — harmful, off by default) ---
    "ft_spa_guard": False,           # Detect Spanish in FT output
    # --- UNK reject gate ---
    "unk_gate_enabled": False,       # Tag low-confidence segments as UNK (reject option)
    "unk_gate_threshold": 0.5,       # Confidence below this → UNK instead of forced label
    # --- Two-pass speaker prior ---
    "two_pass_prior": False,         # Re-count speaker langs using IPA phoneme evidence before prior
    # --- FT-first ---
    "ft_first": False,               # Run NAH finetuned Whisper BEFORE speaker-prior on ALL segments
    # --- Whisper uncertain → preserve IPA lang ---
    "whisper_uncertain_ipa": False,   # Rescue updates text but preserves IPA-based language classification
    # --- Speaker prior reset ---
    "prior_reset": False,            # Re-run speaker prior after two-pass + FT-first corrections
    # --- dʒ+tɕ cross-check NAH marker ---
    "dj_tc_marker": False,           # w2v2 dʒ + allo tɕ agreement → NAH (if no SPA function words)
    # --- Allosaurus full-track mode ---
    "allo_full_track": False,        # Run Allosaurus ONCE on full vocals, map phones to segments by timestamp
    "allo_full_track_emit": 2.0,     # CTC emission rate for full-track (1.0=default=useless, 2.0=good)
    # --- Overlap detection gate ---
    "overlap_gate": False,           # Detect simultaneous speakers via bimodal F0
    "overlap_f0_min_ratio": 0.20,    # Min fraction of voiced frames in each F0 band
    "overlap_hnr_threshold": 10.0,   # HNR below this suggests overlap (single speaker ~15-25)
    "overlap_morphology_weight": 0.3, # Downweight morphology match when overlap detected
    "overlap_separation": False,      # Run MossFormer2 voice separation on overlap turns
    # --- Overlap damping (full stack) ---
    "overlap_damping": False,         # Enable full overlap damping stack
    "overlap_conf_cap": 0.5,         # Max confidence for overlap segments
    "overlap_ipa_weight": 0.3,       # Downweight IPA scores on mixed signal (0=ignore, 1=trust)
}

# --- Prosody scorer (hardcoded LogReg from Hernán-1-3 training) ---
_PROSODY_FEATURES = [
    "f0_mean", "f0_std", "f0_range", "f0_cv", "f0_slope",
    "int_std", "int_range", "npvi", "vu_ratio", "hnr",
    "speech_rate", "jitter", "shimmer", "duration",
]
_PROSODY_SCALER_MEAN = [198.523003, 36.201773, 137.885346, 0.180908, -8.229475, 9.849296, 40.265667, 3.714447, 0.55736, 8.556309, 9.778094, 0.029286, 0.144554, 1.410989]
_PROSODY_SCALER_STD = [70.667195, 28.139811, 95.55815, 0.127939, 79.032094, 1.987586, 8.592765, 1.264831, 0.161312, 3.898336, 3.570998, 0.009631, 0.039196, 1.139702]
_PROSODY_COEF = [-0.159918, 0.505234, 0.265655, -0.577726, -0.048422, -0.265351, 0.068277, 0.643465, -0.290301, 0.98666, -0.174977, 0.209238, -0.059956, 0.426405]
_PROSODY_INTERCEPT = 1.020153


def extract_prosody_features(snd, start_s: float, end_s: float) -> dict | None:
    """Extract prosodic features from a Parselmouth Sound window. Returns None if too short."""
    import numpy as np
    dur = end_s - start_s
    if dur < 0.15:
        return None
    try:
        import parselmouth
        from parselmouth.praat import call
        window = snd.extract_part(start_s, end_s, parselmouth.WindowShape.HAMMING, 1.0, False)
    except Exception:
        return None

    # F0
    try:
        pitch = window.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0_vals = [v for v in (pitch.get_value_at_time(t) for t in pitch.xs()) if not np.isnan(v) and v > 0]
    except Exception:
        f0_vals = []
    if len(f0_vals) < 3:
        return None
    f0 = np.array(f0_vals)

    # Intensity
    try:
        intensity = window.to_intensity(minimum_pitch=75, time_step=0.01)
        int_vals = [v for v in (intensity.get_value(t) for t in intensity.xs()) if not np.isnan(v)]
    except Exception:
        return None
    if len(int_vals) < 3:
        return None

    f0_mean = float(np.mean(f0))
    f0_std = float(np.std(f0))
    f0_cv = f0_std / f0_mean if f0_mean > 0 else 0
    t_norm = np.linspace(0, 1, len(f0))
    int_arr = np.array(int_vals)

    # nPVI
    npvi = 0.0
    if len(int_vals) > 1:
        diffs = [abs(int_vals[i-1] - int_vals[i]) / ((int_vals[i-1] + int_vals[i]) / 2)
                 for i in range(1, len(int_vals)) if (int_vals[i-1] + int_vals[i]) > 0]
        npvi = float(np.mean(diffs) * 100) if diffs else 0.0

    # Voiced/unvoiced ratio
    try:
        pp = window.to_pitch(time_step=0.01)
        n_fr = call(pp, "Get number of frames")
        voiced = sum(1 for fi in range(1, n_fr + 1) if call(pp, "Get value in frame", fi, "Hertz") > 0)
        vu_ratio = voiced / n_fr if n_fr > 0 else 0
    except Exception:
        vu_ratio = 0

    # HNR
    try:
        hnr_obj = call(window, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(hnr_obj, "Get mean", 0, 0)
        if np.isnan(hnr):
            hnr = 0.0
    except Exception:
        hnr = 0.0

    # Speech rate (state transitions / duration)
    states = []
    for t in intensity.xs():
        iv = intensity.get_value(t)
        if np.isnan(iv) or iv < 50:
            states.append(0)
        else:
            pv = pitch.get_value_at_time(t)
            states.append(1 if (np.isnan(pv) or pv <= 0) else 2)
    events = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])

    # Jitter + shimmer
    try:
        pp2 = call(window, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pp2, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([window, pp2], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        if np.isnan(jitter): jitter = 0.0
        if np.isnan(shimmer): shimmer = 0.0
    except Exception:
        jitter, shimmer = 0.0, 0.0

    return {
        "f0_mean": f0_mean, "f0_std": f0_std,
        "f0_range": float(np.max(f0) - np.min(f0)),
        "f0_cv": f0_cv,
        "f0_slope": float(np.polyfit(t_norm, f0, 1)[0]),
        "int_std": float(np.std(int_arr)),
        "int_range": float(np.max(int_arr) - np.min(int_arr)),
        "npvi": npvi, "vu_ratio": vu_ratio, "hnr": hnr,
        "speech_rate": events / dur if dur > 0 else 0,
        "jitter": jitter, "shimmer": shimmer, "duration": dur,
    }


def detect_overlap(snd, start_s: float, end_s: float,
                    f0_min_ratio: float = 0.20,
                    hnr_threshold: float = 10.0) -> tuple[bool, dict]:
    """Detect simultaneous speakers via bimodal F0 distribution + low HNR.

    Returns (is_overlap, details_dict).
    Uses Parselmouth on the vocals track window.
    """
    import numpy as np
    dur = end_s - start_s
    if dur < 0.5:
        return False, {"reason": "too_short"}
    try:
        import parselmouth
        from parselmouth.praat import call
        window = snd.extract_part(start_s, end_s,
                                  parselmouth.WindowShape.HAMMING, 1.0, False)
    except Exception:
        return False, {"reason": "extract_failed"}

    # F0 extraction
    try:
        pitch = window.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=400)
        f0_all = [pitch.get_value_at_time(t) for t in pitch.xs()]
        f0_voiced = np.array([v for v in f0_all if not np.isnan(v) and v > 0])
    except Exception:
        return False, {"reason": "pitch_failed"}

    if len(f0_voiced) < 10:
        return False, {"reason": "too_few_voiced"}

    # Bimodal F0: male 80-180Hz, female 180-350Hz
    low = f0_voiced[(f0_voiced > 80) & (f0_voiced < 180)]
    high = f0_voiced[(f0_voiced > 180) & (f0_voiced < 350)]
    low_ratio = len(low) / len(f0_voiced)
    high_ratio = len(high) / len(f0_voiced)
    bimodal = low_ratio > f0_min_ratio and high_ratio > f0_min_ratio

    # HNR: overlap degrades harmonicity
    try:
        hnr_obj = call(window, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(hnr_obj, "Get mean", 0, 0)
        if np.isnan(hnr):
            hnr = 0.0
    except Exception:
        hnr = 0.0
    low_hnr = hnr < hnr_threshold

    # Intensity variance
    try:
        intensity = window.to_intensity(minimum_pitch=75, time_step=0.01)
        int_vals = np.array([intensity.get_value(t) for t in intensity.xs()])
        int_vals = int_vals[~np.isnan(int_vals)]
        int_cv = float(np.std(int_vals) / np.mean(int_vals)) if len(int_vals) > 3 and np.mean(int_vals) > 0 else 0
    except Exception:
        int_cv = 0
    high_int_var = int_cv > 0.15

    # Combined score: bimodal F0 is the primary signal, others are supporting
    score = (bimodal * 0.5) + (low_hnr * 0.3) + (high_int_var * 0.2)
    # Require bimodal F0 as minimum — low HNR alone is too noisy
    is_overlap = bimodal and score >= 0.5

    details = {
        "bimodal": bimodal, "low_ratio": round(low_ratio, 3),
        "high_ratio": round(high_ratio, 3),
        "f0_mean_low": round(float(np.mean(low)), 1) if len(low) > 0 else 0,
        "f0_mean_high": round(float(np.mean(high)), 1) if len(high) > 0 else 0,
        "hnr": round(hnr, 1), "low_hnr": low_hnr,
        "int_cv": round(int_cv, 3), "high_int_var": high_int_var,
        "score": round(score, 2),
    }
    return is_overlap, details


def score_prosody_nah(features: dict) -> float:
    """Score a segment's prosody for NAH probability (0-1). Uses hardcoded LogReg."""
    import numpy as np
    x = []
    for i, fname in enumerate(_PROSODY_FEATURES):
        val = features.get(fname, 0.0)
        scaled = (val - _PROSODY_SCALER_MEAN[i]) / _PROSODY_SCALER_STD[i] if _PROSODY_SCALER_STD[i] > 0 else 0
        x.append(scaled)
    logit = sum(c * v for c, v in zip(_PROSODY_COEF, x)) + _PROSODY_INTERCEPT
    prob_nah = 1.0 / (1.0 + np.exp(-logit))
    return float(prob_nah)


def _normalize_word_ascii(text: str) -> str:
    text = text.lower()
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _resolve_data_path(rel_path: str) -> Path | None:
    """Resolve data file path — Modal (/data/tenepal/) or local (src/...)."""
    # Modal image path
    basename = Path(rel_path).name
    modal_path = Path("/data/tenepal") / basename
    if modal_path.exists():
        return modal_path
    # Local development path
    local_path = Path(__file__).parent / rel_path
    if local_path.exists():
        return local_path
    return None


def _load_json_list(rel_path: str, key: str) -> set[str]:
    path = _resolve_data_path(rel_path)
    if not path:
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    values = data.get(key)
    if not isinstance(values, list):
        return set()
    return {str(v).lower() for v in values if isinstance(v, str)}


LATIN_KEYWORDS = _load_json_list("src/tenepal/data/lat_lexicon.json", "keywords") or DEFAULT_LATIN_KEYWORDS
MAYA_HINT_WORDS = DEFAULT_MAYA_HINT_WORDS


# ---------------------------------------------------------------------------
# Enhanced Nahuatl Lexicon (corpus-based, min_freq filtered)
# ---------------------------------------------------------------------------
def _load_nah_lexicon_merged(min_freq: int = 50) -> list[dict]:
    """Load merged NAH lexicon with frequency filtering."""
    path = _resolve_data_path("src/tenepal/data/nah_lexicon_merged.json")
    if not path:
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    # Keep curated entries unconditionally, filter corpus by freq
    return [
        e for e in entries
        if e.get("source") == "curated"
        or e.get("freq") is None
        or e.get("freq", 0) >= min_freq
    ]


# ---------------------------------------------------------------------------
# Voice onset/offset trimming — removes silence padding from diarization segments
# ---------------------------------------------------------------------------
def trim_to_voice(
    audio_array,
    sr: int,
    start_s: float,
    end_s: float,
    margin: float = 0.3,
    threshold_db: float = 15.0,
) -> tuple[float, float, dict]:
    """Find exact voice onset/offset within a diarization segment.

    Uses intensity contour analysis to detect when speech actually starts/ends,
    trimming silence padding that causes garbage phoneme output.

    Args:
        audio_array: Audio as numpy array (mono)
        sr: Sample rate
        start_s: Diarization start time (seconds)
        end_s: Diarization end time (seconds)
        margin: Extra context for analysis (seconds)
        threshold_db: dB below peak to consider as voice activity

    Returns:
        (onset_s, offset_s, stats) — trimmed absolute timestamps and stats dict
    """
    import numpy as np
    import parselmouth

    # Convert numpy array to parselmouth Sound
    audio = audio_array.astype(np.float64)
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0
    snd = parselmouth.Sound(audio, sampling_frequency=float(sr))

    # Extract chunk with margin for analysis
    chunk_start = max(0, start_s - margin)
    chunk_end = min(snd.xmax, end_s + margin)

    try:
        chunk = snd.extract_part(from_time=chunk_start, to_time=chunk_end)
        intensity = chunk.to_intensity(time_step=0.01)
    except Exception:
        # Too short for intensity analysis
        return start_s, end_s, {"trimmed": False, "reason": "too_short"}

    times = intensity.xs()
    values = np.array([intensity.get_value(t) for t in times])

    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return start_s, end_s, {"trimmed": False, "reason": "no_valid_intensity"}

    # Peak and threshold
    peak_db = np.nanmax(values)
    thresh = peak_db - threshold_db

    above = np.where(values > thresh)[0]
    if len(above) == 0:
        return start_s, end_s, {"trimmed": False, "reason": "below_threshold"}

    # Convert to absolute time
    onset_abs = times[above[0]] + chunk_start
    offset_abs = times[above[-1]] + chunk_start

    # Safety: don't expand beyond original + 100ms
    onset_abs = max(onset_abs, start_s - 0.1)
    offset_abs = min(offset_abs, end_s + 0.1)

    # Minimum segment duration: 80ms
    if offset_abs - onset_abs < 0.08:
        return start_s, end_s, {"trimmed": False, "reason": "too_short_after_trim"}

    stats = {
        "trimmed": True,
        "trim_start_ms": round((start_s - onset_abs) * 1000),
        "trim_end_ms": round((end_s - offset_abs) * 1000),
        "original_ms": round((end_s - start_s) * 1000),
        "trimmed_ms": round((offset_abs - onset_abs) * 1000),
    }
    return onset_abs, offset_abs, stats


def count_acoustic_events(audio_chunk, sr: int, intensity_threshold_db: float = 50.0) -> int:
    """Count acoustic phone events using Parselmouth intensity + pitch analysis.

    Counts transitions between VOICED, UNVOICED, and SILENCE states.
    Each transition represents approximately one phoneme boundary.

    Args:
        audio_chunk: Audio samples as numpy array
        sr: Sample rate in Hz
        intensity_threshold_db: Intensity below which frames are SILENCE

    Returns:
        Number of acoustic events (state transitions)
    """
    import numpy as np
    import parselmouth

    # Convert to Parselmouth Sound
    audio = audio_chunk.astype(np.float64)
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0
    snd = parselmouth.Sound(audio, sampling_frequency=float(sr))

    # Get intensity and pitch
    try:
        intensity = snd.to_intensity(time_step=0.01)
        pitch = snd.to_pitch(time_step=0.01)
    except Exception:
        return 0

    times = intensity.xs()
    if len(times) < 3:
        return 0

    # Classify each frame: SILENCE (0), UNVOICED (1), VOICED (2)
    states = []
    for t in times:
        int_val = intensity.get_value(t)
        if np.isnan(int_val) or int_val < intensity_threshold_db:
            states.append(0)  # SILENCE
        else:
            pitch_val = pitch.get_value_at_time(t)
            if np.isnan(pitch_val) or pitch_val <= 0:
                states.append(1)  # UNVOICED
            else:
                states.append(2)  # VOICED

    # Count state transitions (= acoustic event boundaries)
    events = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            events += 1

    return events


def detect_tl_in_window(
    snd,
    t_start: float,
    t_end: float,
    min_friction_dur: float = 0.04,
    voicing_threshold: float = 0.30,
    friction_ratio_threshold: float = 0.25,
) -> list[dict]:
    """Detect tɬ candidates inside a time window using acoustic cues."""
    import numpy as np

    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception:
        return []

    duration = t_end - t_start
    if duration < 0.05 or duration > 30.0:
        return []

    try:
        window = snd.extract_part(
            t_start, t_end, parselmouth.WindowShape.RECTANGULAR, 1.0, False
        )
    except Exception:
        return []

    if window.get_total_duration() < 0.05:
        return []

    try:
        intensity = window.to_intensity(minimum_pitch=100, time_step=0.005)
    except Exception:
        return []

    times = intensity.xs()
    values = [intensity.get_value(t) for t in times]
    if not values or all(v is None for v in values):
        return []

    values = [v if v is not None else 0.0 for v in values]
    mean_int = np.mean(values)
    detections: list[dict] = []

    for i in range(1, len(values) - 2):
        if values[i] < mean_int - 6 and values[i + 1] > values[i]:
            burst_time = times[i] + t_start
            analysis_start = times[i] + window.get_start_time()
            analysis_end = min(analysis_start + 0.12, window.get_end_time())
            if analysis_end - analysis_start < min_friction_dur:
                continue

            try:
                post_burst = window.extract_part(
                    analysis_start, analysis_end,
                    parselmouth.WindowShape.HAMMING, 1.0, False,
                )
            except Exception:
                continue

            if post_burst.get_total_duration() < min_friction_dur:
                continue

            try:
                pitch = post_burst.to_pitch(time_step=0.005, pitch_floor=75, pitch_ceiling=500)
                n_frames = call(pitch, "Get number of frames")
                if n_frames > 0:
                    voiced = sum(
                        1
                        for fi in range(1, n_frames + 1)
                        if call(pitch, "Get value in frame", fi, "Hertz") > 0
                    )
                    voicing_ratio = voiced / n_frames
                else:
                    voicing_ratio = 0.5
            except Exception:
                voicing_ratio = 0.5

            try:
                spectrum = post_burst.to_spectrum()
                total_e = call(spectrum, "Get band energy", 0, 0)
                friction_e = call(spectrum, "Get band energy", 3000, 8000)
                if total_e <= 0:
                    continue
                friction_ratio = friction_e / total_e
            except Exception:
                continue

            try:
                centroid = call(spectrum, "Get centre of gravity...", 1)
            except Exception:
                centroid = 5000

            is_unvoiced = voicing_ratio < voicing_threshold
            has_friction = friction_ratio > friction_ratio_threshold
            is_lateral = 2500 < centroid < 5500
            unvoiced_dur = analysis_end - analysis_start
            long_enough = unvoiced_dur > min_friction_dur

            if is_unvoiced and has_friction and long_enough:
                if is_lateral:
                    phone = "tɬ"
                    confidence = min(1.0, (1 - voicing_ratio) * friction_ratio * 2)
                elif centroid > 5500:
                    phone = "tʃ"
                    confidence = min(1.0, (1 - voicing_ratio) * friction_ratio * 1.5)
                else:
                    phone = "tɬ?"
                    confidence = min(1.0, (1 - voicing_ratio) * friction_ratio)

                detections.append(
                    {
                        "time": round(burst_time, 3),
                        "phone": phone,
                        "confidence": round(confidence, 3),
                        "voicing_ratio": round(voicing_ratio, 3),
                        "friction_ratio": round(friction_ratio, 3),
                        "centroid_hz": round(centroid, 0),
                        "duration_ms": round(unvoiced_dur * 1000, 1),
                    }
                )

    return detections


def detect_tl_for_segment(snd, t_start: float, t_end: float) -> dict:
    """Return tɬ segment summary for hard NAH override."""
    detections = detect_tl_in_window(snd, t_start, t_end)
    tl_candidates = [d for d in detections if d.get("phone") in {"tɬ", "tɬ?"}]
    if not tl_candidates:
        return {"tl_detected": False, "tl_confidence": 0.0, "tl_events": 0}

    best_conf = max(float(d.get("confidence", 0.0)) for d in tl_candidates)
    return {
        "tl_detected": True,
        "tl_confidence": round(best_conf, 3),
        "tl_events": len(tl_candidates),
    }


# IPA normalization for lexicon matching
_IPA_SPACING_MODIFIERS = frozenset("ːʲʰˤʼ")
_IPA_ALLOPHONE_MAP = {"ð": "d", "β": "b", "ɣ": "ɡ", "g": "ɡ", "ɸ": "f", "ɻ": "ɾ", "r": "ɾ"}


def _normalize_ipa_token(tok: str) -> str:
    """Normalize a single IPA token (strip modifiers, map allophones)."""
    tok = "".join(c for c in tok if c not in _IPA_SPACING_MODIFIERS)
    return _IPA_ALLOPHONE_MAP.get(tok, tok)


def _check_nah_lexicon_enhanced(
    phonemes: list[str],
    lexicon: list[dict],
    threshold: float = 0.75,
) -> tuple[str | None, float, list[str]]:
    """Enhanced lexicon check with fuzzy matching.

    Returns: (best_word, best_score, all_matched_words)
    """
    if not phonemes or not lexicon:
        return None, 0.0, []

    normalized = [_normalize_ipa_token(p) for p in phonemes]
    input_len = len(normalized)

    matches = []
    best_word = None
    best_score = 0.0

    for entry in lexicon:
        pattern = [_normalize_ipa_token(p) for p in entry["ipa"]]
        pattern_len = len(pattern)

        # Sliding window: check if pattern appears as subsequence
        for start in range(max(1, input_len - pattern_len - 1)):
            window = normalized[start:start + pattern_len]
            if len(window) < 2:
                continue

            # Simple overlap score
            overlap = sum(1 for a, b in zip(window, pattern) if a == b)
            max_len = max(len(window), pattern_len)
            if max_len == 0:
                continue
            score = overlap / max_len

            if score >= threshold:
                matches.append(entry["word"])
                if score > best_score:
                    best_score = score
                    best_word = entry["word"]
                break  # One match per entry is enough

    return best_word, best_score, list(set(matches))


# Load enhanced lexicon at module level (min_freq=50 for ~140 entries)
# P0 fix: Lower min_freq from 50→3 to expand lexicon from ~140 to ~2000 entries
# This dramatically improves NAH detection for short/ambiguous segments
NAH_LEXICON_ENHANCED = _load_nah_lexicon_merged(min_freq=3)

# Build normalized NAH word set (4+ chars) for fuzzy text matching
_NAH_WORDS_4 = set()
for _e in NAH_LEXICON_ENHANCED:
    _w = _e.get("word", "")
    # Strip length marks and diacritics for matching
    _w = _w.replace(":", "").replace("ː", "")
    _w = "".join(
        c for c in unicodedata.normalize("NFD", _w)
        if unicodedata.category(c) != "Mn"
    ).lower()
    if len(_w) >= 4:
        _NAH_WORDS_4.add(_w)


def fuzzy_nah_text_check(text: str) -> bool:
    """Check if text has Nahuatl morphological patterns via fuzzy matching.

    Uses lexicon substring matching (4+ chars) and NAH-exclusive morphemes.
    Designed as supplement to _NAH_TEXT_RX for texts that lack exact pattern matches.

    Returns True if text is likely Nahuatl.
    """
    t = text.lower()
    if len(t) < 4:
        return False

    score = 0.0

    # 1. Lexicon word match (4+ chars — avoids short ambiguous hits)
    for w in _NAH_WORDS_4:
        if w in t:
            score += 0.35 if len(w) >= 5 else 0.20
            break  # one match is enough

    # 2. NAH-exclusive morphemes
    excl = _NAH_EXCLUSIVE_RX.findall(t)
    if excl:
        score += min(0.30, len(excl) * 0.15)

    # 3. Penalty: Spanish consonant clusters (bl, br, cr, dr, fl, fr, gl, gr, pl, pr, tr, str)
    spa_cl = len(re.findall(r"[bcdfgp][rl]|str|ntr", t))
    score -= spa_cl * 0.2

    return score >= 0.15


class PhonemeBackend(ABC):
    """Minimal backend interface for chunk-level phoneme-like recognition."""

    name = "base"

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def recognize_chunk(self, chunk, sample_rate: int) -> str | None:
        pass

    def cleanup(self) -> None:
        pass


class AllosaurusBackend(PhonemeBackend):
    """Allosaurus backend wrapper with volume cache restore/save.

    Supports adaptive blank-bias for recovering phonemes in unknown languages.
    """

    name = "allosaurus"

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self._recognizer = None
        self._blank_idx = None
        self._phone_list = None

    def initialize(self) -> None:
        import allosaurus

        allo_pkg_dir = Path(allosaurus.__file__).parent / "pretrained"
        allo_cache = Path(self.cache_dir) / "allosaurus" / "pretrained"

        if allo_cache.exists() and any(allo_cache.iterdir()):
            import shutil
            if allo_pkg_dir.exists():
                shutil.rmtree(allo_pkg_dir)
            shutil.copytree(allo_cache, allo_pkg_dir)
            print("Allosaurus: restored from volume cache")

        from allosaurus.app import read_recognizer

        self._recognizer = read_recognizer()

        # Cache blank index and phone list for bias decoding
        self._phone_list = self._recognizer.lm.inventory.unit.id_to_unit
        for idx, phone in self._phone_list.items():
            if phone == "<blk>":
                self._blank_idx = idx
                break

        if not allo_cache.exists() or not any(allo_cache.iterdir()):
            import shutil
            allo_cache.mkdir(parents=True, exist_ok=True)
            shutil.copytree(allo_pkg_dir, allo_cache, dirs_exist_ok=True)
            print("Allosaurus: saved to volume cache")

    def recognize_chunk(self, chunk, sample_rate: int, blank_bias: float = 0.0) -> str | None:
        """Recognize phonemes from audio chunk.

        Args:
            chunk: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            blank_bias: CTC blank bias reduction (0=default, 2-3=recommended for NAH)
        """
        import soundfile as sf
        import tempfile

        if self._recognizer is None:
            raise RuntimeError("Allosaurus backend not initialized")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk, sample_rate)
            try:
                # Use bias-adjusted decoding if blank_bias > 0
                if blank_bias > 0 and self._blank_idx is not None:
                    return self._recognize_with_bias(tmp.name, blank_bias)

                # Default: use standard Allosaurus recognize
                result = self._recognizer.recognize(tmp.name)
                return result if result else None
            except Exception:
                return None
            finally:
                os.unlink(tmp.name)

    def _recognize_with_bias(self, temp_path: str, blank_bias: float) -> str | None:
        """CTC decoding with reduced blank bias.

        Manipulates logits before argmax to reduce blank's advantage.
        """
        import numpy as np
        import torch
        from allosaurus.audio import read_audio
        from allosaurus.am.utils import move_to_tensor

        try:
            # Get acoustic features
            audio_allo = read_audio(temp_path)
            feat = self._recognizer.pm.compute(audio_allo)
            feats = np.expand_dims(feat, 0)
            feat_len = np.array([feat.shape[0]], dtype=np.int32)

            tensor_feat, tensor_feat_len = move_to_tensor(
                [feats, feat_len], self._recognizer.config.device_id
            )

            # Get log probabilities from acoustic model
            with torch.no_grad():
                tensor_lprobs = self._recognizer.am(tensor_feat, tensor_feat_len)

            if self._recognizer.config.device_id >= 0:
                lprobs = tensor_lprobs.cpu().numpy()[0]
            else:
                lprobs = tensor_lprobs.numpy()[0]

            # Apply blank bias: subtract from blank logit
            lprobs[:, self._blank_idx] -= blank_bias

            # Greedy CTC decode with collapse
            decoded_indices = np.argmax(lprobs, axis=1)

            # Convert to phones with CTC collapse
            phones = []
            prev_idx = -1
            for idx in decoded_indices:
                if idx != prev_idx:
                    if idx != self._blank_idx:
                        phone = self._phone_list.get(int(idx), f"?{idx}")
                        phones.append(phone)
                    prev_idx = idx

            return " ".join(phones) if phones else None

        except Exception:
            return None

    def recognize_full_track(self, wav_path: str, emit: float = 2.0) -> list[tuple[str, float, float]]:
        """Run Allosaurus ONCE on full audio with timestamps.

        Returns list of (phone, start_s, end_s) tuples. Uses next phone's start
        as current phone's end to avoid gaps from fixed window_size.

        Args:
            emit: CTC emission rate. Default 1.0 suppresses nearly all phones on
                  long files. 2.0 gives ~12 phones/sec on speech regions.
        """
        if self._recognizer is None:
            raise RuntimeError("Allosaurus backend not initialized")
        try:
            result = self._recognizer.recognize(wav_path, timestamp=True, emit=emit)
            if not result:
                return []
            raw = []
            for line in result.strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    raw.append((parts[2], float(parts[0])))
            # Use next phone start as current phone end
            phones = []
            for i, (phone, start) in enumerate(raw):
                end = raw[i + 1][1] if i + 1 < len(raw) else start + 0.03
                phones.append((phone, start, end))
            return phones
        except Exception as e:
            print(f"  [allo-full-track] Failed: {e}")
            return []


def map_full_track_phones(full_phones: list[tuple[str, float, float]],
                          start_t: float, end_t: float) -> str | None:
    """Map full-track Allosaurus phones to a segment by timestamp overlap.

    Returns space-separated IPA string for phones within the segment window.
    A phone is included if its midpoint falls within [start_t, end_t].
    """
    segment_phones = []
    for phone, ps, pe in full_phones:
        mid = (ps + pe) / 2
        if mid >= start_t and mid <= end_t:
            segment_phones.append(phone)
    return " ".join(segment_phones) if segment_phones else None


class Wav2Vec2Backend(PhonemeBackend):
    """Multilingual phoneme backend using facebook/wav2vec2-lv-60-espeak-cv-ft.

    Outputs IPA phonemes directly — no character-to-IPA mapping needed.
    Fine-tuned on CommonVoice with eSpeak phonemization, supports 100+ languages.
    Model size: ~1.26 GB.
    """

    name = "wav2vec2"

    MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

    def __init__(self, cache_dir: str = "") -> None:
        self._model = None
        self._processor = None
        self._sample_rate = 16000
        self._device = "cpu"
        self._cache_dir = cache_dir
        self._blank_id = 0

    def initialize(self) -> None:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        cache_path = f"{self._cache_dir}/wav2vec2-phoneme" if self._cache_dir else None

        self._processor = Wav2Vec2Processor.from_pretrained(
            self.MODEL_ID, cache_dir=cache_path
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            self.MODEL_ID, cache_dir=cache_path
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).eval()
        self._blank_id = int(getattr(self._model.config, "pad_token_id", 0) or 0)
        print(f"Wav2Vec2-Phoneme: {self.MODEL_ID} on {self._device}")

    @staticmethod
    def _clean_espeak(raw: str) -> str:
        """Strip eSpeak artifacts from wav2vec2-lv-60-espeak-cv-ft output.

        The model outputs eSpeak phoneme notation which is mostly IPA but
        includes tone digits (1-5), syllable dots (.), stress marks (ˈˌ),
        and length marks that we strip for clean IPA comparison.
        """
        import re as _re
        # Remove tone digits (eSpeak uses 1-5 for tone levels)
        cleaned = _re.sub(r'[1-5]', '', raw)
        # Remove syllable boundary dots (but keep ː length marker)
        cleaned = cleaned.replace('.', '')
        # Remove primary/secondary stress marks
        cleaned = cleaned.replace('ˈ', '').replace('ˌ', '')
        # Collapse whitespace from removals
        cleaned = _re.sub(r'\s+', ' ', cleaned).strip()
        # Remove empty tokens that result from stripping
        tokens = [t for t in cleaned.split() if t]
        return ' '.join(tokens)

    def recognize_chunk(self, chunk, sample_rate: int) -> str | None:
        import numpy as np
        import torch
        import torchaudio

        if self._model is None or self._processor is None:
            raise RuntimeError("Wav2Vec2 phoneme backend not initialized")

        if len(chunk) < sample_rate * 0.2:
            return None

        # Ensure float32 mono
        chunk_np = np.asarray(chunk, dtype=np.float32)
        if chunk_np.ndim > 1:
            chunk_np = chunk_np.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != self._sample_rate:
            waveform = torch.from_numpy(chunk_np).unsqueeze(0)
            waveform = torchaudio.transforms.Resample(
                sample_rate, self._sample_rate
            )(waveform)
            chunk_np = waveform.squeeze(0).numpy()

        # Process through model
        inputs = self._processor(
            chunk_np, sampling_rate=self._sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self._device)

        with torch.inference_mode():
            logits = self._model(input_values).logits

        # CTC decode → eSpeak phonemes → clean IPA
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)[0]

        phonemes = self._clean_espeak(transcription)
        if not phonemes:
            return None

        tokens = phonemes.split()
        return ' '.join(tokens) if tokens else None

    def recognize_chunk_detailed(
        self,
        chunk,
        sample_rate: int,
        nbest_keep: int = 4,
        alt_per_phone: int = 3,
    ) -> tuple[str | None, dict]:
        import numpy as np
        import torch
        import torchaudio

        if self._model is None or self._processor is None:
            raise RuntimeError("Wav2Vec2 phoneme backend not initialized")

        if len(chunk) < sample_rate * 0.2:
            return None, {"phone_conf": [], "nbest": []}

        chunk_np = np.asarray(chunk, dtype=np.float32)
        if chunk_np.ndim > 1:
            chunk_np = chunk_np.mean(axis=1)

        if sample_rate != self._sample_rate:
            waveform = torch.from_numpy(chunk_np).unsqueeze(0)
            waveform = torchaudio.transforms.Resample(
                sample_rate, self._sample_rate
            )(waveform)
            chunk_np = waveform.squeeze(0).numpy()

        inputs = self._processor(
            chunk_np, sampling_rate=self._sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self._device)

        with torch.inference_mode():
            logits = self._model(input_values).logits[0]  # [T, V]

        probs = torch.softmax(logits, dim=-1)  # [T, V]
        frame_ids = torch.argmax(probs, dim=-1)  # [T]
        frame_max = torch.max(probs, dim=-1).values  # [T]

        # Build CTC collapsed token spans from argmax path.
        spans = []
        prev = None
        start = 0
        for i, tid in enumerate(frame_ids.tolist()):
            if prev is None:
                prev = tid
                start = i
                continue
            if tid != prev:
                if prev != self._blank_id:
                    spans.append((int(prev), start, i))
                prev = tid
                start = i
        if prev is not None and prev != self._blank_id:
            spans.append((int(prev), start, len(frame_ids)))

        if not spans:
            return None, {"phone_conf": [], "nbest": []}

        best_ids = [sid for sid, _, _ in spans]
        best_text = self._clean_espeak(self._processor.decode(best_ids))
        if not best_text:
            return None, {"phone_conf": [], "nbest": []}

        # Per-phone confidence from mean frame max probability on each span.
        best_tokens = best_text.split()
        phone_conf = []
        span_conf = []
        for sid, s0, s1 in spans:
            c = float(torch.mean(frame_max[s0:s1]).item()) if s1 > s0 else 0.0
            span_conf.append(c)
        # Align by order; decoder can occasionally alter count, clamp to min length.
        n_align = min(len(best_tokens), len(span_conf))
        for i in range(n_align):
            phone_conf.append({
                "ph": best_tokens[i],
                "c": round(span_conf[i], 3),
            })

        # Lightweight n-best generation:
        # keep best path and create local alternatives by replacing low-confidence phones.
        # This is not full CTC beam search, but preserves plausible competing phones.
        cand = []
        base_score = float(sum(np.log(max(c, 1e-6)) for c in span_conf[:n_align]))
        cand.append((tuple(best_ids), base_score))

        for idx, (sid, s0, s1) in enumerate(spans[:n_align]):
            seg_probs = torch.mean(probs[s0:s1], dim=0) if s1 > s0 else probs[s0]
            topk = torch.topk(seg_probs, k=min(alt_per_phone + 1, seg_probs.shape[0]))
            alt_ids = topk.indices.tolist()
            alt_ps = topk.values.tolist()
            for aid, ap in zip(alt_ids, alt_ps):
                if aid == sid or aid == self._blank_id:
                    continue
                seq = list(best_ids)
                seq[idx] = int(aid)
                # Swap one phone at a time to keep variants stable/readable.
                score = base_score - np.log(max(span_conf[idx], 1e-6)) + np.log(max(float(ap), 1e-6))
                cand.append((tuple(seq), float(score)))

        # Deduplicate and decode
        nbest = []
        seen = set()
        for seq_ids, score in sorted(cand, key=lambda x: x[1], reverse=True):
            text = self._clean_espeak(self._processor.decode(list(seq_ids)))
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            nbest.append({"ipa": text, "score": round(float(score), 3)})
            if len(nbest) >= nbest_keep:
                break

        return best_text, {"phone_conf": phone_conf, "nbest": nbest}

    def cleanup(self) -> None:
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            del self._processor
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class OmnilingualBackend(PhonemeBackend):
    """Omnilingual ASR backend using omnilingual-asr package.

    Outputs text which is then converted to IPA via G2P.
    """

    name = "omnilingual"
    MODEL_CARDS = {
        "300M": "omniASR_LLM_300M_v2",
        "7B": "omniASR_LLM_7B_v2",
    }

    def __init__(self, cache_dir: str | None = None, model_size: str = "300M"):
        self._cache_dir = cache_dir
        self._model_size = model_size
        self._pipeline = None
        self._device = "cuda"

    def initialize(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        model_card = self.MODEL_CARDS.get(self._model_size, self.MODEL_CARDS["300M"])
        self._pipeline = ASRInferencePipeline(model_card=model_card, device=self._device)
        print(f"Omnilingual: {model_card} on {self._device}")

    def recognize_chunk(self, chunk, sample_rate: int) -> str | None:
        """Recognize a chunk and return IPA string."""
        if self._pipeline is None:
            return None
        if len(chunk) < sample_rate * 0.2:
            return None

        import tempfile
        import soundfile as sf
        import numpy as np

        chunk_np = np.asarray(chunk, dtype=np.float32)
        if chunk_np.ndim > 1:
            chunk_np = chunk_np.mean(axis=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk_np, sample_rate)
            try:
                # Omnilingual outputs text, we convert to IPA
                result = self._pipeline.transcribe([f.name], lang=["spa_Latn"])
                text = ""
                if isinstance(result, list) and result:
                    entry = result[0]
                    if isinstance(entry, dict):
                        text = entry.get("text", "")
                    else:
                        text = str(entry)
                elif isinstance(result, dict):
                    text = result.get("text", "")

                if not text:
                    return None

                # Convert text to IPA using epitran
                try:
                    import epitran
                    epi = epitran.Epitran("spa-Latn")
                    ipa = epi.transliterate(text)
                    return ipa
                except Exception:
                    # Fallback: return raw text if G2P fails
                    return f"[{text}]"
            except Exception as e:
                print(f"Omnilingual error: {e}")
                return None
            finally:
                import os
                os.unlink(f.name)

    def recognize_chunk_detailed(self, chunk, sample_rate: int, nbest_keep: int = 4):
        """Omnilingual doesn't support detailed output yet."""
        ipa = self.recognize_chunk(chunk, sample_rate)
        return ipa, {"phone_conf": [], "nbest": []}

    def cleanup(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# MMS Language Identification Voter (facebook/mms-1b-all)
# ---------------------------------------------------------------------------

class MMSLangID:
    """Language identification using facebook/mms-1b-all.

    Returns ISO 639-3 language codes with confidence scores.
    Supports 1162 languages including nah (Nahuatl), spa, eng, fra, etc.
    """

    MODEL_ID = "facebook/mms-lid-4017"
    # Map MMS output codes to our pipeline codes
    LANG_REMAP = {
        "spa": "spa", "eng": "eng", "deu": "deu", "fra": "fra", "ita": "ita",
        "nah": "nah",  # Central Nahuatl
        "azz": "nah",  # Highland Puebla Nahuatl
        "nhx": "nah",  # Isthmus-Mecayapan Nahuatl
        "nhe": "nah",  # Eastern Huasteca Nahuatl
        "nhi": "nah",  # Tenango Nahuatl
        "nhw": "nah",  # Western Huasteca Nahuatl
        "ncj": "nah",  # Northern Puebla Nahuatl
        "ncl": "nah",  # Michoacán Nahuatl
        "ngu": "nah",  # Guerrero Nahuatl
        "yua": "may",  # Yucatec Maya
        "mam": "may",  # Mam Maya
        "kek": "may",  # Q'eqchi' Maya
        "cak": "may",  # Kaqchikel Maya
        "quc": "may",  # K'iche' Maya
        "tzh": "may",  # Tzeltal Maya
        "tzo": "may",  # Tzotzil Maya
        "lat": "lat",  # Latin
        "por": "spa",  # Portuguese → treat as SPA in our context
        "cat": "spa",  # Catalan → treat as SPA
        "glg": "spa",  # Galician → treat as SPA
    }

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = cache_dir
        self._processor = None
        self._model = None
        self._device = "cuda"

    def initialize(self) -> None:
        import time as _time
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        print(f"Loading MMS LangID ({self.MODEL_ID})...")
        t0 = _time.time()
        cache = self._cache_dir
        self._processor = AutoFeatureExtractor.from_pretrained(
            self.MODEL_ID, cache_dir=cache
        )
        self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.MODEL_ID, cache_dir=cache
        ).to(self._device)
        self._model.eval()
        print(f"MMS LangID loaded in {_time.time() - t0:.1f}s "
              f"({len(self._model.config.id2label)} languages)")

    def identify(self, audio_chunk, sample_rate: int, top_k: int = 5
                 ) -> list[tuple[str, float]]:
        """Identify language of audio chunk.

        Returns list of (lang_code, confidence) tuples, mapped to pipeline codes.
        """
        if self._model is None:
            return []

        import torch
        import numpy as np

        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)

        # MMS expects 16kHz
        if sample_rate != 16000:
            import torchaudio
            waveform = torch.from_numpy(chunk).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            chunk = resampler(waveform).squeeze(0).numpy()

        inputs = self._processor(
            chunk, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_vals, top_ids = probs.topk(top_k)

        results = []
        for val, idx in zip(top_vals.cpu().tolist(), top_ids.cpu().tolist()):
            raw_lang = self._model.config.id2label[idx]
            mapped = self.LANG_REMAP.get(raw_lang, raw_lang)
            results.append((mapped, round(val, 4), raw_lang))
        return results

    def identify_mapped(self, audio_chunk, sample_rate: int
                        ) -> tuple[str, float, str]:
        """Convenience: return (mapped_lang, confidence, raw_lang) for top-1."""
        results = self.identify(audio_chunk, sample_rate, top_k=1)
        if results:
            return results[0]
        return ("other", 0.0, "unk")

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model, self._processor
            self._model = None
            self._processor = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def resolve_phoneme_backend_name(name: str | None) -> tuple[str, str | None]:
    requested = (name or "allosaurus").strip().lower()
    supported = {"allosaurus", "wav2vec2", "espnet"}
    if requested in supported:
        if requested == "espnet":
            return "wav2vec2", "⚠️ ESPnet backend not wired yet, using wav2vec2 compatibility mode"
        return requested, None
    warning = f"⚠️ Unknown phoneme backend '{requested}', falling back to allosaurus"
    return "allosaurus", warning


def get_phoneme_backend(name: str, cache_dir: str) -> PhonemeBackend:
    if name == "allosaurus":
        return AllosaurusBackend(cache_dir)
    if name in {"wav2vec2", "espnet"}:
        return Wav2Vec2Backend(cache_dir)
    # Defensive fallback for direct factory calls.
    return AllosaurusBackend(cache_dir)


def detect_latin_text(text: str, min_matches: int = 3) -> tuple[bool, int]:
    if not text:
        return False, 0
    normalized = _normalize_word_ascii(text)
    words = re.findall(r"[a-z]+", normalized)
    matches = set(words) & LATIN_KEYWORDS
    return (len(matches) >= min_matches), len(matches)


def detect_maya_text_hint(text: str, min_score: float = 1.2) -> tuple[bool, float]:
    if not text:
        return False, 0.0

    t = _normalize_word_ascii(text)
    words = re.findall(r"[a-z']+", t)
    if not words:
        return False, 0.0

    score = 0.0
    apostrophes = t.count("'") + t.count("\u2019")
    if apostrophes >= 1:
        score += 0.7
    if apostrophes >= 2:
        score += 0.4

    # Orthographic clues common in Yucatec romanization.
    for pat in ("k'", "ts'", "ch'", "x", "j", "aa", "oo", "uu"):
        if pat in t:
            score += 0.2

    maya_lex_hits = sum(1 for w in words if w in MAYA_HINT_WORDS)
    score += 0.25 * maya_lex_hits

    return score >= min_score, score


# ---------------------------------------------------------------------------
# FIX 2: Spanish Text Leak Detection
# ---------------------------------------------------------------------------
# Common Spanish words that indicate Whisper hallucination on NAH/MAY audio

SPANISH_LEAK_WORDS = frozenset([
    # Common Spanish words unlikely in NAH/MAY
    "que", "de", "la", "el", "en", "los", "las", "por", "con", "para",
    "una", "uno", "del", "al", "es", "son", "pero", "como", "mas", "muy",
    "este", "esta", "esto", "eso", "ese", "esa", "aqui", "ahi", "alla",
    "si", "no", "ya", "hay", "ser", "estar", "tener", "hacer", "poder",
    "decir", "ir", "ver", "dar", "saber", "querer", "llegar", "pasar",
    # Verbs in common conjugations
    "vamos", "vais", "viene", "tiene", "hace", "dice", "puede", "quiere",
    "estoy", "estas", "esta", "estamos", "estan",
    # Exclamations and imperatives (strong SPA signal)
    "mira", "oye", "veis", "venga", "anda", "cuidado", "espera",
    # Short function words (require 2+ to trigger)
    "a", "mi", "tu", "su", "me", "te", "se", "le", "lo",
    # More verbs
    "empezar", "salir", "entrar", "hablar", "llamar", "llevar",
    "senor", "capitan", "soldado", "hombre", "mujer",
])

# Very strong Spanish indicators (1 word is enough)
SPANISH_STRONG_WORDS = frozenset([
    "vamos", "veis", "mira", "oye", "venga", "cuidado",
    "senor", "capitan", "soldados", "arcabuceros",
])

# ---------------------------------------------------------------------------
# FIX 7: Fuzzy Spanish Word Matching
# ---------------------------------------------------------------------------
# Catches "castellano" in "sankastelanl", "arcabuceros" in "Aokabuférous"

from difflib import SequenceMatcher

# Spanish words that often get garbled by Whisper on indigenous audio
SPANISH_FUZZY_TARGETS = [
    "castellano", "arcabuceros", "soldados", "capitan", "conquistador",
    "espanoles", "cristianos", "hermanos", "companeros", "caballeros",
    "emperador", "gobernador", "senores", "majestades", "excelencia",
]

def fuzzy_spanish_match(text: str, threshold: float = 0.75) -> tuple[bool, str]:
    """Detect Spanish words even when garbled by Whisper.

    Uses sequence matching to find 'castellano' in 'sankastelanl' etc.

    Returns:
        (is_match, matched_word): True if fuzzy match found above threshold
    """
    if not text:
        return False, ""

    # Normalize: lowercase, keep only letters
    normalized = re.sub(r"[^a-z]", "", text.lower())

    for target in SPANISH_FUZZY_TARGETS:
        # Check if target could be a substring (allowing for prefix/suffix noise)
        for i in range(len(normalized) - len(target) + 3):
            window = normalized[i:i + len(target) + 4]  # Allow some extra chars
            if len(window) >= len(target) - 2:
                ratio = SequenceMatcher(None, target, window).ratio()
                if ratio >= threshold:
                    return True, target

    return False, ""


def detect_spanish_text_leak(text: str, ipa: str, lang: str, min_words: int = 2) -> tuple[bool, str]:
    """Detect Spanish text leak: Whisper outputs Spanish but IPA suggests NAH/MAY.

    Returns:
        (is_leak, reason): True if Spanish text detected in non-SPA context
    """
    if not text or lang == "spa":
        return False, ""

    # Normalize and extract words
    t = _normalize_word_ascii(text.lower())
    words = set(re.findall(r"[a-z]+", t))

    # FIX 7: Check fuzzy matches first (catches garbled Spanish)
    is_fuzzy, fuzzy_word = fuzzy_spanish_match(text)
    if is_fuzzy:
        # Check if IPA has NAH/MAY markers that contradict Spanish
        nah_markers = {"tɬ", "kʷ", "ʔ", "ɬ"}
        may_markers = {"kʼ", "tʼ", "tsʼ", "ʔ"}
        ipa_tokens = set(ipa.split()) if ipa else set()
        has_indigenous_markers = bool(ipa_tokens & (nah_markers | may_markers))

        if not has_indigenous_markers:
            return True, f"fuzzy_spa={fuzzy_word}"

    # Spanish orthography (¿¡áéíóúñ) is a strong signal on its own
    if _has_spanish_orthography(text):
        spa_dict_hits = words & SPA_COMMON
        if spa_dict_hits:
            nah_markers = {"tɬ", "kʷ", "ʔ", "ɬ"}
            may_markers = {"kʼ", "tʼ", "tsʼ", "ʔ"}
            ipa_tokens = set(ipa.split()) if ipa else set()
            has_indigenous_markers = bool(ipa_tokens & (nah_markers | may_markers))
            if not has_indigenous_markers:
                return True, f"spa_ortho+dict={list(spa_dict_hits)[:2]}"

    # Check for strong Spanish indicators (1 word is enough)
    strong_hits = words & SPANISH_STRONG_WORDS
    if strong_hits:
        # Check if IPA has NAH/MAY markers that contradict Spanish
        nah_markers = {"tɬ", "kʷ", "ʔ", "ɬ"}
        may_markers = {"kʼ", "tʼ", "tsʼ", "ʔ"}
        ipa_tokens = set(ipa.split()) if ipa else set()
        has_indigenous_markers = bool(ipa_tokens & (nah_markers | may_markers))

        if not has_indigenous_markers:
            return True, f"strong_spa={list(strong_hits)[:2]}"

    # Count regular Spanish word hits (need min_words)
    # Use both curated SPANISH_LEAK_WORDS and the full SPA_COMMON dictionary
    spa_hits = words & (SPANISH_LEAK_WORDS | SPA_COMMON)

    if len(spa_hits) >= min_words:
        # Check if IPA has NAH/MAY markers that contradict Spanish
        nah_markers = {"tɬ", "kʷ", "ʔ", "ɬ"}
        may_markers = {"kʼ", "tʼ", "tsʼ", "ʔ"}

        ipa_tokens = set(ipa.split()) if ipa else set()
        has_indigenous_markers = bool(ipa_tokens & (nah_markers | may_markers))

        if not has_indigenous_markers:
            # Spanish text without indigenous IPA markers = likely SPA
            return True, f"spa_words={list(spa_hits)[:3]}"

    return False, ""


# ---------------------------------------------------------------------------
# FIX 3: Ejective IPA Annotation
# ---------------------------------------------------------------------------

def annotate_ejectives_in_ipa(ipa: str, ejective_count: int, min_count: int = 1) -> str:
    """Add ejective markers to IPA string when acoustic ejectives detected.

    If ejective_count >= min_count, mark potential ejective consonants with ʼ
    """
    if not ipa or ejective_count < min_count:
        return ipa

    # Consonants that can be ejective in Maya
    ejective_targets = ["k", "t", "p", "ts", "tʃ", "tɕ"]

    tokens = ipa.split()
    annotated = []
    ejectives_added = 0

    for token in tokens:
        # Only annotate up to ejective_count consonants
        if ejectives_added < ejective_count and token in ejective_targets:
            annotated.append(token + "ʼ")
            ejectives_added += 1
        else:
            annotated.append(token)

    return " ".join(annotated)


# ---------------------------------------------------------------------------
# FIX 4: Remove English Glosses from LLM Text
# ---------------------------------------------------------------------------

def remove_glosses(text: str) -> str:
    """Remove parenthetical glosses like '(water)' or '(the sun rises)' from text."""
    if not text:
        return text
    return re.sub(r'\s*\([^)]+\)\s*', ' ', text).strip()


# ---------------------------------------------------------------------------
# FIX 5: Phoneme Density Validation
# ---------------------------------------------------------------------------

def validate_phoneme_density(ipa: str, duration: float, min_rate: float = 3.0, max_rate: float = 15.0) -> str:
    """Check if IPA phoneme count is reasonable for segment duration.

    Returns:
        'OK' - normal density
        'COLLAPSED' - too few phones (ASR failure)
        'HALLUCINATED' - too many phones (insertion errors)
        'SHORT' - segment too short to validate
    """
    if duration < 0.3:
        return "SHORT"

    if not ipa:
        return "COLLAPSED"

    phones = len(ipa.split())
    rate = phones / duration

    if rate < min_rate:
        return "COLLAPSED"
    if rate > max_rate:
        return "HALLUCINATED"
    return "OK"


# ---------------------------------------------------------------------------
# FIX 6: LLM IPA-as-Text Detection
# ---------------------------------------------------------------------------

IPA_SYMBOLS = set("ɪɛæɑɔʊəɜɚʌŋθðʃʒɾβɣχʁɴɲɳɻʂʐɕʑɬɮʔɯɨʉɵɤɘɞɐœøɒɶʋɹɰɥʍʜʢɧɦʙʀɺɭɽɢɠɗɓʛʄǃǀǂǁ")

def is_ipa_as_text(text: str) -> bool:
    """Detect if LLM output is IPA symbols instead of actual words.

    LLM sometimes outputs phonetic transcription as 'text' when Whisper fails.
    """
    if not text:
        return False

    # Count IPA-specific characters
    ipa_chars = sum(1 for c in text if c in IPA_SYMBOLS)

    # If >30% of text is IPA symbols, it's likely phonetic output
    if len(text) > 0 and ipa_chars / len(text) > 0.3:
        return True

    # Check for space-separated single characters (phoneme format)
    tokens = text.split()
    if len(tokens) >= 3 and all(len(t) <= 2 for t in tokens):
        return True

    return False


def clean_llm_text(text: str) -> str:
    """Clean LLM text: remove glosses, detect IPA-as-text."""
    if not text:
        return text

    # Remove glosses
    text = remove_glosses(text)

    # If it's IPA-as-text, return empty (use IPA field instead)
    if is_ipa_as_text(text):
        return ""

    return text


# ---------------------------------------------------------------------------
# Acoustic Ejective Detection for Maya
# ---------------------------------------------------------------------------
# Maya ejectives (kʼ, tʼ, tsʼ, chʼ, pʼ) have distinctive acoustic signatures
# that Allosaurus often fails to transcribe. This detector finds them directly
# from the audio waveform using intensity/burst analysis.

class EjectiveDetector:
    """Acoustic ejective detection for Maya language identification.

    Ejective stops have:
    - Long closure duration (glottal + oral closure)
    - High-intensity burst (increased air pressure release)
    - Short positive VOT
    - F0 perturbation after release

    Uses 3-way voting: heuristic + sklearn + wav2vec2 for robust classification.
    """

    # Heuristic thresholds (tuned for better recall)
    MIN_CLOSURE_MS = 45  # Loosened from 60
    MIN_BURST_RELATIVE_DB = 3  # Loosened from 6
    MAX_VOT_MS = 50  # Loosened from 40
    MIN_F0_DROP_HZ = 5  # Loosened from 10

    def __init__(self, use_w2v2: bool = True):
        self._sklearn_model = None
        self._scaler = None
        self._use_w2v2 = use_w2v2
        self._w2v2_model = None
        self._w2v2_processor = None

    def _init_w2v2(self):
        """Lazy-init wav2vec2 model on GPU."""
        if self._w2v2_model is not None:
            return True
        try:
            import torch
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._w2v2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self._w2v2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
            self._w2v2_model.eval()
            self._w2v2_device = device
            return True
        except Exception as e:
            print(f"[ejective] wav2vec2 init failed: {e}")
            return False

    def detect_ejectives(
        self,
        audio: "np.ndarray",
        sr: int,
        start_time: float = 0.0,
        verbose: bool = False,
    ) -> dict:
        """Detect ejective-like stops in audio segment.

        Args:
            audio: Audio samples (mono float32)
            sr: Sample rate
            start_time: Offset for timestamp reporting
            verbose: Print transparent voting details

        Returns:
            {
                "ejective_count": int,  # Consensus count (2+ methods agree)
                "heuristic_count": int,
                "sklearn_count": int,
                "w2v2_count": int,
                "candidates": list[dict],  # Detailed candidate info with votes
                "voting_log": str,  # Transparent voting details
            }
        """
        import numpy as np

        try:
            import parselmouth
        except ImportError:
            return {"ejective_count": 0, "heuristic_count": 0, "sklearn_count": 0, "w2v2_count": 0, "candidates": [], "voting_log": ""}

        empty_result = {"ejective_count": 0, "heuristic_count": 0, "sklearn_count": 0, "w2v2_count": 0, "candidates": [], "voting_log": ""}

        # Need at least 200ms of audio
        if len(audio) < sr * 0.2:
            return empty_result

        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        intensity = sound.to_intensity(minimum_pitch=50, time_step=0.005)
        pitch = sound.to_pitch_cc(time_step=0.01)

        # Find stop candidates via derivative analysis
        times = intensity.xs()
        values = np.array([intensity.get_value(t) or 0 for t in times])

        if len(times) < 2:
            return empty_result

        dt = times[1] - times[0]
        derivative = np.gradient(values, dt)

        # Detection parameters
        drop_threshold = -80  # dB/s
        rise_threshold = 80   # dB/s
        min_gap = 0.015       # 15ms
        max_gap = 0.120       # 120ms

        raw_candidates = []
        i = 0
        while i < len(derivative) - 5:
            if derivative[i] < drop_threshold:
                drop_time = times[i]
                for j in range(i + 3, min(i + int(max_gap / dt), len(derivative))):
                    if derivative[j] > rise_threshold:
                        rise_time = times[j]
                        gap = rise_time - drop_time
                        if gap >= min_gap:
                            raw_candidates.append({
                                "drop_time": drop_time,
                                "rise_time": rise_time,
                                "gap_ms": gap * 1000
                            })
                            i = j
                            break
                    if values[j] < 10:
                        break
            i += 1

        if not raw_candidates:
            return empty_result

        # Deduplicate (merge within 50ms)
        deduped = [raw_candidates[0]]
        for c in raw_candidates[1:]:
            if c["drop_time"] - deduped[-1]["drop_time"] > 0.050:
                deduped.append(c)

        # Extract features for each candidate
        candidates = []
        for rc in deduped:
            burst_time = rc["rise_time"]
            closure_start = rc["drop_time"]
            closure_duration = rc["gap_ms"]

            burst_intensity = intensity.get_value(burst_time) or 60

            # Context intensity
            context_intensities = [
                intensity.get_value(max(0, closure_start - 0.1) + i * 0.01)
                for i in range(10)
            ]
            context_intensities = [v for v in context_intensities if v is not None]
            context_mean = np.mean(context_intensities) if context_intensities else 60
            burst_relative = burst_intensity - context_mean

            # VOT estimation
            vot = 0
            for dt_val in np.arange(0.005, 0.060, 0.005):
                t = burst_time + dt_val
                if t < sound.xmax:
                    f0 = pitch.get_value_at_time(t)
                    if f0 is not None and f0 > 0:
                        vot = dt_val * 1000
                        break

            # Creak detection
            f0_values = []
            for dt_val in np.arange(-0.02, 0.02, 0.005):
                t = burst_time + dt_val
                if 0 < t < sound.xmax:
                    f0 = pitch.get_value_at_time(t)
                    if f0 is not None:
                        f0_values.append(f0)
            has_creak = len(f0_values) >= 3 and np.std(f0_values) > 20

            # F0 perturbation
            f0_before = pitch.get_value_at_time(closure_start)
            f0_after = pitch.get_value_at_time(burst_time + 0.03)
            f0_perturbation = (f0_before - f0_after) if (f0_before and f0_after) else 0

            candidates.append({
                "time": start_time + burst_time,
                "local_time": burst_time,  # Time within this audio chunk
                "closure_ms": closure_duration,
                "vot_ms": vot,
                "burst_rel_db": burst_relative,
                "has_creak": has_creak,
                "f0_drop": f0_perturbation,
                "heuristic_ejective": False,
                "sklearn_ejective": False,
                "w2v2_ejective": False,
            })

        # Heuristic classification
        for c in candidates:
            score = 0
            if c["closure_ms"] >= self.MIN_CLOSURE_MS:
                score += 1
            if c["burst_rel_db"] >= self.MIN_BURST_RELATIVE_DB:
                score += 1
            if 0 < c["vot_ms"] <= self.MAX_VOT_MS:
                score += 1
            if c["f0_drop"] >= self.MIN_F0_DROP_HZ:
                score += 1
            if c["has_creak"]:
                score += 1
            c["heuristic_ejective"] = score >= 3

        # Sklearn anomaly detection
        if len(candidates) >= 5:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler

                features = np.array([
                    [c["closure_ms"], c["vot_ms"], c["burst_rel_db"],
                     1.0 if c["has_creak"] else 0.0, c["f0_drop"]]
                    for c in candidates
                ])

                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                model = IsolationForest(contamination=0.15, random_state=42, n_estimators=50)
                predictions = model.fit_predict(features_scaled)

                for i, c in enumerate(candidates):
                    c["sklearn_ejective"] = predictions[i] == -1
            except Exception:
                pass

        # Wav2vec2 embedding clustering (GPU accelerated)
        w2v2_available = False
        if self._use_w2v2 and len(candidates) >= 5:
            try:
                import torch
                w2v2_available = self._init_w2v2()
                if w2v2_available:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler as W2V2Scaler

                    embeddings = []
                    for c in candidates:
                        # Extract 100ms window around burst
                        local_time = c["local_time"]
                        start_sample = max(0, int((local_time - 0.05) * sr))
                        end_sample = min(len(audio), int((local_time + 0.05) * sr))
                        chunk = audio[start_sample:end_sample]

                        if len(chunk) < sr * 0.02:
                            embeddings.append(np.zeros(768))
                            continue

                        inputs = self._w2v2_processor(
                            chunk, sampling_rate=sr, return_tensors="pt", padding=True
                        )
                        inputs = {k: v.to(self._w2v2_device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self._w2v2_model(**inputs)
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                            embeddings.append(embedding)

                    embeddings = np.array(embeddings)
                    scaler = W2V2Scaler()
                    embeddings_scaled = scaler.fit_transform(embeddings)

                    # 2 clusters: ejective vs non-ejective
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings_scaled)

                    # Smaller cluster = ejectives (minority)
                    cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
                    ejective_cluster = 0 if cluster_sizes[0] < cluster_sizes[1] else 1

                    for i, c in enumerate(candidates):
                        c["w2v2_ejective"] = bool(labels[i] == ejective_cluster)
            except Exception as e:
                if verbose:
                    print(f"[ejective] w2v2 classification failed: {e}")

        # Count results
        heuristic_count = sum(1 for c in candidates if c["heuristic_ejective"])
        sklearn_count = sum(1 for c in candidates if c["sklearn_ejective"])
        w2v2_count = sum(1 for c in candidates if c["w2v2_ejective"])

        # Consensus: 2+ methods agree
        consensus_count = sum(1 for c in candidates
                             if sum([c["heuristic_ejective"], c["sklearn_ejective"], c["w2v2_ejective"]]) >= 2)

        # Build transparent voting log
        voting_lines = []
        voting_lines.append(f"Ejective Detection: {len(candidates)} stops analyzed")
        voting_lines.append(f"  Heuristic: {heuristic_count} (closure≥{self.MIN_CLOSURE_MS}ms, burst≥{self.MIN_BURST_RELATIVE_DB}dB, VOT≤{self.MAX_VOT_MS}ms)")
        voting_lines.append(f"  Sklearn:   {sklearn_count} (IsolationForest anomalies, contamination=15%)")
        voting_lines.append(f"  Wav2Vec2:  {w2v2_count} (embedding cluster minority){' [unavailable]' if not w2v2_available else ''}")
        voting_lines.append(f"  Consensus: {consensus_count} (2+ methods agree)")

        if consensus_count > 0 or verbose:
            voting_lines.append("  Candidates with 2+ votes:")
            for c in candidates:
                votes = sum([c["heuristic_ejective"], c["sklearn_ejective"], c["w2v2_ejective"]])
                if votes >= 2 or verbose:
                    h = "H" if c["heuristic_ejective"] else "-"
                    s = "S" if c["sklearn_ejective"] else "-"
                    w = "W" if c["w2v2_ejective"] else "-"
                    voting_lines.append(
                        f"    {c['time']:6.2f}s: [{h}{s}{w}] closure={c['closure_ms']:3.0f}ms "
                        f"burst={c['burst_rel_db']:+5.1f}dB VOT={c['vot_ms']:2.0f}ms"
                    )

        voting_log = "\n".join(voting_lines)

        return {
            "ejective_count": consensus_count,
            "heuristic_count": heuristic_count,
            "sklearn_count": sklearn_count,
            "w2v2_count": w2v2_count,
            "candidates": candidates,
            "voting_log": voting_log,
        }


# Global detector instance (lazy init)
_ejective_detector = None


def get_ejective_detector() -> EjectiveDetector:
    global _ejective_detector
    if _ejective_detector is None:
        _ejective_detector = EjectiveDetector()
    return _ejective_detector


def detect_ejectives_for_segment(
    audio: "np.ndarray",
    sr: int,
    start_time: float = 0.0,
    verbose: bool = False,
) -> int | dict:
    """Detect ejectives in a segment.

    Args:
        audio: Audio samples (mono float32)
        sr: Sample rate
        start_time: Offset for timestamp reporting
        verbose: If True, return full result dict with voting_log

    Returns:
        int: ejective_count (consensus) when verbose=False
        dict: Full result with voting_log when verbose=True
    """
    detector = get_ejective_detector()
    result = detector.detect_ejectives(audio, sr, start_time, verbose=verbose)
    if verbose:
        return result
    return result["ejective_count"]


def strip_modifiers(phoneme):
    return "".join(c for c in phoneme
                   if c not in SPACING_MODIFIERS
                   and unicodedata.category(c) != "Mn")


def identify_language(phonemes):
    if not phonemes:
        return "other", 0.0

    # Pre-check for NAH core markers (excluding tɬ) for context validation
    NAH_CORE_MARKERS = {"ʔ", "ɬ", "kʷ"}
    has_nah_core = any(
        strip_modifiers(ph) in NAH_CORE_MARKERS or ph in NAH_CORE_MARKERS
        for ph in phonemes
    )

    scores = {}
    for code, prof in PROFILES.items():
        score = 0.0
        for ph in phonemes:
            raw, stripped = ph, strip_modifiers(ph)
            matched = None
            if raw in prof["markers"]:
                matched = raw
            elif stripped != raw and stripped in prof["markers"]:
                matched = stripped
            else:
                norm = ALLOPHONE_MAP.get(stripped)
                if norm and norm in prof["markers"]:
                    matched = norm
            if matched:
                marker_weight = prof["markers"][matched]
                # tɬ context validation for NAH: reduce weight if no other core markers
                if code == "nah" and matched == "tɬ" and not has_nah_core:
                    marker_weight = 0.3  # Reduced from 1.0 for isolated tɬ (e.g., "atlántico")
                score += marker_weight
            norm_neg = ALLOPHONE_MAP.get(stripped, stripped)
            for check in [raw, stripped, norm_neg]:
                if check in prof.get("negative", {}):
                    score -= prof["negative"][check]
                    break
        scores[code] = score

    qualified = {c: s for c, s in scores.items() if s > PROFILES[c]["threshold"]}
    if not qualified:
        return "other", 0.0
    winner = max(qualified, key=lambda c: (qualified[c], PROFILES[c]["priority"]))
    return winner, scores[winner]


def identify_language_with_scores(phonemes):
    """Like identify_language but also returns the full scores dict for Spanish Context Guard."""
    if not phonemes:
        return "other", 0.0, {}

    # Pre-check for NAH core markers (excluding tɬ) for context validation
    NAH_CORE_MARKERS = {"ʔ", "ɬ", "kʷ"}
    has_nah_core = any(
        strip_modifiers(ph) in NAH_CORE_MARKERS or ph in NAH_CORE_MARKERS
        for ph in phonemes
    )

    scores = {}
    for code, prof in PROFILES.items():
        score = 0.0
        for ph in phonemes:
            raw, stripped = ph, strip_modifiers(ph)
            matched = None
            if raw in prof["markers"]:
                matched = raw
            elif stripped != raw and stripped in prof["markers"]:
                matched = stripped
            else:
                norm = ALLOPHONE_MAP.get(stripped)
                if norm and norm in prof["markers"]:
                    matched = norm
            if matched:
                marker_weight = prof["markers"][matched]
                # tɬ context validation for NAH: reduce weight if no other core markers
                if code == "nah" and matched == "tɬ" and not has_nah_core:
                    marker_weight = 0.3
                score += marker_weight
            norm_neg = ALLOPHONE_MAP.get(stripped, stripped)
            for check in [raw, stripped, norm_neg]:
                if check in prof.get("negative", {}):
                    score -= prof["negative"][check]
                    break
        scores[code] = score

    qualified = {c: s for c, s in scores.items() if s > PROFILES[c]["threshold"]}
    if not qualified:
        return "other", 0.0, scores
    winner = max(qualified, key=lambda c: (qualified[c], PROFILES[c]["priority"]))
    return winner, scores[winner], scores


def check_nah_lexicon(phonemes, threshold=0.75, use_enhanced=True):
    """Check if phonemes match a known Nahuatl word.

    Args:
        phonemes: List of IPA phonemes
        threshold: Minimum match score (0.0-1.0)
        use_enhanced: If True, use corpus-based lexicon (140 entries)
                     If False, use original curated lexicon (10 entries)

    Returns:
        Tuple of (word, score) where word is None if no match above threshold
    """
    # Try enhanced lexicon first if available
    if use_enhanced and NAH_LEXICON_ENHANCED:
        best_word, score, _ = _check_nah_lexicon_enhanced(
            phonemes, NAH_LEXICON_ENHANCED, threshold
        )
        if best_word:
            return best_word, score

    # Fallback to original small lexicon
    stripped = [strip_modifiers(p) for p in phonemes]
    best_word = None
    best_score = 0.0
    for entry in NAH_LEXICON:
        pattern = entry["ipa"]
        if abs(len(stripped) - len(pattern)) > 2:
            continue
        max_len = max(len(stripped), len(pattern))
        matches = sum(1 for a, b in zip(stripped, pattern) if a == b)
        if max_len > 0:
            score = matches / max_len
            if score >= threshold and score > best_score:
                best_word = entry["word"]
                best_score = score
    return best_word, best_score


def check_nah_lexicon_score(phonemes, use_enhanced=True):
    """Get lexicon match score without threshold filtering.

    P1: Used for lexicon-override before marker-scoring.
    Returns best match score even if below word-match threshold.
    """
    if use_enhanced and NAH_LEXICON_ENHANCED:
        _, score, _ = _check_nah_lexicon_enhanced(
            phonemes, NAH_LEXICON_ENHANCED, threshold=0.0  # No threshold
        )
        return score

    # Fallback to original lexicon
    stripped = [strip_modifiers(p) for p in phonemes]
    best_score = 0.0
    for entry in NAH_LEXICON:
        pattern = entry["ipa"]
        if abs(len(stripped) - len(pattern)) > 2:
            continue
        max_len = max(len(stripped), len(pattern))
        matches = sum(1 for a, b in zip(stripped, pattern) if a == b)
        if max_len > 0:
            score = matches / max_len
            if score > best_score:
                best_score = score
    return best_score


def validate_whisper(text, avg_log_prob=-0.3):
    if not text or not text.strip():
        return False
    if has_non_latin_script(text):
        return False
    apostrophes = text.count("'") + text.count("\u2019")
    if len(text) > 10 and apostrophes / len(text) > 0.02:
        return False
    words = re.findall(r"[a-záéíóúüñ]+", text.lower())
    if len(words) >= 3:
        known = sum(1 for w in words if w in SPA_COMMON)
        known_ratio = known / len(words)
        # Sliding scale: high Whisper confidence → lower word threshold.
        # avg_log_prob ~ -0.2 (very confident) → threshold 0.15
        # avg_log_prob ~ -0.5 (moderate)       → threshold 0.25
        # avg_log_prob ~ -1.0 (low confidence)  → threshold 0.35
        confidence = max(0.0, min(1.0, (avg_log_prob + 2.0) / 2.0))
        word_threshold = 0.35 - 0.20 * confidence  # range [0.15, 0.35]
        if known_ratio < word_threshold:
            return False
    if avg_log_prob < -1.0:
        return False
    return True


def has_non_latin_script(text: str) -> bool:
    """Return True when text contains alphabetic chars outside Latin script."""
    for ch in text:
        if not ch.isalpha():
            continue
        if "LATIN" not in unicodedata.name(ch, ""):
            return True
    return False


def _transliterate_word_to_spanish_orthography(word: str) -> str:
    """Apply conservative Spanish-orthography mapping to a token."""
    out = word
    out = re.sub(r"sh", "x", out, flags=re.IGNORECASE)
    out = re.sub(r"ts", "tz", out, flags=re.IGNORECASE)
    out = re.sub(r"kw", "cu", out, flags=re.IGNORECASE)
    out = re.sub(r"w", "hu", out, flags=re.IGNORECASE)
    out = re.sub(r"k(?=[eiéí])", "qu", out, flags=re.IGNORECASE)
    out = re.sub(r"k", "c", out, flags=re.IGNORECASE)
    return out


def transliterate_to_spanish_orthography(text: str) -> str:
    """Normalize transcription into Spanish-style orthography."""
    if not text:
        return text
    transliterated = re.sub(
        r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+",
        lambda m: _transliterate_word_to_spanish_orthography(m.group(0)),
        text,
    )
    transliterated = re.sub(r"\s+", " ", transliterated).strip()
    return transliterated


_CANON_EQUIV = {
    "t͡ɕ": "tʃ", "tɕ": "tʃ", "tʂ": "tʃ", "t͡ʃ": "tʃ",
    "ɕ": "ʃ", "ʂ": "ʃ",
    "k̟ʲ": "kʷ", "kw": "kʷ",
    "ð": "d", "β": "b", "ɣ": "ɡ", "ɸ": "f", "ɻ": "ɾ", "r": "ɾ",
}

_NAH_FALLBACK_RULES = [
    ("tla", ["tɬ", "a"]), ("tle", ["tɬ", "e"]), ("tli", ["tɬ", "i"]),
    ("tlo", ["tɬ", "o"]), ("tlu", ["tɬ", "u"]),
    ("tl", ["tɬ"]), ("tz", ["ts"]), ("ts", ["ts"]), ("ch", ["tʃ"]),
    ("sh", ["ʃ"]), ("kw", ["kʷ"]), ("hu", ["w"]), ("uh", ["w"]),
    ("qu", ["k"]), ("cu", ["kʷ"]),
]
_NAH_FALLBACK_SINGLE = {
    "x": "ʃ", "h": "ʔ", "y": "j", "w": "w", "z": "s", "c": "k", "k": "k", "q": "k",
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "b": "b", "d": "d", "f": "f", "g": "ɡ", "m": "m", "n": "n", "p": "p", "t": "t",
    "l": "l", "r": "ɾ", "s": "s",
}
_EPITRAN_CACHE: dict[str, object | None] = {}


def canonicalize_ipa_tokens(
    phonemes: list[str],
    acoustic_tl_detected: bool = False,
) -> list[str]:
    out: list[str] = []
    for ph in phonemes:
        raw = ph.strip()
        if not raw:
            continue
        stripped = strip_modifiers(raw)
        norm = ALLOPHONE_MAP.get(stripped, stripped)
        canon = _CANON_EQUIV.get(norm, norm)
        out.append(canon)

    # Merge split markers that often appear in backend outputs.
    merged: list[str] = []
    i = 0
    while i < len(out):
        if i + 1 < len(out) and out[i] == "k" and out[i + 1] == "w":
            merged.append("kʷ")
            i += 2
            continue
        # Only merge t+l => tɬ when acoustics confirmed lateral affricate.
        if acoustic_tl_detected and i + 1 < len(out) and out[i] == "t" and out[i + 1] == "l":
            merged.append("tɬ")
            i += 2
            continue
        merged.append(out[i])
        i += 1
    return merged


def ipa_agreement(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    ca = Counter(a)
    cb = Counter(b)
    inter = sum(min(ca[k], cb[k]) for k in set(ca) | set(cb))
    p = inter / max(sum(ca.values()), 1)
    r = inter / max(sum(cb.values()), 1)
    return 0.0 if (p + r) == 0 else (2.0 * p * r) / (p + r)


def _align_token_indices(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch alignment on token sequences."""
    m, n = len(a), len(b)
    # costs: match=0, replace=1, gap=1
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,  # match/substitute
                dp[i - 1][j] + 1,         # delete
                dp[i][j - 1] + 1,         # insert
            )
    pairs: list[tuple[int | None, int | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
            continue
        pairs.append((None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


def _confidence_list_for_tokens(
    tokens: list[str],
    phone_conf: list[dict],
    default_conf: float,
) -> list[float]:
    conf = [default_conf] * len(tokens)
    if not phone_conf:
        return conf
    n = min(len(tokens), len(phone_conf))
    for i in range(n):
        try:
            conf[i] = float(phone_conf[i].get("c", default_conf))
        except Exception:
            conf[i] = default_conf
    return conf


def fuse_ipa_by_phone_vote(
    primary_tokens: list[str],
    compare_tokens: list[str],
    primary_phone_conf: list[dict],
    compare_phone_conf: list[dict],
    primary_is_w2v2: bool,
    compare_is_w2v2: bool,
    default_non_w2v2_conf: float = 0.38,
    segment_duration: float = 1.0,  # FIX 8: Duration-based confidence
) -> tuple[list[str], float]:
    """Fuse two IPA token streams by local token voting with confidences.

    FIX 8: On short segments (<0.8s), boost w2v2 confidence since Allosaurus
    tends to collapse on brief audio while w2v2 handles it better.
    """
    if not primary_tokens and not compare_tokens:
        return [], 0.0
    if primary_tokens and not compare_tokens:
        return primary_tokens, 0.0
    if compare_tokens and not primary_tokens:
        return compare_tokens, 0.0

    # FIX 8: Duration-based confidence adjustment
    # Short segments: w2v2 gets boost, Allosaurus gets penalty
    short_boost = 0.0
    if segment_duration < 0.8:
        short_boost = 0.15 * (0.8 - segment_duration) / 0.8  # Max 0.15 at 0s

    w2v2_default = 0.55 + short_boost
    allo_default = max(0.25, default_non_w2v2_conf - short_boost)

    conf_a = _confidence_list_for_tokens(
        primary_tokens,
        primary_phone_conf,
        w2v2_default if primary_is_w2v2 else allo_default,
    )
    conf_b = _confidence_list_for_tokens(
        compare_tokens,
        compare_phone_conf,
        w2v2_default if compare_is_w2v2 else allo_default,
    )
    pairs = _align_token_indices(primary_tokens, compare_tokens)
    fused: list[str] = []
    chosen_conf: list[float] = []
    for ia, ib in pairs:
        if ia is None and ib is None:
            continue
        if ia is None:
            fused.append(compare_tokens[ib])
            chosen_conf.append(conf_b[ib])
            continue
        if ib is None:
            fused.append(primary_tokens[ia])
            chosen_conf.append(conf_a[ia])
            continue
        ta = primary_tokens[ia]
        tb = compare_tokens[ib]
        if ta == tb:
            fused.append(ta)
            chosen_conf.append(max(conf_a[ia], conf_b[ib]))
            continue
        # Prefer the higher-confidence side; small tie bias to w2v2.
        sa = conf_a[ia] + (0.04 if primary_is_w2v2 else 0.0)
        sb = conf_b[ib] + (0.04 if compare_is_w2v2 else 0.0)
        if sb > sa + 0.02:
            fused.append(tb)
            chosen_conf.append(conf_b[ib])
        else:
            fused.append(ta)
            chosen_conf.append(conf_a[ia])
    mean_conf = (sum(chosen_conf) / len(chosen_conf)) if chosen_conf else 0.0
    return fused, max(0.0, min(1.0, mean_conf))


def _epitran_tokens(text: str, lang_code: str) -> list[str]:
    try:
        import epitran
    except Exception:
        return []
    epi = _EPITRAN_CACHE.get(lang_code)
    if epi is None and lang_code not in _EPITRAN_CACHE:
        try:
            epi = epitran.Epitran(lang_code)
        except Exception:
            epi = None
        _EPITRAN_CACHE[lang_code] = epi
    if epi is None:
        return []
    tokens: list[str] = []
    for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’:-]+", text):
        ipa = epi.transliterate(w)
        if not ipa:
            continue
        if " " in ipa:
            tokens.extend([x for x in ipa.split() if x])
        else:
            tokens.extend([c for c in ipa if c.strip()])
    return tokens


def _fallback_text_to_ipa_tokens(text: str) -> list[str]:
    out: list[str] = []
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’:-]+", text.lower())
    for word in words:
        w = word.replace("’", "'")
        i = 0
        while i < len(w):
            matched = False
            for g, ipa in _NAH_FALLBACK_RULES:
                if w.startswith(g, i):
                    out.extend(ipa)
                    i += len(g)
                    matched = True
                    break
            if matched:
                continue
            ch = w[i]
            if ch in _NAH_FALLBACK_SINGLE:
                out.append(_NAH_FALLBACK_SINGLE[ch])
            i += 1
    return out


def text_to_ipa_tokens(text: str, text_lang_hint: str) -> list[str]:
    if not text:
        return []
    lang_code = "nah-Latn" if text_lang_hint == "nah" else "spa-Latn"
    tokens = _epitran_tokens(text, lang_code)
    if not tokens:
        tokens = _fallback_text_to_ipa_tokens(text)
    return canonicalize_ipa_tokens(tokens)


def _has_spanish_orthography(text: str) -> bool:
    """Check if text contains Spanish-specific orthographic features."""
    # Spanish inverted punctuation
    if "¿" in text or "¡" in text:
        return True
    # Spanish accented vowels (common in es, rare in en/de/fr)
    _SPA_ACCENTS = set("áéíóúñ")
    if any(c in _SPA_ACCENTS for c in text.lower()):
        return True
    return False


def guess_language_from_text_markers(text: str, whisper_lang: str) -> tuple[str, float]:
    """Guess language from text markers in Latin/Spanish orthography."""
    if not text:
        return ("other", 0.0)

    is_latin, _ = detect_latin_text(text)
    if is_latin:
        return ("lat", 0.95)

    normalized = _normalize_word_ascii(text)
    words = re.findall(r"[a-z']+", normalized)
    if not words:
        return ("other", 0.0)

    spa_hits = sum(1 for w in words if w in SPA_COMMON)
    spa_ratio = spa_hits / max(len(words), 1)
    spa_ortho = _has_spanish_orthography(text)

    nah_hits = 0
    may_hits = 0
    for w in words:
        if any(m in w for m in NAH_TEXT_MARKERS):
            nah_hits += 1
        if any(m in w for m in MAY_TEXT_MARKERS):
            may_hits += 1

    # SPA detection: trust text content regardless of Whisper's language code
    # Whisper often misdetects es as en/pt/ca — but the transcribed text is correct
    if spa_ratio >= 0.25 and spa_hits >= 2:
        return ("spa", min(0.95, 0.55 + spa_ratio))
    if may_hits >= 2 and may_hits > nah_hits:
        return ("may", min(0.9, 0.45 + 0.15 * may_hits))
    if nah_hits >= 2:
        return ("nah", min(0.9, 0.45 + 0.12 * nah_hits))
    # Spanish orthography (¿¡áéíóú) is strong evidence even with few SPA_COMMON hits
    if spa_ortho and spa_hits >= 1:
        return ("spa", min(0.90, 0.55 + spa_ratio))
    # Short segments (1-3 words) with high SPA ratio: relax hits threshold
    # e.g. "Pero creedme" → spa_hits=1, spa_ratio=0.5, total_words=2
    if spa_hits >= 1 and spa_ratio >= 0.4 and len(words) <= 4:
        return ("spa", min(0.85, 0.50 + spa_ratio))
    if whisper_lang == "es":
        return ("spa", 0.55 if spa_ratio >= 0.1 else 0.45)
    return (LANG_MAP.get(whisper_lang, "other"), 0.4)


def find_gaps(intervals, total_duration, min_gap=0.5):
    if not intervals:
        return [(0.0, total_duration)]
    sorted_iv = sorted(intervals)
    gaps = []
    if sorted_iv[0][0] > min_gap:
        gaps.append((0.0, sorted_iv[0][0]))
    for i in range(len(sorted_iv) - 1):
        gap = sorted_iv[i + 1][0] - sorted_iv[i][1]
        if gap >= min_gap:
            gaps.append((sorted_iv[i][1], sorted_iv[i + 1][0]))
    if total_duration - sorted_iv[-1][1] >= min_gap:
        gaps.append((sorted_iv[-1][1], total_duration))
    return gaps


_BLEED_STOPWORDS = {
    "de", "del", "la", "el", "que", "en", "y", "por", "para", "con", "no", "es", "un", "una",
}


def _norm_text_for_bleed(text: str) -> str:
    t = _normalize_word_ascii(text or "")
    t = re.sub(r"[^a-z0-9\\s]", " ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t


def suppress_bleed_fragments(results: list[dict]) -> tuple[list[dict], int, int]:
    """Merge/prune short text fragments that are likely copied from neighboring lines.

    Preference:
    1) Merge into previous segment if contiguous and same speaker/lang (keeps IPA continuity).
    2) Otherwise prune fragment text only.
    """
    merged = 0
    pruned = 0
    out: list[dict] = []
    i = 0
    n = len(results)
    while i < n:
        r = dict(results[i])
        text = (r.get("text") or "").strip()
        should_check = bool(text)
        bleed = False

        if should_check:
            dur = max(0.0, float(r.get("end", 0.0)) - float(r.get("start", 0.0)))
            norm = _norm_text_for_bleed(text)
            words = norm.split()
            if dur <= 1.2 and 0 < len(words) <= 4 and len(norm) >= 6 and any(w in _BLEED_STOPWORDS for w in words):
                neighbors = []
                if i - 1 >= 0:
                    neighbors.append(results[i - 1])
                if i + 1 < n:
                    neighbors.append(results[i + 1])
                for nb in neighbors:
                    nb_text = (nb.get("text") or "").strip()
                    if not nb_text:
                        continue
                    nb_norm = _norm_text_for_bleed(nb_text)
                    nb_words = nb_norm.split()
                    if len(nb_words) >= len(words) + 3 and norm and norm in nb_norm:
                        bleed = True
                        break

        if bleed and out:
            prev = out[-1]
            same_speaker = prev.get("speaker") == r.get("speaker")
            same_lang = prev.get("lang") == r.get("lang")
            close = (float(r.get("start", 0.0)) - float(prev.get("end", 0.0))) <= 1.2
            if same_speaker and same_lang and close:
                if r.get("ipa"):
                    prev["ipa"] = ((prev.get("ipa") or "").strip() + " " + r["ipa"].strip()).strip()
                if r.get("ipa_compare"):
                    prev["ipa_compare"] = ((prev.get("ipa_compare") or "").strip() + " " + r["ipa_compare"].strip()).strip()
                prev["end"] = max(float(prev.get("end", 0.0)), float(r.get("end", 0.0)))
                prev["backend"] = (prev.get("backend") or "") + "+bleed-merged"
                out[-1] = prev
                merged += 1
                i += 1
                continue

            # Fallback: keep segment but drop suspect fragment text.
            r["text"] = None
            r["whisper_valid"] = False
            r["backend"] = (r.get("backend") or "") + "+bleed-pruned"
            pruned += 1

        out.append(r)
        i += 1

    return out, merged, pruned


def fmt_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# IPA->Text Recovery (reverse G2P for uncertain segment recovery)
# ---------------------------------------------------------------------------

# Reverse G2P maps for converting IPA back to readable text
_NAH_IPA_TO_TEXT = {
    # --- Core Nahuatl consonants ---
    "tɬ": "tl", "ts": "tz", "tʃ": "ch", "ʃ": "x", "kʷ": "cu",
    "ʔ": "h", "j": "y", "w": "hu", "k": "c", "s": "s",
    "p": "p", "t": "t", "m": "m", "n": "n", "l": "l",
    # --- Core vowels ---
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    # --- Allophonic vowel variants (fused pipeline) ---
    "ɒ": "o", "ә": "e", "ʌ": "a", "ɪ": "i", "ə": "e",
    "ɨ": "i", "ɔ": "o", "æ": "a", "ɛ": "e", "ɑ": "a",
    "ʊ": "u", "ɐ": "a", "ø": "e", "y": "i", "ɵ": "o",
    # --- Diphthongs / digraphs ---
    "uə": "ua", "aɪ": "ai", "ij": "i", "ei": "ei",
    "aʊ": "au", "oɪ": "oi", "eɪ": "ei", "iə": "ia",
    # --- Allophonic consonant variants ---
    "ʁ": "r", "ɴ": "n", "ŋ": "n", "ɾ": "r", "d": "t",
    "b": "p", "ɡ": "c", "v": "hu", "f": "p", "h": "h",
    "x": "x", "ɳ": "n", "r": "r", "ɲ": "n", "ʎ": "l",
    "ɹ": "r", "ɻ": "r", "ʙ": "p", "ɖ": "t", "ɟ": "c",
    "ɢ": "c", "ɭ": "l", "ɬ": "l", "ɮ": "l", "ʂ": "x",
    "ʐ": "x", "ɕ": "x", "ʑ": "x", "ç": "x", "ɣ": "c",
    "ʕ": "h", "ɦ": "h", "ʋ": "hu", "β": "p", "ð": "t",
    "θ": "s", "z": "s", "ʒ": "x", "ʝ": "y", "ɰ": "hu",
    # --- Affricates / complex consonants ---
    "kp": "cu", "dʒ": "ch", "tɕ": "ch", "tɕh": "ch",
    "l̪": "l", "ɫ": "l", "pf": "p",
}

_SPA_IPA_TO_TEXT = {
    "tʃ": "ch", "ʃ": "sh", "ɲ": "ñ", "ʎ": "ll", "ɾ": "r",
    "r": "rr", "β": "b", "ð": "d", "ɣ": "g", "x": "j",
    "θ": "z", "b": "b", "d": "d", "ɡ": "g", "f": "f",
    "k": "c", "p": "p", "t": "t", "m": "m", "n": "n",
    "l": "l", "s": "s", "j": "y", "w": "u",
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    # --- Allophonic vowel variants (fused pipeline) ---
    "ɒ": "o", "ә": "e", "ʌ": "a", "ɪ": "i", "ə": "e",
    "ɨ": "i", "ɔ": "o", "æ": "a", "ɛ": "e", "ɑ": "a",
    "ʊ": "u", "ɐ": "a", "ø": "e", "y": "i",
    # --- Allophonic consonant variants ---
    "ʁ": "r", "ɴ": "n", "ŋ": "n", "ɳ": "n", "ɹ": "r",
    "ɻ": "r", "ʋ": "v", "ɕ": "sh", "ʑ": "sh", "ç": "j",
    "ɦ": "j", "ʕ": "j", "ʒ": "sh", "z": "s", "v": "v",
    "h": "j", "ɰ": "u", "ɫ": "l", "ɭ": "l", "l̪": "l",
    # --- Diphthongs ---
    "uə": "ua", "aɪ": "ai", "ij": "i", "ei": "ei",
    "aʊ": "au", "oɪ": "oi", "eɪ": "ei", "iə": "ia",
    # --- Affricates ---
    "dʒ": "ch", "tɕ": "ch", "tɕh": "ch", "kp": "c",
}

_MAY_IPA_TO_TEXT = {
    **_NAH_IPA_TO_TEXT,
    "kʼ": "k'", "tsʼ": "ts'", "tʃʼ": "ch'", "pʼ": "p'", "tʼ": "t'",
}


def _ipa_to_text_recovery(ipa_str: str, lang: str) -> str | None:
    """Convert IPA string back to readable text using reverse G2P mapping.

    Args:
        ipa_str: Space-separated IPA tokens
        lang: Language code (nah/may/spa/other)

    Returns:
        Recovered text string, or None if coverage is too low (<30%)
    """
    # Select reverse map by language
    if lang.lower() in {"nah", "may"}:
        reverse_map = _MAY_IPA_TO_TEXT if lang.lower() == "may" else _NAH_IPA_TO_TEXT
    elif lang.lower() == "spa":
        reverse_map = _SPA_IPA_TO_TEXT
    else:
        return None  # No map available, fall through to LLM

    # Tokenize IPA string
    tokens = ipa_str.strip().split()
    if not tokens:
        return None

    # Recover graphemes using longest-match-first
    recovered = []
    total_ipa_chars = len("".join(tokens))
    matched_ipa_chars = 0

    for token in tokens:
        # Try the token as-is first
        if token in reverse_map:
            recovered.append(reverse_map[token])
            matched_ipa_chars += len(token)
        else:
            # Try longest-prefix match
            matched = False
            for i in range(len(token), 0, -1):
                prefix = token[:i]
                if prefix in reverse_map:
                    recovered.append(reverse_map[prefix])
                    matched_ipa_chars += len(prefix)
                    # Handle remainder recursively if needed
                    if i < len(token):
                        remainder = token[i:]
                        if remainder in reverse_map:
                            recovered.append(reverse_map[remainder])
                            matched_ipa_chars += len(remainder)
                    matched = True
                    break

            if not matched:
                # No mapping found, keep as-is
                recovered.append(token)

    # Check coverage threshold
    coverage = matched_ipa_chars / total_ipa_chars if total_ipa_chars > 0 else 0
    if coverage < 0.3:
        return None  # Insufficient coverage, let LLM try

    # Join without spaces — downstream NAH morphology checks (NAH_TEXT_RX,
    # NAH_TEXT_MARKERS) need contiguous orthography like "itlacam" not "i t l a c a m"
    return "".join(recovered)


def _llm_transliterate_ipa(ipa_str: str, lang: str) -> str | None:
    """Convert IPA to readable text using LLM transliteration.

    This is a stub for future LLM integration. Currently returns None.

    Args:
        ipa_str: Space-separated IPA tokens
        lang: Language code

    Returns:
        LLM-transliterated text, or None if unavailable
    """
    # TODO: wire to LLM API when available
    # Suggested prompt:
    # f"Convert this IPA phonetic transcription to readable {lang} text. IPA: {ipa_str}. Output ONLY the text, no explanations."
    return None


def _recover_uncertain_text(r: dict) -> tuple[str | None, str]:
    """Orchestrate IPA->text recovery for an uncertain segment.

    Args:
        r: Result dict with IPA and language fields

    Returns:
        Tuple of (recovered_text, method) where method is "g2p", "llm", or "none"
    """
    # Get best available IPA
    ipa = r.get("ipa_fused") or r.get("ipa_compare") or r.get("ipa")
    if not ipa or not ipa.strip():
        return (None, "none")

    # Get language
    lang = r.get("lang", "other")

    # Try rule-based recovery first
    recovered = _ipa_to_text_recovery(ipa, lang)
    if recovered:
        return (recovered, "g2p")

    # Try LLM fallback
    recovered = _llm_transliterate_ipa(ipa, lang)
    if recovered:
        return (recovered, "llm")

    return (None, "none")


# ---------------------------------------------------------------------------
# IPA↔Whisper Match Check (prefer IPA when Whisper text doesn't match phonemes)
# ---------------------------------------------------------------------------

# Simplified Text→IPA maps for comparison (forward G2P)
_NAH_TEXT_TO_IPA = {
    "tl": "tɬ", "tz": "ts", "ch": "tʃ", "x": "ʃ", "cu": "kʷ", "qu": "k",
    "hu": "w", "c": "k", "y": "j", "h": "ʔ",
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "p": "p", "t": "t", "m": "m", "n": "n", "l": "l", "s": "s",
}

_SPA_TEXT_TO_IPA = {
    "ch": "tʃ", "ll": "ʎ", "ñ": "ɲ", "rr": "r", "j": "x", "g": "ɡ",
    "qu": "k", "c": "k", "z": "s", "v": "b", "b": "b", "d": "d",
    "f": "f", "h": "", "y": "j", "w": "w",
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "p": "p", "t": "t", "m": "m", "n": "n", "l": "l", "s": "s", "r": "ɾ",
}


def _text_to_ipa_simple(text: str, lang: str) -> str:
    """Convert text to simplified IPA for comparison.

    Args:
        text: Input text (Whisper output)
        lang: Language code (nah/spa/other)

    Returns:
        Simplified IPA representation for comparison
    """
    if not text:
        return ""

    text = text.lower().strip()
    # Remove punctuation and special chars
    text = "".join(c for c in text if c.isalnum() or c.isspace())

    # Select map
    if lang.lower() in {"nah", "may"}:
        g2p_map = _NAH_TEXT_TO_IPA
    elif lang.lower() == "spa":
        g2p_map = _SPA_TEXT_TO_IPA
    else:
        return ""  # No map available

    # Convert using longest-match-first
    result = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue

        matched = False
        # Try 2-char, then 1-char matches
        for length in [2, 1]:
            if i + length <= len(text):
                chunk = text[i:i+length]
                if chunk in g2p_map:
                    if g2p_map[chunk]:  # Skip empty mappings (like 'h' in Spanish)
                        result.append(g2p_map[chunk])
                    i += length
                    matched = True
                    break

        if not matched:
            # Keep as-is if no mapping
            result.append(text[i])
            i += 1

    return " ".join(result)


def _ipa_similarity(ipa1: str, ipa2: str) -> float:
    """Calculate similarity between two IPA strings.

    Uses character-level Jaccard similarity on normalized IPA tokens.

    Returns:
        Similarity score 0.0-1.0
    """
    if not ipa1 or not ipa2:
        return 0.0

    # Normalize: remove spaces, common variants
    def normalize(s):
        s = s.lower().replace(" ", "")
        # Normalize common variants
        s = s.replace("ɾ", "r").replace("ɡ", "g").replace("ʔ", "")
        s = s.replace("ʃ", "sh").replace("tʃ", "ch").replace("tɬ", "tl")
        s = s.replace("ɲ", "ny").replace("ʎ", "ly").replace("kʷ", "kw")
        return s

    n1 = normalize(ipa1)
    n2 = normalize(ipa2)

    if not n1 or not n2:
        return 0.0

    # Character-level Jaccard
    set1 = set(n1)
    set2 = set(n2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def _whisper_matches_ipa(whisper_text: str, ipa_str: str, lang: str, threshold: float = 0.4) -> bool:
    """Check if Whisper text phonetically matches the IPA transcription.

    Args:
        whisper_text: Text output from Whisper
        ipa_str: Actual IPA from phoneme backend
        lang: Language code
        threshold: Minimum similarity score to consider a match

    Returns:
        True if Whisper text matches IPA within threshold
    """
    if not whisper_text or not ipa_str:
        return True  # Can't check, assume match

    # Convert Whisper text to IPA
    whisper_ipa = _text_to_ipa_simple(whisper_text, lang)
    if not whisper_ipa:
        return True  # No conversion possible, assume match

    # Compare
    similarity = _ipa_similarity(whisper_ipa, ipa_str)
    return similarity >= threshold


# ---------------------------------------------------------------------------
# Omnilingual Function — separate container with compatible deps
# ---------------------------------------------------------------------------

@app.function(
    image=omnilingual_image,
    gpu="T4",
    timeout=600,
    volumes={CACHE_DIR: model_cache},
)
def run_omnilingual(audio_bytes: bytes, segments: list[dict]) -> list[dict]:
    """Run Omnilingual ASR on audio segments in separate container.

    Args:
        audio_bytes: WAV audio data
        segments: List of {"start": float, "end": float} dicts

    Returns:
        List of {"start": float, "end": float, "ipa": str} dicts
    """
    import tempfile
    import soundfile as sf
    import numpy as np

    if not segments:
        return []

    # Write audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    # Load audio
    audio_data, sr = sf.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Initialize Omnilingual
    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        pipeline = ASRInferencePipeline(model_card="omniASR_LLM_300M_v2", device="cuda")
        print("Omnilingual: omniASR_LLM_300M_v2 on cuda")
    except Exception as e:
        print(f"Omnilingual init failed: {e}")
        return [{"start": s["start"], "end": s["end"], "ipa": None} for s in segments]

    results = []
    for seg in segments:
        start_s = int(seg["start"] * sr)
        end_s = int(seg["end"] * sr)
        chunk = audio_data[start_s:end_s]

        if len(chunk) < sr * 0.2:
            results.append({"start": seg["start"], "end": seg["end"], "ipa": None})
            continue

        # Write chunk to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as cf:
            sf.write(cf.name, chunk, sr)
            chunk_path = cf.name

        try:
            result = pipeline.transcribe([chunk_path], lang=["spa_Latn"])
            text = ""
            if isinstance(result, list) and result:
                entry = result[0]
                text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
            elif isinstance(result, dict):
                text = result.get("text", "")

            # Convert to IPA using epitran
            ipa = None
            if text:
                try:
                    import epitran
                    epi = epitran.Epitran("spa-Latn")
                    ipa = epi.transliterate(text)
                except Exception:
                    ipa = f"[{text}]"

            results.append({"start": seg["start"], "end": seg["end"], "ipa": ipa})
        except Exception as e:
            print(f"Omnilingual segment error: {e}")
            results.append({"start": seg["start"], "end": seg["end"], "ipa": None})
        finally:
            import os
            os.unlink(chunk_path)

    import os
    os.unlink(audio_path)
    return results


# ---------------------------------------------------------------------------
# SepFormer Voice Separation (Modal GPU)
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=1200,
    volumes={CACHE_DIR: model_cache},
)
def separate_voices_sepformer(
    audio_bytes: bytes,
    filename: str = "mixed.wav",
    start_s: float = 0.0,
    end_s: float = 0.0,
) -> dict:
    """Separate overlapping voices with SpeechBrain SepFormer.

    Args:
        audio_bytes: Input audio bytes (wav/mp4/etc)
        filename: Original filename (used for suffix detection)
        start_s: Optional clip start (seconds); <=0 means from beginning
        end_s: Optional clip end (seconds); <= start_s means full file

    Returns:
        dict with sample_rate and per-speaker WAV bytes.
    """
    import tempfile
    import subprocess
    import soundfile as sf
    import numpy as np
    import torch
    from speechbrain.inference.separation import SepformerSeparation

    os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"

    suffix = Path(filename).suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        input_path = tmpdir_p / f"input{suffix}"
        input_path.write_bytes(audio_bytes)

        wav_path = tmpdir_p / "converted.wav"  # must differ from input for .wav files
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
        ]
        if start_s > 0:
            ffmpeg_cmd.extend(["-ss", f"{start_s:.3f}"])
        if end_s > 0 and end_s > start_s:
            ffmpeg_cmd.extend(["-to", f"{end_s:.3f}"])
        ffmpeg_cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path)])
        r = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {r.stderr[-500:]}")

        model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr",
            savedir=f"{CACHE_DIR}/speechbrain/sepformer-whamr",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        separated = model.separate_file(path=str(wav_path))
        if isinstance(separated, torch.Tensor):
            sep = separated.detach().cpu().numpy()
        else:
            sep = np.array(separated)

        # Normalize shape to [samples, speakers]
        while sep.ndim > 2:
            sep = sep[0]
        if sep.ndim == 1:
            sep = sep[:, None]
        if sep.shape[0] < sep.shape[1]:
            sep = sep.T

        sr = 8000
        try:
            sr = int(getattr(model.hparams, "sample_rate", 8000))
        except Exception:
            pass

        sources = []
        for i in range(sep.shape[1]):
            wav = sep[:, i].astype(np.float32)
            out_path = tmpdir_p / f"speaker_{i+1}.wav"
            sf.write(str(out_path), wav, sr)
            sources.append({
                "speaker_id": f"SPEAKER_{i+1:02d}",
                "duration_s": len(wav) / float(sr),
                "wav_bytes": out_path.read_bytes(),
            })

        return {
            "sample_rate": sr,
            "num_sources": len(sources),
            "sources": sources,
        }


@app.function(
    gpu="T4",
    timeout=1200,
    volumes={CACHE_DIR: model_cache},
)
def separate_voices_convtasnet(
    audio_bytes: bytes,
    filename: str = "mixed.wav",
    start_s: float = 0.0,
    end_s: float = 0.0,
) -> dict:
    """Separate overlapping voices with Asteroid ConvTasNet.

    ConvTasNet is often better than SepFormer for speech separation.
    Model: Cosentino & Pariente pretrained on Libri2Mix.
    """
    import tempfile
    import subprocess
    import soundfile as sf
    import numpy as np
    import torch
    from asteroid.models import ConvTasNet

    os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"

    suffix = Path(filename).suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        input_path = tmpdir_p / f"input{suffix}"
        input_path.write_bytes(audio_bytes)

        # Convert to 8kHz mono (ConvTasNet trained on 8kHz)
        wav_path = tmpdir_p / "converted.wav"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
        ]
        if start_s > 0:
            ffmpeg_cmd.extend(["-ss", f"{start_s:.3f}"])
        if end_s > 0 and end_s > start_s:
            ffmpeg_cmd.extend(["-to", f"{end_s:.3f}"])
        ffmpeg_cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(wav_path)])
        r = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {r.stderr[-500:]}")

        # Load model
        model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()

        # Load audio
        audio, sr = sf.read(str(wav_path))
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # [batch, channels, time]
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()

        # Separate
        with torch.no_grad():
            separated = model(audio_tensor)  # [batch, n_sources, time]

        sep = separated[0].cpu().numpy()  # [n_sources, time]

        sources = []
        for i in range(sep.shape[0]):
            wav = sep[i]
            # Normalize
            wav = wav / (np.max(np.abs(wav)) + 1e-9) * 0.9
            out_path = tmpdir_p / f"source_{i}.wav"
            sf.write(str(out_path), wav, sr)
            sources.append({
                "speaker_id": f"SPEAKER_{i+1:02d}",
                "duration_s": len(wav) / float(sr),
                "wav_bytes": out_path.read_bytes(),
            })

        return {
            "sample_rate": sr,
            "num_sources": len(sources),
            "sources": sources,
            "model": "ConvTasNet_Libri2Mix",
        }


@app.function(
    gpu="T4",
    timeout=1200,
    volumes={CACHE_DIR: model_cache},
)
def separate_voices_pitch(
    audio_bytes: bytes,
    filename: str = "mixed.wav",
    start_s: float = 0.0,
    end_s: float = 0.0,
    n_harmonics: int = 8,
    bandwidth_hz: float = 40.0,
) -> dict:
    """Separate voices based on pitch using PYIN + Wiener filtering.

    Works best when speakers have distinct F0 ranges (male ~100Hz, female ~220Hz).
    No ML model needed - pure signal processing.
    """
    import tempfile
    import subprocess
    import soundfile as sf
    import numpy as np
    import librosa
    from scipy.signal import stft, istft

    suffix = Path(filename).suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        input_path = tmpdir_p / f"input{suffix}"
        input_path.write_bytes(audio_bytes)

        # Convert to 16kHz mono
        wav_path = tmpdir_p / "converted.wav"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
        ]
        if start_s > 0:
            ffmpeg_cmd.extend(["-ss", f"{start_s:.3f}"])
        if end_s > 0 and end_s > start_s:
            ffmpeg_cmd.extend(["-to", f"{end_s:.3f}"])
        ffmpeg_cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path)])
        r = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {r.stderr[-500:]}")

        # Load audio
        audio, sr = sf.read(str(wav_path))

        # Detect F0 in male and female ranges
        f0_male, _, _ = librosa.pyin(
            audio, fmin=60, fmax=180, sr=sr,
            frame_length=2048, hop_length=512, fill_na=0.0
        )
        f0_female, _, _ = librosa.pyin(
            audio, fmin=150, fmax=400, sr=sr,
            frame_length=2048, hop_length=512, fill_na=0.0
        )

        # STFT
        nperseg = 2048
        noverlap = nperseg * 3 // 4
        f, t, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)

        def build_mask(f0_traj):
            """Build harmonic mask from F0 trajectory."""
            mask = np.zeros((len(f), Zxx.shape[1]))
            f0_interp = np.interp(
                np.linspace(0, 1, Zxx.shape[1]),
                np.linspace(0, 1, len(f0_traj)),
                f0_traj
            )
            for ti in range(Zxx.shape[1]):
                f0 = f0_interp[ti]
                if f0 <= 0:
                    continue
                for h in range(1, n_harmonics + 1):
                    harmonic_freq = f0 * h
                    if harmonic_freq > sr / 2:
                        break
                    sigma = bandwidth_hz / 2
                    mask[:, ti] += np.exp(-0.5 * ((f - harmonic_freq) / sigma) ** 2)
            return np.clip(mask, 0, 1)

        mask_male = build_mask(f0_male)
        mask_female = build_mask(f0_female)

        # Wiener filtering
        total_mask = mask_male + mask_female + 1e-8
        mask_male_norm = mask_male / total_mask
        mask_female_norm = mask_female / total_mask

        # Apply masks
        male_spec = Zxx * mask_male_norm
        female_spec = Zxx * mask_female_norm

        # Reconstruct
        _, male_audio = istft(male_spec, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, female_audio = istft(female_spec, fs=sr, nperseg=nperseg, noverlap=noverlap)

        # Normalize
        male_audio = male_audio / (np.max(np.abs(male_audio)) + 1e-9) * 0.9
        female_audio = female_audio / (np.max(np.abs(female_audio)) + 1e-9) * 0.9

        # Save
        sources = []
        for i, (wav, label, f0_arr) in enumerate([
            (male_audio, "male", f0_male),
            (female_audio, "female", f0_female),
        ]):
            out_path = tmpdir_p / f"source_{label}.wav"
            sf.write(str(out_path), wav, sr)
            f0_voiced = f0_arr[f0_arr > 0]
            sources.append({
                "speaker_id": label.upper(),
                "duration_s": len(wav) / float(sr),
                "wav_bytes": out_path.read_bytes(),
                "f0_median": float(np.median(f0_voiced)) if len(f0_voiced) > 0 else 0.0,
                "voiced_pct": float(100 * len(f0_voiced) / len(f0_arr)),
            })

        return {
            "sample_rate": sr,
            "num_sources": 2,
            "sources": sources,
            "model": "PYIN+Wiener",
        }


# ---------------------------------------------------------------------------
# MossFormer2 Voice Separation (16kHz, 2-speaker, State-of-the-art)
# ---------------------------------------------------------------------------

mossformer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "clearvoice",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "soundfile",
        "numpy<2.0",
    )
)

# Dedicated image alias for diarization tasks (pyannote is already installed in tenepal_image).
pyannote_image = tenepal_image


@app.function(
    image=mossformer_image,
    gpu="T4",
    timeout=1200,
    volumes={CACHE_DIR: model_cache},
)
def separate_voices_mossformer2(
    audio_bytes: bytes,
    filename: str = "mixed.wav",
    start_s: float = 0.0,
    end_s: float = 0.0,
) -> dict:
    """Separate 2 overlapping voices with MossFormer2 (ClearVoice).

    State-of-the-art speech separation at 16kHz.
    Better quality than SepFormer's 8kHz for downstream ASR/phoneme recognition.

    Returns:
        dict with sample_rate=16000, num_sources=2, and per-speaker WAV bytes.
    """
    import tempfile
    import subprocess
    import soundfile as sf
    import numpy as np
    from clearvoice import ClearVoice

    os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"

    suffix = Path(filename).suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        input_path = tmpdir_p / f"input{suffix}"
        input_path.write_bytes(audio_bytes)

        # Convert to 16kHz mono (MossFormer2 native rate)
        wav_path = tmpdir_p / "converted.wav"
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(input_path)]
        if start_s > 0:
            ffmpeg_cmd.extend(["-ss", f"{start_s:.3f}"])
        if end_s > 0 and end_s > start_s:
            ffmpeg_cmd.extend(["-to", f"{end_s:.3f}"])
        ffmpeg_cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path)])
        r = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {r.stderr[-500:]}")

        # Load model
        print("Loading MossFormer2_SS_16K...")
        cv = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

        # Separate
        print("Running separation...")
        output = cv(input_path=str(wav_path), online_write=False)

        # output is list of numpy arrays, each shape (1, samples)
        sources = []
        for i, separated in enumerate(output):
            out_path = tmpdir_p / f"source_{i}.wav"
            # Squeeze from (1, samples) to (samples,)
            audio_data = separated.squeeze() if len(separated.shape) > 1 else separated
            sf.write(str(out_path), audio_data, 16000)
            sources.append({
                "speaker_id": f"SOURCE_{i:02d}",
                "duration_s": len(audio_data) / 16000.0,
                "wav_bytes": out_path.read_bytes(),
            })
            print(f"  Source {i}: {len(audio_data)/16000:.1f}s")

        return {
            "sample_rate": 16000,
            "num_sources": len(sources),
            "sources": sources,
            "model": "MossFormer2_SS_16K",
        }


@app.function(
    image=mossformer_image,
    gpu="T4",
    timeout=1200,
    volumes={CACHE_DIR: model_cache},
)
def separate_overlap_segments(
    audio_bytes: bytes,
    segments: list[dict],
) -> dict:
    """Batch-separate overlap segments with MossFormer2.

    Args:
        audio_bytes: Full vocals WAV (16kHz mono).
        segments: [{"idx": int, "start": float, "end": float}, ...]

    Returns:
        dict keyed by segment idx, each value is a list of source dicts:
        {idx: [{"audio": bytes_of_float32, "rms": float, "sr": 16000}, ...]}
    """
    import tempfile
    import soundfile as sf
    import numpy as np
    from clearvoice import ClearVoice

    os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"

    if not segments:
        return {}

    # Load full audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        full_path = f.name
    audio_data, sr = sf.read(full_path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Load model once
    print(f"Loading MossFormer2 for {len(segments)} overlap segments...")
    cv = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

    results = {}
    for seg in segments:
        idx = seg["idx"]
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = audio_data[start_sample:end_sample]

        if len(chunk) < sr * 0.5:
            print(f"  Segment {idx}: too short ({len(chunk)/sr:.2f}s), skip")
            continue

        # Write segment to temp file
        seg_path = os.path.join(tempfile.gettempdir(), f"overlap_seg_{idx}.wav")
        sf.write(seg_path, chunk, sr)

        try:
            output = cv(input_path=seg_path, online_write=False)
            sources = []
            for i, separated in enumerate(output):
                audio_np = separated.squeeze() if len(separated.shape) > 1 else separated
                rms = float(np.sqrt(np.mean(audio_np ** 2)))
                # Serialize as float32 bytes for cross-container transfer
                sources.append({
                    "audio": audio_np.astype(np.float32).tobytes(),
                    "samples": len(audio_np),
                    "rms": rms,
                    "sr": 16000,
                })
            results[idx] = sources
            rms_str = " / ".join(f"{s['rms']:.4f}" for s in sources)
            print(f"  Segment {idx} ({seg['start']:.1f}-{seg['end']:.1f}s): "
                  f"{len(sources)} sources, RMS={rms_str}")
        except Exception as e:
            print(f"  Segment {idx}: separation failed: {e}")

    print(f"Separated {len(results)}/{len(segments)} segments")
    return results


@app.function(
    gpu="T4",
    timeout=1200,
    volumes={CACHE_DIR: model_cache},
)
def separate_voices_3speaker(
    audio_bytes: bytes,
    filename: str = "mixed.wav",
    start_s: float = 0.0,
    end_s: float = 0.0,
) -> dict:
    """Separate 3 overlapping voices with SepFormer-WSJ03Mix.

    Uses SpeechBrain's 3-speaker separation model.
    8kHz output (resample input as needed).

    Returns:
        dict with sample_rate=8000, num_sources=3, and per-speaker WAV bytes.
    """
    import tempfile
    import subprocess
    import soundfile as sf
    import numpy as np
    import torch
    from speechbrain.inference.separation import SepformerSeparation

    os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"

    suffix = Path(filename).suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        input_path = tmpdir_p / f"input{suffix}"
        input_path.write_bytes(audio_bytes)

        # Convert to 8kHz mono (WSJ03Mix trained on 8kHz)
        wav_path = tmpdir_p / "converted.wav"
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(input_path)]
        if start_s > 0:
            ffmpeg_cmd.extend(["-ss", f"{start_s:.3f}"])
        if end_s > 0 and end_s > start_s:
            ffmpeg_cmd.extend(["-to", f"{end_s:.3f}"])
        ffmpeg_cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(wav_path)])
        r = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {r.stderr[-500:]}")

        # Load 3-speaker SepFormer
        print("Loading SepFormer-WSJ03Mix (3-speaker)...")
        model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj03mix",
            savedir=f"{CACHE_DIR}/speechbrain/sepformer-wsj03mix",
            run_opts={"device": "cuda"} if torch.cuda.is_available() else {},
        )

        # Separate
        print("Running 3-speaker separation...")
        est_sources = model.separate_file(path=str(wav_path))
        # est_sources shape: (1, samples, num_sources)

        sources = []
        for i in range(est_sources.shape[2]):
            out_path = tmpdir_p / f"source_{i}.wav"
            audio_data = est_sources[:, :, i].squeeze().detach().cpu().numpy()
            sf.write(str(out_path), audio_data, 8000)
            sources.append({
                "speaker_id": f"SOURCE_{i:02d}",
                "duration_s": len(audio_data) / 8000.0,
                "wav_bytes": out_path.read_bytes(),
            })
            print(f"  Source {i}: {len(audio_data)/8000:.1f}s")

        return {
            "sample_rate": 8000,
            "num_sources": 3,
            "sources": sources,
            "model": "SepFormer-WSJ03Mix",
        }


# ---------------------------------------------------------------------------
# Voice Separation Comparison (3 Approaches)
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=3600,  # 1 hour for comprehensive comparison
    volumes={CACHE_DIR: model_cache},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def compare_separation_methods(
    audio_bytes: bytes,
    filename: str = "input.wav",
    start_s: float = 0.0,
    end_s: float = 0.0,
) -> dict:
    """Compare three voice separation approaches.

    Option A: SepFormer → Pyannote (separate first, then diarize each)
    Option B: Pyannote → SepFormer (diarize first, separate overlaps only)
    Option C: Cascaded 4-Stems (SepFormer twice for 4 total stems)

    Returns dict with results for each approach including:
    - Separated WAV bytes per speaker
    - Speaker segments with timing
    - F0 statistics per speaker
    - Separation quality metrics
    """
    import tempfile
    import subprocess
    import soundfile as sf
    import numpy as np
    import torch
    import json
    import parselmouth
    from collections import defaultdict

    os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"
    os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"

    # Helper: Convert audio to 16kHz mono WAV
    def to_wav_16k(input_bytes: bytes, suffix: str, tmpdir: Path, out_name: str = "audio.wav") -> Path:
        in_path = tmpdir / f"in{suffix}"
        in_path.write_bytes(input_bytes)
        out_path = tmpdir / out_name
        cmd = ["ffmpeg", "-y", "-i", str(in_path), "-ar", "16000", "-ac", "1", str(out_path)]
        subprocess.run(cmd, capture_output=True, check=True)
        return out_path

    # Helper: Run SepFormer on audio file
    def run_sepformer(wav_path: Path, model) -> list:
        separated = model.separate_file(path=str(wav_path))
        if isinstance(separated, torch.Tensor):
            sep = separated.detach().cpu().numpy()
        else:
            sep = np.array(separated)
        while sep.ndim > 2:
            sep = sep[0]
        if sep.ndim == 1:
            sep = sep[:, None]
        if sep.shape[0] < sep.shape[1]:
            sep = sep.T
        return [sep[:, i].astype(np.float32) for i in range(sep.shape[1])]

    # Helper: Run Pyannote diarization
    def run_pyannote(wav_path: Path, pipeline) -> list:
        diarization = pipeline(str(wav_path))
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
        return turns

    # Helper: Run Allosaurus on audio
    def run_allosaurus(wav_path: Path, model) -> str:
        try:
            phones = model.recognize(str(wav_path), lang_id="ipa")
            return phones if phones else ""
        except Exception:
            return ""

    # Helper: F0 analysis with Parselmouth
    def analyze_f0(wav_path: Path) -> dict:
        try:
            sound = parselmouth.Sound(str(wav_path))
            if sound.n_channels > 1:
                sound = sound.convert_to_mono()
            pitch = sound.to_pitch(time_step=0.01)
            freqs = [pitch.get_value_at_time(t) for t in pitch.xs()]
            voiced = [f for f in freqs if f and f > 50]
            if not voiced:
                return {"median": 0, "min": 0, "max": 0, "std": 0, "voiced_ratio": 0}
            return {
                "median": float(np.median(voiced)),
                "min": float(min(voiced)),
                "max": float(max(voiced)),
                "std": float(np.std(voiced)),
                "voiced_ratio": len(voiced) / len(freqs),
            }
        except Exception as e:
            return {"error": str(e)}

    # Helper: Calculate separation quality score
    def separation_quality(f0_stats: list) -> float:
        """Score based on F0 separation between speakers (0-1)."""
        if len(f0_stats) < 2:
            return 0.0
        medians = [s.get("median", 0) for s in f0_stats if s.get("median", 0) > 0]
        if len(medians) < 2:
            return 0.0
        # Higher score if medians are well separated
        spread = max(medians) - min(medians)
        # Good separation: 50+ Hz difference
        return min(1.0, spread / 100.0)

    # Load models
    print("[compare] Loading models...")
    from speechbrain.inference.separation import SepformerSeparation
    from pyannote.audio import Pipeline as PyannotePipeline
    from allosaurus.app import read_recognizer

    sepformer = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr",
        savedir=f"{CACHE_DIR}/speechbrain/sepformer-whamr",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    pyannote = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ.get("HF_TOKEN"),
    )
    pyannote.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    allo = read_recognizer()

    suffix = Path(filename).suffix or ".wav"
    results = {"options": {}}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)

        # Prepare input audio (with optional clipping)
        in_path = tmpdir_p / f"input{suffix}"
        in_path.write_bytes(audio_bytes)
        wav_path = tmpdir_p / "input_16k.wav"
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(in_path)]
        if start_s > 0:
            ffmpeg_cmd.extend(["-ss", f"{start_s:.3f}"])
        if end_s > 0 and end_s > start_s:
            ffmpeg_cmd.extend(["-to", f"{end_s:.3f}"])
        ffmpeg_cmd.extend(["-ar", "16000", "-ac", "1", str(wav_path)])
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

        audio_16k, sr_16k = sf.read(str(wav_path))
        duration = len(audio_16k) / sr_16k
        print(f"[compare] Input: {duration:.1f}s @ {sr_16k}Hz")

        # =====================================================================
        # OPTION A: SepFormer → Pyannote
        # =====================================================================
        print("[compare] === OPTION A: SepFormer → Pyannote ===")
        opt_a = {"name": "SepFormer_then_Pyannote", "speakers": [], "f0_stats": []}

        # 1. SepFormer separation
        sources_a = run_sepformer(wav_path, sepformer)
        print(f"[A] SepFormer produced {len(sources_a)} sources")

        for i, src in enumerate(sources_a):
            src_path = tmpdir_p / f"optA_src{i}.wav"
            sf.write(str(src_path), src, 8000)

            # Upsample to 16kHz for pyannote
            src_16k_path = tmpdir_p / f"optA_src{i}_16k.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(src_path),
                "-ar", "16000", "-ac", "1", str(src_16k_path)
            ], capture_output=True, check=True)

            # 2. Pyannote on each source
            turns = run_pyannote(src_16k_path, pyannote)
            print(f"[A] Source {i}: {len(turns)} speaker turns")

            # 3. Allosaurus IPA
            ipa = run_allosaurus(src_16k_path, allo)

            # 4. F0 analysis
            f0 = analyze_f0(src_16k_path)
            opt_a["f0_stats"].append(f0)

            opt_a["speakers"].append({
                "source_id": f"SRC_{i:02d}",
                "turns": turns,
                "ipa": ipa[:200] if ipa else "",
                "f0": f0,
                "wav_bytes": src_path.read_bytes(),
            })

        opt_a["quality_score"] = separation_quality(opt_a["f0_stats"])
        results["options"]["A"] = opt_a

        # =====================================================================
        # OPTION B: Pyannote → SepFormer (on overlaps only)
        # =====================================================================
        print("[compare] === OPTION B: Pyannote → SepFormer ===")
        opt_b = {"name": "Pyannote_then_SepFormer", "speakers": [], "overlaps": [], "f0_stats": []}

        # 1. Pyannote on original
        turns_b = run_pyannote(wav_path, pyannote)
        print(f"[B] Pyannote found {len(turns_b)} turns")

        # Find overlaps
        overlaps = []
        for i, t1 in enumerate(turns_b):
            for t2 in turns_b[i+1:]:
                if t1["speaker"] != t2["speaker"]:
                    ov_start = max(t1["start"], t2["start"])
                    ov_end = min(t1["end"], t2["end"])
                    if ov_end > ov_start + 0.2:  # Min 200ms overlap
                        overlaps.append({
                            "start": ov_start,
                            "end": ov_end,
                            "speakers": [t1["speaker"], t2["speaker"]],
                        })
        print(f"[B] Found {len(overlaps)} overlap regions")

        # Group turns by speaker
        speaker_turns = defaultdict(list)
        for t in turns_b:
            speaker_turns[t["speaker"]].append(t)

        # For each speaker, get non-overlap segments
        for spk, turns in speaker_turns.items():
            # Concatenate non-overlap audio for this speaker
            spk_audio = []
            for t in turns:
                start_sample = int(t["start"] * sr_16k)
                end_sample = int(t["end"] * sr_16k)
                spk_audio.append(audio_16k[start_sample:end_sample])

            if spk_audio:
                combined = np.concatenate(spk_audio)
                spk_path = tmpdir_p / f"optB_{spk}.wav"
                sf.write(str(spk_path), combined, sr_16k)

                ipa = run_allosaurus(spk_path, allo)
                f0 = analyze_f0(spk_path)
                opt_b["f0_stats"].append(f0)

                opt_b["speakers"].append({
                    "speaker_id": spk,
                    "turns": turns,
                    "ipa": ipa[:200] if ipa else "",
                    "f0": f0,
                    "duration": len(combined) / sr_16k,
                    "wav_bytes": spk_path.read_bytes(),
                })

        # 2. SepFormer only on overlaps
        for ov in overlaps[:3]:  # Limit to first 3 overlaps
            ov_start_sample = int(ov["start"] * sr_16k)
            ov_end_sample = int(ov["end"] * sr_16k)
            ov_audio = audio_16k[ov_start_sample:ov_end_sample]

            ov_path = tmpdir_p / f"optB_ov_{ov['start']:.1f}.wav"
            sf.write(str(ov_path), ov_audio, sr_16k)

            # Downsample to 8kHz for SepFormer
            ov_8k_path = tmpdir_p / f"optB_ov_{ov['start']:.1f}_8k.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(ov_path),
                "-ar", "8000", "-ac", "1", str(ov_8k_path)
            ], capture_output=True, check=True)

            sep_sources = run_sepformer(ov_8k_path, sepformer)
            opt_b["overlaps"].append({
                "time": f"{ov['start']:.1f}-{ov['end']:.1f}",
                "original_speakers": ov["speakers"],
                "separated_count": len(sep_sources),
            })

        opt_b["quality_score"] = separation_quality(opt_b["f0_stats"])
        results["options"]["B"] = opt_b

        # =====================================================================
        # OPTION C: Cascaded 4-Stems
        # =====================================================================
        print("[compare] === OPTION C: Cascaded 4-Stems ===")
        opt_c = {"name": "Cascaded_4Stems", "stems": [], "f0_stats": []}

        # 1. First SepFormer → 2 sources
        sources_c1 = run_sepformer(wav_path, sepformer)
        print(f"[C] Stage 1: {len(sources_c1)} sources")

        stem_idx = 0
        for i, src1 in enumerate(sources_c1):
            src1_path = tmpdir_p / f"optC_s1_{i}.wav"
            sf.write(str(src1_path), src1, 8000)

            # 2. Second SepFormer on each source → 2 more each
            sources_c2 = run_sepformer(src1_path, sepformer)
            print(f"[C] Stage 2 from source {i}: {len(sources_c2)} stems")

            for j, src2 in enumerate(sources_c2):
                stem_path = tmpdir_p / f"optC_stem{stem_idx}.wav"
                sf.write(str(stem_path), src2, 8000)

                # Upsample for analysis
                stem_16k_path = tmpdir_p / f"optC_stem{stem_idx}_16k.wav"
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(stem_path),
                    "-ar", "16000", "-ac", "1", str(stem_16k_path)
                ], capture_output=True, check=True)

                # 3. Pyannote on each stem
                turns = run_pyannote(stem_16k_path, pyannote)

                # 4. Allosaurus
                ipa = run_allosaurus(stem_16k_path, allo)

                # 5. F0
                f0 = analyze_f0(stem_16k_path)
                opt_c["f0_stats"].append(f0)

                opt_c["stems"].append({
                    "stem_id": f"STEM_{stem_idx:02d}",
                    "parent": f"SRC_{i}",
                    "turns": turns,
                    "ipa": ipa[:200] if ipa else "",
                    "f0": f0,
                    "wav_bytes": stem_path.read_bytes(),
                })
                stem_idx += 1

        opt_c["quality_score"] = separation_quality(opt_c["f0_stats"])
        results["options"]["C"] = opt_c

    # Summary
    print("\n[compare] === SUMMARY ===")
    for opt_key in ["A", "B", "C"]:
        opt = results["options"][opt_key]
        n_speakers = len(opt.get("speakers", opt.get("stems", [])))
        quality = opt.get("quality_score", 0)
        print(f"  Option {opt_key} ({opt['name']}): {n_speakers} outputs, quality={quality:.2f}")

    return results


# ---------------------------------------------------------------------------
# Modal Function — the GPU-accelerated pipeline
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=MODAL_TIMEOUT_S,
    volumes={CACHE_DIR: model_cache, NAH_MODEL_DIR: nah_model_vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("mistral-secret"),
    ],
)
@modal.concurrent(max_inputs=1)
def process_film(
    audio_bytes: bytes,
    filename: str = "input.wav",
    whisper_model: str = "medium",
    use_demucs: bool = True,
    vocals_bytes: bytes = None,
    phoneme_backend: str = "allosaurus",
    phoneme_compare: bool = True,  # Always run all backends for comparison
    whisper_rescue: bool = True,
    whisper_prompt: bool = False,
    whisper_prompt_extra: str = "",
    whisper_only: bool = False,
    spanish_orthography: bool = False,
    whisper_force_lang: str = "",
    show_confidence: bool = False,
    uncertain_text_policy: str = "keep",
    suppress_hallucinations: bool = False,
    low_conf_threshold: float = 0.35,
    lang_conf_threshold: float = 0.6,
    min_turn_s: float = 0.2,
    floating_window: bool = False,
    floating_window_shift_s: float = 0.08,
    show_phone_conf: bool = False,
    show_nbest: bool = False,
    nbest_keep: int = 4,
    phone_vote: bool = True,
    vad_refine: bool = False,
    vad_min_speech_ms: int = 120,
    vad_min_silence_ms: int = 80,
    vad_pad_s: float = 0.06,
    vad_expand_s: float = 0.0,
    vad_debug_dump: bool = False,
    recover_uncertain: bool = False,
    ipa_mismatch_override: bool = True,
    ipa_mismatch_threshold: float = 0.4,
    detect_ejectives: bool = True,  # Acoustic ejective detection for Maya
    ejective_maya_boost: float = 0.3,  # Score boost per consensus ejective
    nah_whisper_finetuned: bool = False,  # Use finetuned Whisper for NAH segments
    nah_checkpoint: str = "",  # Checkpoint name (e.g. "checkpoint-3000"), empty = /model
    ejective_min_count: int = 2,  # Minimum consensus ejectives to trigger Maya override
    mms_langid: bool = False,  # Use facebook/mms-1b-all as additional LangID voter
    # --- v7 accuracy fixes (toggleable) ---
    ft_spa_guard: bool = False,  # Fix 1: Detect Spanish in FT output, demote NAH→SPA
    speaker_prior_strong: bool = False,  # Fix 3: Speaker-prior also overrides low-conf SPA↔NAH
    ejective_strict: bool = False,  # Fix 4: Require diverse ejectives, exclude tɬ-like
    noise_gate: bool = False,  # Fix 5: Tag short segments without text as UNK
    noise_gate_max_s: float = 0.4,  # Max duration for noise gate
    ipa_nah_override: bool = False,  # Fix 6: Demote NAH→SPA when IPA lacks NAH-exclusive phonemes
    eq_config: dict | None = None,  # EQ config dict (overrides individual flags)
) -> dict:
    """Whisper-first pipeline: audio bytes in → SRT + stats out."""
    import tempfile
    import time
    import soundfile as sf
    import numpy as np

    # --- EQ: merge config dict with individual flags ---
    eq = dict(DEFAULT_EQ)  # start from defaults
    if eq_config:
        eq.update(eq_config)
    # Individual flags override EQ defaults (CLI flags take precedence)
    if ft_spa_guard:
        eq["ft_spa_guard"] = True
    if speaker_prior_strong:
        eq["speaker_prior_strong"] = True
    if ejective_strict:
        eq["ejective_strict"] = True
    if noise_gate:
        eq["noise_gate_enabled"] = True
    if noise_gate_max_s != 0.4:
        eq["noise_gate_max_s"] = noise_gate_max_s
    if ipa_nah_override:
        eq["ipa_nah_override"] = True
    if ejective_min_count != 2:
        eq["ejective_min_count"] = ejective_min_count
    if ejective_maya_boost != 0.3:
        eq["ejective_maya_boost"] = ejective_maya_boost
    if lang_conf_threshold != 0.6:
        eq["lang_conf_threshold"] = lang_conf_threshold
    if low_conf_threshold != 0.35:
        eq["low_conf_threshold"] = low_conf_threshold
    # Alias back for existing code that reads the flags directly
    ft_spa_guard = eq["ft_spa_guard"]
    speaker_prior_strong = eq["speaker_prior_strong"]
    ejective_strict = eq["ejective_strict"]
    noise_gate = eq["noise_gate_enabled"]
    noise_gate_max_s = eq["noise_gate_max_s"]
    ipa_nah_override = eq["ipa_nah_override"]
    ejective_min_count = eq["ejective_min_count"]
    ejective_maya_boost = eq["ejective_maya_boost"]
    lang_conf_threshold = eq["lang_conf_threshold"]
    low_conf_threshold = eq["low_conf_threshold"]
    print(f"EQ config: {', '.join(f'{k}={v}' for k, v in eq.items() if v != DEFAULT_EQ.get(k))}" or "EQ config: defaults")

    # Redirect torch hub cache into persistent volume
    os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"

    t_start = time.time()

    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    wav_path = audio_path
    if suffix.lower() not in (".wav",):
        wav_path = audio_path + ".wav"
        import subprocess
        subprocess.run([
            "ffmpeg", "-i", audio_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", wav_path, "-y"
        ], capture_output=True)

    audio_data, audio_sr = sf.read(wav_path)
    duration = len(audio_data) / audio_sr
    print(f"Audio: {duration:.0f}s ({duration/60:.1f} min), sr={audio_sr}")

    # Check for invalid flag combination
    if recover_uncertain and not suppress_hallucinations:
        print("⚠️ --recover-uncertain has no effect without --suppress-hallucinations")

    # STEP 0: Demucs vocal isolation (for Allosaurus only, not Whisper)
    vocals_path = None
    vocals_data = None
    vocals_sr = None

    # Use pre-computed vocals if provided
    if vocals_bytes:
        print(f"Using provided vocals track ({len(vocals_bytes)/1e6:.1f} MB)")
        vocals_path = wav_path + ".vocals.wav"
        with open(vocals_path, "wb") as vf:
            vf.write(vocals_bytes)
        vocals_data, vocals_sr = sf.read(vocals_path)
        use_demucs = False

    if use_demucs:
        print("Running Demucs vocal isolation (for Allosaurus)...")
        t_demucs = time.time()
        try:
            import torch
            import torchaudio
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            demucs_model = get_model("htdemucs")
            demucs_model.to("cuda")

            # Load audio for demucs (needs stereo float32)
            audio_np, sr = sf.read(wav_path, dtype="float32")
            if audio_np.ndim == 1:
                audio_np = audio_np[:, None]  # [samples, 1]
            waveform = torch.from_numpy(audio_np.T)  # [channels, samples]
            if sr != demucs_model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, demucs_model.samplerate)
                waveform = resampler(waveform)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            # Normalize, apply, denormalize
            ref = waveform.mean(0)
            ref_mean = ref.mean()
            ref_std = ref.std()
            waveform_norm = (waveform - ref_mean) / (ref_std + 1e-8)
            with torch.no_grad():
                sources = apply_model(
                    demucs_model,
                    waveform_norm.unsqueeze(0).to("cuda"),
                    shifts=1, overlap=0.25,
                )
            vocals = sources[0, 3] * (ref_std + 1e-8) + ref_mean

            # Save vocals track for Allosaurus
            vocals_mono = vocals.mean(dim=0, keepdim=True).cpu()
            if demucs_model.samplerate != 16000:
                resampler_out = torchaudio.transforms.Resample(demucs_model.samplerate, 16000)
                vocals_mono = resampler_out(vocals_mono)

            vocals_path = wav_path + ".vocals.wav"
            sf.write(vocals_path, vocals_mono.squeeze(0).numpy(), 16000)
            vocals_data, vocals_sr = sf.read(vocals_path)

            del demucs_model, sources, vocals, waveform
            torch.cuda.empty_cache()

            print(f"Demucs: vocals isolated in {time.time()-t_demucs:.0f}s")
        except Exception as e:
            print(f"⚠️ Demucs failed ({e}), Allosaurus will use original audio")
            vocals_path = None
    else:
        print("Demucs: skipped")

    # STEP 1: Pyannote Diarization — who speaks when
    import torch
    print("Loading pyannote diarization...")
    t0 = time.time()
    from pyannote.audio import Pipeline as PyannotePipeline
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    from_pretrained_sig = inspect.signature(PyannotePipeline.from_pretrained)
    kwargs = {}
    if hf_token:
        if "token" in from_pretrained_sig.parameters:
            kwargs["token"] = hf_token
        elif "use_auth_token" in from_pretrained_sig.parameters:
            kwargs["use_auth_token"] = hf_token

    diarize = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        **kwargs,
    )
    diarize.to(torch.device("cuda"))
    diarization_output = diarize(wav_path)
    annotation = getattr(diarization_output, "speaker_diarization", diarization_output)

    # Extract speaker turns (compat: Annotation vs DiarizeOutput wrapper)
    speaker_turns = []
    if not hasattr(annotation, "itertracks"):
        raise RuntimeError(
            f"Unexpected diarization output type: {type(diarization_output).__name__}"
        )
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_turns.append({
            "start": turn.start, "end": turn.end, "speaker": speaker,
        })
    print(f"Diarization: {len(speaker_turns)} turns, "
          f"{len(set(t['speaker'] for t in speaker_turns))} speakers, "
          f"{time.time()-t0:.0f}s")

    # Optional: refine diarization boundaries using Silero-VAD speech activity.
    # Additive and opt-in: default flow is unchanged.
    vad_debug = {
        "enabled": bool(vad_refine),
        "changed_turns": 0,
        "total_turns": len(speaker_turns),
        "speech_spans": 0,
        "turns": [],
    }
    if vad_refine and speaker_turns:
        t_vad = time.time()
        try:
            import numpy as np
            import torch
            import torchaudio
            from silero_vad import load_silero_vad, get_speech_timestamps

            vad_audio = vocals_data if vocals_data is not None else audio_data
            vad_sr = vocals_sr if vocals_sr is not None else audio_sr
            if vad_audio is not None and vad_sr is not None:
                vad_np = np.asarray(vad_audio, dtype=np.float32)
                if vad_np.ndim > 1:
                    vad_np = vad_np.mean(axis=1)
                wav = torch.from_numpy(vad_np)
                if vad_sr != 16000:
                    wav = torchaudio.functional.resample(wav, vad_sr, 16000)

                vad_model = load_silero_vad()
                with torch.no_grad():
                    ts = get_speech_timestamps(
                        wav,
                        vad_model,
                        sampling_rate=16000,
                        min_speech_duration_ms=int(vad_min_speech_ms),
                        min_silence_duration_ms=int(vad_min_silence_ms),
                        return_seconds=True,
                    )

                speech_spans = [(float(x["start"]), float(x["end"])) for x in ts if x["end"] > x["start"]]
                vad_debug["speech_spans"] = len(speech_spans)
                if speech_spans:
                    refined = []
                    changed = 0
                    for t in speaker_turns:
                        old_start = float(t["start"])
                        old_end = float(t["end"])
                        s0 = max(0.0, float(t["start"]) - max(0.0, vad_expand_s))
                        s1 = min(duration, float(t["end"]) + max(0.0, vad_expand_s))
                        overlaps = []
                        for a, b in speech_spans:
                            ov0 = max(s0, a)
                            ov1 = min(s1, b)
                            if ov1 > ov0:
                                overlaps.append((ov0, ov1))
                        if overlaps:
                            new_start = max(0.0, overlaps[0][0] - max(0.0, vad_pad_s))
                            new_end = min(duration, overlaps[-1][1] + max(0.0, vad_pad_s))
                            # Keep monotonic and sane
                            if new_end <= new_start:
                                refined.append(t)
                                if vad_debug_dump:
                                    vad_debug["turns"].append({
                                        "speaker": t["speaker"],
                                        "before": {"start": round(old_start, 3), "end": round(old_end, 3)},
                                        "after": {"start": round(old_start, 3), "end": round(old_end, 3)},
                                        "changed": False,
                                        "overlaps": len(overlaps),
                                        "reason": "invalid_refined_window",
                                    })
                                continue
                            was_changed = (abs(new_start - t["start"]) > 1e-3 or abs(new_end - t["end"]) > 1e-3)
                            if was_changed:
                                changed += 1
                            if vad_debug_dump:
                                vad_debug["turns"].append({
                                    "speaker": t["speaker"],
                                    "before": {"start": round(old_start, 3), "end": round(old_end, 3)},
                                    "after": {"start": round(new_start, 3), "end": round(new_end, 3)},
                                    "changed": bool(was_changed),
                                    "overlaps": len(overlaps),
                                    "reason": "refined_from_overlaps",
                                })
                            refined.append({
                                "start": new_start,
                                "end": new_end,
                                "speaker": t["speaker"],
                            })
                        else:
                            refined.append(t)
                            if vad_debug_dump:
                                vad_debug["turns"].append({
                                    "speaker": t["speaker"],
                                    "before": {"start": round(old_start, 3), "end": round(old_end, 3)},
                                    "after": {"start": round(old_start, 3), "end": round(old_end, 3)},
                                    "changed": False,
                                    "overlaps": 0,
                                    "reason": "no_speech_overlap",
                                })
                    speaker_turns = refined
                    vad_debug["changed_turns"] = changed
                    print(
                        f"VAD refine: {changed}/{len(speaker_turns)} turns adjusted, "
                        f"{len(speech_spans)} speech spans, {time.time()-t_vad:.0f}s"
                    )
                else:
                    print(f"VAD refine: no speech spans found, skipped ({time.time()-t_vad:.0f}s)")
        except Exception as e:
            print(f"⚠️ VAD refine failed ({e}), continuing with diarization boundaries")

    # Free diarization model
    del diarize
    torch.cuda.empty_cache()

    # STEP 2: Whisper (always uses ORIGINAL audio — robust to noise)
    print(f"Loading Whisper {whisper_model}...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(
        whisper_model, device="cuda", compute_type="float16",
        download_root=f"{CACHE_DIR}/whisper",
    )
    decode_lang = whisper_force_lang.strip() or ("es" if spanish_orthography else None)
    if decode_lang:
        print(f"Transcribing (original audio, language={decode_lang})...")
    else:
        print("Transcribing (original audio, language=auto)...")
    t0 = time.time()
    segments_raw, info = whisper.transcribe(
        wav_path, language=decode_lang, vad_filter=True,
        beam_size=5, word_timestamps=True,
    )
    whisper_segs = []
    for s in segments_raw:
        st = s.text.strip()
        if has_non_latin_script(st):
            st = ""
        whisper_segs.append(
            {
                "text": st,
                "start": s.start,
                "end": s.end,
                "avg_log_prob": s.avg_logprob,
            }
        )
    file_lang = info.language
    print(f"Whisper: {len(whisper_segs)} segs, lang={file_lang}, {time.time()-t0:.0f}s")

    # STEP 3: Validate Whisper & build unified segment list
    lang_639_3 = LANG_MAP.get(file_lang, "other")

    # Tag all whisper segments with validity
    for seg in whisper_segs:
        seg["valid"] = validate_whisper(seg["text"], seg["avg_log_prob"])
        seg["latin"], seg["latin_keywords"] = detect_latin_text(seg["text"])
        seg["maya_hint"], seg["maya_hint_score"] = detect_maya_text_hint(seg["text"])
    n_valid = sum(1 for s in whisper_segs if s["valid"])
    n_invalid = len(whisper_segs) - n_valid
    print(f"Whisper valid: {n_valid}, uncertain: {n_invalid}")

    # Helper: find best speaker for a time range
    def best_speaker(seg_start, seg_end):
        best, best_ov = None, 0
        for t in speaker_turns:
            ov = max(0, min(seg_end, t["end"]) - max(seg_start, t["start"]))
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        return best

    # Helper: find best whisper match for a time range (dedup: each seg used once)
    _whisper_consumed: set[int] = set()

    def best_whisper(turn_start, turn_end):
        best, best_ov, best_idx = None, 0, -1
        for idx, seg in enumerate(whisper_segs):
            if idx in _whisper_consumed:
                continue
            ov = max(0, min(turn_end, seg["end"]) - max(turn_start, seg["start"]))
            if ov > best_ov:
                best, best_ov, best_idx = seg, ov, idx
        # Only match if overlap covers >30% of whichever is shorter
        if best and best_ov > 0:
            shorter = min(turn_end - turn_start, best["end"] - best["start"])
            if best_ov / max(shorter, 0.01) > 0.3:
                _whisper_consumed.add(best_idx)
                return best
        return None

    # STEP 4: Initialize phoneme backend(s)
    phoneme_audio = vocals_data if vocals_data is not None else audio_data
    phoneme_sr = vocals_sr if vocals_sr is not None else audio_sr
    source_label = "demucs vocals" if vocals_data is not None else "original audio"
    tl_sound = None
    try:
        import numpy as np
        import parselmouth

        tl_audio = phoneme_audio.astype(np.float64)
        if np.abs(tl_audio).max() > 1.0:
            tl_audio = tl_audio / 32768.0
        tl_sound = parselmouth.Sound(tl_audio, sampling_frequency=float(phoneme_sr))
    except Exception as e:
        print(f"⚠️ tɬ detector disabled ({e})")

    primary_backend = None
    compare_backend = None
    phoneme_backend_requested = (phoneme_backend or "allosaurus").strip().lower()
    phoneme_backend_actual = "none"
    if not whisper_only:
        backend_name, backend_warning = resolve_phoneme_backend_name(phoneme_backend)
        if backend_warning:
            print(backend_warning)
        phoneme_backend_actual = backend_name

        # Primary backend
        print(f"Loading phoneme backend: {backend_name} (source: {source_label})")
        primary_backend = get_phoneme_backend(backend_name, CACHE_DIR)
        try:
            primary_backend.initialize()
        except Exception as e:
            if backend_name != "allosaurus":
                print(f"⚠️ Backend '{backend_name}' init failed ({e}), falling back to allosaurus")
                phoneme_backend_actual = "allosaurus"
                primary_backend = get_phoneme_backend("allosaurus", CACHE_DIR)
                primary_backend.initialize()
            else:
                raise

        # Compare backend (only in compare mode)
        if phoneme_compare:
            compare_name = "wav2vec2" if backend_name == "allosaurus" else "allosaurus"
            print(f"Loading compare backend: {compare_name}")
            try:
                compare_backend = get_phoneme_backend(compare_name, CACHE_DIR)
                compare_backend.initialize()
            except Exception as e:
                print(f"⚠️ Compare backend '{compare_name}' init failed ({e}), compare disabled")
                compare_backend = None

        # Omnilingual runs in separate container - we'll call it after collecting segments
        omni_results = {}  # Will be populated after processing turns

    # MMS LangID voter (optional)
    mms_voter = None
    if mms_langid:
        try:
            mms_voter = MMSLangID(cache_dir=CACHE_DIR)
            mms_voter.initialize()
        except Exception as e:
            print(f"⚠️ MMS LangID init failed ({e}), continuing without MMS")
            mms_voter = None

    # STEP 4.5: Full-track Allosaurus recognition (optional)
    # Run Allosaurus ONCE on full vocals track, then map phones to segments by timestamp.
    # Eliminates CTC warmup loss on short segments and boundary artifacts.
    _allo_full_phones = None
    if eq.get("allo_full_track") and not whisper_only:
        _allo_be = None
        _allo_source = vocals_path or wav_path
        if phoneme_backend_actual == "allosaurus" and primary_backend:
            _allo_be = primary_backend
        elif compare_backend and hasattr(compare_backend, "name") and compare_backend.name == "allosaurus":
            _allo_be = compare_backend
        if _allo_be and _allo_source:
            import time as _ft_time
            _ft_t0 = _ft_time.time()
            print(f"  Running Allosaurus full-track on {Path(_allo_source).name}...")
            _ft_emit = eq.get("allo_full_track_emit", 2.0)
            _allo_full_phones = _allo_be.recognize_full_track(str(_allo_source), emit=_ft_emit)
            if _allo_full_phones:
                print(f"  Full-track: {len(_allo_full_phones)} phones in {_ft_time.time()-_ft_t0:.1f}s")
            else:
                print(f"  Full-track: failed, falling back to per-segment Allosaurus")
                _allo_full_phones = None

    # Track trimming stats across all segments
    trim_stats_all = []

    # Helper: run a backend on an audio chunk with voice onset trimming
    def _run_backend(be, start_t, end_t, apply_trim=True, blank_bias=0.0, chunk_override=None):
        # chunk_override: pre-separated audio (skip time-based slicing)
        if chunk_override is not None:
            chunk = chunk_override
        else:
            # Apply voice onset trimming to remove silence padding
            if apply_trim and phoneme_audio is not None:
                trimmed_start, trimmed_end, trim_info = trim_to_voice(
                    phoneme_audio, phoneme_sr, start_t, end_t
                )
                trim_stats_all.append({
                    "orig_start": start_t,
                    "orig_end": end_t,
                    "trim_start": trimmed_start,
                    "trim_end": trimmed_end,
                    **trim_info,
                })
                start_t, end_t = trimmed_start, trimmed_end

            start_s = int(start_t * phoneme_sr)
            end_s = int(end_t * phoneme_sr)
            chunk = phoneme_audio[start_s:end_s]
        if len(chunk) < phoneme_sr * 0.2:
            return None, None
        try:
            if hasattr(be, "recognize_chunk_detailed"):
                ipa_text, detail = be.recognize_chunk_detailed(
                    chunk, phoneme_sr, nbest_keep=max(1, int(nbest_keep))
                )
                # Store trim info in detail
                if detail is None:
                    detail = {}
                if trim_stats_all:
                    detail["trim"] = trim_stats_all[-1]
                return ipa_text, detail
            # Pass blank_bias to AllosaurusBackend (others ignore it)
            if blank_bias > 0 and hasattr(be, "name") and be.name == "allosaurus":
                ipa = be.recognize_chunk(chunk, phoneme_sr, blank_bias=blank_bias)
            else:
                ipa = be.recognize_chunk(chunk, phoneme_sr)
            detail = {}
            if trim_stats_all:
                detail["trim"] = trim_stats_all[-1]
            return ipa, detail if detail else None
        except Exception:
            return None, None

    def run_phoneme_backend(start_t, end_t, blank_bias=0.0):
        # Full-track mode: map pre-computed phones instead of per-segment call
        if _allo_full_phones is not None and phoneme_backend_actual == "allosaurus":
            ipa = map_full_track_phones(_allo_full_phones, start_t, end_t)
            return ipa, None
        return _run_backend(primary_backend, start_t, end_t, blank_bias=blank_bias)

    def run_compare_backend(start_t, end_t):
        if compare_backend is None:
            return None, None
        # Full-track mode for Allo as compare backend
        if _allo_full_phones is not None and hasattr(compare_backend, "name") and compare_backend.name == "allosaurus":
            ipa = map_full_track_phones(_allo_full_phones, start_t, end_t)
            return ipa, None
        return _run_backend(compare_backend, start_t, end_t)

    def run_omni_backend(start_t, end_t):
        # Omnilingual results are fetched after all turns are processed
        # This function returns None during initial processing
        key = f"{start_t:.3f}-{end_t:.3f}"
        return omni_results.get(key), None

    def _norm_logprob(avg_log_prob: float) -> float:
        # Map rough Whisper logprob range [-2, 0] to [0, 1].
        v = (avg_log_prob + 2.0) / 2.0
        return max(0.0, min(1.0, v))

    def _lang_confidence(lang_code: str, raw_score: float | None) -> float:
        """Map raw language score to [0,1] confidence for gating."""
        if raw_score is None:
            return 0.0
        if raw_score <= 0:
            return 0.0
        if lang_code == "nah":
            # NAH threshold is intentionally permissive in profile config; use a fixed scale.
            return max(0.0, min(1.0, raw_score / 1.2))
        thr = float(PROFILES.get(lang_code, {}).get("threshold", 1.0) or 1.0)
        # Slight margin above threshold to avoid overconfidence near decision boundary.
        denom = max(1.0, thr + 0.5)
        return max(0.0, min(1.0, raw_score / denom))

    def decode_whisper_for_turn(turn_start, turn_end, anchor_tokens=None):
        """Decode Whisper on the diarization window (+context pad).

        Runs two decode profiles:
        - semantic: beam search (better for fluent known-language text)
        - phonetic: greedy-ish decode (often closer to acoustic form)
        If anchor_tokens are provided (IPA from audio backends), select candidate by
        agreement+confidence score; otherwise fall back to avg_log_prob.
        """
        import tempfile
        import numpy as np

        decode_audio = vocals_data if vocals_data is not None else audio_data
        decode_sr = vocals_sr if vocals_data is not None else audio_sr

        if decode_audio is None or decode_sr is None:
            return "", -2.0, "semantic", 0.0
        if turn_end - turn_start < 0.2:
            return "", -2.0, "semantic", 0.0

        text = ""
        avg_log_prob = -2.0
        profile_used = "semantic"
        agreement_used = 0.0
        profile_params = [
            {"name": "semantic", "beam_size": 5, "best_of": 5, "temperature": 0.0},
            {"name": "phonetic", "beam_size": 1, "best_of": 1, "temperature": 0.2},
        ]
        window_specs = [
            {"name": "base", "lpad": 0.50, "rpad": 0.50, "shift": 0.0},
        ]
        if floating_window:
            s = max(0.0, float(floating_window_shift_s))
            window_specs = [
                {"name": "tight", "lpad": 0.18, "rpad": 0.08, "shift": 0.0},
                {"name": "left", "lpad": 0.35, "rpad": 0.10, "shift": 0.0},
                {"name": "right", "lpad": 0.12, "rpad": 0.35, "shift": 0.0},
                {"name": "sym", "lpad": 0.35, "rpad": 0.35, "shift": 0.0},
                {"name": "shift_l", "lpad": 0.30, "rpad": 0.12, "shift": -s},
                {"name": "shift_r", "lpad": 0.12, "rpad": 0.30, "shift": s},
                {"name": "wide", "lpad": 0.50, "rpad": 0.50, "shift": 0.0},
            ]

        candidates = []
        for ws in window_specs:
            t0 = max(0.0, turn_start + float(ws["shift"]))
            t1 = min(duration, turn_end + float(ws["shift"]))
            if t1 <= t0:
                continue
            chunk_start = max(0.0, t0 - float(ws["lpad"]))
            chunk_end = min(duration, t1 + float(ws["rpad"]))
            i0 = int(chunk_start * decode_sr)
            i1 = int(chunk_end * decode_sr)
            chunk = decode_audio[i0:i1]
            if len(chunk) == 0:
                continue

            silence = np.zeros(int(0.5 * decode_sr), dtype=np.float32)
            if getattr(chunk, "ndim", 1) > 1:
                silence = np.zeros((int(0.5 * decode_sr), chunk.shape[1]), dtype=np.float32)
            chunk_padded = np.concatenate([silence, chunk, silence])

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk_padded, decode_sr)
                try:
                    for prof in profile_params:
                        segs, _seg_info = whisper.transcribe(
                            tmp.name,
                            language=decode_lang,
                            vad_filter=False,
                            beam_size=prof["beam_size"],
                            best_of=prof["best_of"],
                            temperature=prof["temperature"],
                            word_timestamps=True,
                            condition_on_previous_text=False,
                        )
                        for s in segs:
                            st = s.text.strip()
                            if st and not has_non_latin_script(st):
                                cand = {
                                    "text": st,
                                    "avg_log_prob": s.avg_logprob,
                                    "profile": prof["name"],
                                    "window": ws["name"],
                                    "agreement": 0.0,
                                }
                                if anchor_tokens:
                                    hint, _ = guess_language_from_text_markers(st, file_lang or "es")
                                    cand_tokens = text_to_ipa_tokens(st, hint)
                                    cand["agreement"] = ipa_agreement(anchor_tokens, cand_tokens) if cand_tokens else 0.0
                                candidates.append(cand)
                except Exception:
                    pass
                finally:
                    os.unlink(tmp.name)

        if candidates:
            if anchor_tokens:
                # Agreement to acoustic IPA anchor dominates; logprob stabilizes ties.
                best = max(
                    candidates,
                    key=lambda c: (0.55 * c["agreement"] + 0.45 * _norm_logprob(c["avg_log_prob"])),
                )
            else:
                best = max(candidates, key=lambda c: c["avg_log_prob"])
            text = best["text"]
            avg_log_prob = best["avg_log_prob"]
            profile_used = f"{best['profile']}@{best['window']}"
            agreement_used = best["agreement"]
        return text, avg_log_prob, profile_used, agreement_used

    # STEP 5: Build unified results
    if whisper_only:
        print(f"Processing {len(speaker_turns)} speaker turns (whisper-only forced decode)...")
    else:
        print(f"Processing {len(speaker_turns)} speaker turns (whisper + {phoneme_backend_actual})...")
    all_results = []
    allo_count = 0
    if whisper_only:
        # Force decode each diarized turn so all audio gets text output.
        # If diarization unexpectedly returns nothing, fall back to Whisper segments.
        if not speaker_turns:
            for seg in whisper_segs:
                text = seg["text"]
                if text and has_non_latin_script(text):
                    text = ""
                if spanish_orthography:
                    text = transliterate_to_spanish_orthography(text)
                whisper_valid = validate_whisper(text, seg["avg_log_prob"])
                lang_guess, _ = guess_language_from_text_markers(text, file_lang or "es")
                all_results.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": best_speaker(seg["start"], seg["end"]),
                    "lang": lang_guess,
                    "text": text,
                    "whisper_valid": whisper_valid,
                    "ipa": None,
                    "ipa_compare": None,
                    "ipa_omni": None,
                    "backend": "whisper-only",
                })
        else:
            for turn in speaker_turns:
                if turn["end"] - turn["start"] < min_turn_s:
                    continue

                text, avg_log_prob, profile_used, _agreement_used = decode_whisper_for_turn(
                    turn["start"], turn["end"], anchor_tokens=None
                )

                if spanish_orthography and text:
                    text = transliterate_to_spanish_orthography(text)

                whisper_valid = validate_whisper(text, avg_log_prob) if text else False
                lang_guess, _ = guess_language_from_text_markers(text, file_lang or "es")
                all_results.append({
                    "start": turn["start"],
                    "end": turn["end"],
                    "speaker": turn["speaker"],
                    "lang": lang_guess,
                    "text": text,
                    "whisper_valid": whisper_valid,
                    "ipa": None,
                    "ipa_compare": None,
                    "ipa_omni": None,
                    "backend": f"whisper-only:{profile_used}",
                })
    else:
        # Speaker-prior: track per-speaker language history for voter influence
        from collections import Counter as _Counter
        _speaker_lang_history: dict[str, _Counter] = {}

        # Pre-compute overlap flags for all turns (before classification loop)
        _turn_overlap_flags = {}
        if eq.get("overlap_gate", False) and tl_sound is not None:
            _f0_min_ratio = eq.get("overlap_f0_min_ratio", 0.20)
            _hnr_threshold = eq.get("overlap_hnr_threshold", 10.0)
            _pre_overlap_count = 0
            for ti, turn in enumerate(speaker_turns):
                dur = turn["end"] - turn["start"]
                if dur < 0.5:
                    continue
                is_ov, _ = detect_overlap(
                    tl_sound, turn["start"], turn["end"],
                    f0_min_ratio=_f0_min_ratio,
                    hnr_threshold=_hnr_threshold,
                )
                if is_ov:
                    _turn_overlap_flags[ti] = True
                    _pre_overlap_count += 1
            if _pre_overlap_count:
                print(f"  [overlap-pre] {_pre_overlap_count}/{len(speaker_turns)} turns flagged")

        # Pre-separate overlap turns with MossFormer2 (runs on separate GPU container)
        _overlap_separated = {}  # turn_idx -> [{"audio": np.array, "rms": float}, ...]
        if eq.get("overlap_separation", False) and _turn_overlap_flags:
            overlap_segs = [
                {"idx": ti, "start": speaker_turns[ti]["start"], "end": speaker_turns[ti]["end"]}
                for ti in sorted(_turn_overlap_flags.keys())
            ]
            print(f"  [overlap-sep] Sending {len(overlap_segs)} segments to MossFormer2...")
            try:
                import io as _io
                _voc_buf = _io.BytesIO()
                sf.write(_voc_buf, phoneme_audio, phoneme_sr, format="WAV")
                _voc_wav_bytes = _voc_buf.getvalue()
                _sep_raw = separate_overlap_segments.remote(_voc_wav_bytes, overlap_segs)
                # Deserialize float32 bytes back to numpy arrays
                for _seg_idx, _sources in _sep_raw.items():
                    _seg_idx_int = int(_seg_idx) if isinstance(_seg_idx, str) else _seg_idx
                    _deserialized = []
                    for _src in _sources:
                        _audio_np = np.frombuffer(_src["audio"], dtype=np.float32).copy()
                        _deserialized.append({
                            "audio": _audio_np,
                            "rms": _src["rms"],
                            "sr": _src["sr"],
                        })
                    _overlap_separated[_seg_idx_int] = _deserialized
                print(f"  [overlap-sep] Got {len(_overlap_separated)} separated segments")
            except Exception as _sep_err:
                print(f"  [overlap-sep] Failed: {_sep_err}")

        for _turn_idx, turn in enumerate(speaker_turns):
            if turn["end"] - turn["start"] < min_turn_s:
                continue
            _is_overlap_turn = _turn_overlap_flags.get(_turn_idx, False)
            score_used = None
            agree_primary = 0.0
            agree_compare = 0.0
            agree_cross = 0.0

            # Run phoneme backend(s)
            _sep_ipa_sources = None  # Will hold per-voice IPA if overlap-separated
            if _is_overlap_turn and _turn_idx in _overlap_separated:
                # Overlap-separated: run IPA on each separated voice independently
                _sep_sources = _overlap_separated[_turn_idx]
                _sep_ipa_sources = []
                for _si, _src in enumerate(_sep_sources):
                    _src_ipa, _src_detail = _run_backend(
                        primary_backend, turn["start"], turn["end"],
                        apply_trim=False, chunk_override=_src["audio"],
                    )
                    _src_lang = None
                    if _src_ipa:
                        _src_phones = canonicalize_ipa_tokens(
                            _src_ipa.split(), acoustic_tl_detected=False
                        )
                        _src_lang, _src_score = identify_language(_src_phones)
                    _sep_ipa_sources.append({
                        "ipa": _src_ipa,
                        "detail": _src_detail,
                        "rms": _src["rms"],
                        "lang": _src_lang,
                    })
                    print(f"  [overlap-sep] turn {_turn_idx} src{_si}: "
                          f"rms={_src['rms']:.4f} ipa={'yes' if _src_ipa else 'no'} "
                          f"lang={_src_lang}")
                # Source selection: prefer the source whose IPA-language differs from
                # the other source. When one source is NAH and the other is not,
                # pick the non-NAH source — overlap bleed causes SPA→NAH FP,
                # so the cleaner (non-NAH) source is usually the true speaker.
                _sep_with_ipa = [s for s in _sep_ipa_sources if s["ipa"]]
                if len(_sep_with_ipa) >= 2:
                    langs = [s["lang"] for s in _sep_with_ipa]
                    if langs[0] != langs[1]:
                        # Sources disagree — pick non-NAH if one is NAH
                        non_nah = [s for s in _sep_with_ipa if s["lang"] != "nah"]
                        if non_nah:
                            _pick = non_nah[0]
                            print(f"  [overlap-sep] turn {_turn_idx}: picked non-NAH source "
                                  f"(lang={_pick['lang']})")
                        else:
                            _pick = _sep_with_ipa[0]
                    else:
                        _pick = _sep_with_ipa[0]  # both agree, use either
                elif _sep_with_ipa:
                    _pick = _sep_with_ipa[0]
                else:
                    _pick = _sep_ipa_sources[0]
                ipa = _pick["ipa"]
                ipa_detail = _pick["detail"]
            else:
                ipa, ipa_detail = run_phoneme_backend(turn["start"], turn["end"])

            # Adaptive blank-bias: if PM sees way more acoustic events than Allo decoded
            if ipa is not None and phoneme_backend_actual == "allosaurus":
                allo_phones = len(ipa.split()) if ipa else 0
                # Get audio chunk for PM event counting
                chunk_start_sample = int(turn["start"] * phoneme_sr)
                chunk_end_sample = int(turn["end"] * phoneme_sr)
                chunk = phoneme_audio[chunk_start_sample:chunk_end_sample]
                if len(chunk) >= phoneme_sr * 0.1:  # At least 100ms
                    try:
                        pm_events = count_acoustic_events(chunk, phoneme_sr)
                        # If PM sees 2x+ more events than Allo decoded, re-run with bias
                        if pm_events > 2 * max(allo_phones, 1):
                            ipa_biased, detail_biased = run_phoneme_backend(
                                turn["start"], turn["end"], blank_bias=3.0
                            )
                            if ipa_biased and len(ipa_biased.split()) > allo_phones:
                                new_phones = len(ipa_biased.split())
                                print(f"  [adaptive-bias] PM={pm_events} > 2×Allo={allo_phones} → bias=3.0 → {new_phones} phones (+{new_phones - allo_phones})")
                                ipa = ipa_biased
                                ipa_detail = detail_biased
                                # Mark that we used adaptive bias
                                if ipa_detail is None:
                                    ipa_detail = {}
                                ipa_detail["adaptive_bias"] = 3.0
                                ipa_detail["pm_events"] = pm_events
                                ipa_detail["allo_phones_before"] = allo_phones
                    except Exception:
                        pass  # PM analysis failed, continue with original IPA

            ipa_compare, ipa_compare_detail = run_compare_backend(turn["start"], turn["end"])
            ipa_omni, _ = run_omni_backend(turn["start"], turn["end"])
            if ipa:
                allo_count += 1
            ipa_fused = None
            ipa_fused_conf = None
            seg_duration = turn["end"] - turn["start"]  # FIX 8: For duration-based fusion

            tl_result = {"tl_detected": False, "tl_confidence": 0.0, "tl_events": 0}
            if tl_sound is not None:
                try:
                    tl_result = detect_tl_for_segment(tl_sound, turn["start"], turn["end"])
                except Exception:
                    pass
            tl_detected = bool(tl_result.get("tl_detected"))

            if phone_vote and ipa and ipa_compare:
                primary_is_w2v2 = phoneme_backend_actual == "wav2vec2"
                compare_is_w2v2 = phoneme_backend_actual == "allosaurus"
                pt = canonicalize_ipa_tokens(ipa.split(), acoustic_tl_detected=tl_detected)
                ct = canonicalize_ipa_tokens(ipa_compare.split(), acoustic_tl_detected=tl_detected)
                ft, fconf = fuse_ipa_by_phone_vote(
                    primary_tokens=pt,
                    compare_tokens=ct,
                    primary_phone_conf=(ipa_detail or {}).get("phone_conf", []),
                    compare_phone_conf=(ipa_compare_detail or {}).get("phone_conf", []),
                    primary_is_w2v2=primary_is_w2v2,
                    compare_is_w2v2=compare_is_w2v2,
                    segment_duration=seg_duration,  # FIX 8
                )
                if ft:
                    ipa_fused = " ".join(ft)
                    ipa_fused_conf = round(float(fconf), 3)

            # Decode text on the exact diarization turn window (no cross-turn borrowing),
            # with IPA anchor when available.
            anchor_tokens = []
            if ipa:
                anchor_tokens.extend(canonicalize_ipa_tokens(ipa.split(), acoustic_tl_detected=tl_detected))
            if ipa_compare:
                anchor_tokens.extend(canonicalize_ipa_tokens(ipa_compare.split(), acoustic_tl_detected=tl_detected))
            text, avg_log_prob, profile_used, decode_agree = decode_whisper_for_turn(
                turn["start"], turn["end"], anchor_tokens=anchor_tokens or None
            )
            if spanish_orthography and text:
                text = transliterate_to_spanish_orthography(text)
            whisper_valid = validate_whisper(text, avg_log_prob) if text else False
            wseg = {
                "text": text,
                "valid": whisper_valid,
                "latin": detect_latin_text(text)[0] if text else False,
                "maya_hint": detect_maya_text_hint(text)[0] if text else False,
            }

            # Acoustic ejective detection for Maya
            ejective_count = 0
            if detect_ejectives:
                try:
                    ej_audio = vocals_data if vocals_data is not None else audio_data
                    ej_sr = vocals_sr if vocals_data is not None else audio_sr
                    start_s = int(turn["start"] * ej_sr)
                    end_s = int(turn["end"] * ej_sr)
                    chunk = ej_audio[start_s:end_s]
                    if len(chunk) >= ej_sr * 0.3:  # At least 300ms
                        ejective_count = detect_ejectives_for_segment(chunk, ej_sr, turn["start"])
                except Exception:
                    pass
            wseg["ejective_count"] = ejective_count

            # MMS LangID voter (optional)
            mms_lang = None
            mms_conf = 0.0
            mms_raw = None
            if mms_voter is not None:
                try:
                    mms_audio = vocals_data if vocals_data is not None else audio_data
                    mms_sr = vocals_sr if vocals_sr is not None else audio_sr
                    ms_start = int(turn["start"] * mms_sr)
                    ms_end = int(turn["end"] * mms_sr)
                    mms_chunk = mms_audio[ms_start:ms_end]
                    if len(mms_chunk) >= mms_sr * 0.3:  # At least 300ms
                        mms_lang, mms_conf, mms_raw = mms_voter.identify_mapped(
                            mms_chunk, mms_sr
                        )
                        # Log top-5 for analysis
                        mms_top5 = mms_voter.identify(mms_chunk, mms_sr, top_k=5)
                        mms_top5_str = " ".join(
                            f"{m}({r})={c:.2f}" for m, c, r in mms_top5
                        )
                        print(f"  [mms] {mms_lang}({mms_raw}) conf={mms_conf:.3f}  top5: {mms_top5_str}")
                except Exception as e:
                    print(f"  [mms] error: {e}")

            # Determine language
            lang_conf = 0.0
            lang_conf_source = "none"
            uncertain = False
            lexicon_hit = False
            # Pre-compute spa_score for Spanish Context Guard (used in multiple paths)
            spa_score_early = 0.0
            indigenous_count_early = 0
            STRONG_INDIGENOUS_MARKERS_EARLY = {"ʔ", "ɬ", "tɬ", "kʷ", "kʼ", "tʼ", "pʼ"}
            if ipa:
                ipa_for_spa_check = ipa_fused or ipa
                phonemes_for_spa = canonicalize_ipa_tokens(ipa_for_spa_check.split(), acoustic_tl_detected=tl_detected)
                _, _, scores_for_spa = identify_language_with_scores(phonemes_for_spa)
                spa_score_early = scores_for_spa.get("spa", 0.0)
                indigenous_count_early = sum(1 for m in STRONG_INDIGENOUS_MARKERS_EARLY if m in ipa_for_spa_check)
            suppress_may_early = spa_score_early > 0.5 and indigenous_count_early < 2

            # Guard: Spanish function words in Whisper text suppress tɬ override.
            # Spanish has "tl" clusters (atlántico, extremaunción) that Allosaurus
            # misreads as tɬ. If Whisper confidently transcribed Spanish, trust it.
            # Use broad set including contractions (del=de+el, al=a+el).
            _SPA_FUNC_WORDS = {
                "la", "el", "de", "que", "en", "los", "las", "un", "una", "por", "con",
                "del", "al", "se", "su", "más", "mas", "como", "hay", "pero", "ya",
                "no", "es", "le", "lo", "y", "a",
            }
            _text_lower = (text or "").lower()
            # Strip leading ¿¡ and trailing punctuation for cleaner word matching
            _text_words = set(w.strip("¿¡?!.,;:\"'") for w in _text_lower.split())
            _spa_func_hits = len(_text_words & _SPA_FUNC_WORDS)
            _whisper_conf = _norm_logprob(avg_log_prob) if text else 0.0
            _whisper_is_spanish = (file_lang == "es") or lang_639_3 == "spa"
            _tl_spa_guard = (
                _spa_func_hits >= 3
                or (_whisper_is_spanish and _whisper_conf >= 0.85 and _spa_func_hits >= 1)
            )
            _tl_active = tl_result["tl_detected"] and not _tl_spa_guard
            # Overlap gate: suppress tl-override when simultaneous speakers detected
            if _tl_active and _is_overlap_turn:
                _tl_active = False
                print(f"  [tl-overlap] tɬ detected but overlap gate fired → skip tɬ override")

            if tl_result["tl_detected"] and _tl_spa_guard:
                print(
                    f"  [tl-suppressed] tɬ detected but Spanish guard fired "
                    f"(func_words={_spa_func_hits}, w_conf={_whisper_conf:.2f}, "
                    f"w_spa={_whisper_is_spanish}) → skip tɬ override"
                )

            if _tl_active:
                if ejective_count >= ejective_min_count:
                    # tɬ + ejective conflict: tɬ is NAH-exclusive (does not exist in Maya),
                    # so tɬ always wins. Ejective detector false positives are common
                    # on non-Maya audio (La Otra Conquista showed 29 MAY FPs from this path).
                    lang = "nah"
                    backend_tag = "tl-wins-over-ejective"
                    lang_conf = 0.85
                    lang_conf_source = "tl-over-ejective"
                    uncertain = False
                    print(
                        f"  [tl-wins] tɬ+ejective conflict: tɬ is NAH-exclusive → NAH "
                        f"(tl_conf={tl_result['tl_confidence']:.2f}, ejectives={ejective_count})"
                    )
                else:
                    lang = "nah"
                    backend_tag = "tl-acoustic-conditional-override"
                    lang_conf = 1.0
                    lang_conf_source = "tl-acoustic-conditional-override"
                    uncertain = False
                    print(f"  [tl-override] tɬ detected → NAH (conf={tl_result['tl_confidence']:.2f})")
            elif text and wseg.get("latin"):
                # Latin liturgical text override even when Whisper is "uncertain".
                lang = "lat"
                backend_tag = "whisper-latin+allo" if ipa else "whisper-latin"
                lang_conf = 1.0
                lang_conf_source = "latin-rule"
            elif text and wseg.get("maya_hint"):
                # Maya orthography hint from Whisper text (apostrophes/k' patterns).
                lang = "may"
                backend_tag = "whisper-maya+allo" if ipa else "whisper-maya"
                lang_conf = 1.0
                lang_conf_source = "maya-rule"
            elif whisper_valid and text:
                # Trusted Whisper → use detected file language, but override
                # when text is clearly Spanish despite Whisper saying en/de/etc.
                _text_lang_guess, _text_lang_conf = guess_language_from_text_markers(text, file_lang or "es")
                if _text_lang_guess == "spa" and lang_639_3 != "spa" and _text_lang_conf >= 0.55:
                    lang = "spa"
                    lang_conf = _text_lang_conf
                    lang_conf_source = f"whisper+spa-text-override:{_text_lang_conf:.2f}"
                    print(
                        f"  [spa-text-override] Whisper lang={file_lang} but text is SPA "
                        f"(conf={_text_lang_conf:.2f}): '{text[:40]}'"
                    )
                else:
                    lang = lang_639_3
                    lang_conf = _norm_logprob(avg_log_prob)
                    lang_conf_source = "whisper"
                backend_tag = "whisper+allo" if ipa else "whisper"
            elif ipa:
                # No trusted Whisper → use Allosaurus language ID
                phonemes_primary = canonicalize_ipa_tokens((ipa_fused or ipa).split(), acoustic_tl_detected=tl_detected)
                word, lex_score = check_nah_lexicon(phonemes_primary)
                lexicon_hit = bool(word)

                # P1: Lexicon-override before marker-scoring
                # If lexicon score >= 0.6 (even without full word match), force NAH
                # This prevents marker-scoring from mis-classifying partial matches
                # Always get full scores dict for Spanish Context Guard
                _, _, all_scores = identify_language_with_scores(phonemes_primary)
                if word:
                    lang_primary, score_primary = "nah", 1.0
                elif lex_score >= 0.6:
                    # Strong lexicon signal but not full word match
                    lang_primary, score_primary = "nah", lex_score
                    lexicon_hit = True  # Count as lexicon hit for stats
                else:
                    lang_primary, score_primary = identify_language(phonemes_primary)

                lang = lang_primary
                score_used = score_primary

                # Overlap damping: reduce IPA score weight on mixed signal
                if _is_overlap_turn and eq.get("overlap_damping", False):
                    _ipa_w = eq.get("overlap_ipa_weight", 0.3)
                    score_used = score_used * _ipa_w
                    if score_used < 0.3:
                        lang = "oth"  # Too uncertain on mixed signal
                    print(f"  [overlap-damp] IPA score dampened {score_primary:.2f}→{score_used:.2f} (w={_ipa_w})")

                # Optional compare arbitration: choose backend that agrees better
                # with text-derived IPA anchor when available.
                if ipa_compare:
                    phonemes_compare = canonicalize_ipa_tokens(ipa_compare.split(), acoustic_tl_detected=tl_detected)
                    word_cmp, lex_score_cmp = check_nah_lexicon(phonemes_compare)
                    # P1: Apply same lexicon-override logic for compare backend
                    if word_cmp:
                        lang_compare, score_compare = "nah", 1.0
                    elif lex_score_cmp >= 0.6:
                        lang_compare, score_compare = "nah", lex_score_cmp
                    else:
                        lang_compare, score_compare = identify_language(phonemes_compare)
                    text_hint, _ = guess_language_from_text_markers(text or "", file_lang or "es")
                    anchor = text_to_ipa_tokens(text or "", text_hint)
                    agree_primary = ipa_agreement(phonemes_primary, anchor) if anchor else 0.0
                    agree_compare = ipa_agreement(phonemes_compare, anchor) if anchor else 0.0
                    agree_cross = ipa_agreement(phonemes_primary, phonemes_compare)
                    # Promote compare backend if it aligns better to text anchor.
                    if (agree_compare > agree_primary + 0.12 and score_compare >= score_primary * 0.7):
                        lang = lang_compare
                        score_used = score_compare
                        backend_override = True
                    else:
                        backend_override = False
                else:
                    agree_primary = 0.0
                    agree_compare = 0.0
                    backend_override = False
                if text:
                    # Fallback tie-breaker: Maya hint in text can rescue MAY tags.
                    maya_hint, _ = detect_maya_text_hint(text)
                    if maya_hint and lang in {"other", "spa", "eng"}:
                        lang = "may"
                # Ejective differentiation: NAH vs MAY vs ambiguous
                # tɬ is NAH-exclusive (does not exist in Maya).
                # Ejectives (ʼ) can be NAH or MAY — differentiate by co-occurring markers.
                # Implosive ɓ is a MAY-exclusive signal.
                _EJECTIVE_PHONES = {"tʼ", "kʼ", "pʼ", "tsʼ", "tʃʼ"}
                NAH_EXCLUSIVE_MARKERS = {"tɬ", "kʷ", "ɬ"}  # Do not exist in Maya
                ipa_for_guard = ipa_fused or ipa or ""
                has_nah_exclusive = any(m in ipa_for_guard for m in NAH_EXCLUSIVE_MARKERS)
                has_ejective_ipa = any(m in ipa_for_guard for m in _EJECTIVE_PHONES)
                has_implosive_b = "ɓ" in ipa_for_guard
                spa_score = all_scores.get("spa", 0.0)

                # Ejective voter scoring
                _ej_nah_boost = 0.0
                _ej_may_boost = 0.0
                _ej_tag = ""
                if has_nah_exclusive:
                    _ej_nah_boost += 2.0  # tɬ confirmed → strong NAH
                    _ej_tag = "ejective-nah"
                    if has_ejective_ipa:
                        _ej_nah_boost += 1.0  # tɬ + ejective → very strong NAH (+3 total)
                if has_implosive_b:
                    _ej_may_boost += 1.0  # ɓ alone → MAY signal
                    _ej_tag = "ejective-may" if not has_nah_exclusive else "ejective-nah"
                    if has_ejective_ipa:
                        _ej_may_boost += 1.0  # ɓ + ejective → strong MAY (+2 total)
                if has_ejective_ipa and not has_nah_exclusive and not has_implosive_b:
                    _ej_nah_boost += 0.5  # Ejective alone → weak NAH
                    _ej_tag = "ejective-ambiguous"

                # Acoustic ejective detection (from detector, not IPA)
                spa_guard_triggered = False
                _ej_min = ejective_min_count
                if ejective_strict:
                    # Strict mode: require ≥3 ejectives AND NAH-exclusive markers
                    # must NOT be present (tɬ → NAH, not MAY)
                    _ej_min = max(_ej_min, 3)
                if ejective_count >= _ej_min and lang in {"other", "spa", "eng", "nah"}:
                    suppress_ejective_boost = (
                        (spa_score > 0.5 and not has_nah_exclusive and not has_implosive_b)
                        or (has_nah_exclusive)  # tɬ wins, don't flip to MAY
                    )
                    # Strict mode: also suppress if IPA has NAH phonemes (kʷ, ɬ, tɬ)
                    if ejective_strict and not suppress_ejective_boost:
                        _nah_phonemes = {"tɬ", "kʷ", "ɬ"}
                        _ipa_str = ipa_fused or ipa or ""
                        if any(p in _ipa_str for p in _nah_phonemes):
                            suppress_ejective_boost = True
                    if suppress_ejective_boost:
                        spa_guard_triggered = True
                    else:
                        lang = "may"
                        lang_conf_source = f"ejective-{ejective_count}"
                        _ej_tag = "ejective-may"

                # Apply ejective-based NAH boost if IPA has strong NAH markers
                if _ej_nah_boost >= 2.0 and lang in {"other", "may"}:
                    lang = "nah"
                    _ej_tag = "ejective-nah"

                if _ej_tag:
                    print(f"  [{_ej_tag}] tɬ={has_nah_exclusive} ej_ipa={has_ejective_ipa} "
                          f"ɓ={has_implosive_b} ej_acoustic={ejective_count} "
                          f"→ nah_boost={_ej_nah_boost:.1f} may_boost={_ej_may_boost:.1f}")

                backend_tag = "allo+whisper" if text else "allosaurus"
                if backend_override:
                    backend_tag += "+compare-vote"
                if agree_cross > 0.0:
                    backend_tag += f":agr{agree_cross:.2f}"
                if spa_guard_triggered:
                    backend_tag += "+spa-guard"
                lang_conf = 1.0 if lexicon_hit else _lang_confidence(lang, score_used)
                lang_conf_source = "ipa-lexicon" if lexicon_hit else "ipa-score"
                # Apply ejective Maya boost AFTER base confidence assignment.
                if lang == "may" and ejective_count >= ejective_min_count:
                    lang_conf = min(1.0, lang_conf + ejective_count * ejective_maya_boost)
                    lang_conf_source += f"+ejective-{ejective_count}"
                # Apply NAH ejective IPA boost
                if _ej_nah_boost > 0 and lang == "nah":
                    lang_conf = min(1.0, lang_conf + _ej_nah_boost * 0.1)
                    lang_conf_source += f"+ej-nah-{_ej_nah_boost:.0f}"

                # Speaker-prior voter: if this speaker has a strong history, nudge confidence
                _spk = turn.get("speaker")
                if _spk and _spk in _speaker_lang_history:
                    _spk_hist = _speaker_lang_history[_spk]
                    _spk_total = sum(_spk_hist.values())
                    if _spk_total >= 3:
                        _spk_best_lang, _spk_best_count = _spk_hist.most_common(1)[0]
                        _spk_ratio = _spk_best_count / _spk_total
                        if _spk_ratio >= 0.7:
                            if _spk_best_lang == "nah" and lang in {"other", "nah", "may"}:
                                if lang == "may" and _spk_ratio >= 0.8 and not has_implosive_b:
                                    # Strong NAH speaker + MAY from ejective FP → correct to NAH
                                    lang = "nah"
                                    lang_conf = min(1.0, lang_conf + 0.1)
                                    lang_conf_source += "+speaker-prior-nah"
                                    print(f"  [speaker-prior-nah] {_spk}: {_spk_best_count}/{_spk_total} NAH → MAY→NAH")
                                elif lang in {"other", "nah"}:
                                    lang_conf = min(1.0, lang_conf + 0.1)
                                    if lang == "other":
                                        lang = "nah"
                                        uncertain = False
                                    lang_conf_source += "+speaker-prior-nah"
                                    print(f"  [speaker-prior-nah] {_spk}: {_spk_best_count}/{_spk_total} NAH → boost")
                            elif _spk_best_lang == "spa" and lang in {"other", "nah"}:
                                if lang == "nah" and not has_nah_exclusive:
                                    # SPA speaker with NAH classification but no tɬ → demote NAH confidence
                                    lang_conf = max(0.0, lang_conf - 0.1)
                                    lang_conf_source += "+speaker-prior-spa"
                                    print(f"  [speaker-prior-spa] {_spk}: {_spk_best_count}/{_spk_total} SPA → NAH demote")

                # MMS voter: override when MMS is confident and disagrees
                if mms_lang and mms_conf >= 0.5:
                    if mms_lang != lang:
                        # MMS disagrees with pipeline — log and potentially override
                        _mms_override = False
                        if mms_conf >= 0.8 and lang in {"other", "eng"}:
                            # High-confidence MMS overrides OTH/ENG
                            _mms_override = True
                        elif mms_conf >= 0.6 and lang == "spa" and mms_lang == "nah":
                            # MMS says NAH with decent confidence, pipeline says SPA
                            # (Whisper hallucinated Spanish on Nahuatl audio)
                            if not (_spa_func_hits >= 3 and _whisper_conf >= 0.8):
                                _mms_override = True
                        elif mms_conf >= 0.6 and lang == "nah" and mms_lang == "spa":
                            # MMS says SPA, pipeline says NAH — trust MMS if no tɬ
                            if not has_nah_exclusive:
                                _mms_override = True
                        elif mms_conf >= 0.7 and lang == "other" and mms_lang in {"nah", "spa", "may"}:
                            _mms_override = True

                        if _mms_override:
                            print(
                                f"  [mms-override] {lang}→{mms_lang} "
                                f"(mms_conf={mms_conf:.2f}, was {lang_conf_source})"
                            )
                            lang = mms_lang
                            lang_conf = mms_conf
                            lang_conf_source += f"+mms-override({mms_raw}:{mms_conf:.2f})"
                        else:
                            print(
                                f"  [mms-disagree] pipeline={lang} mms={mms_lang}({mms_conf:.2f}) "
                                f"— kept pipeline ({lang_conf_source})"
                            )
                    else:
                        # MMS agrees — boost confidence
                        _mms_boost = min(0.15, mms_conf * 0.15)
                        lang_conf = min(1.0, lang_conf + _mms_boost)
                        lang_conf_source += f"+mms-agree({mms_raw}:{mms_conf:.2f})"

                if lang not in {"lat", "may"} and lang_conf < lang_conf_threshold:
                    # Conservative gate: uncertain IPA language guesses go to OTH.
                    uncertain = True
                    lang = "other"
                    backend_tag += "+uncertain"

                # FIX 8: SPA Text Rescue — when IPA says OTH but Whisper text is Spanish
                # Uses guess_language_from_text_markers (broad SPA_COMMON set) as evidence.
                # Only rescues if IPA has no strong indigenous markers contradicting SPA.
                _NAH_STRONG_IPA = {"tɬ", "kʷ", "ɬ"}
                _MAY_STRONG_IPA = {"kʼ", "tʼ", "tsʼ", "pʼ"}
                if lang == "other" and text:
                    _text_lang, _text_conf = guess_language_from_text_markers(text, file_lang or "es")
                    if _text_lang == "spa" and _text_conf >= 0.5:
                        _ipa_check = ipa_fused or ipa or ""
                        _has_indigenous = any(m in _ipa_check for m in _NAH_STRONG_IPA | _MAY_STRONG_IPA)
                        if not _has_indigenous:
                            lang = "spa"
                            lang_conf = _text_conf
                            lang_conf_source = f"spa-text-rescue:{_text_conf:.2f}"
                            backend_tag += "+spa-text-rescue"
                            uncertain = False
                            print(
                                f"  [spa-rescue] Whisper text looks SPA "
                                f"(conf={_text_conf:.2f}), IPA has no indigenous markers → SPA"
                            )

                # FIX 2: Spanish text leak detection
                # If Whisper outputs Spanish words but lang is NAH/MAY/OTH/ENG, override to SPA
                if lang in {"nah", "may", "other", "eng"} and text:
                    is_spa_leak, leak_reason = detect_spanish_text_leak(text, ipa, lang)
                    if is_spa_leak:
                        lang = "spa"
                        lang_conf = 0.7  # Moderate confidence
                        lang_conf_source = f"spa-leak:{leak_reason}"
                        backend_tag += "+spa-leak-fix"
                        uncertain = False

                # FIX 1: OTH Recovery for short IPA on long audio
                # If IPA has < 5 tokens but audio > 2s, it's likely a backend failure
                turn_duration = turn["end"] - turn["start"]
                ipa_token_count = len(ipa.split()) if ipa else 0
                if lang == "other" and turn_duration > 2.0 and ipa_token_count < 5:
                    # Mark as potential recovery target (for future backend retry)
                    backend_tag += f"+short-ipa({ipa_token_count}t/{turn_duration:.1f}s)"
                    # Don't change lang here - just flag it for analysis

                # FIX 3: Ejective annotation in IPA
                if ejective_count >= ejective_min_count and ipa:
                    ipa = annotate_ejectives_in_ipa(ipa, ejective_count, ejective_min_count)
                    if ipa_fused:
                        ipa_fused = annotate_ejectives_in_ipa(ipa_fused, ejective_count, ejective_min_count)

                # FIX 4: Clean LLM text (remove glosses, detect IPA-as-text)
                if text:
                    original_text = text
                    text = clean_llm_text(text)
                    if text != original_text:
                        if not text:
                            backend_tag += "+ipa-text-cleaned"
                        elif "(" in original_text:
                            backend_tag += "+gloss-removed"

                # FIX 5: Phoneme density validation
                density_status = validate_phoneme_density(ipa, turn_duration)
                if density_status == "COLLAPSED":
                    backend_tag += f"+collapsed({ipa_token_count}p/{turn_duration:.1f}s)"
                elif density_status == "HALLUCINATED":
                    backend_tag += f"+hallucinated({ipa_token_count}p/{turn_duration:.1f}s)"
                elif density_status == "SHORT":
                    backend_tag += "+short-seg"
            else:
                # Neither produced useful output
                continue
            backend_tag += f":{profile_used}"

            # --- Overlap damping: cap confidence on overlap segments ---
            if _is_overlap_turn and eq.get("overlap_damping", False):
                _conf_cap = eq.get("overlap_conf_cap", 0.5)
                if lang_conf > _conf_cap:
                    print(f"  [overlap-damp] conf capped {lang_conf:.2f}→{_conf_cap} on overlap turn")
                    lang_conf = _conf_cap
                    lang_conf_source = f"overlap-capped:{lang_conf_source}"

            # Extract trim info from primary backend detail
            trim_info = (ipa_detail or {}).get("trim", {})

            all_results.append({
                "start": turn["start"], "end": turn["end"],
                "speaker": turn["speaker"],
                "lang": lang,
                "text": text,           # Whisper text (even "hallucinated")
                "whisper_valid": whisper_valid,
                "ipa": ipa,             # Primary backend IPA
                "ipa_compare": ipa_compare,  # Compare backend IPA (None if not in compare mode)
                "ipa_omni": ipa_omni,   # Omnilingual backend IPA
                "ipa_fused": ipa_fused,
                "ipa_fused_conf": ipa_fused_conf,
                "ipa_phone_conf": (ipa_detail or {}).get("phone_conf", []),
                "ipa_nbest": (ipa_detail or {}).get("nbest", []),
                "ipa_compare_phone_conf": (ipa_compare_detail or {}).get("phone_conf", []),
                "ipa_compare_nbest": (ipa_compare_detail or {}).get("nbest", []),
                "ipa_conf": round(score_used, 3) if score_used is not None else None,
                "lang_conf": round(lang_conf, 3),
                "lang_conf_source": lang_conf_source,
                "lang_uncertain": bool(uncertain),
                "ipa_text_agree": round(agree_primary, 3),
                "ipa_compare_text_agree": round(agree_compare, 3),
                "ipa_cross_agree": round(agree_cross, 3),
                "decode_ipa_agree": round(decode_agree, 3),
                "backend": backend_tag,
                "ejective_count": ejective_count,  # Acoustic ejective detection for Maya
                "tl_detected": bool(tl_result.get("tl_detected")),
                "tl_confidence": float(tl_result.get("tl_confidence", 0.0)),
                "tl_events": int(tl_result.get("tl_events", 0)),
                "trim_info": trim_info,  # Voice onset trimming stats
                "mms_lang": mms_lang,
                "mms_conf": round(mms_conf, 4) if mms_lang else None,
                "mms_raw": mms_raw,
                "overlap": _is_overlap_turn,
                "overlap_separated": _sep_ipa_sources is not None,
                "overlap_sources": [
                    {"lang": s["lang"], "rms": round(s["rms"], 4),
                     "ipa": (s["ipa"] or "")[:80]}
                    for s in _sep_ipa_sources
                ] if _sep_ipa_sources else None,
            })

            # Update speaker language history for voter-prior
            _spk_update = turn.get("speaker")
            if _spk_update and lang not in ("other", "silence"):
                if _spk_update not in _speaker_lang_history:
                    _speaker_lang_history[_spk_update] = _Counter()
                _speaker_lang_history[_spk_update][lang] += 1

    print(f"Unified: {len(all_results)} segments, "
          f"{sum(1 for r in all_results if r['text'])} with text, "
          f"{allo_count} with IPA")

    # STEP 6: Whisper rescue pass — retry missing/low-quality text with vad_filter=False
    rescue_count = 0
    if whisper_rescue and not whisper_only:
        rescue_targets = [
            i for i, r in enumerate(all_results)
            if (not r.get("text")) or (r.get("text") and not r.get("whisper_valid"))
        ]
        if rescue_targets:
            no_text = sum(1 for i in rescue_targets if not all_results[i].get("text"))
            low_quality = len(rescue_targets) - no_text
            print(
                f"Rescue pass: {len(rescue_targets)} targets "
                f"({no_text} missing text, {low_quality} low-quality text)"
            )

            # Free phoneme backends to make VRAM room
            primary_backend.cleanup()
            if compare_backend:
                compare_backend.cleanup()

            # Build vocab prompt from successful transcriptions (if enabled)
            vocab_prompt = None
            if whisper_prompt:
                seen_words = set()
                if whisper_prompt_extra:
                    for w in re.findall(r"[a-záéíóúüñ]+", whisper_prompt_extra.lower()):
                        if len(w) > 1:
                            seen_words.add(w)
                for r in all_results:
                    if r.get("whisper_valid") and r.get("text"):
                        for w in re.findall(r"[a-záéíóúüñ]+", r["text"].lower()):
                            if len(w) > 2:
                                seen_words.add(w)
                # Top words by frequency in SPA_COMMON + any proper nouns
                prompt_words = sorted(seen_words - {"que", "de", "la", "el", "en", "no", "es"})
                if prompt_words:
                    vocab_prompt = ", ".join(prompt_words[:50])
                    print(f"  Vocab prompt: {vocab_prompt[:80]}...")

            # Reload Whisper for rescue
            print(f"  Reloading Whisper {whisper_model} for rescue...")
            whisper_rescue_model = WhisperModel(
                whisper_model, device="cuda", compute_type="float16",
                download_root=f"{CACHE_DIR}/whisper",
            )

            # Rescue works better on denoised vocals when available.
            rescue_audio = vocals_data if vocals_data is not None else audio_data
            rescue_sr = vocals_sr if vocals_data is not None else audio_sr

            for idx in rescue_targets:
                r = all_results[idx]
                # Extract chunk with 500ms padding on each side
                pad = 0.5
                chunk_start = max(0, r["start"] - pad)
                chunk_end = min(duration, r["end"] + pad)
                i0 = int(chunk_start * rescue_sr)
                i1 = int(chunk_end * rescue_sr)
                chunk = rescue_audio[i0:i1]

                # Add 500ms silence padding for clean Whisper boundaries
                silence = np.zeros(int(0.5 * rescue_sr), dtype=np.float32)
                if chunk.ndim > 1:
                    silence = np.zeros((int(0.5 * rescue_sr), chunk.shape[1]), dtype=np.float32)
                chunk_padded = np.concatenate([silence, chunk, silence])

                # Write to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, chunk_padded, rescue_sr)
                    try:
                        kwargs = {
                            "vad_filter": False,
                            "beam_size": 5,
                            "word_timestamps": True,
                            # Keep rescue anchored to the same file-level language
                            # as the main Whisper pass to avoid random script drift.
                            "language": decode_lang or file_lang or None,
                        }
                        if vocab_prompt:
                            kwargs["initial_prompt"] = vocab_prompt
                            kwargs["condition_on_previous_text"] = False

                        segs, seg_info = whisper_rescue_model.transcribe(tmp.name, **kwargs)
                        rescue_texts = []
                        for s in segs:
                            st = s.text.strip()
                            if st and not has_non_latin_script(st):
                                rescue_texts.append({
                                    "text": st,
                                    "avg_log_prob": s.avg_logprob,
                                })

                        if rescue_texts:
                            best = max(rescue_texts, key=lambda t: t["avg_log_prob"])
                            best_valid = validate_whisper(best["text"], best["avg_log_prob"])
                            # Only accept rescue output if it passes validation.
                            # This prevents random-script or off-topic text from replacing
                            # an existing segment.
                            if best_valid:
                                r["text"] = best["text"]
                                r["whisper_valid"] = True
                                r["backend"] = r["backend"].replace("allosaurus", "allo+rescue")
                                if "allo" not in r["backend"]:
                                    r["backend"] = "rescue+" + r["backend"]
                                # EQ: whisper_uncertain_ipa — preserve IPA-based language
                                # when rescue provides text. Don't let rescue override
                                # language classification that IPA already determined.
                                if eq.get("whisper_uncertain_ipa") and r.get("lang_conf_source", "").startswith(("allo", "ipa", "tl-")):
                                    pass  # Keep IPA-based lang, only update text
                                else:
                                    # Re-classify language from rescued text
                                    _rescue_lang, _rescue_conf = guess_language_from_text_markers(
                                        best["text"], file_lang or "es"
                                    )
                                    if _rescue_lang == "spa" and _rescue_conf >= 0.55:
                                        r["lang"] = "spa"
                                        r["lang_conf"] = _rescue_conf
                                        r["lang_conf_source"] = f"rescue-spa:{_rescue_conf:.2f}"
                                rescue_count += 1
                    except Exception as e:
                        pass  # Silent fail, segment keeps IPA-only
                    finally:
                        os.unlink(tmp.name)

            del whisper_rescue_model
            torch.cuda.empty_cache()
            print(f"  Rescued: {rescue_count}/{len(rescue_targets)} segments")

            # Re-initialize phoneme backends if needed for cleanup
            # (they're about to be cleaned up anyway, so just mark as None)
            primary_backend = None
            compare_backend = None
        else:
            print("Rescue pass: no orphan segments, skipping")

    # NOTE: do not cross-merge/post-prune text here; diarization turns are authoritative.

    policy = (uncertain_text_policy or "keep").strip().lower()
    if policy not in {"keep", "suppress", "ipa"}:
        policy = "keep"

    # Helper for suppress_hallucinations AND-gate logic
    def _should_suppress_segment(r, speaker_validated_counts, low_conf_threshold):
        """Returns True if segment should have text suppressed.

        Broadened gate: suppress when Whisper text is unreliable.
        Two entry paths:
          Path A: lang_uncertain=True (IPA path uncertain about language)
          Path B: whisper_valid=False (Whisper text failed validation)
        Either path can trigger suppression, subject to speaker exemption
        and confidence checks.
        """
        is_uncertain = r.get("lang_uncertain", False)
        is_whisper_invalid = not r.get("whisper_valid", True)

        # Must have at least one unreliability signal
        if not is_uncertain and not is_whisper_invalid:
            return False

        # Speaker exemption: 5+ validated segments = benefit of doubt
        speaker = r.get("speaker")
        if speaker and speaker_validated_counts.get(speaker, 0) >= 5:
            return False

        # For whisper-invalid segments: suppress if Whisper failed validation
        # (the text is hallucinated/garbage regardless of IPA confidence)
        if is_whisper_invalid:
            # Short segments: always suppress when Whisper is invalid
            duration = r.get("end", 0.0) - r.get("start", 0.0)
            if duration < 1.0:
                return True
            # Longer segments: suppress if IPA confidence is also low,
            # OR if lang_uncertain (both paths agree it's bad)
            ipa_conf = r.get("ipa_conf")
            if is_uncertain:
                return True  # Both paths agree: suppress
            if ipa_conf is None or ipa_conf < low_conf_threshold:
                return True  # IPA confidence also low: suppress
            # Whisper invalid but IPA is confident: suppress the Whisper text
            # but note this is a borderline case. The IPA evidence is good
            # so [REC] recovery will likely produce useful output.
            return True

        # Pure lang_uncertain path (whisper IS valid but IPA is uncertain)
        # This is the original conservative gate — only suppress when
        # Whisper confidence is also low
        whisper_low = False
        if r.get("lang_conf") is not None and r["lang_conf"] < low_conf_threshold:
            whisper_low = True

        if not whisper_low:
            return False

        duration = r.get("end", 0.0) - r.get("start", 0.0)
        if duration < 1.0:
            return True

        ipa_conf = r.get("ipa_conf")
        if ipa_conf is None or ipa_conf < low_conf_threshold:
            return True

        return False

    # Build speaker context: count validated segments per speaker
    speaker_validated_counts = {}
    if suppress_hallucinations:
        for r in all_results:
            if r.get("whisper_valid") and r.get("speaker"):
                speaker = r["speaker"]
                speaker_validated_counts[speaker] = speaker_validated_counts.get(speaker, 0) + 1

    # STEP 6.37: (removed — overlap detection now runs pre-loop with optional MossFormer2 separation)

    # STEP 6.38: FT-first — run NAH finetuned Whisper on ALL segments BEFORE speaker-prior.
    # Uses FT output as classification signal: if FT produces recognizable NAH morphology
    # on a SPA-tagged segment, override to NAH. This breaks the chicken-and-egg problem
    # at the source: FT doesn't depend on speaker-prior, so its output is unbiased.
    _ft_first_done = False
    if eq.get("ft_first", False) and nah_whisper_finetuned:
        nah_model_path = f"{NAH_MODEL_DIR}/checkpoints/{nah_checkpoint}" if nah_checkpoint else f"{NAH_MODEL_DIR}/model"
        _is_lora = os.path.exists(f"{nah_model_path}/adapter_config.json")
        _is_merged = os.path.exists(f"{nah_model_path}/config.json")
        if os.path.exists(nah_model_path) and (_is_lora or _is_merged):
            # Target: ALL non-silence segments (SPA + OTH + NAH + everything)
            ft_first_targets = [
                i for i, r in enumerate(all_results)
                if r.get("lang") not in ("silence",) and (r["end"] - r["start"]) >= 0.8
            ]
            if ft_first_targets:
                print(f"  [ft-first] Running FT on ALL {len(ft_first_targets)} segments (before speaker-prior)")

                # Free main Whisper to reclaim GPU memory
                if whisper is not None:
                    del whisper
                    whisper = None
                    torch.cuda.empty_cache()

                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                print(f"  [ft-first] Loading finetuned Whisper from {nah_model_path}...")
                nah_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
                if _is_lora:
                    import json as _json
                    from safetensors.torch import load_file as _load_safetensors
                    nah_model = WhisperForConditionalGeneration.from_pretrained(
                        "openai/whisper-large-v3", torch_dtype=torch.float16,
                    )
                    _adapter_cfg = _json.loads(open(f"{nah_model_path}/adapter_config.json").read())
                    _lora_alpha = _adapter_cfg.get("lora_alpha", 64)
                    _lora_r = _adapter_cfg.get("r", 32)
                    _scaling = _lora_alpha / _lora_r
                    _adapter_weights = _load_safetensors(f"{nah_model_path}/adapter_model.safetensors")
                    _state = nah_model.state_dict()
                    _merged = 0
                    for key, val in _adapter_weights.items():
                        if "lora_A" in key:
                            _base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
                            _b_key = key.replace("lora_A", "lora_B")
                            if _base_key in _state and _b_key in _adapter_weights:
                                _a = val.to(torch.float32)
                                _b = _adapter_weights[_b_key].to(torch.float32)
                                _delta = (_b @ _a) * _scaling
                                _state[_base_key] = _state[_base_key].to(torch.float32) + _delta
                                _state[_base_key] = _state[_base_key].to(torch.float16)
                                _merged += 1
                    nah_model.load_state_dict(_state)
                    nah_model = nah_model.to("cuda")
                    print(f"  [ft-first] Merged {_merged} LoRA layers")
                else:
                    nah_model = WhisperForConditionalGeneration.from_pretrained(
                        nah_model_path, torch_dtype=torch.float16,
                    ).to("cuda")
                nah_model.eval()

                # NAH morphology detector (same as STEP 6.55)
                _NAH_TEXT_RX = re.compile(
                    r"tl[aeiou]|tz[aeiou]|hu[aeiou]|cu[aeiou]|ua[ctnlshp]"
                    r"|xic|xoc|xol|chih|chihu|chiw|moch|noch|tech|nech"
                    r"|nik[aeiou]|tik[aeiou]|ntek|ntik|necin|necua"
                    r"|xk[aeiou]|kw[aeiou]|hw[aeiou]|[aeiou]wk"
                    r"|atl|etl|itl|otl|utl|[aeiou]tzin|coch|poch|toch|calc|calt|tepet"
                    r"|teca|tlaca|xica|naca|meca|titec|piltec"
                    r"|teka|tlaka|xika|naka|meka|titek|piltek"
                    r"|iske|isce|isc[aeiou]|isk[aeiou]"
                    r"|monec|monek|oneki|oneci"
                    r"|mict[aeiou]|tequi|chiua|cipac|mani|pach[aeiou]"
                    r"|tict|ticm|ticn|nicn|nicm|nict|quin[aeiou]"
                    r"|nican|ican|tlaco|tlam[aeiou]|ilis|ilia|cali"
                )
                _ft_first_count = 0
                _ft_first_overrides = 0
                for idx in ft_first_targets:
                    r = all_results[idx]
                    seg_start, seg_end = r["start"], r["end"]
                    start_sample = int(seg_start * audio_sr)
                    end_sample = int(seg_end * audio_sr)
                    seg_audio = audio_data[start_sample:end_sample]
                    if seg_audio.ndim > 1:
                        seg_audio = seg_audio.mean(axis=1)
                    seg_audio = seg_audio.astype(np.float32)
                    if len(seg_audio) < 160:
                        continue

                    input_features = nah_processor.feature_extractor(
                        seg_audio, sampling_rate=audio_sr, return_tensors="pt",
                    ).input_features.to("cuda", dtype=torch.float16)

                    with torch.no_grad():
                        predicted_ids = nah_model.generate(
                            input_features, task="transcribe",
                            language="spanish", max_new_tokens=225,
                        )
                    ft_text = nah_processor.tokenizer.batch_decode(
                        predicted_ids, skip_special_tokens=True,
                    )[0].strip()

                    if ft_text:
                        r["nah_ft_text"] = ft_text
                        _ft_first_count += 1

                        # Check NAH morphology: if SPA segment has NAH FT → override
                        has_nah_morphology = bool(
                            _NAH_TEXT_RX.search(ft_text.lower())
                            or fuzzy_nah_text_check(ft_text)
                        )
                        if r.get("lang") == "spa" and has_nah_morphology:
                            # Overlap gate: skip override if simultaneous speakers detected
                            if r.get("overlap", False):
                                print(f"  [ft-first] cue~{idx} {r.get('speaker','?')}: SPA→NAH "
                                      f"BLOCKED by overlap gate (FT='{ft_text[:40]}')")
                                continue
                            old_conf = r.get("lang_conf", 0.5)
                            r["lang"] = "nah"
                            r["lang_conf"] = 0.65  # Moderate confidence (above UNK gate)
                            r["lang_conf_source"] = f"ft-first-override:{old_conf:.2f}"
                            r["backend"] = r.get("backend", "") + "+ft-first"
                            _ft_first_overrides += 1
                            print(f"  [ft-first] cue~{idx} {r.get('speaker','?')}: SPA→NAH "
                                  f"(FT='{ft_text[:40]}', morphology=True)")

                del nah_model, nah_processor
                torch.cuda.empty_cache()
                print(f"  [ft-first] Done: {_ft_first_count} FT texts, {_ft_first_overrides} SPA→NAH overrides")
                _ft_first_done = True

    # STEP 6.39: Two-pass speaker prior — IPA phoneme evidence re-counts speaker languages
    # before the speaker-prior runs. Fixes chicken-and-egg: Whisper maps NAH→SPA cognates,
    # so speaker profile accumulates wrong language, reinforcing the error.
    # Solution: check IPA for non-Spanish phonemes (ts, tɬ, kʷ) and count those segments
    # as NAH for the speaker profile, regardless of what Whisper/classification said.
    if eq.get("two_pass_prior", False):
        from collections import defaultdict as _dd
        _NAH_IPA_MARKERS = {"ts", "tɬ", "kʷ", "tʃʼ", "kʼ", "tɕ"}  # phonemes absent from Spanish
        _speaker_ipa_nah = _dd(int)
        _speaker_total = _dd(int)
        _ipa_overrides = 0

        for r in all_results:
            spk = r.get("speaker")
            if not spk:
                continue
            lang = r.get("lang", "other")
            if lang in ("silence",):
                continue
            _speaker_total[spk] += 1

            # Check fused IPA for non-Spanish phonemes
            ipa_str = r.get("ipa_fused") or r.get("ipa") or ""
            ipa_tokens = set(ipa_str.split())
            has_nah_marker = bool(ipa_tokens & _NAH_IPA_MARKERS)
            if has_nah_marker:
                _speaker_ipa_nah[spk] += 1

        # For speakers with significant IPA-NAH evidence, override their SPA segments
        print(f"  [two-pass-prior] IPA-NAH evidence per speaker:")
        for spk in sorted(_speaker_ipa_nah, key=lambda s: -_speaker_ipa_nah[s]):
            nah_count = _speaker_ipa_nah[spk]
            total = _speaker_total[spk]
            nah_ratio = nah_count / total if total else 0
            spa_count = sum(1 for r in all_results if r.get("speaker") == spk and r.get("lang") == "spa")
            nah_tagged = sum(1 for r in all_results if r.get("speaker") == spk and r.get("lang") == "nah")
            print(f"    {spk}: {nah_count}/{total} IPA-NAH ({nah_ratio:.0%}), "
                  f"tagged {nah_tagged} NAH / {spa_count} SPA")

        for spk, nah_count in _speaker_ipa_nah.items():
            total = _speaker_total[spk]
            if total < 5:
                continue
            nah_ratio = nah_count / total
            if nah_ratio < 0.15:  # need at least 15% IPA-NAH evidence
                continue

            # Count how many are currently tagged SPA vs NAH
            spa_count = sum(1 for r in all_results if r.get("speaker") == spk and r.get("lang") == "spa")
            nah_tagged = sum(1 for r in all_results if r.get("speaker") == spk and r.get("lang") == "nah")

            # If IPA says significant NAH evidence → override low-conf SPA to NAH
            # Use IPA ratio, not current tag ratio (tags are already polluted by inline prior)
            if nah_count >= 3:
                print(f"  [two-pass-prior] {spk}: IPA-NAH evidence {nah_count}/{total} ({nah_ratio:.0%}), "
                      f"tagged {nah_tagged} NAH vs {spa_count} SPA — overriding low-conf SPA→NAH")
                for r in all_results:
                    if r.get("speaker") == spk and r.get("lang") == "spa":
                        old_conf = r.get("lang_conf", 0.5)
                        if old_conf < 0.7:
                            r["lang"] = "nah"
                            r["lang_conf_source"] = f"two-pass-ipa:{old_conf:.2f}"
                            r["backend"] = r.get("backend", "") + "+two-pass-prior"
                            _ipa_overrides += 1

        if _ipa_overrides > 0:
            print(f"  Two-pass prior: {_ipa_overrides} SPA→NAH overrides from IPA evidence")

    # STEP 6.4: Speaker-Prior — propagate majority language per speaker
    # Fixes NAH detection: SPEAKER_12/09/07/14 are 100% NAH in Hernán but per-segment
    # detection misclassifies ~47% as OTH and ~29% as SPA. Speaker-prior recovers these.
    def apply_speaker_prior(results, min_segments=3, min_ratio=0.6, strong_mode=False):
        """Override segment language with speaker's majority language.

        For indigenous languages (NAH, MAY), speakers typically use only one language
        throughout a scene. This propagates the majority vote to all their segments.

        Args:
            min_segments: Minimum segments before locking (default 3)
            min_ratio: Minimum ratio for majority language (default 0.6 = 60%)
            strong_mode: If True, also override low-confidence SPA↔NAH (not just OTH)

        Returns:
            Count of overridden segments
        """
        from collections import defaultdict

        # Count languages per speaker (excluding OTH since it's uncertain)
        speaker_langs = defaultdict(lambda: defaultdict(int))
        for r in results:
            spk = r.get("speaker")
            lang = r.get("lang", "other")
            if spk and lang not in ("other", "silence"):
                speaker_langs[spk][lang] += 1

        # Determine majority language per speaker
        speaker_primary = {}
        for speaker, langs in speaker_langs.items():
            total = sum(langs.values())
            if total >= min_segments:
                best_lang, best_count = max(langs.items(), key=lambda x: x[1])
                if best_count / total >= min_ratio:
                    speaker_primary[speaker] = best_lang
                    print(f"  [speaker-prior] {speaker}: {best_lang.upper()} "
                          f"({best_count}/{total} = {best_count/total*100:.0f}%)")

        # Apply overrides
        override_count = 0
        for r in results:
            spk = r.get("speaker")
            if spk in speaker_primary:
                old_lang = r.get("lang", "other")
                new_lang = speaker_primary[spk]
                if old_lang == new_lang:
                    continue
                # Default: only upgrade OTH → known language
                if old_lang == "other" and new_lang != "other":
                    r["lang"] = new_lang
                    r["lang_prior"] = True
                    override_count += 1
                # Strong mode: also override misclassified segments
                # (e.g., SPA speaker with NAH tag, or NAH speaker with SPA/MAY tag)
                elif strong_mode and old_lang not in ("silence",):
                    # Only override if confidence is low (< 0.7) — high-confidence
                    # classifications should be trusted over the prior.
                    old_conf = r.get("lang_conf", 0.5)
                    if old_conf < 0.7:
                        print(f"  [speaker-prior-strong] {spk}: {old_lang.upper()}"
                              f"→{new_lang.upper()} (conf={old_conf:.2f})")
                        r["lang"] = new_lang
                        r["lang_prior"] = True
                        override_count += 1

        return override_count

    prior_overrides = apply_speaker_prior(
        all_results,
        min_segments=eq["speaker_prior_min_segments"],
        min_ratio=eq["speaker_prior_min_ratio"],
        strong_mode=speaker_prior_strong,
    )
    if prior_overrides > 0:
        print(f"  Speaker-prior: {prior_overrides} segments overridden"
              f"{' (strong mode)' if speaker_prior_strong else ' (OTH only)'}")

    # STEP 6.45: Noise gate — tag very short segments without meaningful text as UNK.
    if noise_gate:
        _noise_gated = 0
        for r in all_results:
            dur = r["end"] - r["start"]
            if dur > noise_gate_max_s:
                continue
            # Only gate if there's no meaningful Whisper text
            _text = (r.get("text") or "").strip()
            # Strip tags like [FT], [LLM], speaker labels
            _clean = re.sub(r"\[.*?\]", "", _text).strip()
            # Remove single-character utterances ("A.", "Mm.", etc.)
            _words = [w for w in re.findall(r"[a-záéíóúñ]+", _clean.lower()) if len(w) > 1]
            if len(_words) >= 2:
                continue  # Has real text content, keep it
            old_lang = r.get("lang", "other")
            if old_lang == "silence":
                continue
            r["lang"] = "other"  # Will map to OTH/UNK in SRT
            r["lang_conf"] = 0.0
            r["lang_conf_source"] = f"noise-gate:{dur:.2f}s"
            r["backend"] = r.get("backend", "") + "+noise-gate"
            _noise_gated += 1
        if _noise_gated > 0:
            print(f"  Noise gate: {_noise_gated} segments ≤{noise_gate_max_s}s → OTH")

    # STEP 6.46: UNK reject gate — tag low-confidence segments as UNK instead of forced label.
    # "Reject first, classify second": if all voters couldn't push confidence above threshold,
    # honest UNK is better than a coin flip. Runs after speaker-prior and noise-gate so those
    # had a chance to boost confidence. Runs before NAH FT to avoid wasting inference on garbage.
    if eq["unk_gate_enabled"]:
        _unk_threshold = eq["unk_gate_threshold"]
        _unk_gated = 0
        for r in all_results:
            lang = r.get("lang", "other")
            # Skip already-unknown/silence segments and hard-override LAT (text-based, reliable)
            if lang in ("other", "silence", "unknown", "lat"):
                continue
            conf = r.get("lang_conf", 0.0)
            if conf < _unk_threshold:
                old_lang = lang
                r["lang"] = "unknown"
                r["lang_conf_source"] = f"unk-gate:{conf:.2f}<{_unk_threshold}"
                r["backend"] = r.get("backend", "") + "+unk-gate"
                _unk_gated += 1
        if _unk_gated > 0:
            print(f"  UNK gate: {_unk_gated} segments with conf<{_unk_threshold} → UNK")

    # STEP 6.5: NAH Finetuned Whisper pass — re-transcribe NAH segments with finetuned model
    # Runs AFTER speaker-prior so OTH→NAH upgrades are included
    # Skip if ft_first already processed all segments
    nah_ft_count = 0
    if nah_whisper_finetuned and not _ft_first_done:
        nah_model_path = f"{NAH_MODEL_DIR}/checkpoints/{nah_checkpoint}" if nah_checkpoint else f"{NAH_MODEL_DIR}/model"
        _is_lora = os.path.exists(f"{nah_model_path}/adapter_config.json")
        _is_merged = os.path.exists(f"{nah_model_path}/config.json")
        if os.path.exists(nah_model_path) and (_is_lora or _is_merged):
            nah_targets = [
                i for i, r in enumerate(all_results)
                if r.get("lang") == "nah"
            ]
            if nah_targets:
                print(f"  NAH finetuned pass: {len(nah_targets)} NAH segments to re-transcribe")

                # Free main Whisper to reclaim GPU memory
                if whisper is not None:
                    del whisper
                    whisper = None
                    torch.cuda.empty_cache()

                # Load finetuned HuggingFace Whisper model
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                print(f"  Loading finetuned Whisper from {nah_model_path} ({nah_checkpoint or 'default'})...")
                nah_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
                if _is_lora:
                    # LoRA adapter checkpoint — load base model + manually merge adapter weights
                    import json as _json
                    from safetensors.torch import load_file as _load_safetensors
                    print(f"  LoRA adapter detected, loading base model + merging adapter...")
                    nah_model = WhisperForConditionalGeneration.from_pretrained(
                        "openai/whisper-large-v3", torch_dtype=torch.float16,
                    )
                    _adapter_cfg = _json.loads(open(f"{nah_model_path}/adapter_config.json").read())
                    _lora_alpha = _adapter_cfg.get("lora_alpha", 64)
                    _lora_r = _adapter_cfg.get("r", 32)
                    _scaling = _lora_alpha / _lora_r
                    _adapter_weights = _load_safetensors(f"{nah_model_path}/adapter_model.safetensors")
                    _state = nah_model.state_dict()
                    _merged = 0
                    for key, val in _adapter_weights.items():
                        # LoRA keys: base_model.model.{module}.lora_A.weight / lora_B.weight
                        if "lora_A" in key:
                            _base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
                            _b_key = key.replace("lora_A", "lora_B")
                            if _base_key in _state and _b_key in _adapter_weights:
                                _a = val.to(torch.float32)
                                _b = _adapter_weights[_b_key].to(torch.float32)
                                _delta = (_b @ _a) * _scaling
                                _state[_base_key] = _state[_base_key].to(torch.float32) + _delta
                                _state[_base_key] = _state[_base_key].to(torch.float16)
                                _merged += 1
                    nah_model.load_state_dict(_state)
                    nah_model = nah_model.to("cuda")
                    print(f"  Merged {_merged} LoRA layers (r={_lora_r}, alpha={_lora_alpha}, scaling={_scaling:.1f})")
                else:
                    # Pre-merged model
                    nah_model = WhisperForConditionalGeneration.from_pretrained(
                        nah_model_path, torch_dtype=torch.float16,
                    ).to("cuda")
                nah_model.eval()

                _ft_skipped_short = 0
                for idx in nah_targets:
                    r = all_results[idx]
                    seg_start, seg_end = r["start"], r["end"]
                    seg_dur = seg_end - seg_start

                    # Min duration gate: FT model halluzinates speaker names on short segments
                    if seg_dur < 0.8:
                        _ipa_tokens = len((r.get("ipa") or "").split())
                        if _ipa_tokens < 4:
                            # Very short + very few phonemes → classify as OTH, skip entirely
                            r["lang"] = "other"
                            r["backend"] = r.get("backend", "") + "+ft-skipped-too-short"
                            print(f"  [ft-skipped-too-short] {seg_start:.2f}s ({seg_dur:.2f}s, {_ipa_tokens} phones) → OTH")
                        else:
                            # Short but has IPA content → skip FT, keep stock Whisper text
                            r["backend"] = r.get("backend", "") + "+ft-skipped-too-short"
                            print(f"  [ft-skipped-too-short] {seg_start:.2f}s ({seg_dur:.2f}s) → keep stock Whisper")
                        _ft_skipped_short += 1
                        continue

                    start_sample = int(seg_start * audio_sr)
                    end_sample = int(seg_end * audio_sr)
                    seg_audio = audio_data[start_sample:end_sample]
                    if seg_audio.ndim > 1:
                        seg_audio = seg_audio.mean(axis=1)
                    seg_audio = seg_audio.astype(np.float32)

                    if len(seg_audio) < 160:  # < 10ms at 16kHz
                        continue

                    input_features = nah_processor.feature_extractor(
                        seg_audio, sampling_rate=audio_sr, return_tensors="pt",
                    ).input_features.to("cuda", dtype=torch.float16)

                    with torch.no_grad():
                        predicted_ids = nah_model.generate(
                            input_features,
                            task="transcribe",
                            language="spanish",  # proxy language used during training
                            max_new_tokens=225,
                        )
                    ft_text = nah_processor.tokenizer.batch_decode(
                        predicted_ids, skip_special_tokens=True,
                    )[0].strip()

                    if ft_text:
                        r["nah_ft_text"] = ft_text
                        nah_ft_count += 1

                del nah_model, nah_processor
                torch.cuda.empty_cache()
                print(f"  NAH finetuned: {nah_ft_count}/{len(nah_targets)} segments transcribed"
                      f"{f', {_ft_skipped_short} skipped (too short)' if _ft_skipped_short else ''}")
            else:
                print("  NAH finetuned pass: no NAH segments found, skipping")
        else:
            print(f"  NAH finetuned model not found at {nah_model_path}, skipping")

    # STEP 6.55: SPA Reclaim — demote NAH segments whose FT text is clearly Spanish.
    # The FT model blindly follows the pipeline's lang label. If the pipeline
    # misclassified SPA→NAH (tɬ-cascade, speaker-prior), the FT model will produce
    # garbled-Spanish or even clean Spanish text. Detect and reclaim.
    spa_reclaim_count = 0
    _NAH_FT_SPA_STRONG = {"tɬ", "kʷ", "ɬ"}  # IPA markers that justify NAH over SPA
    # Nahuatl morphology patterns in FT text — if present, text is real Nahuatl
    _NAH_TEXT_RX = re.compile(
        r"tl[aeiou]|tz[aeiou]|hu[aeiou]|cu[aeiou]|ua[ctnlshp]"
        r"|xic|xoc|xol|chih|chihu|chiw|moch|noch|tech|nech"
        r"|nik[aeiou]|tik[aeiou]|ntek|ntik|necin|necua"
        r"|xk[aeiou]|kw[aeiou]|hw[aeiou]|[aeiou]wk"
        r"|atl|etl|itl|otl|utl|[aeiou]tzin|coch|poch|toch|calc|calt|tepet"
        r"|teca|tlaca|xica|naca|meca|titec|piltec"
        r"|teka|tlaka|xika|naka|meka|titek|piltek"
        r"|iske|isce|isc[aeiou]|isk[aeiou]"
        r"|monec|monek|oneki|oneci"
        r"|mict[aeiou]|tequi|chiua|cipac|mani|pach[aeiou]"
        r"|tict|ticm|ticn|nicn|nicm|nict|quin[aeiou]"
        r"|nican|ican|tlaco|tlam[aeiou]|ilis|ilia|cali"
    )
    for r in all_results:
        if r.get("lang") != "nah":
            continue
        ft_text = r.get("nah_ft_text") or r.get("text") or ""
        if not ft_text:
            continue
        # Guard: if FT text has Nahuatl morphology, this is real NAH — skip
        if _NAH_TEXT_RX.search(ft_text.lower()) or fuzzy_nah_text_check(ft_text):
            continue
        # Check BOTH FT text and original Whisper text for SPA evidence.
        # The FT model garbles Spanish ("Sonde" for "Donde"), but the original
        # Whisper text often has better SPA signal. Take the higher SPA score.
        def _spa_score(text_str):
            n = _normalize_word_ascii(text_str.lower())
            words = re.findall(r"[a-z]+", n)
            if len(words) < 2:
                return 0, 0.0
            hits = sum(1 for w in words if w in SPA_COMMON)
            return hits, hits / len(words)
        _ft_hits, _ft_ratio = _spa_score(ft_text)
        _orig_text = r.get("text") or ""
        _orig_hits, _orig_ratio = _spa_score(_orig_text)
        # Use whichever text has stronger SPA evidence
        _best_hits = max(_ft_hits, _orig_hits)
        _best_ratio = max(_ft_ratio, _orig_ratio)
        # Thresholds from EQ (ft_spa_guard overrides to lower values)
        _min_hits = 1 if ft_spa_guard else eq["spa_reclaim_min_hits"]
        _min_ratio = 0.25 if ft_spa_guard else eq["spa_reclaim_min_ratio"]
        if _best_hits < _min_hits or _best_ratio < _min_ratio:
            continue
        text_conf = min(0.95, 0.55 + _best_ratio)
        # Guard: keep NAH if IPA has strong NAH-exclusive markers
        ipa_check = r.get("ipa_fused") or r.get("ipa") or ""
        has_nah_exclusive = any(m in ipa_check for m in _NAH_FT_SPA_STRONG)
        if has_nah_exclusive:
            continue
        # Reclaim to SPA
        old_backend = r.get("backend", "")
        r["lang"] = "spa"
        r["lang_conf"] = text_conf
        r["lang_conf_source"] = f"spa-reclaim:{text_conf:.2f}"
        r["backend"] = old_backend + "+spa-reclaim"
        r["nah_ft_text"] = None  # Clear FT text, use original Whisper
        spa_reclaim_count += 1
        print(f"  [spa-reclaim] NAH→SPA: '{ft_text[:50]}' (spa_conf={text_conf:.2f})")
    if spa_reclaim_count > 0:
        print(f"  SPA reclaim: {spa_reclaim_count} segments demoted NAH→SPA")

    # STEP 6.56: IPA-NAH-Override — demote NAH when IPA lacks NAH-exclusive phonemes.
    # If pipeline says NAH but Allosaurus/Wav2Vec2 IPA has no tɬ/kʷ/ɬ, and tɬ was NOT
    # acoustically detected, the NAH classification is likely a Whisper hallucination.
    _ipa_nah_override_count = 0
    if ipa_nah_override:
        _NAH_IPA_MARKERS = {"tɬ", "kʷ", "ɬ"}
        for r in all_results:
            if r.get("lang") != "nah":
                continue
            # Guard 1: tɬ acoustically detected → real NAH
            if r.get("tl_detected"):
                continue
            # Guard 2: FT text has Nahuatl morphology → real NAH
            _ft_text = r.get("nah_ft_text") or r.get("text") or ""
            if _ft_text and (_NAH_TEXT_RX.search(_ft_text.lower()) or fuzzy_nah_text_check(_ft_text)):
                continue
            # Guard 3: need IPA data to make a decision
            _ipa_check = r.get("ipa_fused") or r.get("ipa") or ""
            if not _ipa_check:
                continue
            # Core check: does IPA contain NAH-exclusive phonemes?
            if any(m in _ipa_check for m in _NAH_IPA_MARKERS):
                continue  # IPA confirms NAH
            # No NAH evidence in IPA → demote to SPA
            r["lang"] = "spa"
            r["lang_conf"] = 0.6
            r["lang_conf_source"] = "ipa-nah-override"
            r["backend"] = r.get("backend", "") + "+ipa-nah-override"
            r["nah_ft_text"] = None
            _ipa_nah_override_count += 1
            print(f"  [ipa-nah-override] NAH→SPA: '{(_ft_text or '')[:50]}' (no NAH phonemes in IPA)")
        if _ipa_nah_override_count:
            print(f"  IPA-NAH-Override: {_ipa_nah_override_count} segments demoted NAH→SPA")

    # STEP 6.57: dʒ+tɕ cross-check NAH marker
    # When w2v2 hears dʒ AND allosaurus hears tɕ for the same segment, they agree on
    # an affricate that doesn't exist in Spanish. If text lacks SPA function words → NAH.
    _dj_tc_count = 0
    if eq.get("dj_tc_marker"):
        _SPA_FW = {
            "la", "el", "de", "que", "en", "los", "las", "un", "una", "por", "con",
            "del", "al", "se", "su", "más", "mas", "como", "hay", "pero", "ya",
            "no", "es", "le", "lo", "y", "a",
        }
        for r in all_results:
            if r.get("lang") in ("nah", "may", "lat", "silence"):
                continue  # already non-SPA or irrelevant
            ipa_w2v = r.get("ipa_compare") or ""
            ipa_allo = r.get("ipa") or ""
            if "dʒ" not in ipa_w2v or "tɕ" not in ipa_allo:
                continue
            # Guard: SPA function words → likely Spanish
            _txt = (r.get("text") or "").lower()
            _words = set(w.strip("¿¡?!.,;:\"'") for w in _txt.split())
            _spa_hits = len(_words & _SPA_FW)
            if _spa_hits >= 2:
                continue
            old_lang = r.get("lang", "other")
            r["lang"] = "nah"
            r["lang_conf"] = 0.65
            r["lang_conf_source"] = "dj-tc-marker"
            r["backend"] = r.get("backend", "") + "+dj-tc"
            _dj_tc_count += 1
            print(f"  [dj-tc-marker] {old_lang.upper()}→NAH: dʒ(w2v2)+tɕ(allo) "
                  f"'{_txt[:40]}' (spa_fw={_spa_hits})")
        if _dj_tc_count:
            print(f"  dʒ+tɕ marker: {_dj_tc_count} segments promoted to NAH")

    # STEP 6.58: Prior reset — re-run speaker prior after all corrections.
    # After two-pass, FT-first, SPA reclaim, IPA-NAH-Override, dj-tc, language tags are
    # more accurate. Re-running speaker prior produces cleaner speaker profiles.
    if eq.get("prior_reset"):
        print("  [prior-reset] Re-running speaker prior with corrected language tags...")
        _reset_overrides = apply_speaker_prior(
            all_results,
            min_segments=eq["speaker_prior_min_segments"],
            min_ratio=eq["speaker_prior_min_ratio"],
            strong_mode=speaker_prior_strong,
        )
        if _reset_overrides > 0:
            print(f"  Prior reset: {_reset_overrides} additional overrides"
                  f"{' (strong mode)' if speaker_prior_strong else ''}")

    # STEP 6.6: Prosody voter — use acoustic prosody to vote NAH vs SPA
    _prosody_votes = 0
    _prosody_overrides = 0
    if eq["prosody_enabled"] and tl_sound is not None:
        print("  Prosody voter: extracting features...")
        for r in all_results:
            seg_dur = r["end"] - r["start"]
            if seg_dur < eq["prosody_min_dur"]:
                continue
            if r.get("lang") not in ("nah", "spa"):
                continue
            feats = extract_prosody_features(tl_sound, r["start"], r["end"])
            if feats is None:
                continue
            p_nah = score_prosody_nah(feats)
            r["prosody_nah"] = round(p_nah, 3)
            _prosody_votes += 1

            # Apply prosody vote: shift language if prosody strongly disagrees
            prosody_says_nah = p_nah >= eq["prosody_threshold"]
            current_lang = r.get("lang")
            weight = eq["prosody_weight"]

            if current_lang == "nah" and not prosody_says_nah:
                # Prosody says SPA, pipeline says NAH → weaken NAH confidence
                old_conf = r.get("lang_conf", 0.5)
                new_conf = old_conf * (1 - weight) + (1 - p_nah) * weight
                if new_conf < 0.4:  # Below threshold → flip to SPA
                    r["lang"] = "spa"
                    r["lang_conf"] = round(new_conf, 3)
                    r["lang_conf_source"] = f"prosody-override:{p_nah:.2f}"
                    r["backend"] = r.get("backend", "") + "+prosody-spa"
                    _prosody_overrides += 1
                else:
                    r["lang_conf"] = round(new_conf, 3)
            elif current_lang == "spa" and prosody_says_nah:
                # Prosody says NAH, pipeline says SPA → weaken SPA confidence
                old_conf = r.get("lang_conf", 0.5)
                new_conf = old_conf * (1 - weight) + p_nah * weight
                if new_conf < 0.4:  # Below threshold → flip to NAH
                    r["lang"] = "nah"
                    r["lang_conf"] = round(new_conf, 3)
                    r["lang_conf_source"] = f"prosody-override:{p_nah:.2f}"
                    r["backend"] = r.get("backend", "") + "+prosody-nah"
                    _prosody_overrides += 1
                else:
                    r["lang_conf"] = round(new_conf, 3)
        if _prosody_votes:
            print(f"  Prosody voter: {_prosody_votes} segments scored, {_prosody_overrides} overrides")

    # STEP 7: SRT with speaker labels — always show both text + IPA
    srt_lines = []
    suppressed_count = 0
    recovered_count = 0
    for i, r in enumerate(all_results, 1):
        lang_tag = {"other": "OTH", "silence": "SIL", "unknown": "UNK"}.get(r["lang"], r["lang"].upper())
        spk = r.get("speaker") or "?"
        text_out = r.get("text") or ""

        # Suppress Whisper hallucinations for unknown languages (OTH)
        # Whisper hallucinates random English/Italian/French when it doesn't understand
        if r["lang"] == "other" and text_out:
            text_out = ""  # Clear hallucinated text, keep IPA only

        # Apply suppress_hallucinations AND-gate (takes precedence over uncertain_text_policy)
        segment_suppressed = False
        segment_recovered = False
        recovery_method = ""
        if suppress_hallucinations and _should_suppress_segment(r, speaker_validated_counts, low_conf_threshold):
            text_out = ""
            segment_suppressed = True

            # Try recovery if recover_uncertain is enabled
            if recover_uncertain:
                recovered_text, method = _recover_uncertain_text(r)
                if recovered_text:
                    text_out = recovered_text
                    segment_recovered = True
                    recovery_method = method
                    recovered_count += 1
                else:
                    suppressed_count += 1
            else:
                suppressed_count += 1
        elif r.get("lang_uncertain"):
            # Legacy uncertain_text_policy for backward compatibility
            if policy == "suppress":
                text_out = ""
            elif policy == "ipa":
                text_out = f"[ipa] {r.get('ipa_fused') or r.get('ipa_compare') or r.get('ipa') or ''}".strip()

        # IPA-Whisper mismatch override: if Whisper text doesn't match IPA, use IPA transliteration
        ipa_overridden = False
        if ipa_mismatch_override and text_out and not segment_suppressed and not segment_recovered:
            ipa_str = r.get("ipa_fused") or r.get("ipa_compare") or r.get("ipa") or ""
            lang = r.get("lang", "other")
            if ipa_str and not _whisper_matches_ipa(text_out, ipa_str, lang, ipa_mismatch_threshold):
                # Whisper text doesn't match IPA — try to recover from IPA
                recovered_text, method = _recover_uncertain_text(r)
                if recovered_text:
                    text_out = recovered_text
                    ipa_overridden = True
                    recovery_method = method
                    print(f"  [ipa-override] seg={i-1} whisper='{r.get('text','')}' -> ipa='{recovered_text}' ({method})")

        # Diagnostic debug output for suppression reasoning
        seg_idx = i - 1  # enumerate starts at 1
        if suppress_hallucinations and (seg_idx % 50 == 0 or segment_suppressed):
            print(f"  [suppress] seg={seg_idx} spk={r.get('speaker','?')} "
                  f"lang={r.get('lang','?')} unc={r.get('lang_uncertain')} "
                  f"w_valid={r.get('whisper_valid')} ipa_c={r.get('ipa_conf')} "
                  f"lang_c={r.get('lang_conf')} -> {'SUPPRESSED' if segment_suppressed else 'kept'}")

        valid_mark = "" if r.get("whisper_valid") else "? " if text_out and policy == "keep" else ""
        lines = []

        # LLM fallback for indigenous languages: Whisper hallucinates garbage for NAH/MAY/OTH
        # Force LLM transcription and suppress Whisper text entirely
        segment_lang = r.get("lang", "").upper()
        force_llm = segment_lang in ("NAH", "MAY", "OTH")
        nah_ft_text = r.get("nah_ft_text")  # Finetuned Whisper text (if available)

        llm_text = None
        text_empty = not text_out.strip()
        # Skip LLM if finetuned text is available for NAH
        if (text_empty or force_llm) and not segment_suppressed and not nah_ft_text:
            ipa_for_llm = r.get("ipa_fused") or r.get("ipa") or r.get("ipa_compare") or ""
            if ipa_for_llm:
                reason = f"force={segment_lang}" if force_llm and text_out else "empty"
                print(f"  [llm] trying seg={i-1} ({reason}) whisper='{text_out[:30]}...' ipa='{ipa_for_llm[:30]}...'")
                llm_text = llm_transcribe_ipa(ipa_for_llm, lang_hint=r.get("lang", "unknown"))
                if llm_text:
                    print(f"  [llm] seg={i-1} -> '{llm_text}'")

        # Text line: finetuned Whisper > LLM > IPA for indigenous, Whisper for others
        if segment_recovered:
            lines.append(f"[REC] {text_out}")
        elif segment_suppressed:
            lines.append("[UNC]")
        elif nah_ft_text:
            # Finetuned Whisper text for NAH — trained on Puebla-Nahuatl corpus
            # Takes priority over ipa_overridden and LLM since it's a real ASR model
            lines.append(f"[FT] {nah_ft_text}")
        elif ipa_overridden:
            lines.append(f"[LLM] {text_out}")  # IPA-override also uses LLM for G2P
        elif llm_text:
            # For forced LLM (NAH/MAY/OTH): suppress Whisper hallucination entirely
            lines.append(f"[LLM] {llm_text}")
        elif force_llm and text_out:
            # LLM failed but we still don't trust Whisper for indigenous languages
            lines.append(f"[IPA] {r.get('ipa_fused') or r.get('ipa') or ''}")
        else:
            # Standard languages: show Whisper text
            lines.append(f"{valid_mark}{text_out}" if text_out else "")

        # IPA lines: always show all available backends with labels
        ipa_primary = r.get("ipa")
        ipa_compare = r.get("ipa_compare")
        ipa_fused = r.get("ipa_fused")
        ipa_omni = r.get("ipa_omni")

        # Determine labels based on primary backend
        if phoneme_backend_actual == "allosaurus":
            primary_label, compare_label = "allo", "w2v2"
        else:
            primary_label, compare_label = "w2v2", "allo"

        # Always show all backends that have data
        if ipa_primary:
            lines.append(f"♫{primary_label}: {ipa_primary}")
        if ipa_compare:
            lines.append(f"♫{compare_label}: {ipa_compare}")
        if ipa_omni:
            lines.append(f"♫omni: {ipa_omni}")
        if ipa_fused:
            lines.append(f"♫fused: {ipa_fused}")
        # Show trimming info if segment was trimmed
        trim_info = r.get("trim_info", {})
        if trim_info.get("trimmed"):
            trim_start = trim_info.get("trim_start_ms", 0)
            trim_end = trim_info.get("trim_end_ms", 0)
            lines.append(f"♫trim: {trim_start:+d}ms/{trim_end:+d}ms")
        # Always show confidence for suppressed segments, or when show_confidence is True
        if show_confidence or segment_suppressed:
            c_ipa = r.get("ipa_conf")
            c_ag = r.get("ipa_text_agree")
            c_cmp = r.get("ipa_compare_text_agree")
            c_cross = r.get("ipa_cross_agree")
            c_dec = r.get("decode_ipa_agree")
            c_lang = r.get("lang_conf")
            c_fused = r.get("ipa_fused_conf")
            conf_parts = []
            if c_ipa is not None:
                conf_parts.append(f"ipa={c_ipa:.2f}")
            if c_lang is not None:
                conf_parts.append(f"lang={c_lang:.2f}")
            if c_ag is not None:
                conf_parts.append(f"txt={c_ag:.2f}")
            if c_cmp is not None:
                conf_parts.append(f"cmp={c_cmp:.2f}")
            if c_cross is not None:
                conf_parts.append(f"x={c_cross:.2f}")
            if c_dec is not None:
                conf_parts.append(f"dec={c_dec:.2f}")
            if c_fused is not None:
                conf_parts.append(f"fused={c_fused:.2f}")
            if c_ipa is not None and c_ipa < low_conf_threshold:
                conf_parts.append("LOW")
            if r.get("lang_uncertain"):
                conf_parts.append("UNC")
            if segment_recovered and recovery_method:
                conf_parts.append(f"recovery={recovery_method}")
            if conf_parts:
                lines.append("♫conf: " + " ".join(conf_parts))
        if show_phone_conf:
            if r.get("ipa_phone_conf"):
                pc = " ".join(
                    f"{p['ph']}:{p['c']:.2f}{'*' if p['c'] < low_conf_threshold else ''}"
                    for p in r["ipa_phone_conf"]
                )
                lines.append(f"♫pc:{pc}")
            if r.get("ipa_compare_phone_conf"):
                pcc = " ".join(
                    f"{p['ph']}:{p['c']:.2f}{'*' if p['c'] < low_conf_threshold else ''}"
                    for p in r["ipa_compare_phone_conf"]
                )
                lines.append(f"♫pc_cmp:{pcc}")
        if show_nbest:
            if r.get("ipa_nbest"):
                nb = " | ".join(f"{x['ipa']} ({x['score']:.2f})" for x in r["ipa_nbest"])
                lines.append(f"♫nbest:{nb}")
            if r.get("ipa_compare_nbest"):
                nbc = " | ".join(f"{x['ipa']} ({x['score']:.2f})" for x in r["ipa_compare_nbest"])
                lines.append(f"♫nbest_cmp:{nbc}")
        content = "\n".join(lines) if lines else "[empty]"
        srt_lines.append(
            f"{i}\n{fmt_srt_time(r['start'])} --> {fmt_srt_time(r['end'])}\n"
            f"[{lang_tag}|{spk}] {content}\n"
        )
    srt_text = "\n".join(srt_lines)
    # Machine-parseable debug footer: Whisper's raw ISO-639-1 file-level language
    # Consumed by evaluate.py _extract_whisper_lang() via regex "whisper_lang[=:]"
    srt_text += f"\n# whisper_lang={file_lang}"

    from collections import Counter
    lang_counts = dict(Counter(r["lang"] for r in all_results).most_common())
    backend_counts = dict(Counter(r["backend"] for r in all_results).most_common())
    speaker_counts = dict(Counter(r.get("speaker", "?") for r in all_results).most_common())
    ejective_total = sum(r.get("ejective_count", 0) for r in all_results)
    ejective_segments = sum(1 for r in all_results if r.get("ejective_count", 0) >= ejective_min_count)

    # Compute trimming statistics
    trimmed_segments = sum(1 for r in all_results if r.get("trim_info", {}).get("trimmed"))
    trim_start_total = sum(r.get("trim_info", {}).get("trim_start_ms", 0) for r in all_results)
    trim_end_total = sum(r.get("trim_info", {}).get("trim_end_ms", 0) for r in all_results)
    trim_avg_start = trim_start_total / max(trimmed_segments, 1)
    trim_avg_end = trim_end_total / max(trimmed_segments, 1)

    elapsed = time.time() - t_start

    model_cache.commit()

    stats = {
        "duration_s": round(duration, 1),
        "processing_s": round(elapsed, 1),
        "realtime_factor": round(duration / max(elapsed, 1), 1),
        "total_segments": len(all_results),
        "suppressed_segments": suppressed_count if suppress_hallucinations else 0,
        "recovered_segments": recovered_count if recover_uncertain else 0,
        "languages": lang_counts,
        "backends": backend_counts,
        "whisper_model": whisper_model,
        "demucs": use_demucs,
        "phoneme_backend_requested": phoneme_backend_requested,
        "phoneme_backend_actual": phoneme_backend_actual,
        "phoneme_compare": phoneme_compare,
        "show_confidence": show_confidence,
        "show_phone_conf": show_phone_conf,
        "show_nbest": show_nbest,
        "nbest_keep": nbest_keep,
        "phone_vote": phone_vote,
        "suppress_hallucinations": suppress_hallucinations,
        "recover_uncertain": recover_uncertain,
        "low_conf_threshold": low_conf_threshold,
        "lang_conf_threshold": lang_conf_threshold,
        "min_turn_s": min_turn_s,
        "floating_window": floating_window,
        "floating_window_shift_s": floating_window_shift_s,
        "vad_refine": vad_refine,
        "vad_min_speech_ms": vad_min_speech_ms,
        "vad_min_silence_ms": vad_min_silence_ms,
        "vad_pad_s": vad_pad_s,
        "vad_expand_s": vad_expand_s,
        "speakers": speaker_counts,
        "gpu": "T4",
        "detect_ejectives": detect_ejectives,
        "ejective_min_count": ejective_min_count,
        "ejective_total": ejective_total,
        "ejective_segments": ejective_segments,
        "trimmed_segments": trimmed_segments,
        "trim_avg_start_ms": round(trim_avg_start, 1),
        "trim_avg_end_ms": round(trim_avg_end, 1),
        "nah_finetuned": nah_whisper_finetuned,
        "nah_ft_segments": nah_ft_count,
        "mms_langid": mms_langid,
    }
    # MMS voter summary
    if mms_langid and mms_voter is not None:
        mms_overrides = sum(
            1 for r in all_results if "mms-override" in (r.get("lang_conf_source") or "")
        )
        mms_agrees = sum(
            1 for r in all_results if "mms-agree" in (r.get("lang_conf_source") or "")
        )
        mms_disagrees = sum(
            1 for r in all_results if r.get("mms_lang") and r.get("mms_lang") != r.get("lang")
            and "mms-override" not in (r.get("lang_conf_source") or "")
        )
        stats["mms_overrides"] = mms_overrides
        stats["mms_agrees"] = mms_agrees
        stats["mms_disagrees"] = mms_disagrees
        print(f"   MMS voter: {mms_overrides} overrides, {mms_agrees} agrees, {mms_disagrees} disagrees (kept pipeline)")

    print(f"\n✅ Done in {elapsed:.0f}s, {duration/max(elapsed,1):.1f}x realtime")
    print(f"   Languages: {lang_counts}")
    print(f"   Speakers:  {speaker_counts}")
    if trimmed_segments > 0:
        print(f"   Trimming:  {trimmed_segments}/{len(all_results)} segments, avg {trim_avg_start:.0f}ms start / {trim_avg_end:.0f}ms end")
    if detect_ejectives and ejective_total > 0:
        print(f"   Ejectives: {ejective_total} total, {ejective_segments} segments with >={ejective_min_count} (Maya indicator)")
    if nah_whisper_finetuned and nah_ft_count > 0:
        print(f"   NAH FT:    {nah_ft_count} segments transcribed with finetuned Whisper")
    result = {"srt": srt_text, "stats": stats}
    if vad_debug_dump:
        # Keep dump compact and deterministic for post-run inspection.
        if len(vad_debug["turns"]) > 5000:
            vad_debug["turns"] = vad_debug["turns"][:5000]
        result["vad_refine_debug"] = vad_debug

    # Read vocals BEFORE cleanup
    if vocals_path and os.path.exists(vocals_path):
        result["vocals_bytes"] = Path(vocals_path).read_bytes()
        print(f"   Vocals track: {len(result['vocals_bytes'])/1e6:.1f} MB")

    # Cleanup temp files
    os.unlink(audio_path)
    if wav_path != audio_path:
        os.unlink(wav_path)
    if vocals_path and vocals_path != wav_path and os.path.exists(vocals_path):
        os.unlink(vocals_path)
    if primary_backend is not None:
        primary_backend.cleanup()
    if compare_backend is not None:
        compare_backend.cleanup()
    if mms_voter is not None:
        mms_voter.cleanup()
    if whisper is not None:
        del whisper
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Regression test entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_regression(
    manifest: str = "tools/regression/manifest.json",
    whisper_model: str = "small",  # Use small for speed in regression
    verbose: bool = False,
    no_report: bool = False,
):
    """Run regression test suite on Modal.

    Usage:
        modal run tenepal_modal.py::run_regression
        modal run tenepal_modal.py::run_regression --whisper-model medium
        modal run tenepal_modal.py::run_regression --verbose
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from tools.regression.runner import run_regression as _run_regression
    from tools.regression.report import format_summary, generate_report

    print(f"Running regression suite from {manifest}")
    print(f"Whisper model: {whisper_model}")

    result = _run_regression(
        manifest_path=manifest,
        modal_args={
            "whisper_model": whisper_model,
            "phoneme_compare": True,  # Always use full comparison mode
            "phone_vote": True,
            "detect_ejectives": True,
        },
        verbose=verbose,
        save_report=not no_report,
    )

    # Print formatted summary
    if result.get("total_clips", 0) > 0 and not result.get("dry_run"):
        report = generate_report(result)
        print(format_summary(report))

        # Exit with error code if any failures
        if result["failed"] > 0:
            print(f"\n{result['failed']} test(s) FAILED")
            sys.exit(1)
        else:
            print(f"\nAll {result['passed']} test(s) PASSED")
            sys.exit(0)
    else:
            print("No clips processed (dry run or empty manifest)")


@app.local_entrypoint()
def separate_clip(
    input: str = "validation_video/Hernán-1-3.vocals.wav",
    start_s: float = 1220.0,
    end_s: float = 1240.0,
    out_dir: str = "validation_video/sepformer_test",
):
    """Run SepFormer separation on a local clip via Modal GPU.

    Example:
        modal run tenepal_modal.py::separate_clip
    """
    input_path = Path(input)
    if not input_path.exists():
        print(f"❌ File not found: {input}")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"📤 Uploading {input_path.name} ({input_path.stat().st_size / 1e6:.1f} MB)")
    print(f"✂️  Clip: {start_s:.3f}s -> {end_s:.3f}s")
    result = separate_voices_sepformer.remote(
        audio_bytes=input_path.read_bytes(),
        filename=input_path.name,
        start_s=start_s,
        end_s=end_s,
    )

    stem = input_path.stem
    print(f"🎯 Sources detected: {result.get('num_sources', 0)}")
    for src in result.get("sources", []):
        spk = src.get("speaker_id", "SPEAKER_XX")
        out_file = out_path / f"{stem}_{int(start_s)}-{int(end_s)}_{spk}.wav"
        out_file.write_bytes(src["wav_bytes"])
        print(f"   - {out_file} ({src.get('duration_s', 0):.2f}s)")

    print(f"✅ Separation complete. Output dir: {out_path}")


@app.local_entrypoint()
def compare_separation(
    input: str = "validation_video/Hernán-1-3.vocals.wav",
    start_s: float = 1220.0,
    end_s: float = 1240.0,
    out_dir: str = "validation_video/separation_comparison",
):
    """Compare three voice separation approaches on a local clip.

    Options compared:
      A: SepFormer → Pyannote (separate first, diarize each)
      B: Pyannote → SepFormer (diarize first, separate overlaps)
      C: Cascaded 4-Stems (double SepFormer)

    Example:
        modal run tenepal_modal.py::compare_separation
    """
    input_path = Path(input)
    if not input_path.exists():
        print(f"❌ File not found: {input}")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"📤 Uploading {input_path.name} ({input_path.stat().st_size / 1e6:.1f} MB)")
    print(f"✂️  Clip: {start_s:.3f}s -> {end_s:.3f}s")
    print(f"🔬 Running 3-way comparison...")
    print()

    result = compare_separation_methods.remote(
        audio_bytes=input_path.read_bytes(),
        filename=input_path.name,
        start_s=start_s,
        end_s=end_s,
    )

    # Save results
    stem = input_path.stem
    clip_tag = f"{int(start_s)}-{int(end_s)}"

    for opt_key in ["A", "B", "C"]:
        opt = result["options"].get(opt_key, {})
        opt_dir = out_path / f"option_{opt_key}"
        opt_dir.mkdir(exist_ok=True)

        # Save WAVs
        speakers = opt.get("speakers", opt.get("stems", []))
        for spk in speakers:
            spk_id = spk.get("speaker_id", spk.get("source_id", spk.get("stem_id", "UNK")))
            wav_bytes = spk.get("wav_bytes")
            if wav_bytes:
                wav_file = opt_dir / f"{stem}_{clip_tag}_{spk_id}.wav"
                wav_file.write_bytes(wav_bytes)

        # Save JSON stats
        stats = {
            "name": opt.get("name", f"Option {opt_key}"),
            "quality_score": opt.get("quality_score", 0),
            "num_outputs": len(speakers),
            "f0_stats": opt.get("f0_stats", []),
        }
        if opt.get("overlaps"):
            stats["overlaps"] = opt["overlaps"]

        stats_file = opt_dir / "stats.json"
        stats_file.write_text(json.dumps(stats, indent=2, default=str))

        # Generate SRT (simplified)
        srt_lines = []
        cue_idx = 1
        for spk in speakers:
            spk_id = spk.get("speaker_id", spk.get("source_id", spk.get("stem_id", "UNK")))
            f0 = spk.get("f0", {})
            median_f0 = f0.get("median", 0)
            # Estimate language from F0 (rough heuristic)
            lang = "NAH" if median_f0 > 180 else "SPA" if median_f0 > 100 else "OTH"

            turns = spk.get("turns", [])
            for turn in turns[:10]:  # Limit turns per speaker
                start = turn.get("start", 0)
                end = turn.get("end", start + 1)
                start_tc = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                end_tc = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
                srt_lines.append(f"{cue_idx}")
                srt_lines.append(f"{start_tc} --> {end_tc}")
                srt_lines.append(f"[{lang}|{spk_id}] (F0={median_f0:.0f}Hz)")
                srt_lines.append(f"IPA: {spk.get('ipa', '')[:50]}...")
                srt_lines.append("")
                cue_idx += 1

        srt_file = opt_dir / f"{stem}_{clip_tag}.srt"
        srt_file.write_text("\n".join(srt_lines))

        print(f"Option {opt_key} ({opt.get('name', '')}): quality={stats['quality_score']:.2f}")
        print(f"   Outputs: {len(speakers)}, saved to {opt_dir}/")

    # Save summary
    summary = {
        "input": str(input_path),
        "clip": f"{start_s}-{end_s}s",
        "options": {
            k: {
                "name": v.get("name"),
                "quality_score": v.get("quality_score"),
                "num_outputs": len(v.get("speakers", v.get("stems", []))),
            }
            for k, v in result["options"].items()
        },
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"✅ Comparison complete. Results in {out_path}/")
    print(f"   summary.json - Overall comparison")
    print(f"   option_A/ - SepFormer → Pyannote")
    print(f"   option_B/ - Pyannote → SepFormer")
    print(f"   option_C/ - Cascaded 4-Stems")


@app.local_entrypoint()
def test_separation_methods(
    input: str = "validation_video/Hernán-1-3.vocals.wav",
    start_s: float = 1210.0,
    end_s: float = 1242.0,
    out_dir: str = "validation_video/separation_comparison/method_test",
):
    """Test different separation methods: SepFormer, ConvTasNet, Pitch-based.

    Example:
        modal run tenepal_modal.py::test_separation_methods --start-s 1210 --end-s 1242
    """
    input_path = Path(input)
    if not input_path.exists():
        print(f"❌ File not found: {input}")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"📤 Testing separation methods on {input_path.name}")
    print(f"✂️  Clip: {start_s:.3f}s -> {end_s:.3f}s")
    print()

    audio_bytes = input_path.read_bytes()

    methods = {
        "sepformer": separate_voices_sepformer,
        "convtasnet": separate_voices_convtasnet,
        "pitch": separate_voices_pitch,
    }

    results = {}
    for method_name, method_fn in methods.items():
        print(f"🔬 Testing {method_name}...")
        try:
            result = method_fn.remote(
                audio_bytes=audio_bytes,
                filename=input_path.name,
                start_s=start_s,
                end_s=end_s,
            )
            results[method_name] = result

            # Save outputs
            method_dir = out_path / method_name
            method_dir.mkdir(exist_ok=True)

            for i, src in enumerate(result["sources"]):
                spk_id = src.get("speaker_id", f"SRC_{i:02d}")
                wav_bytes = src.get("wav_bytes")
                if wav_bytes:
                    wav_file = method_dir / f"{spk_id}.wav"
                    wav_file.write_bytes(wav_bytes)

            # Save stats
            stats = {
                "model": result.get("model", method_name),
                "sample_rate": result.get("sample_rate"),
                "num_sources": result.get("num_sources"),
                "sources": [
                    {
                        "speaker_id": s.get("speaker_id"),
                        "duration_s": s.get("duration_s"),
                        "f0_median": s.get("f0_median"),
                        "voiced_pct": s.get("voiced_pct"),
                    }
                    for s in result["sources"]
                ],
            }
            (method_dir / "stats.json").write_text(json.dumps(stats, indent=2))

            print(f"   ✅ {method_name}: {result.get('num_sources', 0)} sources @ {result.get('sample_rate', 0)}Hz")

        except Exception as e:
            print(f"   ❌ {method_name} failed: {e}")
            results[method_name] = {"error": str(e)}

    # Summary
    print()
    print("=== SUMMARY ===")
    for method_name, result in results.items():
        if "error" in result:
            print(f"  {method_name}: FAILED - {result['error'][:100]}")
        else:
            sr = result.get("sample_rate", 0)
            n_src = result.get("num_sources", 0)
            model = result.get("model", method_name)
            print(f"  {method_name}: {n_src} sources @ {sr}Hz ({model})")

    print()
    print(f"✅ Results saved to {out_path}/")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    input: str = "input.wav",
    output: str = "",
    model: str = "medium",
    no_demucs: bool = False,
    vocals: str = "",
    phoneme_backend: str = "allosaurus",
    compare: bool = True,  # Always run all backends for comparison
    whisper_prompt: bool = False,
    whisper_prompt_extra: str = "",
    whisper_only: bool = False,
    spanish_orthography: bool = False,
    whisper_force_lang: str = "",
    show_confidence: bool = False,
    uncertain_text_policy: str = "keep",
    suppress_hallucinations: bool = False,
    low_conf_threshold: float = 0.35,
    lang_conf_threshold: float = 0.6,
    min_turn_s: float = 0.2,
    floating_window: bool = False,
    floating_window_shift_s: float = 0.08,
    show_phone_conf: bool = False,
    show_nbest: bool = False,
    nbest_keep: int = 4,
    phone_vote: bool = True,
    vad_refine: bool = False,
    vad_min_speech_ms: int = 120,
    vad_min_silence_ms: int = 80,
    vad_pad_s: float = 0.06,
    vad_expand_s: float = 0.0,
    vad_debug_dump: bool = False,
    recover_uncertain: bool = False,
    ipa_mismatch_override: bool = True,
    ipa_mismatch_threshold: float = 0.4,
    detect_ejectives: bool = True,
    ejective_maya_boost: float = 0.3,
    ejective_min_count: int = 2,
    nah_finetuned: bool = False,
    nah_checkpoint: str = "",
    mms: bool = False,
    # --- v7 accuracy fixes ---
    ft_spa_guard: bool = False,
    speaker_prior_strong: bool = False,
    ejective_strict: bool = False,
    noise_gate: bool = False,
    noise_gate_max_s: float = 0.4,
    ipa_nah_override: bool = False,
    # --- EQ system ---
    eq: str = "",  # Path to EQ JSON config file
    prosody: bool = False,  # Enable prosody voter
    prosody_weight: float = 0.5,  # Prosody voter weight (0-1)
    prosody_threshold: float = 0.65,  # P(NAH) threshold
    # --- UNK reject gate ---
    unk_gate: bool = False,  # Tag low-confidence segments as UNK
    unk_gate_threshold: float = 0.5,  # Confidence below this → UNK
    # --- Two-pass speaker prior ---
    two_pass_prior: bool = False,  # IPA-evidence re-count before speaker-prior
    # --- FT-first ---
    ft_first: bool = False,  # Run NAH FT on ALL segments before speaker-prior
):
    """Process a local audio/video file through Tenepal on Modal GPU."""
    input_path = Path(input)
    if not input_path.exists():
        print(f"❌ File not found: {input}")
        return
    if not output:
        output = str(input_path.with_suffix(".srt"))

    # Load pre-computed vocals track: explicit --vocals flag or auto-detect
    v_bytes = None
    if vocals:
        vocals_path = Path(vocals)
        if not vocals_path.exists():
            print(f"❌ Vocals file not found: {vocals}")
            return
        v_bytes = vocals_path.read_bytes()
        print(f"🎵 Vocals: {vocals_path.name} ({len(v_bytes)/1e6:.1f} MB)")
    else:
        # Auto-detect vocals from previous Demucs run
        auto_vocals = input_path.with_suffix(".vocals.wav")
        if auto_vocals.exists():
            v_bytes = auto_vocals.read_bytes()
            print(f"🎵 Vocals (auto-detected): {auto_vocals.name} ({len(v_bytes)/1e6:.1f} MB)")

    demucs_on = not no_demucs and not v_bytes
    print(f"📤 Uploading {input_path.name} ({input_path.stat().st_size / 1e6:.1f} MB)...")
    print(f"⏳ Modal timeout: {MODAL_TIMEOUT_S}s ({MODAL_TIMEOUT_S/60:.1f} min)")
    print(f"🎵 Demucs: {'OFF (vocals provided)' if v_bytes else 'ON' if demucs_on else 'OFF'}")
    print(f"🧠 Whisper prompt context: {'ON' if whisper_prompt else 'OFF'}")
    if whisper_prompt and whisper_prompt_extra.strip():
        print(f"🧠 Whisper extra prompt terms: {whisper_prompt_extra}")
    print(f"📝 Whisper-only mode: {'ON' if whisper_only else 'OFF'}")
    print(f"🔤 Spanish orthography: {'ON' if spanish_orthography else 'OFF'}")
    print(f"🎯 Confidence lines: {'ON' if show_confidence else 'OFF'} (LOW<{low_conf_threshold:.2f})")
    print(f"🧯 Uncertain text policy: {uncertain_text_policy}")
    print(f"🛡️ Suppress hallucinations: {'ON' if suppress_hallucinations else 'OFF'}")
    print(f"🔄 Recover uncertain: {'ON' if recover_uncertain else 'OFF'}")
    print(f"🔀 IPA mismatch override: {'ON' if ipa_mismatch_override else 'OFF'} (threshold={ipa_mismatch_threshold:.2f})")
    print(f"🗳️ Language confidence gate: OTH fallback if lang<{lang_conf_threshold:.2f} (IPA path)")
    print(f"🪟 Floating window decode: {'ON' if floating_window else 'OFF'} (shift={floating_window_shift_s:.2f}s)")
    print(f"🔎 Phone confidence: {'ON' if show_phone_conf else 'OFF'}")
    print(f"🧪 N-best IPA: {'ON' if show_nbest else 'OFF'} (keep={nbest_keep})")
    print(f"🗳️ Phone voting fusion: {'ON' if phone_vote else 'OFF'}")
    print(
        f"🧭 VAD refine: {'ON' if vad_refine else 'OFF'} "
        f"(min_speech={vad_min_speech_ms}ms, min_silence={vad_min_silence_ms}ms, "
        f"pad={vad_pad_s:.2f}s, expand={vad_expand_s:.2f}s)"
    )
    print(f"🧾 VAD debug dump: {'ON' if vad_debug_dump else 'OFF'}")
    print(f"⏱️ Min turn length: {min_turn_s:.2f}s")
    effective_lang = whisper_force_lang if whisper_force_lang else ("es" if spanish_orthography else "auto")
    print(f"🗣️ Forced Whisper language: {effective_lang}")
    print(
        f"🎯 Ejective detection (Maya): {'ON' if detect_ejectives else 'OFF'} "
        f"(boost={ejective_maya_boost:.2f}, min_count={ejective_min_count})"
    )
    _nah_ckpt_label = nah_checkpoint or "model (1-epoch)" if nah_finetuned else "OFF"
    print(f"🔬 NAH finetuned Whisper: {_nah_ckpt_label}")
    print(f"🌐 MMS LangID voter: {'ON' if mms else 'OFF'}")
    _v7_flags = [f for f, v in [
        ("ft-spa-guard", ft_spa_guard), ("speaker-prior-strong", speaker_prior_strong),
        ("ejective-strict", ejective_strict), ("noise-gate", noise_gate),
        ("ipa-nah-override", ipa_nah_override),
    ] if v]
    print(f"🔧 v7 accuracy fixes: {', '.join(_v7_flags) if _v7_flags else 'OFF'}")

    # Build EQ config from JSON file + CLI flags
    _eq_config = {}
    if eq:
        eq_path = Path(eq)
        if eq_path.exists():
            _eq_config = json.loads(eq_path.read_text())
            print(f"🎛️ EQ config loaded: {eq} ({len(_eq_config)} overrides)")
        else:
            print(f"⚠️ EQ file not found: {eq}, using defaults")
    # CLI prosody flags override EQ file
    if prosody:
        _eq_config["prosody_enabled"] = True
    if prosody_weight != 0.5:
        _eq_config["prosody_weight"] = prosody_weight
    if prosody_threshold != 0.65:
        _eq_config["prosody_threshold"] = prosody_threshold
    # CLI UNK gate flags override EQ file
    if unk_gate:
        _eq_config["unk_gate_enabled"] = True
    if unk_gate_threshold != 0.5:
        _eq_config["unk_gate_threshold"] = unk_gate_threshold
    if two_pass_prior:
        _eq_config["two_pass_prior"] = True
    if ft_first:
        _eq_config["ft_first"] = True
    if _eq_config:
        _eq_active = {k: v for k, v in _eq_config.items() if v != DEFAULT_EQ.get(k)}
        if _eq_active:
            print(f"🎛️ EQ overrides: {', '.join(f'{k}={v}' for k, v in _eq_active.items())}")

    audio_bytes = input_path.read_bytes()

    print(f"🚀 Processing on Modal GPU (Whisper {model})...")
    result = process_film.remote(
        audio_bytes=audio_bytes,
        filename=input_path.name,
        whisper_model=model,
        use_demucs=demucs_on,
        vocals_bytes=v_bytes,
        phoneme_backend=phoneme_backend,
        phoneme_compare=compare,
        whisper_prompt=whisper_prompt,
        whisper_prompt_extra=whisper_prompt_extra,
        whisper_only=whisper_only,
        spanish_orthography=spanish_orthography,
        whisper_force_lang=whisper_force_lang,
        show_confidence=show_confidence,
        uncertain_text_policy=uncertain_text_policy,
        suppress_hallucinations=suppress_hallucinations,
        low_conf_threshold=low_conf_threshold,
        lang_conf_threshold=lang_conf_threshold,
        min_turn_s=min_turn_s,
        floating_window=floating_window,
        floating_window_shift_s=floating_window_shift_s,
        show_phone_conf=show_phone_conf,
        show_nbest=show_nbest,
        nbest_keep=nbest_keep,
        phone_vote=phone_vote,
        vad_refine=vad_refine,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_pad_s=vad_pad_s,
        vad_expand_s=vad_expand_s,
        vad_debug_dump=vad_debug_dump,
        recover_uncertain=recover_uncertain,
        ipa_mismatch_override=ipa_mismatch_override,
        ipa_mismatch_threshold=ipa_mismatch_threshold,
        detect_ejectives=detect_ejectives,
        ejective_maya_boost=ejective_maya_boost,
        ejective_min_count=ejective_min_count,
        nah_whisper_finetuned=nah_finetuned,
        nah_checkpoint=nah_checkpoint,
        mms_langid=mms,
        ft_spa_guard=ft_spa_guard,
        speaker_prior_strong=speaker_prior_strong,
        ejective_strict=ejective_strict,
        noise_gate=noise_gate,
        noise_gate_max_s=noise_gate_max_s,
        ipa_nah_override=ipa_nah_override,
        eq_config=_eq_config if _eq_config else None,
    )

    Path(output).write_text(result["srt"], encoding="utf-8")
    if vad_debug_dump and "vad_refine_debug" in result:
        dump_path = str(Path(output).with_suffix(".vad_refine.json"))
        Path(dump_path).write_text(
            json.dumps(result["vad_refine_debug"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"🧾 VAD refine dump: {dump_path}")

    # Save vocals track if returned
    if "vocals_bytes" in result:
        vocals_out = str(input_path.with_suffix(".vocals.wav"))
        Path(vocals_out).write_bytes(result["vocals_bytes"])
        print(f"🎙️ Vocals saved: {vocals_out} ({len(result['vocals_bytes'])/1e6:.1f} MB)")

    stats = result["stats"]
    print(f"\n{'='*60}")
    print(f"✅ SRT saved: {output}")
    print(f"   Audio:      {stats['duration_s']}s ({stats['duration_s']/60:.1f} min)")
    print(f"   Processing: {stats['processing_s']}s ({stats['realtime_factor']}x realtime)")
    print(f"   Segments:   {stats['total_segments']}")
    print(f"   Languages:  {stats['languages']}")
    print(f"   Speakers:   {stats['speakers']}")
    print(f"   Backends:   {stats['backends']}")
    compare_info = " (compare mode)" if stats.get('phoneme_compare') else ""
    print(f"   Phoneme backend: requested={stats.get('phoneme_backend_requested')} actual={stats.get('phoneme_backend_actual')}{compare_info}")
    if stats.get('suppress_hallucinations'):
        print(f"   Suppressed: {stats.get('suppressed_segments', 0)} segments")
    if stats.get('recover_uncertain'):
        print(f"   Recovered:  {stats.get('recovered_segments', 0)} segments")
    if stats.get('detect_ejectives') and stats.get('ejective_total', 0) > 0:
        print(
            f"   Ejectives:  {stats.get('ejective_total', 0)} total, "
            f"{stats.get('ejective_segments', 0)} segments with >={stats.get('ejective_min_count', 1)} "
            f"(Maya indicator)"
        )


# ---------------------------------------------------------------------------
# Test Entrypoints for Voice Separation Methods
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def test_mossformer2(
    input_file: str = "validation_video/separation_comparison/e03_20m10s/original_clip.wav",
    output_dir: str = "validation_video/separation_comparison/method_test/mossformer2",
):
    """Test MossFormer2 voice separation on Modal GPU.

    Usage:
        modal run tenepal_modal.py::test_mossformer2
        modal run tenepal_modal.py::test_mossformer2 --input-file path/to/audio.wav
    """
    from pathlib import Path
    import json

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎵 Testing MossFormer2 on: {input_path}")
    print(f"📤 Uploading {input_path.stat().st_size / 1e6:.1f} MB...")

    result = separate_voices_mossformer2.remote(
        audio_bytes=input_path.read_bytes(),
        filename=input_path.name,
    )

    print(f"\n✅ Separation complete: {result['num_sources']} sources @ {result['sample_rate']}Hz")
    print(f"   Model: {result['model']}")

    # Save outputs
    for src in result["sources"]:
        out_path = out_dir / f"{src['speaker_id']}.wav"
        out_path.write_bytes(src["wav_bytes"])
        print(f"   Saved: {out_path} ({src['duration_s']:.1f}s)")

    # Save stats
    stats = {
        "model": result["model"],
        "sample_rate": result["sample_rate"],
        "num_sources": result["num_sources"],
        "sources": [
            {"speaker_id": s["speaker_id"], "duration_s": s["duration_s"]}
            for s in result["sources"]
        ],
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"   Stats: {stats_path}")


@app.local_entrypoint()
def test_3speaker(
    input_file: str = "validation_video/separation_comparison/e03_20m10s/original_clip.wav",
    output_dir: str = "validation_video/separation_comparison/method_test/sepformer_3spk",
):
    """Test 3-speaker SepFormer separation on Modal GPU.

    Usage:
        modal run tenepal_modal.py::test_3speaker
        modal run tenepal_modal.py::test_3speaker --input-file path/to/audio.wav
    """
    from pathlib import Path
    import json

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎵 Testing SepFormer-WSJ03Mix (3-speaker) on: {input_path}")
    print(f"📤 Uploading {input_path.stat().st_size / 1e6:.1f} MB...")

    result = separate_voices_3speaker.remote(
        audio_bytes=input_path.read_bytes(),
        filename=input_path.name,
    )

    print(f"\n✅ Separation complete: {result['num_sources']} sources @ {result['sample_rate']}Hz")
    print(f"   Model: {result['model']}")

    # Save outputs
    for src in result["sources"]:
        out_path = out_dir / f"{src['speaker_id']}.wav"
        out_path.write_bytes(src["wav_bytes"])
        print(f"   Saved: {out_path} ({src['duration_s']:.1f}s)")

    # Save stats
    stats = {
        "model": result["model"],
        "sample_rate": result["sample_rate"],
        "num_sources": result["num_sources"],
        "sources": [
            {"speaker_id": s["speaker_id"], "duration_s": s["duration_s"]}
            for s in result["sources"]
        ],
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"   Stats: {stats_path}")


@app.function(
    image=mossformer_image,
    gpu="T4",
    timeout=900,
    volumes={CACHE_DIR: model_cache},
)
def separate_voices_2pass(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """2-Pass MossFormer2: split into up to 4 speakers via recursive separation."""
    from clearvoice import ClearVoice
    import soundfile as sf
    from pathlib import Path
    import tempfile

    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        input_path = tmpdir_p / filename
        input_path.write_bytes(audio_bytes)

        # Pass 1: Split into 2 stems
        print("Pass 1: Splitting into 2 stems...")
        output_wav_p1 = myClearVoice(input_path=str(input_path), online_write=False)

        all_sources = []

        # Pass 2: Split each stem again
        for i, stem_wav in enumerate(output_wav_p1):
            stem_audio = stem_wav.squeeze() if hasattr(stem_wav, "squeeze") else stem_wav
            stem_path = tmpdir_p / f"stem_{i}.wav"
            sf.write(str(stem_path), stem_audio, 16000)

            print(f"Pass 2.{i}: Splitting stem_{i}...")
            output_wav_p2 = myClearVoice(input_path=str(stem_path), online_write=False)

            for j, speaker_wav in enumerate(output_wav_p2):
                speaker_audio = speaker_wav.squeeze() if hasattr(speaker_wav, "squeeze") else speaker_wav
                speaker_id = f"SPEAKER_{i*2 + j:02d}"
                out_path = tmpdir_p / f"{speaker_id}.wav"
                sf.write(str(out_path), speaker_audio, 16000)

                all_sources.append({
                    "speaker_id": speaker_id,
                    "parent_stem": f"stem_{i}",
                    "duration_s": len(speaker_audio) / 16000.0,
                    "wav_bytes": out_path.read_bytes(),
                })
                print(f"  {speaker_id}: {len(speaker_audio)/16000:.1f}s")

        return {
            "sample_rate": 16000,
            "num_sources": len(all_sources),
            "sources": all_sources,
            "model": "MossFormer2_SS_16K (2-pass)",
        }


@app.local_entrypoint()
def test_2pass(
    input_file: str = "validation_video/separation_comparison/e03_20m10s/original_clip.wav",
    output_dir: str = "validation_video/separation_comparison/method_test/mossformer2_2pass",
):
    """Test 2-pass MossFormer2 for 4-speaker separation."""
    from pathlib import Path
    import json

    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎵 Testing 2-Pass MossFormer2 on: {input_path}")
    result = separate_voices_2pass.remote(
        audio_bytes=input_path.read_bytes(),
        filename=input_path.name,
    )

    print(f"\n✅ 2-Pass complete: {result['num_sources']} speakers")
    for src in result["sources"]:
        out_path = out_dir / f"{src['speaker_id']}.wav"
        out_path.write_bytes(src["wav_bytes"])
        print(f"   {src['speaker_id']} (from {src['parent_stem']}): {src['duration_s']:.1f}s")

    # Save stats
    stats = {
        "model": result["model"],
        "sample_rate": result["sample_rate"],
        "num_sources": result["num_sources"],
        "sources": [
            {
                "speaker_id": s["speaker_id"],
                "parent_stem": s["parent_stem"],
                "duration_s": s["duration_s"],
            }
            for s in result["sources"]
        ],
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))


@app.function(
    image=pyannote_image,
    gpu="T4",
    timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_DIR: model_cache},
)
def diarize_stem(audio_bytes: bytes, filename: str) -> dict:
    """Run Pyannote diarization on a single separated stem."""
    from pyannote.audio import Pipeline
    import tempfile
    from pathlib import Path
    import torch

    os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ.get("HF_TOKEN"),
    )
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / filename
        audio_path.write_bytes(audio_bytes)

        diarization = pipeline(str(audio_path))

        segments = []
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })
            speakers.add(str(speaker))

        return {
            "filename": filename,
            "num_speakers": len(speakers),
            "num_segments": len(segments),
            "segments": segments,
        }


@app.local_entrypoint()
def test_sep_diar():
    """Test: Separation -> Diarization pipeline on existing MossFormer2 stems."""
    from pathlib import Path
    import json

    stems = [
        "validation_video/separation_comparison/method_test/mossformer2/SOURCE_00.wav",
        "validation_video/separation_comparison/method_test/mossformer2/SOURCE_01.wav",
    ]

    results = []
    for stem_path in stems:
        p = Path(stem_path)
        if not p.exists():
            raise FileNotFoundError(f"Stem not found: {p}")
        print(f"Diarizing {p.name}...")
        result = diarize_stem.remote(p.read_bytes(), p.name)
        results.append(result)
        print(f"  -> {result['num_speakers']} speakers, {result['num_segments']} segments")

    # Save results
    out_dir = Path("validation_video/separation_comparison/method_test/sep_diar")
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        out_path = out_dir / f"{Path(r['filename']).stem}_diarization.json"
        out_path.write_text(json.dumps(r, indent=2))

    total_segments = sum(r["num_segments"] for r in results)
    total_speakers = sum(r["num_speakers"] for r in results)
    print("\nSummary:")
    for r in results:
        print(f"  {r['filename']}: {r['num_speakers']} speakers, {r['num_segments']} segments")
    print(f"  Total (stems): {total_speakers} speakers across {total_segments} segments")


@app.local_entrypoint()
def test_sep_transcribe(
    input_dir: str = "validation_video/separation_comparison/method_test/mossformer2",
    output_dir: str = "validation_video/separation_comparison/method_test/sep_transcribe",
):
    """Test: full Tenepal pipeline on separated MossFormer2 stems."""
    from pathlib import Path

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = list(in_dir.glob("SOURCE_*.wav"))
    if not stems:
        raise FileNotFoundError(f"No SOURCE_*.wav files in {in_dir}")

    for stem_path in sorted(stems):
        print(f"\n{'='*60}")
        print(f"Processing {stem_path.name}...")
        print(f"{'='*60}")

        result = process_film.remote(
            audio_bytes=stem_path.read_bytes(),
            filename=stem_path.name,
            whisper_model="large-v3",
            use_demucs=False,  # Already separated
            phoneme_backend="allosaurus",
            spanish_orthography=True,
        )

        # Save SRT
        srt_path = out_dir / f"{stem_path.stem}.srt"
        srt_path.write_text(result["srt"], encoding="utf-8")
        print(f"Saved: {srt_path}")

        stats = result["stats"]
        print(f"  Segments: {stats['total_segments']}")
        print(f"  Languages: {stats['languages']}")
        print(f"  Speakers: {stats['speakers']}")


@app.function(
    gpu="T4",
    timeout=600,
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def debug_ejective_segments(
    audio_bytes: bytes,
    srt_text: str,
    segment_ids: list[int] = [5, 8, 9, 16],
) -> dict:
    """Run verbose acoustic ejective detection for selected SRT segment IDs."""
    import re
    from datetime import datetime
    import numpy as np
    import soundfile as sf
    import io

    def _parse_srt_segments(text: str) -> dict[int, tuple[float, float]]:
        segs: dict[int, tuple[float, float]] = {}
        idx = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.isdigit():
                idx = int(line)
                continue
            if idx is not None and "-->" in line:
                start_tc, end_tc = [x.strip() for x in line.split("-->")]
                start = datetime.strptime(start_tc, "%H:%M:%S,%f")
                end = datetime.strptime(end_tc, "%H:%M:%S,%f")
                start_s = (
                    start.hour * 3600
                    + start.minute * 60
                    + start.second
                    + start.microsecond / 1_000_000.0
                )
                end_s = (
                    end.hour * 3600
                    + end.minute * 60
                    + end.second
                    + end.microsecond / 1_000_000.0
                )
                segs[idx] = (start_s, end_s)
                idx = None
        return segs

    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)

    seg_map = _parse_srt_segments(srt_text)
    results: list[dict] = []
    for seg_id in segment_ids:
        if seg_id not in seg_map:
            results.append({
                "segment_id": seg_id,
                "error": "segment not found in SRT",
            })
            continue

        start_s, end_s = seg_map[seg_id]
        start_i = max(0, int(start_s * sr))
        end_i = min(len(audio), int(end_s * sr))
        chunk = audio[start_i:end_i]
        duration_ms = (len(chunk) / sr) * 1000.0

        if len(chunk) < int(sr * 0.3):
            results.append({
                "segment_id": seg_id,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_ms": round(duration_ms, 1),
                "skipped": True,
                "reason": "<300ms gate",
            })
            print(f"[ejective-debug] seg={seg_id} skipped (<300ms), dur={duration_ms:.1f}ms")
            continue

        det = detect_ejectives_for_segment(
            chunk,
            sr,
            start_time=start_s,
            verbose=True,
        )
        out = {
            "segment_id": seg_id,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration_ms": round(duration_ms, 1),
            "ejective_count": int(det.get("ejective_count", 0)),
            "heuristic_count": int(det.get("heuristic_count", 0)),
            "sklearn_count": int(det.get("sklearn_count", 0)),
            "w2v2_count": int(det.get("w2v2_count", 0)),
            "candidate_count": len(det.get("candidates", [])),
            "voting_log": det.get("voting_log", ""),
        }
        results.append(out)

        print(
            f"[ejective-debug] seg={seg_id} dur={duration_ms:.1f}ms "
            f"count={out['ejective_count']} "
            f"(H={out['heuristic_count']} S={out['sklearn_count']} W={out['w2v2_count']})"
        )
        voting_log = out["voting_log"].strip()
        if voting_log:
            for line in voting_log.splitlines():
                print(f"  {line}")

    return {
        "sample_rate": int(sr),
        "segments": results,
    }


@app.local_entrypoint()
def test_debug_ejectives(
    wav_path: str = "validation_video/separation_comparison/method_test/mossformer2/SOURCE_00.wav",
    srt_path: str = "validation_video/separation_comparison/method_test/mossformer2/SOURCE_00.srt",
    segment_ids: str = "5,8,9,16",
):
    """Run Modal ejective debug for selected segment IDs from SOURCE_00."""
    from pathlib import Path
    import json

    wav_p = Path(wav_path)
    srt_p = Path(srt_path)
    if not wav_p.exists():
        raise FileNotFoundError(f"WAV not found: {wav_p}")
    if not srt_p.exists():
        raise FileNotFoundError(f"SRT not found: {srt_p}")

    seg_ids = [int(x.strip()) for x in segment_ids.split(",") if x.strip()]
    print(f"🔎 Ejective debug on {wav_p.name}, segments={seg_ids}")

    result = debug_ejective_segments.remote(
        audio_bytes=wav_p.read_bytes(),
        srt_text=srt_p.read_text(encoding="utf-8"),
        segment_ids=seg_ids,
    )

    out_dir = Path("validation_video/separation_comparison/method_test/sep_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{wav_p.stem}_ejective_debug.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")

    for seg in result.get("segments", []):
        if seg.get("error"):
            print(f"  seg {seg['segment_id']}: ERROR {seg['error']}")
            continue
        if seg.get("skipped"):
            print(
                f"  seg {seg['segment_id']}: skipped ({seg['duration_ms']}ms, {seg['reason']})"
            )
            continue
        print(
            f"  seg {seg['segment_id']}: ejective={seg['ejective_count']} "
            f"H={seg['heuristic_count']} S={seg['sklearn_count']} W={seg['w2v2_count']} "
            f"cand={seg['candidate_count']}"
        )


# ---------------------------------------------------------------------------
# VibeVoice-ASR-7B — Microsoft's multilingual ASR with speaker diarization
# ---------------------------------------------------------------------------

@app.function(
    image=vibevoice_image,
    gpu="A10G",
    timeout=MODAL_TIMEOUT_S,
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=1)
def run_vibevoice(
    audio_bytes: bytes,
    filename: str = "input.wav",
    context_info: str = "",
    chunk_offset_s: float = 0.0,
    max_new_tokens: int = 16384,
) -> dict:
    """Run VibeVoice-ASR-7B on audio chunk, return structured transcript."""
    import torch
    import soundfile as sf
    import time

    t0 = time.time()

    tmp_wav = "/tmp/vibevoice_input.wav"
    with open(tmp_wav, "wb") as f:
        f.write(audio_bytes)

    info = sf.info(tmp_wav)
    duration_s = info.duration
    print(f"Chunk: {chunk_offset_s:.0f}s-{chunk_offset_s+duration_s:.0f}s "
          f"({duration_s:.1f}s, sr={info.samplerate})")

    # Load model
    print("Loading VibeVoice-ASR-7B...")
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    model_path = "microsoft/VibeVoice-ASR"
    cache_dir = f"{CACHE_DIR}/vibevoice"

    processor = VibeVoiceASRProcessor.from_pretrained(
        model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
        cache_dir=cache_dir,
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.eval()
    t_load = time.time() - t0
    print(f"Model loaded in {t_load:.1f}s")

    # Load audio
    audio_data, sr = sf.read(tmp_wav, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Context with hotwords
    default_context = (
        "This is a scene from a historical drama set in 16th-century Mexico. "
        "Dialogue alternates between Spanish and Nahuatl (an indigenous language). "
        "Known names: Xicotelcatl, Malinche, Cortés, Tlaxcala, Tenochtitlan, "
        "Moctezuma, Huitzilopochtli, Quetzalcoatl. "
        "Nahuatl words: tlatoani, cenca, nikan, tlakameh, amo, mochiwa."
    )
    ctx = context_info if context_info else default_context

    inputs = processor(
        audio=audio_data,
        sampling_rate=sr,
        context_info=ctx,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    print(f"Generating (max_new_tokens={max_new_tokens})...")
    t_gen = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.pad_id,
            do_sample=False,
        )
    t_gen = time.time() - t_gen
    print(f"Generation: {t_gen:.1f}s ({duration_s/t_gen:.1f}x realtime)")

    raw_text = processor.batch_decode(output_ids, skip_special_tokens=True)
    decoded = raw_text[0]
    print(f"Raw output: {len(decoded)} chars")

    # Robust JSON extraction — parse individual objects even if array is truncated
    segments = []
    json_start = decoded.find("[{")
    if json_start >= 0:
        json_str = decoded[json_start:]
        # Try full array parse first
        try:
            # Find proper end
            bracket_depth = 0
            json_end = len(json_str)
            for ci, ch in enumerate(json_str):
                if ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        json_end = ci + 1
                        break
            raw_segs = json.loads(json_str[:json_end])
            for s in raw_segs:
                segments.append(_vv_normalize_seg(s, chunk_offset_s))
        except json.JSONDecodeError:
            # Fallback: extract individual {…} objects
            import re
            for m in re.finditer(r'\{[^{}]+\}', json_str):
                try:
                    s = json.loads(m.group())
                    if "Start" in s or "Start time" in s:
                        segments.append(_vv_normalize_seg(s, chunk_offset_s))
                except json.JSONDecodeError:
                    continue

    # Filter out degenerated segments (repetition hallucination)
    clean_segments = []
    for seg in segments:
        text = seg["text"]
        # Detect repetition: if any 5+ word phrase repeats 3+ times
        words = text.split()
        if len(words) > 15:
            chunk_5 = " ".join(words[:5])
            if text.count(chunk_5) >= 3:
                print(f"  [halluc-filter] Skipping repetition at {seg['start_time']:.1f}s: {text[:60]}...")
                continue
        clean_segments.append(seg)
    segments = clean_segments

    print(f"Parsed {len(segments)} segments")

    total_time = time.time() - t0
    speakers = {}
    for seg in segments:
        spk = seg.get("speaker_id", "?")
        speakers[spk] = speakers.get(spk, 0) + 1
    print(f"✅ Chunk done in {total_time:.1f}s — {len(segments)} segs, speakers={speakers}")

    return {
        "chunk_offset_s": chunk_offset_s,
        "duration_s": duration_s,
        "processing_time_s": total_time,
        "num_segments": len(segments),
        "segments": segments,
        "raw_output": decoded,
    }


def _vv_normalize_seg(s: dict, offset: float) -> dict:
    """Normalize VibeVoice segment keys and apply chunk time offset."""
    return {
        "start_time": (s.get("Start", s.get("Start time", 0)) or 0) + offset,
        "end_time": (s.get("End", s.get("End time", 0)) or 0) + offset,
        "speaker_id": str(s.get("Speaker", s.get("Speaker ID", ""))),
        "text": s.get("Content", s.get("Text", "")),
    }


@app.local_entrypoint()
def vibevoice(
    input: str = "validation_video/Hernán-1-3.wav",
    output: str = "",
    context: str = "",
    chunk_s: int = 300,
):
    """Run VibeVoice-ASR-7B on audio in chunks. Outputs JSON + SRT."""
    from pathlib import Path
    import subprocess
    import tempfile
    import time

    input_path = Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"📤 Audio: {input_path.name} ({input_path.stat().st_size / 1e6:.1f} MB)")

    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(input_path)],
        capture_output=True, text=True
    )
    total_duration = float(probe.stdout.strip())
    num_chunks = int(total_duration // chunk_s) + (1 if total_duration % chunk_s > 0 else 0)
    print(f"⏱️ Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"🔪 Splitting into {num_chunks} chunks of {chunk_s}s each")

    # Split audio into chunks using ffmpeg
    chunks = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_chunks):
            start = i * chunk_s
            chunk_path = Path(tmpdir) / f"chunk_{i:02d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path),
                 "-ss", str(start), "-t", str(chunk_s),
                 "-ac", "1", "-ar", "16000",
                 str(chunk_path)],
                capture_output=True
            )
            if chunk_path.exists():
                chunks.append((start, chunk_path.read_bytes()))
                print(f"  chunk {i}: {start}s-{start+chunk_s}s ({len(chunks[-1][1])/1e6:.1f} MB)")

        # Run all chunks sequentially (single GPU)
        print(f"\n🚀 Running VibeVoice-ASR-7B on {len(chunks)} chunks (Modal A10G)...")
        t0 = time.time()
        all_segments = []
        all_raw = []
        all_speakers = set()

        for i, (offset, audio_bytes) in enumerate(chunks):
            print(f"\n--- Chunk {i+1}/{len(chunks)} (offset={offset}s) ---")
            result = run_vibevoice.remote(
                audio_bytes=audio_bytes,
                filename=f"{input_path.stem}_chunk{i:02d}",
                context_info=context,
                chunk_offset_s=offset,
            )
            segs = result.get("segments", [])
            all_segments.extend(segs)
            all_raw.append(result.get("raw_output", ""))
            for s in segs:
                all_speakers.add(s.get("speaker_id", ""))
            print(f"  → {len(segs)} segments, total so far: {len(all_segments)}")

    total_time = time.time() - t0

    # Sort by start time
    all_segments.sort(key=lambda s: s["start_time"])

    # Build final result
    import json
    final = {
        "filename": input_path.name,
        "duration_s": total_duration,
        "model": "VibeVoice-ASR-7B",
        "processing_time_s": total_time,
        "realtime_factor": total_duration / total_time if total_time > 0 else 0,
        "num_chunks": num_chunks,
        "chunk_s": chunk_s,
        "num_segments": len(all_segments),
        "num_speakers": len(all_speakers),
        "segments": all_segments,
    }

    # Output paths
    stem = input_path.stem
    out_base = output if output else f"eq_comparison_results/vibevoice_{stem}"
    out_json = Path(f"{out_base}.json")
    out_srt = Path(f"{out_base}.srt")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📄 JSON saved: {out_json}")

    # SRT
    srt_lines = []
    for i, seg in enumerate(all_segments, 1):
        start = seg.get("start_time", 0)
        end = seg.get("end_time", start + 1)
        speaker = seg.get("speaker_id", "?")
        text = seg.get("text", "").strip()

        def _ts(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            ms = int((s % 1) * 1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        srt_lines.append(f"{i}")
        srt_lines.append(f"{_ts(start)} --> {_ts(end)}")
        srt_lines.append(f"[Speaker {speaker}] {text}")
        srt_lines.append("")

    out_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    print(f"📝 SRT saved: {out_srt}")

    # Speaker distribution
    speakers = {}
    for seg in all_segments:
        spk = seg.get("speaker_id", "?")
        speakers[spk] = speakers.get(spk, 0) + 1

    print(f"\n{'='*60}")
    print(f"✅ VibeVoice-ASR-7B — chunked results:")
    print(f"   Duration:    {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"   Processing:  {total_time:.1f}s ({total_duration/total_time:.1f}x realtime)")
    print(f"   Chunks:      {num_chunks} × {chunk_s}s")
    print(f"   Segments:    {len(all_segments)}")
    print(f"   Speakers:    {speakers}")
    print(f"   Output:      {out_json}, {out_srt}")

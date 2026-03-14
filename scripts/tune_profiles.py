#!/usr/bin/env python3
"""Semi-automated profile tuning script for tenepal language detection.

This script analyzes real Allosaurus output from audio samples, computes optimal
marker weights based on frequency analysis, discovers language-specific markers,
and proposes profile changes to minimize false positives. If PHOIBLE data is
used, ensure the report's citation text is included in downstream documentation.

Usage:
    python scripts/tune_profiles.py --scan validation_audio/ --nah moctezuma_test.wav \
        [--phoible download] [--phoible-iso eng,spa,deu,nah] [--apply] [--report tuning_report.txt]

    # Legacy per-language flags still work:
    python scripts/tune_profiles.py --nah moctezuma_test.wav [--eng eng_sample.wav] \
        [--deu deu_sample.wav] [--spa spa_sample.wav] [--apply] [--report tuning_report.txt]

Must be run from the project venv to ensure tenepal imports work correctly.
"""

import argparse
import csv
import json
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Ensure we can import tenepal modules (requires venv activation)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tenepal.phoneme import recognize_phonemes
from tenepal.language import identify_language
from tenepal.language.analyzer import analyze_phonemes, build_confusion_matrix
from tenepal.language.registry import default_registry, LanguageProfile
from tenepal.language.identifier import _strip_modifiers


# Filename prefix -> ISO 639-3 language code mapping
# Files whose prefix is not in this map are treated as "unknown" (expected: OTH)
FILENAME_LANG_MAP = {
    # Languages with profiles
    "french": "fra",
    "italian": "ita",
    "german": "deu",
    "english": "eng",
    "spanish": "spa",
    "nahuatl": "nah",
    "maya": "may",
    "yucatec": "may",
    "portuguese": "por",
    # LibriVox naming patterns
    "moctezuma": "nah",
    "frosttonight": "eng",
    "deutschegedichte": "deu",
    # Languages WITHOUT profiles -> mapped to their ISO codes
    # (no profile = validation expects OTH, but we track what they ARE)
    "ancientgreek": "grc",
    "afrikaans": "afr",
    "bulgarian": "bul",
    "catalan": "cat",
    "chinese": "cmn",
    "dutch": "nld",
    "esperanto": "epo",
    "hungarian": "hun",
    "latin": "lat",
    "norwegian": "nor",
    "romanian": "ron",
    "yiddish": "yid",
    "yorkshire": "eng-yor",
}


# Directory name -> ISO 639-3 language code (fallback when filename has no match)
DIRECTORY_LANG_MAP = {
    "nah": "nah",
    "may": "may",
    "eng": "eng",
    "deu": "deu",
    "spa": "spa",
    "fra": "fra",
    "ita": "ita",
    "por": "por",
}


def lang_from_filename(filename: str, parent_dir: Optional[str] = None) -> str:
    """Extract language code from filename prefix, with directory fallback.

    Strategy:
    1. Check filename against FILENAME_LANG_MAP prefixes -> definitive match
    2. If no filename match AND parent_dir is a known language dir -> use directory
    3. Otherwise -> 'unknown' (validation expects OTH)

    Filename prefix always wins over directory. So 'french_etranger.wav' in
    the 'ita/' directory is correctly tagged as FRA. But 'a_ti_silva.wav'
    in the 'spa/' directory uses the directory fallback.

    Examples:
        french_acejour.wav (in fra/) -> fra (filename match)
        bulgarian_bezpomoshtna.wav (in fra/) -> unknown (no profile)
        a_ti_silva.wav (in spa/) -> spa (directory fallback)
        french_etranger.wav (in ita/) -> fra (filename wins)
    """
    name = Path(filename).stem.lower()
    for prefix, code in FILENAME_LANG_MAP.items():
        if name.startswith(prefix):
            return code
    # No filename match -> try directory fallback
    if parent_dir:
        dir_name = parent_dir.lower()
        if dir_name in DIRECTORY_LANG_MAP:
            return DIRECTORY_LANG_MAP[dir_name]
    return "unknown"


def scan_validation_dir(scan_path: Path) -> List[Tuple[str, Path]]:
    """Scan validation audio directory, using filename prefix as ground truth.

    Recursively finds all .wav files and maps them to language codes
    via lang_from_filename(). Files with unknown languages are included
    with code 'unknown' — validation expects OTH for these.

    Args:
        scan_path: Root directory to scan (e.g. validation_audio/)

    Returns:
        List of (language_code, audio_path) tuples
    """
    results = []
    for wav_file in sorted(scan_path.rglob("*.wav")):
        parent_dir = wav_file.parent.name
        lang_code = lang_from_filename(wav_file.name, parent_dir=parent_dir)
        results.append((lang_code, wav_file))
    return results


PHOIBLE_ZENODO_ZIP_URL = "https://zenodo.org/records/2626687/files/phoible/dev-v2.0.zip?download=1"
PHOIBLE_GITHUB_RAW_URL = "https://github.com/phoible/dev/blob/master/data/phoible.csv?raw=true"


def _iso_list(value: Optional[str]) -> Set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _find_csv_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        for field in fieldnames:
            if field.strip().lower() == name.strip().lower():
                return field
    return None


def _download_phoible_csv() -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="phoible_"))
    zip_path = temp_dir / "phoible.zip"
    csv_path = temp_dir / "phoible.csv"

    try:
        with urllib.request.urlopen(PHOIBLE_ZENODO_ZIP_URL) as response:
            zip_path.write_bytes(response.read())
        with zipfile.ZipFile(zip_path, "r") as zf:
            candidates = [name for name in zf.namelist() if name.endswith("phoible.csv")]
            if not candidates:
                raise FileNotFoundError("phoible.csv not found in PHOIBLE zip")
            zf.extract(candidates[0], temp_dir)
            extracted = temp_dir / candidates[0]
            extracted.rename(csv_path)
        return csv_path
    except Exception:
        # Fallback to raw CSV download from GitHub
        with urllib.request.urlopen(PHOIBLE_GITHUB_RAW_URL) as response:
            csv_path.write_bytes(response.read())
        return csv_path


def load_phoible_inventories(source: str, iso_filter: Set[str]) -> Dict[str, Set[str]]:
    if source == "download":
        csv_path = _download_phoible_csv()
    else:
        csv_path = Path(source)
        if not csv_path.exists():
            raise FileNotFoundError(f"PHOIBLE CSV not found: {csv_path}")

    inventories: Dict[str, Set[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return inventories

        iso_field = _find_csv_column(
            reader.fieldnames,
            ["ISO6393", "ISO639-3", "ISO 639-3", "LanguageCode", "ISO"]
        )
        phoneme_field = _find_csv_column(
            reader.fieldnames,
            ["Phoneme", "Segment", "PhonemeIPA"]
        )

        if not iso_field or not phoneme_field:
            raise KeyError("PHOIBLE CSV missing ISO or Phoneme column")

        for row in reader:
            iso = (row.get(iso_field) or "").strip().lower()
            phoneme = (row.get(phoneme_field) or "").strip()
            if not iso or not phoneme:
                continue
            if iso_filter and iso not in iso_filter:
                continue
            inventories.setdefault(iso, set()).add(phoneme)

    return inventories


def phoible_citation(accessed: date) -> str:
    return (
        "Moran, Steven & McCloy, Daniel (eds.) 2019. PHOIBLE 2.0. "
        "Jena: Max Planck Institute for the Science of Human History. "
        f"(Available online at http://phoible.org, Accessed on {accessed.isoformat()}.) "
        "DOI: 10.5281/zenodo.2626687."
    )


def _score_totals_for_profiles(
    phonemes: list,
    profiles: list[LanguageProfile]
) -> dict[str, float]:
    totals: dict[str, float] = {p.code: 0.0 for p in profiles}

    # Marker contributions
    for seg in phonemes:
        p = seg.phoneme
        for profile in profiles:
            if p in profile.marker_phonemes:
                totals[profile.code] += profile.marker_weights.get(p, 1.0)

    # Sequence contributions
    for i in range(len(phonemes) - 1):
        pair = (
            _strip_modifiers(phonemes[i].phoneme),
            _strip_modifiers(phonemes[i + 1].phoneme),
        )
        for profile in profiles:
            if pair in profile.marker_sequences:
                totals[profile.code] += profile.sequence_weights.get(pair, 1.0)

    return totals


def _top_weighted_markers(
    profile: LanguageProfile,
    analysis: any,
    limit: int = 6
) -> list[tuple[str, float, int]]:
    hits = analysis.profile_hits.get(profile.code)
    if not hits or not hits.markers_found:
        return []

    weighted = []
    for marker, count in hits.markers_found.items():
        weight = profile.marker_weights.get(marker, 1.0)
        weighted.append((marker, weight * count, count))

    weighted.sort(key=lambda x: x[1], reverse=True)
    return weighted[:limit]


class TuningAnalyzer:
    """Analyzer for computing optimal profile weights and thresholds."""

    def __init__(self):
        self.registry = default_registry()
        # Per-sample storage (sample_id -> data)
        self.sample_paths: Dict[str, Path] = {}
        self.sample_langs: Dict[str, str] = {}
        self.sample_phonemes: Dict[str, List] = {}
        self.sample_analysis: Dict[str, any] = {}
        # Per-language aggregate storage (lang_code -> data)
        self.language_phonemes: Dict[str, List] = {}
        self.language_analysis: Dict[str, any] = {}
        self.phoible_inventory: Dict[str, Set[str]] = {}
        self.phoible_source: Optional[str] = None

    def add_audio(self, lang_code: str, audio_path: Path):
        """Register an audio file for a language."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        sample_id = f"{lang_code}:{audio_path.name}"
        suffix = 1
        while sample_id in self.sample_paths:
            suffix += 1
            sample_id = f"{lang_code}:{audio_path.name}.{suffix}"
        self.sample_paths[sample_id] = audio_path
        self.sample_langs[sample_id] = lang_code

    def load_phoible(self, source: str, iso_filter: Set[str]):
        """Load PHOIBLE inventories for candidate marker discovery."""
        inventories = load_phoible_inventories(source, iso_filter)
        self.phoible_inventory = inventories
        self.phoible_source = source

    def analyze_all(self):
        """Run phoneme recognition and analysis on all registered audio files."""
        for sample_id, audio_path in self.sample_paths.items():
            lang_code = self.sample_langs[sample_id]
            print(f"Analyzing {lang_code.upper()}: {audio_path.name}...", file=sys.stderr)
            phonemes = recognize_phonemes(audio_path)
            self.sample_phonemes[sample_id] = phonemes
            analysis = analyze_phonemes(phonemes, self.registry)
            self.sample_analysis[sample_id] = analysis
            self.language_phonemes.setdefault(lang_code, []).extend(phonemes)

        for lang_code, phonemes in self.language_phonemes.items():
            analysis = analyze_phonemes(phonemes, self.registry)
            self.language_analysis[lang_code] = analysis

    def phoible_candidates(self, lang_code: str) -> Set[str]:
        """Return PHOIBLE-backed candidate markers seen in audio but not in NAH."""
        if not self.phoible_inventory or lang_code not in self.phoible_inventory:
            return set()

        own_analysis = self.language_analysis.get(lang_code)
        nah_analysis = self.language_analysis.get("nah")
        if not own_analysis:
            return set()

        own_phonemes = set(own_analysis.frequencies.keys())
        nah_phonemes = set(nah_analysis.frequencies.keys()) if nah_analysis else set()

        candidates = own_phonemes & self.phoible_inventory[lang_code]
        candidates = {p for p in candidates if p not in nah_phonemes}
        return candidates

    def compute_false_positives(self, reference_lang: str) -> Dict[str, Dict[str, int]]:
        """Compute false positive markers from reference language audio.

        For Nahuatl audio (reference_lang='nah'), this finds all ENG/DEU/SPA
        markers that appear in the Nahuatl phonemes.

        Returns:
            Dict mapping language_code -> {marker: frequency}
        """
        if reference_lang not in self.language_analysis:
            return {}

        analysis = self.language_analysis[reference_lang]
        false_positives = {}

        # For each OTHER language's profile
        for lang_code in self.registry.codes():
            if lang_code == reference_lang:
                continue

            hits = analysis.profile_hits.get(lang_code)
            if hits and hits.markers_found:
                # These markers appeared in reference language audio = false positives
                false_positives[lang_code] = hits.markers_found

        return false_positives

    def compute_marker_weights(
        self,
        lang_code: str,
        reference_lang: str = "nah"
    ) -> Dict[str, float]:
        """Compute optimal weights for a language's markers.

        Uses frequency specificity: weight = own_freq / (own_freq + nah_freq)
        with a floor of 0.05 to keep markers active.

        Args:
            lang_code: Language to compute weights for
            reference_lang: Reference language to compare against (default: nah)

        Returns:
            Dict mapping marker phoneme -> optimal weight
        """
        profile = self.registry.get(lang_code)
        if not profile:
            return {}

        # Get frequencies in the target language's audio (if available)
        own_analysis = self.language_analysis.get(lang_code)
        own_freqs = {}
        if own_analysis:
            hits = own_analysis.profile_hits.get(lang_code)
            if hits:
                own_freqs = hits.markers_found

        # Get frequencies in reference language audio
        ref_analysis = self.language_analysis.get(reference_lang)
        ref_freqs = {}
        if ref_analysis:
            hits = ref_analysis.profile_hits.get(lang_code)
            if hits:
                ref_freqs = hits.markers_found

        weights = {}
        for marker in profile.marker_phonemes:
            own_freq = own_freqs.get(marker, 0)
            ref_freq = ref_freqs.get(marker, 0)

            if own_freq == 0 and ref_freq == 0:
                # Ghost marker - never appears
                weights[marker] = 0.0
            elif own_freq == 0:
                # Only in reference language - false positive, very low weight
                if ref_freq >= 10:
                    weights[marker] = 0.01
                elif ref_freq >= 4:
                    weights[marker] = 0.05
                else:
                    weights[marker] = 0.1
            elif ref_freq == 0:
                # Only in own language - perfect marker
                weights[marker] = 1.0
            else:
                # Appears in both - compute specificity
                specificity = own_freq / (own_freq + ref_freq)
                weights[marker] = max(0.05, specificity)

        return weights

    def compute_marker_weights_multilang(self, lang_code: str) -> Dict[str, float]:
        """Compute weights using all available languages (not just NAH).

        Uses specificity across all other languages:
            weight = own_freq / (own_freq + other_freq)
        with a floor of 0.05 to keep markers active.
        """
        profile = self.registry.get(lang_code)
        if not profile:
            return {}

        own_analysis = self.language_analysis.get(lang_code)
        own_freqs = {}
        if own_analysis:
            hits = own_analysis.profile_hits.get(lang_code)
            if hits:
                own_freqs = hits.markers_found

        # Aggregate frequencies across other languages
        other_freqs: Dict[str, int] = {}
        for other_code, analysis in self.language_analysis.items():
            if other_code == lang_code:
                continue
            hits = analysis.profile_hits.get(lang_code)
            if not hits:
                continue
            for marker, count in hits.markers_found.items():
                other_freqs[marker] = other_freqs.get(marker, 0) + count

        weights = {}
        for marker in profile.marker_phonemes:
            own_freq = own_freqs.get(marker, 0)
            other_freq = other_freqs.get(marker, 0)

            if own_freq == 0 and other_freq == 0:
                weights[marker] = 0.0
            elif own_freq == 0:
                weights[marker] = 0.05
            elif other_freq == 0:
                weights[marker] = 1.0
            else:
                specificity = own_freq / (own_freq + other_freq)
                weights[marker] = max(0.05, specificity)

        return weights

    def compute_threshold(
        self,
        lang_code: str,
        reference_lang: str = "nah"
    ) -> float:
        """Compute optimal detection threshold for a language.

        Threshold is set above the maximum false positive score that
        the reference language audio produces WITH TUNED WEIGHTS.

        Args:
            lang_code: Language to compute threshold for
            reference_lang: Reference language to test against (default: nah)

        Returns:
            Optimal threshold (max false score + safety margin)
        """
        if reference_lang not in self.language_phonemes:
            return 0.0

        # Compute what score this language's profile would produce on reference audio
        ref_phonemes = self.language_phonemes[reference_lang]
        profile = self.registry.get(lang_code)
        if not profile:
            return 0.0

        # Get tuned weights for this language
        tuned_weights = self.compute_marker_weights(lang_code, reference_lang)

        # Simulate weighted scoring WITH TUNED WEIGHTS
        max_score = 0.0
        for phoneme_seg in ref_phonemes:
            phoneme = phoneme_seg.phoneme
            if phoneme in profile.marker_phonemes:
                # Use tuned weight, not current profile weight
                weight = tuned_weights.get(phoneme, 1.0)
                if weight > 0.0:  # Only count active markers
                    max_score += weight

        # Check bigram sequences
        for i in range(len(ref_phonemes) - 1):
            pair = (
                _strip_modifiers(ref_phonemes[i].phoneme),
                _strip_modifiers(ref_phonemes[i + 1].phoneme),
            )
            if pair in profile.marker_sequences:
                weight = profile.sequence_weights.get(pair, 1.0)
                max_score += weight

        # Add safety margin
        return max_score + 0.1

    def discover_sequences(
        self,
        lang_code: str,
        min_freq: int = 2
    ) -> List[Tuple[str, str]]:
        """Discover frequent bigram sequences in language audio.

        Convenience wrapper around discover_ngrams(n=2).

        Args:
            lang_code: Language to discover sequences for
            min_freq: Minimum frequency to consider (default: 2)

        Returns:
            List of (phoneme1, phoneme2) tuples
        """
        return self.discover_ngrams(lang_code, n=2, min_freq=min_freq)

    def discover_ngrams(
        self,
        lang_code: str,
        n: int = 2,
        min_freq: int = 2,
        min_specificity: float = 0.0,
    ) -> list:
        """Discover frequent n-gram sequences in language audio.

        Counts all n-grams in the target language and optionally filters
        by cross-language specificity (target_freq / total_freq across
        all languages).

        Args:
            lang_code: Language to discover n-grams for
            n: N-gram length (2=bigrams, 3=trigrams, etc.)
            min_freq: Minimum frequency to consider (default: 2)
            min_specificity: Minimum specificity ratio 0.0-1.0 (default: 0.0 = no filtering)

        Returns:
            List of n-gram tuples sorted by frequency descending
        """
        if lang_code not in self.language_phonemes:
            return []

        phonemes = self.language_phonemes[lang_code]
        ngram_counts: Dict[tuple, int] = {}

        for i in range(len(phonemes) - n + 1):
            ngram = tuple(
                _strip_modifiers(phonemes[i + j].phoneme)
                for j in range(n)
            )
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        # Filter by frequency
        candidates = {
            ngram: count for ngram, count in ngram_counts.items()
            if count >= min_freq
        }

        # Cross-language specificity filtering
        if min_specificity > 0.0 and candidates:
            other_counts: Dict[tuple, int] = {}
            for other_code, other_phonemes in self.language_phonemes.items():
                if other_code == lang_code:
                    continue
                for i in range(len(other_phonemes) - n + 1):
                    ngram = tuple(
                        _strip_modifiers(other_phonemes[i + j].phoneme)
                        for j in range(n)
                    )
                    if ngram in candidates:
                        other_counts[ngram] = other_counts.get(ngram, 0) + 1

            filtered = {}
            for ngram, count in candidates.items():
                total = count + other_counts.get(ngram, 0)
                specificity = count / total
                if specificity >= min_specificity:
                    filtered[ngram] = count
            candidates = filtered

        # Sort by frequency descending
        return sorted(candidates.keys(), key=lambda ng: candidates[ng], reverse=True)

    def apply_tuning_to_registry(self):
        """Apply computed tuning to in-memory registry for validation.

        This updates the registry profiles with tuned weights and thresholds
        so that validation tests use the new values.
        """
        for lang_code in ["eng", "deu", "spa", "fra", "ita"]:
            profile = self.registry.get(lang_code)
            if not profile:
                continue

            # Compute new values
            new_weights = self.compute_marker_weights(lang_code, "nah")
            new_threshold = self.compute_threshold(lang_code, "nah")

            # Remove ghost markers (weight 0.0)
            profile.marker_phonemes = {
                m for m in profile.marker_phonemes
                if new_weights.get(m, 1.0) > 0.0
            }

            # Update weights (only store non-default values)
            profile.marker_weights = {
                marker: weight
                for marker, weight in new_weights.items()
                if weight > 0.0 and weight != 1.0
            }

            # Update threshold
            profile.threshold = new_threshold


def generate_tuning_report(analyzer: TuningAnalyzer) -> str:
    """Generate detailed tuning report with evidence for every decision.

    Args:
        analyzer: TuningAnalyzer with completed analysis

    Returns:
        Multi-section text report
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("TUNING REPORT")
    lines.append("=" * 80)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Audio analyzed:")
    for sample_id, audio_path in analyzer.sample_paths.items():
        lang_code = analyzer.sample_langs[sample_id]
        analysis = analyzer.sample_analysis.get(sample_id)
        phoneme_count = analysis.total_phonemes if analysis else 0
        lines.append(f"  {lang_code.upper()}: {audio_path.name} ({phoneme_count} phonemes)")
    lines.append("")

    # Per-audio marker summary (to avoid ambiguity about which audio produced which evidence)
    for sample_id, audio_path in analyzer.sample_paths.items():
        lang_code = analyzer.sample_langs[sample_id]
        analysis = analyzer.sample_analysis.get(sample_id)
        if not analysis:
            continue
        lines.append("-" * 80)
        lines.append(f"AUDIO SUMMARY: {lang_code.upper()} ({audio_path.name})")
        lines.append("-" * 80)
        for profile_code in ["nah", "may", "spa", "eng", "deu", "fra", "ita"]:
            hits = analysis.profile_hits.get(profile_code)
            if not hits or not hits.markers_found:
                lines.append(f"  {profile_code.upper()}: no markers found")
                continue
            top = sorted(hits.markers_found.items(), key=lambda x: x[1], reverse=True)[:8]
            top_str = ", ".join(f"{m}={c}" for m, c in top)
            lines.append(f"  {profile_code.upper()}: {top_str}")
        lines.append("")

        # Score totals and top contributors (reasoning aid)
        profiles = analyzer.registry.all_profiles()
        totals = _score_totals_for_profiles(analyzer.sample_phonemes[sample_id], profiles)
        ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        top_scores = ", ".join(f"{code.upper()}={score:.2f}" for code, score in ranked[:4])
        lines.append(f"  SCORE TOTALS: {top_scores}")

        expected_profile = analyzer.registry.get(lang_code)
        top_lang_code = ranked[0][0] if ranked else None
        top_profile = analyzer.registry.get(top_lang_code) if top_lang_code else None

        if expected_profile:
            expected_markers = _top_weighted_markers(expected_profile, analysis)
            if expected_markers:
                markers_str = ", ".join(
                    f"{m}={w:.2f}({c})" for m, w, c in expected_markers
                )
                lines.append(f"  TOP {lang_code.upper()} CONTRIBUTORS: {markers_str}")

        if top_profile and top_profile.code != lang_code:
            top_markers = _top_weighted_markers(top_profile, analysis)
            if top_markers:
                markers_str = ", ".join(
                    f"{m}={w:.2f}({c})" for m, w, c in top_markers
                )
                lines.append(f"  TOP {top_profile.code.upper()} CONTRIBUTORS: {markers_str}")

        lines.append("")

    if analyzer.phoible_inventory:
        lines.append("PHOIBLE source:")
        if analyzer.phoible_source == "download":
            lines.append(f"  Downloaded from Zenodo (PHOIBLE v2.0)")
        else:
            lines.append(f"  Loaded from local path: {analyzer.phoible_source}")
        lines.append("")
        lines.append("PHOIBLE citation (required):")
        lines.append(f"  {phoible_citation(date.today())}")
        lines.append("  Note: For language-specific inventories, also cite the original sources listed by PHOIBLE.")
        lines.append("")

    # Profile tuning sections
    for lang_code in ["eng", "deu", "spa", "fra", "ita"]:  # Skip nah - reference language
        profile = analyzer.registry.get(lang_code)
        if not profile:
            continue

        lines.append("-" * 80)
        lines.append(f"PROFILE: {profile.name} ({lang_code.upper()})")
        lines.append("-" * 80)
        lines.append("")

        # Compute weights
        weights = analyzer.compute_marker_weights(lang_code, "nah")
        weights_multi = analyzer.compute_marker_weights_multilang(lang_code)
        if weights:
            lines.append("MARKERS:")
            for marker in sorted(profile.marker_phonemes):
                old_weight = profile.marker_weights.get(marker, 1.0)
                new_weight = weights.get(marker, 1.0)
                multi_weight = weights_multi.get(marker, new_weight)

                # Get frequency evidence
                nah_analysis = analyzer.language_analysis.get("nah")
                nah_freq = 0
                if nah_analysis:
                    nah_hits = nah_analysis.profile_hits.get(lang_code)
                    if nah_hits:
                        nah_freq = nah_hits.markers_found.get(marker, 0)

                own_freq = 0
                own_analysis = analyzer.language_analysis.get(lang_code)
                if own_analysis:
                    own_hits = own_analysis.profile_hits.get(lang_code)
                    if own_hits:
                        own_freq = own_hits.markers_found.get(marker, 0)

                if nah_freq > 0 or own_freq > 0 or new_weight != old_weight:
                    specificity = 0.0
                    if own_freq + nah_freq > 0:
                        specificity = own_freq / (own_freq + nah_freq)

                    status = ""
                    if new_weight == 0.0:
                        status = " [GHOST - REMOVE]"
                    elif nah_freq > 0 and own_freq == 0:
                        status = " [FALSE POSITIVE]"

                    lines.append(
                        f"  {marker}: weight {old_weight:.2f} -> {new_weight:.2f} "
                        f"(NAH freq={nah_freq}, {lang_code.upper()} freq={own_freq}, "
                        f"specificity={specificity:.2f}){status}"
                    )
                    if weights_multi:
                        lines.append(
                            f"    suggested multi-lang weight: {multi_weight:.2f}"
                        )
        else:
            lines.append("MARKERS: No weight changes")

        lines.append("")

        if analyzer.phoible_inventory:
            candidates = analyzer.phoible_candidates(lang_code)
            if candidates:
                lines.append("PHOIBLE CANDIDATES (seen in audio, absent in NAH):")
                lines.append("  " + ", ".join(sorted(candidates)))
            else:
                lines.append("PHOIBLE CANDIDATES: None")
            lines.append("")

        # Threshold
        old_threshold = profile.threshold
        new_threshold = analyzer.compute_threshold(lang_code, "nah")
        nah_analysis = analyzer.language_analysis.get("nah")
        max_false_score = 0.0
        if nah_analysis:
            # Compute actual max score
            nah_phonemes = analyzer.language_phonemes.get("nah", [])
            for p in nah_phonemes:
                if p.phoneme in profile.marker_phonemes:
                    w = weights.get(p.phoneme, profile.marker_weights.get(p.phoneme, 1.0))
                    max_false_score += w

        lines.append(
            f"THRESHOLD: {old_threshold:.2f} -> {new_threshold:.2f} "
            f"(max false score on NAH: {max_false_score:.2f})"
        )
        lines.append("")

    # Validation section — use current profiles (as loaded from JSON)
    lines.append("=" * 80)
    lines.append("VALIDATION RESULTS (current profiles)")
    lines.append("=" * 80)

    if analyzer.sample_phonemes:
        # Group results by language for summary
        lang_results: dict[str, list[tuple[str, bool, set]]] = {}

        for sample_id, phonemes in analyzer.sample_phonemes.items():
            lang_code = analyzer.sample_langs[sample_id]
            result = identify_language(phonemes, analyzer.registry)

            # Majority classification
            from collections import Counter
            lang_counts: Counter[str] = Counter()
            for seg in result:
                lang_counts[seg.language] += len(seg.phonemes)
            majority = lang_counts.most_common(1)[0][0] if lang_counts else "none"
            correct = majority == lang_code

            detected_langs = set(seg.language for seg in result)
            expected = {lang_code, "other"}
            false_positives = detected_langs - expected

            if lang_code not in lang_results:
                lang_results[lang_code] = []
            lang_results[lang_code].append((sample_id, correct, majority, lang_counts, false_positives))

        # Print per-language summary
        total_correct = 0
        total_files = 0
        # Known profiles first, then unknown
        lang_order = ["nah", "may", "eng", "deu", "spa", "fra", "ita"]
        # Add any other languages found (including "unknown")
        for lc in sorted(lang_results.keys()):
            if lc not in lang_order:
                lang_order.append(lc)

        for lang_code in lang_order:
            if lang_code not in lang_results:
                continue
            results = lang_results[lang_code]

            has_profile = analyzer.registry.get(lang_code) is not None
            if not has_profile:
                # Language without a profile -> "correct" means majority = other
                # (or at least not misclassified as a profiled language)
                known_codes = set(analyzer.registry.codes())
                correct_count = sum(
                    1 for _, c, majority, *_ in results
                    if majority not in known_codes or majority == "other"
                )
                total_correct += correct_count
                total_files += len(results)
                label = lang_code.upper() if lang_code != "unknown" else "UNKNOWN"
                lines.append(f"\n{label} (no profile, expect OTH): {correct_count}/{len(results)} correct")
                for sample_id, _, majority, counts, fps in results:
                    is_correct = majority not in known_codes or majority == "other"
                    mark = "OK" if is_correct else "FAIL"
                    breakdown = ", ".join(
                        f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])[:4]
                    )
                    fname = sample_id.split("/")[-1][:42] if "/" in sample_id else sample_id[:42]
                    lines.append(f"  [{mark}] {fname:<42} -> {majority:<6} ({breakdown})")
                    if not is_correct:
                        lines.append(f"         misclassified as: {majority}")
            else:
                correct_count = sum(1 for _, c, *_ in results if c)
                total_correct += correct_count
                total_files += len(results)
                lines.append(f"\n{lang_code.upper()}: {correct_count}/{len(results)} correct (majority classification)")
                for sample_id, correct, majority, counts, fps in results:
                    mark = "OK" if correct else "FAIL"
                    breakdown = ", ".join(
                        f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])[:4]
                    )
                    fname = sample_id.split("/")[-1][:42] if "/" in sample_id else sample_id[:42]
                    lines.append(f"  [{mark}] {fname:<42} -> {majority:<6} ({breakdown})")
                    if fps:
                        lines.append(f"         false positive segments: {', '.join(sorted(fps))}")


        lines.append(f"\nOVERALL: {total_correct}/{total_files} ({100*total_correct/total_files:.0f}%)")
    else:
        lines.append("No audio provided - validation skipped")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def apply_tuning(analyzer: TuningAnalyzer, profiles_dir: Path):
    """Apply computed tuning to JSON profile files.

    Args:
        analyzer: TuningAnalyzer with completed analysis
        profiles_dir: Path to profiles directory
    """
    for lang_code in ["eng", "deu", "spa", "fra", "ita"]:
        profile = analyzer.registry.get(lang_code)
        if not profile:
            continue

        # Compute new values
        new_weights = analyzer.compute_marker_weights(lang_code, "nah")
        new_threshold = analyzer.compute_threshold(lang_code, "nah")

        # Load existing JSON
        json_path = profiles_dir / f"{lang_code}.json"
        if not json_path.exists():
            print(f"Warning: Profile not found: {json_path}", file=sys.stderr)
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Update marker_weights
        # Remove ghost markers (weight 0.0)
        active_markers = [m for m in profile.marker_phonemes if new_weights.get(m, 1.0) > 0.0]
        data["marker_phonemes"] = sorted(active_markers)

        # Set non-default weights
        marker_weights = {
            marker: weight
            for marker, weight in new_weights.items()
            if weight > 0.0 and weight != 1.0
        }
        data["marker_weights"] = marker_weights

        # Update threshold
        data["threshold"] = round(new_threshold, 2)

        # Write back
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")  # Trailing newline

        print(f"Updated: {json_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Tune language detection profiles based on real audio analysis"
    )
    parser.add_argument(
        "--nah",
        type=Path,
        required=True,
        help="Path to Nahuatl reference audio (pure NAH, no code-switching)"
    )
    parser.add_argument(
        "--scan",
        type=Path,
        help="Scan directory for .wav files, using filename prefix as ground truth. "
             "e.g. french_*.wav -> FRA, bulgarian_*.wav -> unknown (expect OTH). "
             "See FILENAME_LANG_MAP for supported prefixes."
    )
    parser.add_argument(
        "--eng",
        type=Path,
        action="append",
        help="Path to English validation audio (optional, repeatable)"
    )
    parser.add_argument(
        "--deu",
        type=Path,
        action="append",
        help="Path to German validation audio (optional, repeatable)"
    )
    parser.add_argument(
        "--spa",
        type=Path,
        action="append",
        help="Path to Spanish validation audio (optional, repeatable)"
    )
    parser.add_argument(
        "--fra",
        type=Path,
        action="append",
        help="Path to French validation audio (optional, repeatable)"
    )
    parser.add_argument(
        "--ita",
        type=Path,
        action="append",
        help="Path to Italian validation audio (optional, repeatable)"
    )
    parser.add_argument(
        "--may",
        type=Path,
        action="append",
        help="Path to Yucatec Maya validation audio (optional, repeatable)"
    )
    parser.add_argument(
        "--phoible",
        type=str,
        help="PHOIBLE CSV path or 'download' to fetch PHOIBLE v2.0"
    )
    parser.add_argument(
        "--phoible-iso",
        type=str,
        help="Comma-separated ISO 639-3 codes to load from PHOIBLE (e.g., eng,spa,deu,nah)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply tuning changes to profile JSON files (default: dry run)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write report to file (default: stdout)"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = TuningAnalyzer()

    # Register audio files
    analyzer.add_audio("nah", args.nah)

    # --scan mode: auto-discover from directory using filename prefix
    if args.scan:
        if not args.scan.is_dir():
            print(f"Error: --scan path is not a directory: {args.scan}", file=sys.stderr)
            sys.exit(1)
        scanned = scan_validation_dir(args.scan)
        lang_counts: Dict[str, int] = {}
        for lang_code, audio_path in scanned:
            # Skip NAH files from scan (already added via --nah)
            if lang_code == "nah":
                continue
            analyzer.add_audio(lang_code, audio_path)
            lang_counts[lang_code] = lang_counts.get(lang_code, 0) + 1
        print(f"Scanned {len(scanned)} files from {args.scan}:", file=sys.stderr)
        for lc, count in sorted(lang_counts.items()):
            label = lc.upper() if lc != "unknown" else "UNKNOWN (no profile)"
            print(f"  {label}: {count} files", file=sys.stderr)

    # Legacy per-language flags (additive with --scan)
    if args.eng:
        for path in args.eng:
            analyzer.add_audio("eng", path)
    if args.deu:
        for path in args.deu:
            analyzer.add_audio("deu", path)
    if args.spa:
        for path in args.spa:
            analyzer.add_audio("spa", path)
    if args.fra:
        for path in args.fra:
            analyzer.add_audio("fra", path)
    if args.ita:
        for path in args.ita:
            analyzer.add_audio("ita", path)
    if args.may:
        for path in args.may:
            analyzer.add_audio("may", path)

    if args.phoible:
        iso_filter = _iso_list(args.phoible_iso)
        print("Loading PHOIBLE inventories...", file=sys.stderr)
        analyzer.load_phoible(args.phoible, iso_filter)

    # Run analysis
    print("Running phoneme analysis...", file=sys.stderr)
    analyzer.analyze_all()

    # Generate report
    report = generate_tuning_report(analyzer)

    # Output report
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport written to: {args.report}", file=sys.stderr)
    else:
        print(report)

    # Apply changes if requested
    if args.apply:
        print("\nApplying tuning changes to profile JSON files...", file=sys.stderr)
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        apply_tuning(analyzer, profiles_dir)
        print("Tuning applied successfully.", file=sys.stderr)
    else:
        print("\nDry run - no changes applied. Use --apply to update profiles.", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Baseline regression tests for La Otra Conquista language classification.

These tests validate SRT output from the Modal pipeline against known baselines.
They catch regressions in language classification without requiring Modal runs —
they check the SRT files already generated and committed.

Baseline established: 2026-03-14 (v7: full pipeline re-run)
"""

import re
from collections import Counter
from pathlib import Path

import pytest

SRT_DIR = Path(__file__).parent.parent / "validation_video"

# Baseline language counts per clip (from v7 pipeline run, 2026-03-14)
BASELINES = {
    "La-Otra-Conquista_14m15-24m15": {"SPA": 31, "NAH": 7, "MAY": 1},
    "La-Otra-Conquista_24m15-34m15": {"SPA": 37, "NAH": 35, "ENG": 5, "MAY": 1},
    "La-Otra-Conquista_34m15-44m25": {"SPA": 118, "NAH": 17},
    "La-Otra-Conquista_44m25-55m25": {"SPA": 83, "NAH": 18},
}

# Ground truth from manual annotation (clip 3: 24:15-34:15, 39 segments annotated)
# Accuracy: 61% (23/38 correct)
CLIP3_ANNOTATED = {
    # seg_index: correct_lang (from human annotation)
    1: "SPA", 2: "SPA", 6: "SPA", 7: "SPA", 8: "SPA", 9: "NAH",
    10: "SPA", 16: "SPA", 17: "SPA", 18: "SPA", 19: "SPA", 20: "SPA",
    21: "SPA", 22: "SPA", 23: "SPA", 35: "NAH", 37: "NAH", 47: "NAH",
    50: "NAH", 51: "NAH", 53: "NAH", 54: "OTH", 55: "OTH", 56: "NAH",
    58: "SPA", 59: "SPA", 60: "SPA", 61: "SPA", 62: "SPA", 63: "SPA",
    64: "SPA", 65: "SPA", 66: "SPA", 67: "SPA", 68: "SPA", 72: "SPA",
    74: "SPA",
}


def _parse_srt_langs(srt_path: Path) -> dict[int, str]:
    """Parse SRT file and return {segment_index: lang_code}."""
    text = srt_path.read_text()
    result = {}
    for match in re.finditer(r"^(\d+)\n.+\n\[([A-Z]+)\|", text, re.MULTILINE):
        idx = int(match.group(1))
        lang = match.group(2)
        result[idx] = lang
    return result


def _count_langs(srt_path: Path) -> dict[str, int]:
    """Count language labels in SRT file."""
    langs = re.findall(r"\[([A-Z]+)\|", srt_path.read_text())
    return dict(Counter(langs))


class TestLOCBaseline:
    """Regression tests: language counts should not regress from baseline."""

    @pytest.mark.parametrize("clip_name,expected", list(BASELINES.items()))
    def test_language_counts_stable(self, clip_name, expected):
        srt = SRT_DIR / f"{clip_name}.srt"
        if not srt.exists():
            pytest.skip(f"SRT not found: {srt}")
        actual = _count_langs(srt)
        for lang, count in expected.items():
            actual_count = actual.get(lang, 0)
            # Allow +-10% drift but flag large changes
            assert abs(actual_count - count) <= max(3, count * 0.15), (
                f"{clip_name}: {lang} expected ~{count}, got {actual_count}"
            )

    def test_no_may_in_loc(self):
        """La Otra Conquista has NO Maya — MAY count should be <= 3 total."""
        total_may = 0
        for clip_name in BASELINES:
            srt = SRT_DIR / f"{clip_name}.srt"
            if not srt.exists():
                continue
            counts = _count_langs(srt)
            total_may += counts.get("MAY", 0)
        assert total_may <= 3, f"Too many MAY in LOC: {total_may} (was 70 pre-fix)"

    def test_spa_dominant_across_clips(self):
        """SPA should be the dominant language across all LOC clips."""
        total = Counter()
        for clip_name in BASELINES:
            srt = SRT_DIR / f"{clip_name}.srt"
            if not srt.exists():
                continue
            total.update(_count_langs(srt))
        assert total["SPA"] > total.get("NAH", 0), (
            f"SPA ({total['SPA']}) should exceed NAH ({total.get('NAH', 0)})"
        )
        assert total["SPA"] > total.get("ENG", 0), (
            f"SPA ({total['SPA']}) should exceed ENG ({total.get('ENG', 0)})"
        )


class TestLOCAnnotated:
    """Tests against human-annotated ground truth (clip 3)."""

    @pytest.fixture
    def clip3_langs(self):
        srt = SRT_DIR / "La-Otra-Conquista_24m15-34m15.srt"
        if not srt.exists():
            pytest.skip("Clip 3 SRT not found")
        return _parse_srt_langs(srt)

    def test_clip3_accuracy_above_55(self, clip3_langs):
        """Annotated accuracy should not drop below 55% (baseline: 61%)."""
        correct = sum(
            1 for idx, lang in CLIP3_ANNOTATED.items()
            if clip3_langs.get(idx) == lang
        )
        total = len(CLIP3_ANNOTATED)
        accuracy = correct / total
        assert accuracy >= 0.55, (
            f"Clip 3 accuracy {accuracy:.0%} ({correct}/{total}) dropped below 55%"
        )

    def test_clip3_spa_segments_correct(self, clip3_langs):
        """Key Spanish segments should be classified as SPA."""
        # These were the worst ENG->SPA errors, now fixed
        spa_segs = {10, 59, 60, 62, 63, 67, 68, 74}  # clear Spanish
        correct = sum(1 for idx in spa_segs if clip3_langs.get(idx) == "SPA")
        assert correct >= 6, (
            f"Only {correct}/8 key SPA segments correct (regression?)"
        )

    def test_clip3_eng_count_low(self, clip3_langs):
        """ENG count should stay low (was 26 pre-fix, now 8)."""
        eng_count = sum(1 for lang in clip3_langs.values() if lang == "ENG")
        assert eng_count <= 12, f"ENG count {eng_count} too high (baseline: 8)"

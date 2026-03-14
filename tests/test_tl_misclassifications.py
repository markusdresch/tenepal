import json
from pathlib import Path


FIXTURE_PATH = Path("tests/regression/fixtures/tl_misclassified_11.json")


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_tl_misclassification_fixture_shape():
    fixture = _load_fixture()
    segments = fixture.get("segments", [])

    assert fixture.get("version") == "1.0"
    assert len(segments) == 11, "Expected 11 known SPA->NAH misclassification segments"


def test_all_segments_have_tl_in_ipa_output():
    fixture = _load_fixture()
    segments = fixture["segments"]

    for seg in segments:
        fused = str(seg.get("fused", ""))
        assert seg.get("ipa_contains_tl") is True, f"{seg['id']} missing tl_in_ipa marker"
        assert ("tɬ" in fused) or ("t l" in fused), f"{seg['id']} fused IPA does not show tɬ/t l"


def test_known_bug_documentation_spa_tagged_but_should_be_nah():
    """Regression marker for current bug: these segments are tagged SPA but should be NAH."""
    fixture = _load_fixture()
    segments = fixture["segments"]

    for seg in segments:
        assert seg["expected_lang"] == "NAH"
        assert seg["current_lang"] == "SPA"

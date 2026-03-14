"""Tests for language formatter output and labels."""

from tenepal.language.formatter import _language_label, format_language_segments
from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme import PhonemeSegment


def _seg(language: str, start: float = 0.0) -> LanguageSegment:
    return LanguageSegment(
        language=language,
        phonemes=[PhonemeSegment("a", start, 0.1)],
        start_time=start,
        end_time=start + 0.1,
    )


def test_language_label_includes_may():
    """Maya language code maps to MAY label."""
    assert _language_label("may") == "MAY"


def test_formatter_summary_includes_maya():
    """Summary line includes Maya count instead of dropping to Other."""
    output = format_language_segments([
        _seg("may", 0.0),
        _seg("nah", 0.2),
        _seg("other", 0.4),
    ], use_color=False)

    assert "[MAY]" in output
    assert "Maya: 1" in output
    assert "Nahuatl: 1" in output
    assert "Other: 1" in output

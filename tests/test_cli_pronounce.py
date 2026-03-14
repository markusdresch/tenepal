"""Tests for --pronounce flag and rendering integration."""

from types import SimpleNamespace

import pytest

from tenepal.cli import build_parser, _print_live_segment
from tenepal.language.formatter import format_language_segments
from tenepal.subtitle.srt import format_srt
from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme.backend import PhonemeSegment


class TestPronounceCLI:
    def test_build_parser_pronounce_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--pronounce", "de", "test.wav"])
        assert args.pronounce == "de"
        args = parser.parse_args(["--pronounce", "es", "test.wav"])
        assert args.pronounce == "es"
        args = parser.parse_args(["--pronounce", "en", "test.wav"])
        assert args.pronounce == "en"

    def test_build_parser_pronounce_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["test.wav"])
        assert args.pronounce is None

    def test_build_parser_pronounce_invalid_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--pronounce", "xx", "test.wav"])

    def test_print_live_segment_with_pronounce(self, capsys, monkeypatch):
        segment = LanguageSegment(
            "nah",
            [PhonemeSegment("ʃ", 0.0, 0.1), PhonemeSegment("a", 0.1, 0.1)],
            0.0,
            0.2,
        )
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        _print_live_segment(segment, pronounce="de")
        captured = capsys.readouterr()
        assert "sch" in captured.out
        assert "ʃ" not in captured.out

    def test_print_live_segment_without_pronounce(self, capsys, monkeypatch):
        segment = LanguageSegment(
            "nah",
            [PhonemeSegment("ʃ", 0.0, 0.1), PhonemeSegment("a", 0.1, 0.1)],
            0.0,
            0.2,
        )
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        _print_live_segment(segment)
        captured = capsys.readouterr()
        assert "ʃ" in captured.out


class TestPronounceIntegration:
    def test_format_language_segments_with_pronounce(self):
        segment = LanguageSegment(
            "nah",
            [PhonemeSegment("ʃ", 0.0, 0.1), PhonemeSegment("a", 0.1, 0.1)],
            0.0,
            0.2,
        )
        output = format_language_segments([segment], use_color=False, pronounce="de")
        assert "sch" in output
        assert "ʃ" not in output

    def test_format_language_segments_without_pronounce(self):
        segment = LanguageSegment(
            "nah",
            [PhonemeSegment("ʃ", 0.0, 0.1), PhonemeSegment("a", 0.1, 0.1)],
            0.0,
            0.2,
        )
        output = format_language_segments([segment], use_color=False)
        assert "ʃ" in output

    def test_format_srt_with_pronounce(self):
        segment = LanguageSegment(
            "nah",
            [PhonemeSegment("ʃ", 0.0, 0.1), PhonemeSegment("a", 0.1, 0.1)],
            0.0,
            0.2,
        )
        output = format_srt([segment], pronounce="es")
        assert "sh" in output
        assert "ʃ" not in output

    def test_format_srt_without_pronounce(self):
        segment = LanguageSegment(
            "nah",
            [PhonemeSegment("ʃ", 0.0, 0.1), PhonemeSegment("a", 0.1, 0.1)],
            0.0,
            0.2,
        )
        output = format_srt([segment])
        assert "ʃ" in output

    def test_phoneme_data_unchanged_after_render(self):
        segment = LanguageSegment(
            "nah",
            [PhonemeSegment("ʃ", 0.0, 0.1), PhonemeSegment("a", 0.1, 0.1)],
            0.0,
            0.2,
        )
        _ = format_language_segments([segment], use_color=False, pronounce="de")
        _ = format_srt([segment], pronounce="en")
        assert segment.phonemes[0].phoneme == "ʃ"
        assert segment.phonemes[1].phoneme == "a"

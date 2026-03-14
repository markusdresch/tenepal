"""Tests for SRT subtitle generation."""

from pathlib import Path

import pytest

from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme import PhonemeSegment
from tenepal.subtitle.srt import format_srt, format_timestamp, merge_consecutive_segments, write_srt


class TestFormatTimestamp:
    """Test SRT timestamp formatting."""

    def test_zero_seconds(self):
        """Zero seconds should format as 00:00:00,000."""
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_one_minute_five_seconds(self):
        """65.123 seconds should format as 00:01:05,123."""
        assert format_timestamp(65.123) == "00:01:05,123"

    def test_one_hour_one_minute_one_second(self):
        """3661.5 seconds should format as 01:01:01,500."""
        assert format_timestamp(3661.5) == "01:01:01,500"

    def test_milliseconds_rounding(self):
        """Milliseconds should be rounded correctly."""
        assert format_timestamp(1.2345) == "00:00:01,234"
        assert format_timestamp(1.2356) == "00:00:01,236"

    def test_hours_minutes_seconds(self):
        """Test various combinations of hours, minutes, seconds."""
        assert format_timestamp(3600.0) == "01:00:00,000"  # Exactly 1 hour
        assert format_timestamp(60.0) == "00:01:00,000"  # Exactly 1 minute
        assert format_timestamp(0.999) == "00:00:00,999"  # Just under 1 second


class TestMergeConsecutiveSegments:
    """Test consecutive segment merging logic."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert merge_consecutive_segments([]) == []

    def test_single_segment(self):
        """Single segment should remain unchanged."""
        seg = LanguageSegment(
            language="nah",
            phonemes=[
                PhonemeSegment("t", 0.0, 0.1),
                PhonemeSegment("ɬ", 0.1, 0.1),
            ],
            start_time=0.0,
            end_time=0.2
        )
        result = merge_consecutive_segments([seg])
        assert len(result) == 1
        assert result[0].language == "nah"
        assert len(result[0].phonemes) == 2

    def test_merge_two_consecutive_same_language(self):
        """Two consecutive segments with same language should merge."""
        seg1 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("t", 0.0, 0.1)],
            start_time=0.0,
            end_time=0.1
        )
        seg2 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("ɬ", 0.1, 0.1)],
            start_time=0.1,
            end_time=0.2
        )
        result = merge_consecutive_segments([seg1, seg2])
        assert len(result) == 1
        assert result[0].language == "nah"
        assert len(result[0].phonemes) == 2
        assert result[0].start_time == 0.0
        assert result[0].end_time == 0.2

    def test_no_merge_different_languages(self):
        """Segments with different languages should not merge."""
        seg1 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("t", 0.0, 0.1)],
            start_time=0.0,
            end_time=0.1
        )
        seg2 = LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment("e", 0.1, 0.1)],
            start_time=0.1,
            end_time=0.2
        )
        result = merge_consecutive_segments([seg1, seg2])
        assert len(result) == 2
        assert result[0].language == "nah"
        assert result[1].language == "spa"

    def test_merge_multiple_consecutive(self):
        """Multiple consecutive same-language segments should merge into one."""
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1),
            LanguageSegment("nah", [PhonemeSegment("ɬ", 0.1, 0.1)], 0.1, 0.2),
            LanguageSegment("nah", [PhonemeSegment("a", 0.2, 0.1)], 0.2, 0.3),
        ]
        result = merge_consecutive_segments(segments)
        assert len(result) == 1
        assert result[0].language == "nah"
        assert len(result[0].phonemes) == 3
        assert result[0].start_time == 0.0
        assert result[0].end_time == 0.3

    def test_mixed_languages_partial_merge(self):
        """Pattern [nah, nah, spa] should become [nah_merged, spa]."""
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1),
            LanguageSegment("nah", [PhonemeSegment("ɬ", 0.1, 0.1)], 0.1, 0.2),
            LanguageSegment("spa", [PhonemeSegment("e", 0.2, 0.1)], 0.2, 0.3),
        ]
        result = merge_consecutive_segments(segments)
        assert len(result) == 2
        assert result[0].language == "nah"
        assert len(result[0].phonemes) == 2
        assert result[1].language == "spa"
        assert len(result[1].phonemes) == 1

    def test_no_merge_alternating_languages(self):
        """Pattern [nah, spa, nah] should stay as three segments."""
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1),
            LanguageSegment("spa", [PhonemeSegment("e", 0.1, 0.1)], 0.1, 0.2),
            LanguageSegment("nah", [PhonemeSegment("a", 0.2, 0.1)], 0.2, 0.3),
        ]
        result = merge_consecutive_segments(segments)
        assert len(result) == 3
        assert result[0].language == "nah"
        assert result[1].language == "spa"
        assert result[2].language == "nah"


class TestFormatSrt:
    """Test full SRT output generation."""

    def test_empty_list(self):
        """Empty segment list should return empty string."""
        assert format_srt([]) == ""

    def test_single_segment(self):
        """Single segment should produce valid SRT with one cue."""
        seg = LanguageSegment(
            language="nah",
            phonemes=[
                PhonemeSegment("t", 0.0, 0.1),
                PhonemeSegment("ɬ", 0.1, 0.1),
                PhonemeSegment("a", 0.2, 0.1),
                PhonemeSegment("k", 0.3, 0.1),
            ],
            start_time=0.0,
            end_time=0.4
        )
        result = format_srt([seg])
        expected = "1\n00:00:00,000 --> 00:00:00,400\n[NAH] t ɬ a k\n\n"
        assert result == expected

    def test_multiple_segments_different_languages(self):
        """Multiple segments with different languages should produce separate cues."""
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1),
            LanguageSegment("spa", [PhonemeSegment("e", 0.1, 0.1)], 0.1, 0.2),
        ]
        result = format_srt(segments)
        lines = result.split("\n")

        # Should have: cue1 num, cue1 time, cue1 text, blank, cue2 num, cue2 time, cue2 text, blank, final blank
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:00,100" in lines[1]
        assert "[NAH] t" in lines[2]
        assert lines[3] == ""
        assert lines[4] == "2"
        assert "00:00:00,100 --> 00:00:00,200" in lines[5]
        assert "[SPA] e" in lines[6]

    def test_consecutive_same_language_segments_merge(self):
        """Consecutive segments with same language should merge before formatting."""
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1),
            LanguageSegment("nah", [PhonemeSegment("ɬ", 0.1, 0.1)], 0.1, 0.2),
            LanguageSegment("spa", [PhonemeSegment("e", 0.2, 0.1)], 0.2, 0.3),
        ]
        result = format_srt(segments)
        lines = result.split("\n")

        # Should produce 2 cues (nah merged, spa)
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:00,200" in lines[1]
        assert "[NAH] t ɬ" in lines[2]
        assert lines[4] == "2"
        assert "[SPA] e" in lines[6]

    def test_language_labels(self):
        """Test language label generation for different language codes."""
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1),
            LanguageSegment("may", [PhonemeSegment("kʼ", 0.05, 0.1)], 0.05, 0.15),
            LanguageSegment("spa", [PhonemeSegment("e", 0.1, 0.1)], 0.1, 0.2),
            LanguageSegment("other", [PhonemeSegment("ə", 0.2, 0.1)], 0.2, 0.3),
        ]
        result = format_srt(segments)

        assert "[NAH]" in result
        assert "[MAY]" in result
        assert "[SPA]" in result
        assert "[OTH]" in result

    def test_phoneme_spacing(self):
        """Phonemes should be space-separated within cue text."""
        seg = LanguageSegment(
            language="nah",
            phonemes=[
                PhonemeSegment("t", 0.0, 0.1),
                PhonemeSegment("ɬ", 0.1, 0.1),
                PhonemeSegment("a", 0.2, 0.1),
            ],
            start_time=0.0,
            end_time=0.3
        )
        result = format_srt([seg])

        # Should have space-separated phonemes
        assert "[NAH] t ɬ a" in result

    def test_cue_numbering(self):
        """Cues should be numbered sequentially starting at 1."""
        # Use alternating languages to prevent merging
        languages = ["nah", "spa", "nah", "spa", "nah"]
        segments = [
            LanguageSegment(languages[i], [PhonemeSegment("t", i * 0.1, 0.1)], i * 0.1, (i + 1) * 0.1)
            for i in range(5)
        ]
        result = format_srt(segments)
        lines = result.split("\n")

        # Extract cue numbers (first line of each cue)
        cue_numbers = [line for line in lines if line.strip().isdigit()]
        assert cue_numbers == ["1", "2", "3", "4", "5"]

    def test_blank_line_separation(self):
        """Cues should be separated by blank lines."""
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1),
            LanguageSegment("spa", [PhonemeSegment("e", 0.1, 0.1)], 0.1, 0.2),
        ]
        result = format_srt(segments)

        # Should end with double newline after each cue
        # Structure: "num\ntime\ntext\n\n"
        assert "\n\n" in result


class TestWriteSrt:
    """Test SRT file writing functionality."""

    def test_file_created_at_path(self, tmp_path):
        """File should be created at the specified path."""
        output_path = tmp_path / "output.srt"
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1)
        ]

        result = write_srt(segments, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_file_content_matches_format_srt(self, tmp_path):
        """Written file content should match format_srt output."""
        output_path = tmp_path / "output.srt"
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1),
            LanguageSegment("spa", [PhonemeSegment("e", 0.1, 0.1)], 0.1, 0.2),
        ]

        write_srt(segments, output_path)

        # Read written content
        written_content = output_path.read_text(encoding="utf-8")
        expected_content = format_srt(segments)

        assert written_content == expected_content

    def test_utf8_encoding_with_ipa_characters(self, tmp_path):
        """File should support IPA characters via UTF-8 encoding."""
        output_path = tmp_path / "ipa_test.srt"

        # Use IPA characters: tɬ (voiceless lateral affricate), kʷ (labialized velar), ʃ (voiceless postalveolar fricative)
        segments = [
            LanguageSegment(
                "nah",
                [
                    PhonemeSegment("t", 0.0, 0.1),
                    PhonemeSegment("ɬ", 0.1, 0.1),
                    PhonemeSegment("kʷ", 0.2, 0.1),
                    PhonemeSegment("ʃ", 0.3, 0.1),
                ],
                0.0,
                0.4
            )
        ]

        write_srt(segments, output_path)

        # Read back with explicit UTF-8 encoding
        content = output_path.read_text(encoding="utf-8")

        # Verify IPA characters preserved
        assert "ɬ" in content
        assert "kʷ" in content
        assert "ʃ" in content
        assert "[NAH] t ɬ kʷ ʃ" in content

    def test_parent_directory_creation(self, tmp_path):
        """Parent directories should be created if they don't exist."""
        output_path = tmp_path / "nested" / "dir" / "output.srt"
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1)
        ]

        # Parent directories don't exist yet
        assert not output_path.parent.exists()

        write_srt(segments, output_path)

        # Parent directories should now exist
        assert output_path.parent.exists()
        assert output_path.exists()

    def test_empty_segments_list(self, tmp_path):
        """Empty segments should produce an empty file (valid edge case)."""
        output_path = tmp_path / "empty.srt"

        write_srt([], output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert content == ""

    def test_return_value_is_path_object(self, tmp_path):
        """Return value should be a Path object."""
        output_path = tmp_path / "output.srt"
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1)
        ]

        result = write_srt(segments, output_path)

        assert isinstance(result, Path)
        assert result == output_path

    def test_accepts_string_path(self, tmp_path):
        """Function should accept string path (not just Path object)."""
        output_path = str(tmp_path / "output.srt")
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1)
        ]

        result = write_srt(segments, output_path)

        assert isinstance(result, Path)
        assert result.exists()


class TestSrtFormatValidation:
    """Test SRT format compliance and player compatibility."""

    def test_srt_spec_compliance(self):
        """Generated SRT should comply with SubRip format specification."""
        # Multi-segment input with all three language types
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1), PhonemeSegment("ɬ", 0.1, 0.1)], 0.0, 0.2),
            LanguageSegment("spa", [PhonemeSegment("e", 0.2, 0.1), PhonemeSegment("s", 0.3, 0.1)], 0.2, 0.4),
            LanguageSegment("other", [PhonemeSegment("ə", 0.4, 0.1)], 0.4, 0.5),
        ]

        result = format_srt(segments)
        lines = result.split("\n")

        # Verify cue numbers start at 1
        assert lines[0] == "1"
        assert lines[4] == "2"
        assert lines[8] == "3"

        # Verify timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm
        import re
        timestamp_pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        assert re.match(timestamp_pattern, lines[1])
        assert re.match(timestamp_pattern, lines[5])
        assert re.match(timestamp_pattern, lines[9])

        # Verify blank line separators between cues
        assert lines[3] == ""
        assert lines[7] == ""
        assert lines[11] == ""

        # Verify all three tag types present
        assert "[NAH]" in result
        assert "[SPA]" in result
        assert "[OTH]" in result


class TestSrtWithSpeakerTags:
    """Test SRT output with speaker diarization tags."""

    def test_segment_with_speaker_shows_combined_tag(self):
        """Speaker + language should produce [Speaker A | NAH] format."""
        seg = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("t", 0.0, 0.1)],
            start_time=0.0,
            end_time=0.1,
            speaker="Speaker A"
        )
        result = format_srt([seg])
        assert "[Speaker A | NAH]" in result

    def test_segment_without_speaker_shows_language_only(self):
        """No speaker should produce [NAH] format (backward-compatible)."""
        seg = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("t", 0.0, 0.1)],
            start_time=0.0,
            end_time=0.1
        )
        result = format_srt([seg])
        assert "[NAH]" in result
        assert "Speaker" not in result

    def test_merge_same_speaker_and_language(self):
        """Consecutive segments with same speaker AND language should merge."""
        seg1 = LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1, speaker="Speaker A")
        seg2 = LanguageSegment("nah", [PhonemeSegment("\u026c", 0.1, 0.1)], 0.1, 0.2, speaker="Speaker A")
        result = merge_consecutive_segments([seg1, seg2])
        assert len(result) == 1
        assert result[0].speaker == "Speaker A"

    def test_no_merge_different_speaker_same_language(self):
        """Same language but different speaker should NOT merge."""
        seg1 = LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1, speaker="Speaker A")
        seg2 = LanguageSegment("nah", [PhonemeSegment("\u026c", 0.1, 0.1)], 0.1, 0.2, speaker="Speaker B")
        result = merge_consecutive_segments([seg1, seg2])
        assert len(result) == 2

    def test_multiple_speakers_srt_output(self):
        """Multiple speakers should produce separate cues with speaker tags."""
        segments = [
            LanguageSegment("nah", [PhonemeSegment("t", 0.0, 0.1)], 0.0, 0.1, speaker="Speaker A"),
            LanguageSegment("spa", [PhonemeSegment("e", 0.1, 0.1)], 0.1, 0.2, speaker="Speaker B"),
        ]
        result = format_srt(segments)
        assert "[Speaker A | NAH]" in result
        assert "[Speaker B | SPA]" in result


class TestSrtWithTranscription:
    """Test SRT output with Whisper transcription text."""

    def test_srt_with_transcription_text(self):
        """Segment with transcription attribute should output text instead of phonemes."""
        seg = LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment("o", 0.0, 0.1), PhonemeSegment("l", 0.1, 0.1), PhonemeSegment("a", 0.2, 0.1)],
            start_time=0.0,
            end_time=0.3
        )
        # Attach transcription text (done by TranscriptionRouter in pipeline)
        seg.transcription = "Hola mundo"
        seg.transcription_backend = "whisper"

        result = format_srt([seg])

        # Should contain the text, not the phonemes
        assert "Hola mundo" in result
        assert "[SPA] Hola mundo" in result
        # Should NOT contain IPA phonemes
        assert "o l a" not in result

    def test_srt_without_transcription_falls_back(self):
        """Segment without transcription should output IPA phonemes (backward compat)."""
        seg = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("t", 0.0, 0.1), PhonemeSegment("ɬ", 0.1, 0.1)],
            start_time=0.0,
            end_time=0.2
        )
        # No transcription attribute

        result = format_srt([seg])

        # Should contain phonemes
        assert "t ɬ" in result
        assert "[NAH]" in result

    def test_srt_mixed_transcription_and_phonemes(self):
        """Mix of segments with and without transcription should output correctly."""
        seg1 = LanguageSegment(
            language="spa",
            phonemes=[PhonemeSegment("o", 0.0, 0.1)],
            start_time=0.0,
            end_time=0.1
        )
        seg1.transcription = "Hola"

        seg2 = LanguageSegment(
            language="nah",
            phonemes=[PhonemeSegment("t", 0.1, 0.1), PhonemeSegment("ɬ", 0.2, 0.1)],
            start_time=0.1,
            end_time=0.3
        )
        # No transcription for seg2

        result = format_srt([seg1, seg2])

        # First segment should have text
        assert "Hola" in result
        # Second segment should have phonemes
        assert "t ɬ" in result

    def test_srt_transcription_with_speaker(self):
        """Segment with transcription AND speaker label should format correctly."""
        seg = LanguageSegment(
            language="eng",
            phonemes=[PhonemeSegment("h", 0.0, 0.1)],
            start_time=0.0,
            end_time=0.1,
            speaker="Speaker A"
        )
        seg.transcription = "Hello world"

        result = format_srt([seg])

        # Should show speaker, language, and text
        assert "[Speaker A | ENG] Hello world" in result
        # Should NOT show phonemes
        assert " h " not in result

"""Tests for Whisper-only orthography mode and text-marker language guessing."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from tenepal.phoneme.whisper_backend import WhisperAutoSegment


def test_transliterate_to_spanish_orthography_basic():
    from tenepal.pipeline import transliterate_to_spanish_orthography

    # k->c/qu, w->hu, ts->tz, sh->x
    out = transliterate_to_spanish_orthography("kuali wika tsitsin shali")
    assert "cuali" in out
    assert "huica" in out
    assert "tzitzin" in out
    assert "xali" in out


def test_guess_language_from_text_markers_nahuatl():
    from tenepal.pipeline import _guess_language_from_text_markers

    mock_lat = MagicMock()
    mock_lat.check_text.return_value = (False, 0)

    lang, conf = _guess_language_from_text_markers(
        "in tlatoani de tenochtitlan",
        whisper_lang="es",
        latin_lexicon=mock_lat,
    )
    assert lang == "nah"
    assert conf > 0.0


def test_process_whisper_text_only_assigns_transcription_and_language():
    from tenepal.pipeline import process_whisper_text_only

    segs = [
        WhisperAutoSegment(
            text="in tlatoani wika",
            start=0.0,
            end=1.0,
            language="es",
            avg_log_prob=-0.2,
        ),
        WhisperAutoSegment(
            text="donde esta",
            start=1.0,
            end=2.0,
            language="es",
            avg_log_prob=-0.2,
        ),
    ]

    mock_whisper = MagicMock()
    mock_whisper.transcribe_auto.return_value = segs

    mock_lat = MagicMock()
    mock_lat.check_text.return_value = (False, 0)

    with patch("tenepal.phoneme.whisper_backend.WhisperBackend", return_value=mock_whisper), \
         patch("tenepal.language.latin_lexicon.LatinLexicon", return_value=mock_lat):
        results = process_whisper_text_only(
            Path("/fake/audio.wav"),
            whisper_model="base",
            enable_diarization=False,
            spanish_orthography=True,
        )

    assert len(results) == 2
    assert getattr(results[0], "transcription_backend", None) == "whisper-only"
    assert "huica" in results[0].transcription
    assert results[0].language == "nah"
    assert results[1].language == "spa"

"""Diarize-first pipeline: audio -> speakers -> phonemes -> language ID.

Also provides whisper-first pipeline for film processing where Whisper
handles known languages and Allosaurus handles the rest.
"""

import logging
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

from tenepal.audio import load_audio, preprocess_audio, save_wav
from tenepal.phoneme import recognize_phonemes
from tenepal.language import identify_language, smooth_by_speaker
from tenepal.language.identifier import LanguageSegment
from tenepal.phoneme.backend import PhonemeSegment
from tenepal.speaker import diarize, slice_audio_by_speaker
from tenepal.validation.confidence_tiers import split_by_confidence_tier
from tenepal.language.speaker_profile import build_speaker_profiles, apply_speaker_inheritance

logger = logging.getLogger(__name__)


_SPA_STOPWORDS = {
    "de", "la", "el", "los", "las", "y", "que", "en", "un", "una", "por", "para",
    "con", "se", "es", "del", "al", "como", "más", "mas", "su", "sus", "ya",
    "pero", "si", "sí", "no", "le", "lo", "a", "o", "u", "yo", "tu", "tú", "mi",
    "me", "te", "nos", "vos", "fue", "era", "son", "ser", "ha", "han", "donde",
    "dónde", "esta", "está", "qué", "que",
}

_NAH_MARKERS = {
    "tl", "tz", "hu", "cuauh", "quauh", "teotl", "tlatoani", "tenochtitlan",
    "mexica", "motecuhzoma", "xicotencatl", "tzintli", "itz", "xoch", "coatl",
}

_MAYA_MARKERS = {
    "k'", "ts'", "ch'", "x", "halach", "ajaw", "kukul", "k'uk", "yuum", "uay",
    "kaax", "balam", "chan", "k'in", "kab",
}


def _normalize_ascii(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped


def _transliterate_word_to_spanish_orthography(word: str) -> str:
    """Apply conservative Spanish-orthography mapping to a token."""
    original = word
    w = word

    # Normalize common digraphs seen in modernized Nahuatl/Maya transcriptions.
    w = re.sub(r"sh", "x", w)
    w = re.sub(r"ts", "tz", w)
    w = re.sub(r"kw", "cu", w)
    w = re.sub(r"w", "hu", w)

    # k -> c/qu mapping by following vowel.
    w = re.sub(r"k(?=[eiéí])", "qu", w)
    w = re.sub(r"k", "c", w)

    # Keep punctuation shape from original token boundaries.
    return w if w else original


def transliterate_to_spanish_orthography(text: str) -> str:
    """Normalize transcription into Spanish-style orthography.

    This is a deterministic transliteration pass aimed at harmonizing
    classical-source spellings. It is intentionally conservative and does
    not attempt linguistic reconstruction.
    """
    if not text:
        return text

    # Preserve punctuation between words by replacing only word bodies.
    def repl(match):
        token = match.group(0)
        return _transliterate_word_to_spanish_orthography(token)

    transliterated = re.sub(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+", repl, text)
    transliterated = re.sub(r"\s+", " ", transliterated).strip()
    return transliterated


def _guess_language_from_text_markers(text: str, whisper_lang: str, latin_lexicon) -> tuple[str, float]:
    """Guess language code from text markers in Latin/Spanish orthography."""
    if not text:
        return ("other", 0.0)

    raw = text
    norm = _normalize_ascii(text)
    words = re.findall(r"[a-z']+", norm)

    if not words:
        return ("other", 0.0)

    # Latin liturgical check first (high precision when it hits).
    is_latin, latin_count = latin_lexicon.check_text(raw)
    if is_latin:
        return ("lat", 0.95)

    # Spanish stopword density is a strong SPA signal.
    spa_hits = sum(1 for w in words if w in _SPA_STOPWORDS)
    spa_ratio = spa_hits / max(len(words), 1)

    # Nahuatl/Maya orthographic marker hits.
    nah_hits = 0
    may_hits = 0
    for w in words:
        if any(m in w for m in _NAH_MARKERS):
            nah_hits += 1
        if any(m in w for m in _MAYA_MARKERS):
            may_hits += 1

    # If Whisper says Spanish and text looks Spanish, keep SPA.
    if whisper_lang == "es" and spa_ratio >= 0.25 and spa_hits >= 2:
        return ("spa", min(0.95, 0.55 + spa_ratio))

    # Distinctive marker-based routing for Nahuatl/Maya.
    if may_hits >= 2 and may_hits > nah_hits:
        return ("may", min(0.9, 0.45 + 0.15 * may_hits))
    if nah_hits >= 2:
        return ("nah", min(0.9, 0.45 + 0.12 * nah_hits))

    # Fallback to Whisper language when marker evidence is weak.
    if whisper_lang == "es":
        return ("spa", 0.55 if spa_ratio >= 0.1 else 0.45)
    if whisper_lang in {"en", "de", "fr", "it"}:
        return ("other", 0.4)
    return ("other", 0.3)


def process_whisper_text_only(
    audio_path: Path,
    whisper_model: str = "medium",
    output_path: Path | None = None,
    enable_diarization: bool = True,
    spanish_orthography: bool = False,
) -> list[LanguageSegment]:
    """Whisper-only segment transcription with text-marker language tagging.

    No Allosaurus/Omnilingual pass is executed in this mode.
    """
    from tenepal.phoneme.whisper_backend import WhisperBackend
    from tenepal.language.latin_lexicon import LatinLexicon

    audio_path = Path(audio_path)
    whisper = WhisperBackend(model_size=whisper_model)
    latin_lexicon = LatinLexicon()

    whisper_segments = whisper.transcribe_auto(audio_path)
    results: list[LanguageSegment] = []

    for seg in whisper_segments:
        text = seg.text.strip()
        if spanish_orthography:
            text = transliterate_to_spanish_orthography(text)

        lang_code, lang_conf = _guess_language_from_text_markers(text, seg.language, latin_lexicon)

        placeholder = PhonemeSegment(
            phoneme=text or "empty",
            start_time=seg.start,
            duration=seg.end - seg.start,
        )
        out = LanguageSegment(
            language=lang_code,
            phonemes=[placeholder],
            start_time=seg.start,
            end_time=seg.end,
            confidence=lang_conf,
        )
        out.transcription = text
        out.transcription_backend = "whisper-only"
        results.append(out)

    if enable_diarization and results:
        speaker_segments = diarize(audio_path)
        is_fallback = (len(speaker_segments) == 1 and speaker_segments[0].speaker == "Speaker ?")
        if is_fallback:
            for seg in results:
                seg.speaker = "Speaker ?"
        else:
            _assign_speakers(results, speaker_segments)

    whisper.unload()

    results.sort(key=lambda s: s.start_time)
    if output_path is not None:
        from tenepal.subtitle import write_srt
        write_srt(results, output_path)
    return results


def process_audio(
    audio_path: Union[str, Path],
    enable_diarization: bool = True,
    backend: str = "allosaurus",
    model_size: str = "300M",
    use_docker: bool = False,
    whisper_model: str | None = None,
) -> list:
    """Process audio through the diarize-first pipeline.

    Pipeline: audio -> diarize -> per-speaker audio -> phonemes -> language ID

    When diarization is enabled and available:
    1. Run pyannote diarization to get speaker segments
    2. Slice audio by speaker
    3. For each speaker: run Allosaurus phoneme recognition
    4. For each speaker: run language identification
    5. Combine all segments, sorted by start_time

    When diarization is disabled or unavailable:
    Falls back to single-stream processing (existing v1 behavior)
    with speaker="Speaker ?" if diarization was requested but unavailable.

    Args:
        audio_path: Path to audio file
        enable_diarization: Whether to attempt speaker diarization

    Returns:
        List of LanguageSegment with speaker labels, sorted by start_time
    """
    audio_path = Path(audio_path)

    if not enable_diarization:
        return _process_single_stream(
            audio_path,
            speaker=None,
            backend=backend,
            model_size=model_size,
            whisper_model=whisper_model,
        )

    # Step 1: Diarize
    speaker_segments = diarize(audio_path, use_docker=use_docker)

    # Check if fallback (single "Speaker ?" segment)
    is_fallback = (len(speaker_segments) == 1
                   and speaker_segments[0].speaker == "Speaker ?")

    if is_fallback:
        return _process_single_stream(
            audio_path,
            speaker="Speaker ?",
            backend=backend,
            model_size=model_size,
            whisper_model=whisper_model,
        )

    # Step 2: Load and preprocess audio for slicing
    audio = load_audio(audio_path)
    audio = preprocess_audio(audio, target_sr=16000)

    # Step 3: Slice audio by speaker
    speaker_audio_pairs = slice_audio_by_speaker(audio, speaker_segments)

    # Step 4: Per-speaker phoneme recognition + language ID
    all_segments = []
    for spk_segment, spk_audio in speaker_audio_pairs:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save_wav(spk_audio, tmp_path)
            backend_kwargs = {"model_size": model_size} if backend in ("omnilingual", "dual") else {}
            phonemes = recognize_phonemes(tmp_path, backend=backend, **backend_kwargs)

            # Adjust phoneme timestamps to absolute time
            for p in phonemes:
                p.start_time += spk_segment.start_time

            # Run language ID on this speaker's phonemes
            lang_segments = identify_language(
                phonemes,
                audio_data=(spk_audio.samples, spk_audio.sample_rate),
            )

            # Tag each segment with speaker label
            for seg in lang_segments:
                seg.speaker = spk_segment.speaker

            all_segments.extend(lang_segments)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # Step 5: Transcription routing (if Whisper enabled)
    if whisper_model is not None:
        try:
            from tenepal.transcription import TranscriptionRouter
            router = TranscriptionRouter(whisper_model=whisper_model)
            transcription_results = router.transcribe_segments(all_segments, audio_path)
            _attach_transcriptions(all_segments, transcription_results)
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "faster-whisper not installed. Install: pip install 'tenepal[transcription]'. "
                "Falling back to phoneme-only output."
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Transcription routing failed: %s. Falling back to phoneme-only.", exc
            )

    # Step 6: Apply speaker-level language smoothing
    if enable_diarization and not is_fallback:
        all_segments = smooth_by_speaker(all_segments)

    # Step 7: Sort all segments by start_time for chronological output
    all_segments.sort(key=lambda s: (s.start_time, s.speaker or ""))

    return all_segments


def _attach_transcriptions(segments, transcription_results):
    """Attach transcription text to LanguageSegments.

    Maps TranscriptionResults back onto LanguageSegments by matching timing.
    For Whisper results, sets segment.transcription and segment.transcription_backend.
    For Allosaurus results, leaves segment unchanged (phonemes already present).

    Args:
        segments: List of LanguageSegment objects (modified in place)
        transcription_results: List of TranscriptionResult objects from router
    """
    # Create mapping by start_time for quick lookup
    result_map = {result.start_time: result for result in transcription_results}

    for segment in segments:
        result = result_map.get(segment.start_time)
        if result and result.backend == "whisper":
            # Attach Whisper transcription text
            segment.transcription = result.text
            segment.transcription_backend = result.backend


def _process_single_stream(
    audio_path,
    speaker=None,
    backend: str = "allosaurus",
    model_size: str = "300M",
    whisper_model: str | None = None,
):
    """Process audio as single stream (no diarization)."""
    backend_kwargs = {"model_size": model_size} if backend in ("omnilingual", "dual") else {}
    phonemes = recognize_phonemes(audio_path, backend=backend, **backend_kwargs)

    # Load audio for prosody extraction
    audio = load_audio(audio_path)
    audio = preprocess_audio(audio, target_sr=16000)
    lang_segments = identify_language(
        phonemes,
        audio_data=(audio.samples, audio.sample_rate),
    )

    if speaker is not None:
        for seg in lang_segments:
            seg.speaker = speaker

    # Transcription routing (if Whisper enabled)
    if whisper_model is not None:
        try:
            from tenepal.transcription import TranscriptionRouter
            router = TranscriptionRouter(whisper_model=whisper_model)
            transcription_results = router.transcribe_segments(lang_segments, audio_path)
            _attach_transcriptions(lang_segments, transcription_results)
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "faster-whisper not installed. Install: pip install 'tenepal[transcription]'. "
                "Falling back to phoneme-only output."
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Transcription routing failed: %s. Falling back to phoneme-only.", exc
            )

    return lang_segments


def _find_gaps(
    segments: list,
    audio_duration: float,
    min_gap_duration: float = 0.3,
) -> list[tuple[float, float]]:
    """Find uncovered time ranges between Whisper segments.

    Identifies time gaps that Whisper didn't transcribe, including:
    - Start gap: from time 0 to first segment
    - Interior gaps: between non-adjacent segments
    - End gap: from last segment to audio duration

    Args:
        segments: List of objects with .start and .end attributes
            (WhisperAutoSegment or similar), need not be sorted
        audio_duration: Total audio duration in seconds
        min_gap_duration: Minimum gap duration to report (default: 0.3s)

    Returns:
        List of (start, end) tuples representing uncovered time ranges,
        sorted by start time
    """
    if not segments:
        if audio_duration > min_gap_duration:
            return [(0.0, audio_duration)]
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s.start)

    # Track the "covered up to" watermark
    covered_up_to = 0.0
    gaps = []

    for seg in sorted_segs:
        # If this segment starts after current watermark, there's a gap
        if seg.start > covered_up_to + 1e-6:  # epsilon for float comparison
            gap_start = covered_up_to
            gap_end = seg.start
            if (gap_end - gap_start) >= min_gap_duration:
                gaps.append((gap_start, gap_end))
        # Advance watermark (handle overlapping segments)
        covered_up_to = max(covered_up_to, seg.end)

    # Check end gap
    if audio_duration > covered_up_to + 1e-6:
        gap_start = covered_up_to
        gap_end = audio_duration
        if (gap_end - gap_start) >= min_gap_duration:
            gaps.append((gap_start, gap_end))

    return gaps


def _process_allosaurus_fallback(
    time_ranges: list[tuple[float, float]],
    samples: np.ndarray,
    sample_rate: int,
    backend_tag: str = "allosaurus",
) -> list[LanguageSegment]:
    """Process time ranges through Allosaurus for phoneme recognition and language ID.

    Args:
        time_ranges: List of (start, end) tuples in seconds
        samples: Audio samples array
        sample_rate: Sample rate in Hz
        backend_tag: Tag to apply to transcription_backend field

    Returns:
        List of LanguageSegment objects with phoneme and language data
    """
    import numpy as np

    results = []
    for start_time, end_time in time_ranges:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_samples = samples[start_sample:end_sample]

        if len(segment_samples) == 0:
            continue

        # Write slice to temp WAV
        fd, temp_path_str = tempfile.mkstemp(suffix=".wav", prefix="tenepal_wf_")
        os.close(fd)
        temp_path = Path(temp_path_str)

        try:
            sf.write(str(temp_path), segment_samples, sample_rate)
            phonemes = recognize_phonemes(temp_path)

            # Adjust phoneme timestamps to absolute time
            for p in phonemes:
                p.start_time += start_time

            # Run language identification
            lang_segments = identify_language(
                phonemes,
                audio_data=(segment_samples, sample_rate),
            )

            # Tag segments with backend identifier
            for seg in lang_segments:
                seg.transcription_backend = backend_tag

            results.extend(lang_segments)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return results


def _apply_cross_segment_nah_absorption(segments: list[LanguageSegment]) -> list[LanguageSegment]:
    """Apply Nahuatl context absorption across the full segment list.

    Scans the sorted segment list for NAH-OTH-NAH patterns where the OTH
    segment is short (duration <= 2.0 seconds). Reclassifies such OTH segments
    as "nah" to handle the film audio pattern where brief unidentified segments
    appear between Nahuatl speech.

    This is a simpler version than identifier.py's OTH absorption since these
    are already LanguageSegments (no phoneme-level analysis needed).

    Args:
        segments: List of LanguageSegment objects sorted by start_time

    Returns:
        List of LanguageSegment objects with OTH absorption applied
    """
    if len(segments) < 3:
        return segments

    changed = True
    while changed:
        changed = False
        result = []
        i = 0

        while i < len(segments):
            seg = segments[i]

            # Check if this is an OTH segment between two NAH segments
            if seg.language == "other" and i > 0 and i < len(segments) - 1:
                prev_lang = result[-1].language if result else None
                next_seg = segments[i + 1]

                if prev_lang == "nah" and next_seg.language == "nah":
                    # Check if OTH is short enough to absorb
                    duration = seg.end_time - seg.start_time

                    if duration <= 2.0:
                        # Absorb: reclassify OTH as NAH
                        seg.language = "nah"
                        changed = True
                        logger.info(
                            "Absorbed OTH segment (%.1fs-%.1fs, %.2fs) between NAH segments",
                            seg.start_time, seg.end_time, duration
                        )

            result.append(seg)
            i += 1

        segments = result

    return segments


def _assign_speakers(
    lang_segments: list[LanguageSegment],
    speaker_segments: list,
) -> None:
    """Assign speaker labels to language segments by best time overlap.

    For each language segment, finds the speaker segment with the most
    temporal overlap and assigns that speaker label. Modifies in place.

    Args:
        lang_segments: List of LanguageSegment objects (modified in place)
        speaker_segments: List of SpeakerSegment objects from diarization
    """
    for lang_seg in lang_segments:
        best_speaker = None
        best_overlap = 0.0

        for spk_seg in speaker_segments:
            # Calculate overlap
            overlap_start = max(lang_seg.start_time, spk_seg.start_time)
            overlap_end = min(lang_seg.end_time, spk_seg.end_time)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk_seg.speaker

        if best_speaker is not None:
            lang_seg.speaker = best_speaker


def _deduplicate_whisper_to_turns(
    whisper_segments: list,
    speaker_segments: list,
) -> tuple[list[tuple], list]:
    """Deduplicate Whisper segments to speaker turns using 1:1 maximum-overlap assignment.

    Each Whisper segment is assigned to the ONE speaker turn with maximum temporal
    overlap. Each turn can receive at most one Whisper segment. Turns without
    assignments are returned as unassigned for Allosaurus processing.

    Algorithm:
    1. For each Whisper segment, compute overlap with all speaker turns
    2. Find the turn with maximum overlap for this Whisper segment
    3. If multiple Whisper segments want the same turn, the one with highest
       overlap wins (greedy assignment)
    4. Unassigned turns become gaps for Allosaurus fallback

    Args:
        whisper_segments: List of WhisperAutoSegment objects
        speaker_segments: List of SpeakerSegment objects from pyannote

    Returns:
        Tuple of (assigned_pairs, unassigned_turns):
        - assigned_pairs: List of (WhisperAutoSegment, SpeakerSegment) tuples
        - unassigned_turns: List of SpeakerSegment objects without Whisper match
    """
    if not whisper_segments:
        return [], list(speaker_segments)

    if not speaker_segments:
        return [], []

    # Build mapping: each Whisper segment index → (best_turn, overlap_duration)
    whisper_to_turn = []
    for w_idx, w_seg in enumerate(whisper_segments):
        best_turn = None
        best_overlap = 0.0

        for spk_seg in speaker_segments:
            # Calculate temporal overlap
            overlap_start = max(w_seg.start, spk_seg.start_time)
            overlap_end = min(w_seg.end, spk_seg.end_time)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = spk_seg

        # Store best match (even if overlap is 0, meaning no overlap)
        if best_turn is not None and best_overlap > 0:
            whisper_to_turn.append((w_idx, best_turn, best_overlap))

    # Resolve conflicts: if multiple Whisper segments want the same turn,
    # the one with highest overlap wins
    turn_assignments = {}  # turn -> (whisper_index, overlap)
    for w_idx, turn, overlap in whisper_to_turn:
        if turn not in turn_assignments:
            turn_assignments[turn] = (w_idx, overlap)
        else:
            # Conflict: check if this Whisper has better overlap
            existing_w_idx, existing_overlap = turn_assignments[turn]
            if overlap > existing_overlap:
                turn_assignments[turn] = (w_idx, overlap)

    # Build result pairs and unassigned turns
    assigned_pairs = [
        (whisper_segments[w_idx], turn)
        for turn, (w_idx, _) in turn_assignments.items()
    ]
    assigned_turn_set = set(turn for turn, _ in turn_assignments.items())
    unassigned_turns = [turn for turn in speaker_segments if turn not in assigned_turn_set]

    return assigned_pairs, unassigned_turns


def _process_medium_confidence(
    medium_segments: list,
    samples: np.ndarray,
    sample_rate: int,
    latin_lexicon,
) -> list[LanguageSegment]:
    """Process medium-confidence Whisper segments: keep text, use Allosaurus for language.

    These segments have decent Whisper text (-0.3 to -0.7 avg_log_prob) but
    uncertain language detection. We trust the text but run Allosaurus on the
    same audio to determine the actual language.

    Args:
        medium_segments: List of WhisperAutoSegment objects with medium confidence
        samples: Audio samples array
        sample_rate: Sample rate in Hz
        latin_lexicon: LatinLexicon instance for keyword detection

    Returns:
        List of LanguageSegment with Allosaurus language and Whisper text
    """
    from tenepal.transcription.languages import WHISPER_LANG_REVERSE

    results = []
    for seg in medium_segments:
        # Run Allosaurus on this segment's time range
        start_sample = int(seg.start * sample_rate)
        end_sample = int(seg.end * sample_rate)
        segment_samples = samples[start_sample:end_sample]

        if len(segment_samples) == 0:
            continue

        # Write slice to temp WAV
        fd, temp_path_str = tempfile.mkstemp(suffix=".wav", prefix="tenepal_med_")
        os.close(fd)
        temp_path = Path(temp_path_str)

        try:
            sf.write(str(temp_path), segment_samples, sample_rate)
            phonemes = recognize_phonemes(temp_path)

            # Adjust timestamps to absolute time
            for p in phonemes:
                p.start_time += seg.start

            # Get language from Allosaurus analysis
            lang_segments = identify_language(
                phonemes,
                audio_data=(segment_samples, sample_rate),
            )

            # Check for Latin liturgical keywords FIRST
            is_latin, latin_count = latin_lexicon.check_text(seg.text)
            if is_latin:
                allosaurus_lang = "lat"
                logger.info(
                    "Latin detected in medium-confidence segment (%.1fs-%.1fs): %d keywords in '%s'",
                    seg.start, seg.end, latin_count, seg.text[:50]
                )
            else:
                # Use Allosaurus's language determination
                if lang_segments:
                    # Use the dominant language from Allosaurus
                    allosaurus_lang = lang_segments[0].language
                else:
                    allosaurus_lang = "other"

            # Create LanguageSegment with Allosaurus language + Whisper text
            placeholder = PhonemeSegment(
                phoneme=seg.text,
                start_time=seg.start,
                duration=seg.end - seg.start,
            )
            lang_seg = LanguageSegment(
                language=allosaurus_lang,
                phonemes=[placeholder],
                start_time=seg.start,
                end_time=seg.end,
            )
            lang_seg.transcription = seg.text
            lang_seg.transcription_backend = "whisper"
            results.append(lang_seg)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return results


def _remove_overlaps(
    allosaurus_segments: list[LanguageSegment],
    whisper_intervals: list[tuple[float, float]],
) -> list[LanguageSegment]:
    """Remove Allosaurus segments that overlap with accepted Whisper intervals.

    Prevents mixed IPA+text in SRT by ensuring Allosaurus segments don't
    duplicate time ranges already covered by high-confidence Whisper output.

    Args:
        allosaurus_segments: Segments from Allosaurus fallback/gap processing
        whisper_intervals: List of (start, end) tuples from accepted Whisper segments

    Returns:
        Filtered list of Allosaurus segments with no Whisper overlap
    """
    if not whisper_intervals:
        return allosaurus_segments

    filtered = []
    for seg in allosaurus_segments:
        overlaps = False
        for w_start, w_end in whisper_intervals:
            # Check for any overlap (not just containment)
            if seg.start_time < w_end and seg.end_time > w_start:
                overlaps = True
                break
        if not overlaps:
            filtered.append(seg)
        else:
            logger.info(
                "Removed overlapping Allosaurus segment (%.1fs-%.1fs)",
                seg.start_time, seg.end_time,
            )
    return filtered


def _build_vocab_prompt(whisper_segments: list) -> str:
    """Build vocabulary prompt from recognized Whisper segments.

    Extracts unique words from high-confidence Whisper transcriptions to
    prime the rescue pass. This helps Whisper recognize short utterances
    like "soldado" instead of hallucinating "saludo" or "a su lado".

    Args:
        whisper_segments: List of WhisperAutoSegment with .text attribute

    Returns:
        Comma-separated string of unique words, or empty string
    """
    words = set()
    for seg in whisper_segments:
        for word in seg.text.split():
            # Strip punctuation for cleaner prompt
            cleaned = word.strip(".,!?¡¿;:\"'()[]{}…")
            if cleaned and len(cleaned) > 1:
                words.add(cleaned)
    if not words:
        return ""
    return ", ".join(sorted(words))


def _whisper_rescue_pass(
    results: list,
    unassigned_turns: list,
    whisper,
    audio_path: Path,
    high_confidence: list,
    latin_lexicon,
    validator,
) -> list:
    """Re-try unassigned turns through Whisper with VAD disabled and vocab prompt.

    Turns that got no Whisper text in the main pass (music-only, short utterances
    that VAD filtered out) are individually re-processed with:
    - vad_filter=False: Force Whisper to decode even "silent" segments
    - Silence padding: 500ms before/after for cleaner boundary detection
    - Vocabulary prompt: Words from already-recognized segments prime the decoder

    Successfully rescued segments replace their Allosaurus-only counterparts.

    Args:
        results: Current list of LanguageSegment (modified in place via replacement)
        unassigned_turns: Speaker turns without Whisper text
        whisper: WhisperBackend instance (still loaded)
        audio_path: Path to audio file
        high_confidence: High-confidence WhisperAutoSegments for vocab prompt
        latin_lexicon: LatinLexicon for keyword detection
        validator: WhisperValidator for hallucination checks

    Returns:
        Updated results list with rescued segments replacing Allosaurus ones
    """
    from tenepal.transcription.languages import WHISPER_LANG_REVERSE

    if not unassigned_turns:
        return results

    # Build vocabulary prompt from already-recognized text
    vocab_prompt = _build_vocab_prompt(high_confidence)
    if vocab_prompt:
        logger.info("Rescue pass: vocab prompt = '%s'", vocab_prompt[:100])

    # Load audio for segment extraction
    audio_data = load_audio(audio_path)
    samples = audio_data.samples
    sample_rate = audio_data.sample_rate

    rescued_count = 0
    for turn in unassigned_turns:
        # Extract audio for this turn
        start_sample = int(turn.start_time * sample_rate)
        end_sample = int(turn.end_time * sample_rate)
        segment_samples = samples[start_sample:end_sample]

        if len(segment_samples) == 0:
            continue

        # Run Whisper rescue: no VAD, with padding and prompt
        rescue_segments = whisper.transcribe_segment(
            segment_samples,
            sample_rate,
            vad_filter=False,
            initial_prompt=vocab_prompt or None,
            pad_seconds=0.5,
        )

        if not rescue_segments:
            continue

        # Take the best segment (first one, typically only one for short audio)
        rescue_seg = rescue_segments[0]

        # Validate: skip hallucinations
        validation = validator.validate(
            rescue_seg.text, avg_log_prob=rescue_seg.avg_log_prob
        )
        if not validation.is_valid:
            logger.info(
                "Rescue rejected (%.1fs-%.1fs): '%s' [%s]",
                turn.start_time, turn.end_time, rescue_seg.text[:50], validation.reason,
            )
            continue

        # Determine language
        is_latin, latin_count = latin_lexicon.check_text(rescue_seg.text)
        if is_latin:
            lang_639_3 = "lat"
        else:
            lang_639_3 = WHISPER_LANG_REVERSE.get(rescue_seg.language, "other")

        # Create replacement LanguageSegment
        placeholder = PhonemeSegment(
            phoneme=rescue_seg.text,
            start_time=turn.start_time,
            duration=turn.end_time - turn.start_time,
        )
        new_seg = LanguageSegment(
            language=lang_639_3,
            phonemes=[placeholder],
            start_time=turn.start_time,
            end_time=turn.end_time,
        )
        new_seg.transcription = rescue_seg.text
        new_seg.transcription_backend = "whisper-rescue"
        new_seg.speaker = turn.speaker

        # Replace matching allosaurus-turn segment in results
        replaced = False
        for i, existing in enumerate(results):
            if (
                getattr(existing, "transcription_backend", None) == "allosaurus-turn"
                and abs(existing.start_time - turn.start_time) < 0.05
            ):
                results[i] = new_seg
                replaced = True
                break

        if not replaced:
            # No exact match found — append (turn may have produced no Allosaurus output)
            results.append(new_seg)

        rescued_count += 1
        logger.info(
            "Rescued turn (%.1fs-%.1fs): '%s' [%s, logprob=%.2f]",
            turn.start_time, turn.end_time, rescue_seg.text[:50],
            lang_639_3, rescue_seg.avg_log_prob,
        )

    if rescued_count > 0:
        logger.info("Rescue pass: %d/%d turns recovered", rescued_count, len(unassigned_turns))
        # Re-sort after replacements
        results.sort(key=lambda s: s.start_time)

    return results


def process_whisper_first(
    audio_path: Path,
    whisper_model: str = "medium",
    confidence_threshold: float = -0.5,
    allosaurus_fallback: bool = True,
    output_path: Path | None = None,
    enable_diarization: bool = True,
    whisper_rescue: bool = False,
) -> list[LanguageSegment]:
    """Whisper-first processing pipeline for film audio.

    Lets Whisper go first with auto-detect. Uses three-tier confidence routing:
    - HIGH (>-0.3): Use Whisper language unconditionally
    - MEDIUM (-0.7 to -0.3): Keep Whisper text, use Allosaurus for language
    - LOW (<-0.7): Use Allosaurus for phonemes and language

    Speaker profiles are built from high-confidence segments and applied for
    language inheritance to uncertain/short segments.

    Args:
        audio_path: Path to audio file
        whisper_model: Whisper model size (tiny/base/small/medium/large/turbo)
        confidence_threshold: DEPRECATED - now using fixed tier thresholds (-0.3/-0.7).
            Parameter kept for backward compatibility but not used.
        allosaurus_fallback: Whether to run Allosaurus on low-confidence segments
        output_path: Optional path to write SRT output
        enable_diarization: Whether to run speaker diarization (default: True)
        whisper_rescue: Re-try unassigned turns through Whisper with VAD disabled
            and vocabulary prompt built from already-recognized text (default: False)

    Returns:
        List of LanguageSegment sorted by start_time
    """
    import numpy as np
    from tenepal.phoneme.whisper_backend import WhisperBackend
    from tenepal.transcription.languages import WHISPER_LANG_REVERSE
    from tenepal.validation import WhisperValidator
    from tenepal.language.latin_lexicon import LatinLexicon

    audio_path = Path(audio_path)

    # Initialize Latin lexicon singleton
    latin_lexicon = LatinLexicon()

    # Step 1: Run Whisper with auto-detect
    whisper = WhisperBackend(model_size=whisper_model)
    auto_segments = whisper.transcribe_auto(audio_path)

    # Get audio duration for gap detection
    audio_duration = sf.info(str(audio_path)).duration

    # Step 1.5: Validate Whisper segments for hallucinations
    validator = WhisperValidator()
    validated_segments = []
    hallucinated_segments = []

    for seg in auto_segments:
        result = validator.validate(seg.text, avg_log_prob=seg.avg_log_prob)
        if result.is_valid:
            validated_segments.append(seg)
        else:
            logger.info(
                "Whisper hallucination detected (%.1fs-%.1fs): %s [reason: %s]",
                seg.start, seg.end, seg.text[:50], result.reason
            )
            hallucinated_segments.append(seg)

    # Step 2: Split VALIDATED segments into HIGH/MEDIUM/LOW confidence tiers
    high_confidence, medium_confidence, low_confidence = split_by_confidence_tier(validated_segments)

    results: list[LanguageSegment] = []

    # Step 2.5: Diarization early (if enabled) for deduplication
    speaker_segments = []
    unassigned_turns = []
    assigned_pairs = []
    speaker_profiles = {}
    if enable_diarization:
        speaker_segments = diarize(audio_path)
        is_fallback = (len(speaker_segments) == 1
                       and speaker_segments[0].speaker == "Speaker ?")
        if not is_fallback:
            # Deduplicate: assign each Whisper segment to ONE turn
            assigned_pairs, unassigned_turns = _deduplicate_whisper_to_turns(
                high_confidence, speaker_segments
            )
            logger.info(
                "Deduplication: %d Whisper→turn assignments, %d unassigned turns",
                len(assigned_pairs), len(unassigned_turns)
            )
            # Build speaker profiles from high-confidence Whisper segments
            speaker_profiles = build_speaker_profiles(assigned_pairs)
            logger.info("Built %d speaker profiles", len(speaker_profiles))
        else:
            # Fallback mode - no real diarization
            pass

    # Step 3: Convert high-confidence Whisper segments to LanguageSegments
    if enable_diarization and speaker_segments and not (len(speaker_segments) == 1 and speaker_segments[0].speaker == "Speaker ?"):
        # With diarization: create LanguageSegments from assigned pairs
        # Use TURN boundaries and TURN speaker, not Whisper boundaries
        for whisper_seg, turn in assigned_pairs:
            # Check for Latin liturgical text first
            is_latin, latin_count = latin_lexicon.check_text(whisper_seg.text)
            if is_latin:
                lang_639_3 = "lat"
                logger.info(
                    "Latin detected in Whisper segment (%.1fs-%.1fs): %d keywords in '%s'",
                    whisper_seg.start, whisper_seg.end, latin_count, whisper_seg.text[:50]
                )
            else:
                lang_639_3 = WHISPER_LANG_REVERSE.get(whisper_seg.language, "other")

            # Use TURN boundaries, not Whisper boundaries
            placeholder = PhonemeSegment(
                phoneme=whisper_seg.text,
                start_time=turn.start_time,
                duration=turn.end_time - turn.start_time,
            )
            lang_seg = LanguageSegment(
                language=lang_639_3,
                phonemes=[placeholder],
                start_time=turn.start_time,
                end_time=turn.end_time,
            )
            # Store transcription text and speaker from turn
            lang_seg.transcription = whisper_seg.text
            lang_seg.transcription_backend = "whisper"
            lang_seg.speaker = turn.speaker
            results.append(lang_seg)
    else:
        # Without diarization: use Whisper boundaries as before
        for seg in high_confidence:
            # Check for Latin liturgical text first
            is_latin, latin_count = latin_lexicon.check_text(seg.text)
            if is_latin:
                lang_639_3 = "lat"
                logger.info(
                    "Latin detected in Whisper segment (%.1fs-%.1fs): %d keywords in '%s'",
                    seg.start, seg.end, latin_count, seg.text[:50]
                )
            else:
                lang_639_3 = WHISPER_LANG_REVERSE.get(seg.language, "other")

            # Create a placeholder PhonemeSegment to carry the text
            placeholder = PhonemeSegment(
                phoneme=seg.text,
                start_time=seg.start,
                duration=seg.end - seg.start,
            )
            lang_seg = LanguageSegment(
                language=lang_639_3,
                phonemes=[placeholder],
                start_time=seg.start,
                end_time=seg.end,
            )
            # Store transcription text for SRT output
            lang_seg.transcription = seg.text
            lang_seg.transcription_backend = "whisper"
            results.append(lang_seg)

    # Step 3.5: Process medium-confidence segments (keep Whisper text, use Allosaurus for language)
    if medium_confidence:
        logger.info("Processing %d medium-confidence segments (Whisper text, Allosaurus language)", len(medium_confidence))
        # Load audio if not already loaded
        audio_data = load_audio(audio_path)
        samples = audio_data.samples
        sample_rate = audio_data.sample_rate

        medium_results = _process_medium_confidence(
            medium_confidence, samples, sample_rate, latin_lexicon
        )
        results.extend(medium_results)

    # Step 4: Run Allosaurus fallback on low-confidence + hallucinated segments + unassigned turns
    fallback_segments = low_confidence + hallucinated_segments
    if allosaurus_fallback:
        # Load audio if we need Allosaurus processing (fallback or gaps or unassigned turns)
        audio_data = None
        samples = None
        sample_rate = None

        # Process unassigned turns (turns without Whisper match)
        if unassigned_turns:
            logger.info("Processing %d unassigned turns through Allosaurus", len(unassigned_turns))
            audio_data = load_audio(audio_path)
            samples = audio_data.samples
            sample_rate = audio_data.sample_rate

            unassigned_ranges = [(turn.start_time, turn.end_time) for turn in unassigned_turns]
            unassigned_results = _process_allosaurus_fallback(
                unassigned_ranges, samples, sample_rate, backend_tag="allosaurus-turn"
            )
            # Assign speakers to unassigned turn results
            for i, turn in enumerate(unassigned_turns):
                if i < len(unassigned_results):
                    for seg in unassigned_results:
                        if abs(seg.start_time - turn.start_time) < 0.01:  # Match by start time
                            seg.speaker = turn.speaker
            results.extend(unassigned_results)

        if fallback_segments:
            if audio_data is None:
                audio_data = load_audio(audio_path)
                samples = audio_data.samples
                sample_rate = audio_data.sample_rate

            # Process fallback segments through Allosaurus
            fallback_ranges = [(seg.start, seg.end) for seg in fallback_segments]
            fallback_results = _process_allosaurus_fallback(
                fallback_ranges, samples, sample_rate, backend_tag="allosaurus"
            )
            results.extend(fallback_results)

        # Step 4.5: Detect and process time gaps Whisper didn't cover
        gaps = _find_gaps(auto_segments, audio_duration)
        if gaps:
            logger.info("Found %d time gap(s) to process through Allosaurus", len(gaps))
            # Load audio now if not already loaded
            if audio_data is None:
                audio_data = load_audio(audio_path)
                samples = audio_data.samples
                sample_rate = audio_data.sample_rate

            gap_results = _process_allosaurus_fallback(
                gaps, samples, sample_rate, backend_tag="allosaurus-gap"
            )
            results.extend(gap_results)
    else:
        # Low-confidence but validated segments get Whisper text
        for seg in low_confidence:
            # Check for Latin liturgical text first
            is_latin, latin_count = latin_lexicon.check_text(seg.text)
            if is_latin:
                lang_639_3 = "lat"
                logger.info(
                    "Latin detected in low-confidence Whisper segment (%.1fs-%.1fs): %d keywords",
                    seg.start, seg.end, latin_count
                )
            else:
                lang_639_3 = WHISPER_LANG_REVERSE.get(seg.language, "other")

            placeholder = PhonemeSegment(
                phoneme=seg.text,
                start_time=seg.start,
                duration=seg.end - seg.start,
            )
            lang_seg = LanguageSegment(
                language=lang_639_3,
                phonemes=[placeholder],
                start_time=seg.start,
                end_time=seg.end,
            )
            lang_seg.transcription = seg.text
            lang_seg.transcription_backend = "whisper"
            results.append(lang_seg)
        # NOTE: hallucinated_segments are DROPPED when fallback=False
        # They are hallucinations, not valid transcription

    # Step 4.7: Remove Allosaurus segments that overlap with Whisper intervals
    # When deduplication is used, whisper intervals are the TURN boundaries, not original Whisper boundaries
    if enable_diarization and speaker_segments and not (len(speaker_segments) == 1 and speaker_segments[0].speaker == "Speaker ?"):
        # Use turn boundaries for overlap checking
        whisper_intervals = [(turn.start_time, turn.end_time) for _, turn in assigned_pairs] if assigned_pairs else []
    else:
        # Use original Whisper boundaries
        whisper_intervals = [(seg.start, seg.end) for seg in high_confidence]

    whisper_results = [r for r in results if getattr(r, 'transcription_backend', None) == 'whisper']
    allosaurus_results = [r for r in results if getattr(r, 'transcription_backend', None) != 'whisper']
    allosaurus_results = _remove_overlaps(allosaurus_results, whisper_intervals)
    results = whisper_results + allosaurus_results

    # Step 5: Sort by start_time
    results.sort(key=lambda s: s.start_time)

    # Step 5.5: Apply Nahuatl lexicon-based OTH absorption across full results
    results = _apply_cross_segment_nah_absorption(results)

    # Step 5.7: Diarization post-processing
    if enable_diarization:
        # Check if we already have diarization from deduplication
        if not speaker_segments or (len(speaker_segments) == 1 and speaker_segments[0].speaker == "Speaker ?"):
            # Fallback: run diarization now
            speaker_segments = diarize(audio_path)
            is_fallback = (len(speaker_segments) == 1
                           and speaker_segments[0].speaker == "Speaker ?")
            if not is_fallback:
                _assign_speakers(results, speaker_segments)
                # Apply speaker inheritance before smoothing
                if speaker_profiles:
                    results = apply_speaker_inheritance(results, speaker_profiles)
                results = smooth_by_speaker(results)
        else:
            # We already assigned speakers during deduplication
            # Only assign to segments that don't have speakers yet (e.g., gap fills)
            segments_without_speaker = [seg for seg in results if seg.speaker is None]
            if segments_without_speaker:
                _assign_speakers(segments_without_speaker, speaker_segments)
            # Apply speaker inheritance before smoothing
            if speaker_profiles:
                results = apply_speaker_inheritance(results, speaker_profiles)
            results = smooth_by_speaker(results)

    # Step 6: Whisper rescue pass for unassigned turns (optional)
    if whisper_rescue and unassigned_turns:
        results = _whisper_rescue_pass(
            results, unassigned_turns, whisper, audio_path,
            high_confidence, latin_lexicon, validator,
        )

    # Unload Whisper model to free GPU memory
    whisper.unload()

    # Step 7: Write SRT if requested
    if output_path is not None:
        from tenepal.subtitle import write_srt
        write_srt(results, output_path)

    return results

"""Command-line interface for Tenepal phonetic transcription tool."""

import argparse
import signal
import sys
import tempfile
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from tenepal.audio import AudioCapture, load_audio, save_wav, preprocess_audio
from tenepal.phoneme import recognize_phonemes
from tenepal.language import identify_language, print_language_segments, format_language_segments, analyze_phonemes, build_confusion_matrix, format_analysis, smooth_by_speaker
from tenepal.pipeline import process_audio, process_whisper_first, process_whisper_text_only
from tenepal.speaker import format_speaker_stats, diarize, slice_audio_by_speaker, LabelLocker
from tenepal.subtitle import write_srt

# Global flag for signal handling
_stop_capture = False

# Speaker color palette for live mode (ANSI 256-color)
SPEAKER_COLORS = [
    "\033[38;5;39m",   # Blue (Speaker A)
    "\033[38;5;208m",  # Orange (Speaker B)
    "\033[38;5;48m",   # Green (Speaker C)
    "\033[38;5;213m",  # Pink (Speaker D)
    "\033[38;5;226m",  # Yellow (Speaker E)
    "\033[38;5;123m",  # Cyan (Speaker F)
]


def process_file(
    audio_path: Path,
    output_path: Optional[Path] = None,
    enable_diarization: bool = True,
    backend: str = "allosaurus",
    model_size: str = "300M",
    pronounce: Optional[str] = None,
    whisper_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single audio file through the phonetic transcription pipeline.

    Args:
        audio_path: Path to input audio file
        output_path: Optional override for output SRT file path
        enable_diarization: Whether to enable speaker diarization

    Returns:
        Dictionary containing processing statistics:
        - duration: Audio duration in seconds
        - phonemes: Number of phoneme segments
        - srt_path: Path to output SRT file

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio format is unsupported
        Exception: For other processing errors
    """
    # Validate file exists
    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    # Validate file extension
    extension = audio_path.suffix.lower()
    supported_formats = {".wav", ".mp3", ".flac"}
    if extension not in supported_formats:
        raise ValueError(
            f"Unsupported format: {extension}. "
            f"Supported: {', '.join(supported_formats)}"
        )

    print(f"Processing: {audio_path.name}")

    # Load audio to get metadata
    try:
        audio_data = load_audio(audio_path)
    except Exception as e:
        # Check for missing ffmpeg on MP3/FLAC
        if extension in {".mp3", ".flac"}:
            error_msg = str(e).lower()
            if "ffmpeg" in error_msg or "couldn't find ffmpeg" in error_msg or "decoder" in error_msg:
                raise RuntimeError(
                    "MP3/FLAC support requires ffmpeg. Install: sudo pacman -S ffmpeg"
                ) from e
        raise

    # Run diarize-first pipeline (phoneme recognition + language identification)
    language_segments = process_audio(
        audio_path,
        enable_diarization=enable_diarization,
        backend=backend,
        model_size=model_size,
        whisper_model=whisper_model,
    )

    # Display color-coded output in terminal
    print_language_segments(language_segments, pronounce=pronounce)

    # Check if diarization was requested but unavailable (all "Speaker ?")
    if enable_diarization and language_segments:
        all_fallback = all(
            seg.speaker == "Speaker ?" for seg in language_segments if seg.speaker is not None
        )
        if all_fallback:
            print("Warning: Speaker diarization unavailable. Set HUGGINGFACE_TOKEN for speaker detection.", file=sys.stderr)

    # Display speaker statistics if diarization was enabled
    if enable_diarization and language_segments:
        stats = format_speaker_stats(language_segments)
        print(f"\n{stats}")

    # Determine output path
    if output_path is None:
        srt_path = audio_path.with_suffix(".srt")
    else:
        srt_path = output_path

    # Write SRT file
    write_srt(language_segments, srt_path, pronounce=pronounce)

    print(f"Saved: {srt_path}")

    # Count phonemes across all segments
    total_phonemes = sum(len(seg.phonemes) for seg in language_segments)

    # Return statistics
    return {
        "duration": audio_data.duration,
        "phonemes": total_phonemes,
        "srt_path": str(srt_path)
    }


def _signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM gracefully."""
    global _stop_capture
    _stop_capture = True


def _backend_kwargs(backend: str, model_size: str) -> Dict[str, str]:
    if backend in ("omnilingual", "dual"):
        return {"model_size": model_size}
    return {}


def _recognize_from_audio(
    audio_data,
    backend: str = "allosaurus",
    model_size: str = "300M",
) -> List:
    """Helper to recognize phonemes from AudioData object.

    Saves audio to temp WAV file and runs recognition.

    Args:
        audio_data: AudioData object

    Returns:
        List of PhonemeSegment objects
    """
    # Create temp file for recognition
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Save audio to temp file
        save_wav(audio_data, tmp_path)

        # Run recognition
        phoneme_segments = recognize_phonemes(
            tmp_path,
            backend=backend,
            **_backend_kwargs(backend, model_size),
        )

        return phoneme_segments
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


def _print_live_segment(
    segment,
    speaker_colors: Optional[Dict[str, str]] = None,
    pronounce: Optional[str] = None,
) -> None:
    """Print a single language segment with color coding.

    Args:
        segment: LanguageSegment to print
        speaker_colors: Optional dict mapping speaker label -> ANSI color code.
                       If provided, colors by speaker. Otherwise colors by language.
    """
    label_map = {
        "nah": "NAH",
        "spa": "SPA",
        "eng": "ENG",
        "deu": "DEU",
        "fra": "FRA",
        "ita": "ITA",
        "may": "MAY",
        "lat": "LAT",
    }
    label = label_map.get(segment.language, "OTH")

    if segment.speaker:
        tag = f"[{segment.speaker} | {label}]"
    else:
        tag = f"[{label}]"

    time_range = f"{segment.start_time:.2f}s - {segment.end_time:.2f}s"

    # Check for transcription text (from Whisper)
    transcription = getattr(segment, 'transcription', None)
    if transcription:
        phonemes_str = transcription
    elif pronounce:
        from tenepal.pronunciation import render_pronunciation

        phonemes_str = render_pronunciation([p.phoneme for p in segment.phonemes], pronounce)
    else:
        phonemes_str = " ".join(p.phoneme for p in segment.phonemes)

    if sys.stdout.isatty():
        # If speaker colors provided, use speaker color; else use language color
        if speaker_colors and segment.speaker:
            color = speaker_colors.get(segment.speaker, "\033[2m")  # dim gray fallback
        else:
            # Language-based coloring (backward compatible)
            color_map = {
                "nah": "\033[32m",
                "may": "\033[36m",
                "spa": "\033[33m",
                "eng": "\033[34m",
                "deu": "\033[35m",
                "lat": "\033[31m",
                "other": "\033[2m"
            }
            color = color_map.get(segment.language, color_map["other"])

        reset = "\033[0m"
        print(f"{color}{time_range:<20} {tag} {phonemes_str}{reset}")
    else:
        print(f"{time_range:<20} {tag} {phonemes_str}")


def run_live(
    output_path: Optional[Path] = None,
    enable_diarization: bool = True,
    backend: str = "allosaurus",
    model_size: str = "300M",
    pronounce: Optional[str] = None,
) -> None:
    """Run live audio capture mode with streaming display.

    Captures system audio in real-time, displays color-coded language-tagged
    phonemes as they're recognized, and optionally saves to SRT file.

    When diarization enabled:
    - Accumulates audio chunks into buffer
    - Re-diarizes every 30s of accumulated audio
    - Uses LabelLocker for stable speaker labels across passes
    - Colors output by speaker (not language)
    - Shows [Speaker X | LANG] tags and [new speaker detected] notifications
    - Displays speaker count in status bar
    - Prints speaker stats after capture

    When diarization disabled:
    - Uses sliding window phoneme buffering
    - Colors by language (v1 behavior)

    Args:
        output_path: Optional path to save SRT output
        enable_diarization: Whether to enable speaker diarization

    Raises:
        RuntimeError: If PulseAudio/PipeWire not available
    """
    global _stop_capture
    _stop_capture = False

    # Setup signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Test audio system availability before starting
    try:
        # Create AudioCapture to test availability (don't start yet)
        test_capture = AudioCapture()
    except RuntimeError as e:
        print("Error: Live mode requires PulseAudio or PipeWire. Neither detected on this system.")
        sys.exit(1)

    if not enable_diarization:
        # Fall back to v1 sliding window behavior (no diarization)
        _run_live_no_diarize(
            output_path,
            backend=backend,
            model_size=model_size,
            pronounce=pronounce,
        )
        return

    # === Diarization-enabled mode ===

    # Re-diarize every 30s of accumulated audio
    REDIARIZE_INTERVAL = 30.0

    # All emitted segments (for SRT output and speaker stats)
    emitted_segments = []
    # Accumulated audio buffer (list of numpy arrays)
    audio_buffer_chunks = []
    # Metadata for accumulated audio
    buffer_duration = 0.0
    buffer_sample_rate = 16000  # Allosaurus uses 16kHz
    # Last time we ran diarization
    last_diarize_time = 0.0
    # Label locker for stable speaker labels
    label_locker = LabelLocker()
    # Track seen speakers for [new speaker detected] and status bar
    seen_speakers = set()
    # Speaker colors mapping
    speaker_colors = {}
    # Track if diarization is actually available
    diarization_available = True

    print("Live capture started (Ctrl+C to stop)")

    try:
        with AudioCapture() as capture:
            for audio_chunk in capture.chunks():
                # Check stop flag
                if _stop_capture:
                    break

                # Preprocess to 16kHz for consistency
                from tenepal.audio.loader import AudioData
                chunk_audio = AudioData(
                    samples=audio_chunk.samples,
                    sample_rate=audio_chunk.sample_rate,
                    duration=audio_chunk.duration,
                    source_format="raw"
                )
                chunk_audio = preprocess_audio(chunk_audio, target_sr=16000)

                # Accumulate into buffer
                audio_buffer_chunks.append(chunk_audio.samples)
                buffer_duration += chunk_audio.duration
                buffer_sample_rate = chunk_audio.sample_rate

                # Check if it's time to re-diarize
                time_since_diarize = buffer_duration - last_diarize_time
                should_diarize = time_since_diarize >= REDIARIZE_INTERVAL

                if should_diarize and diarization_available:
                    # Concatenate all accumulated audio
                    full_buffer = np.concatenate(audio_buffer_chunks)

                    # Save to temp file for diarization
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = Path(tmp.name)

                    try:
                        full_audio = AudioData(
                            samples=full_buffer,
                            sample_rate=buffer_sample_rate,
                            duration=buffer_duration,
                            source_format="raw"
                        )
                        save_wav(full_audio, tmp_path)

                        # Run diarization on accumulated audio
                        speaker_segments = diarize(tmp_path)

                        # Check if this is first diarization and it's fallback
                        if last_diarize_time == 0.0:
                            is_fallback = (len(speaker_segments) == 1
                                          and speaker_segments[0].speaker == "Speaker ?")
                            if is_fallback:
                                diarization_available = False
                                print("Warning: Speaker diarization unavailable. Set HUGGINGFACE_TOKEN for speaker detection.", file=sys.stderr)
                                # Continue processing this pass with fallback

                        # Stabilize labels
                        speaker_segments = label_locker.stabilize(speaker_segments)

                        # Build speaker color mapping (assign colors to new speakers)
                        for seg in speaker_segments:
                            if seg.speaker not in speaker_colors:
                                color_idx = len(speaker_colors) % len(SPEAKER_COLORS)
                                speaker_colors[seg.speaker] = SPEAKER_COLORS[color_idx]

                        # Get NEW audio since last diarization
                        new_audio_start = last_diarize_time
                        start_sample = int(new_audio_start * buffer_sample_rate)
                        new_audio_samples = full_buffer[start_sample:]
                        new_audio_duration = len(new_audio_samples) / buffer_sample_rate
                        new_audio = AudioData(
                            samples=new_audio_samples,
                            sample_rate=buffer_sample_rate,
                            duration=new_audio_duration,
                            source_format="raw"
                        )

                        # Filter speaker segments to only those in new audio range
                        new_segments = [
                            seg for seg in speaker_segments
                            if seg.end_time > new_audio_start
                        ]

                        # Adjust segment times to be relative to new audio
                        adjusted_segments = []
                        for seg in new_segments:
                            adjusted_start = max(0.0, seg.start_time - new_audio_start)
                            adjusted_end = seg.end_time - new_audio_start
                            from tenepal.speaker.diarizer import SpeakerSegment
                            adjusted_segments.append(SpeakerSegment(
                                speaker=seg.speaker,
                                start_time=adjusted_start,
                                end_time=adjusted_end
                            ))

                        # Slice new audio by speaker
                        speaker_audio_pairs = slice_audio_by_speaker(new_audio, adjusted_segments)

                        # Process each speaker's audio
                        new_language_segments = []
                        for spk_segment, spk_audio in speaker_audio_pairs:
                            # Recognize phonemes
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as spk_tmp:
                                spk_tmp_path = Path(spk_tmp.name)
                            try:
                                save_wav(spk_audio, spk_tmp_path)
                                phonemes = recognize_phonemes(
                                    spk_tmp_path,
                                    backend=backend,
                                    **_backend_kwargs(backend, model_size),
                                )

                                # Adjust phoneme timestamps to absolute time
                                for p in phonemes:
                                    p.start_time += new_audio_start + spk_segment.start_time

                                # Run language ID
                                lang_segments = identify_language(
                                    phonemes,
                                    audio_data=(spk_audio.samples, spk_audio.sample_rate),
                                )

                                # Tag with speaker
                                for seg in lang_segments:
                                    seg.speaker = spk_segment.speaker

                                new_language_segments.extend(lang_segments)
                            finally:
                                if spk_tmp_path.exists():
                                    spk_tmp_path.unlink()

                        # Sort by start time
                        new_language_segments.sort(key=lambda s: (s.start_time, s.speaker or ""))

                        # Apply speaker-level language smoothing
                        new_language_segments = smooth_by_speaker(new_language_segments)

                        # Emit segments
                        if new_language_segments:
                            sys.stdout.write('\r' + ' ' * 80 + '\r')
                            sys.stdout.flush()

                            for seg in new_language_segments:
                                # Check for new speaker
                                if seg.speaker and seg.speaker not in seen_speakers:
                                    print(f"[new speaker detected]")
                                    seen_speakers.add(seg.speaker)

                                _print_live_segment(
                                    seg,
                                    speaker_colors=speaker_colors,
                                    pronounce=pronounce,
                                )
                                emitted_segments.append(seg)

                        # Update diarization timestamp
                        last_diarize_time = buffer_duration

                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()

                # Print status bar
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                speaker_count = len(seen_speakers)
                if speaker_count > 0:
                    status = f"[LIVE] {buffer_duration:.0f}s | {len(emitted_segments)} segments | {speaker_count} speakers | listening..."
                else:
                    status = f"[LIVE] {buffer_duration:.0f}s | {len(emitted_segments)} segments | listening..."
                sys.stdout.write(status)
                sys.stdout.flush()

    finally:
        # Process any remaining audio not yet diarized
        if audio_buffer_chunks and buffer_duration > last_diarize_time and diarization_available:
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()
            print("Processing final audio...")

            full_buffer = np.concatenate(audio_buffer_chunks)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                full_audio = AudioData(
                    samples=full_buffer,
                    sample_rate=buffer_sample_rate,
                    duration=buffer_duration,
                    source_format="raw"
                )
                save_wav(full_audio, tmp_path)

                speaker_segments = diarize(tmp_path)
                speaker_segments = label_locker.stabilize(speaker_segments)

                # Assign colors to any new speakers
                for seg in speaker_segments:
                    if seg.speaker not in speaker_colors:
                        color_idx = len(speaker_colors) % len(SPEAKER_COLORS)
                        speaker_colors[seg.speaker] = SPEAKER_COLORS[color_idx]

                # Process new audio since last diarization
                new_audio_start = last_diarize_time
                start_sample = int(new_audio_start * buffer_sample_rate)
                new_audio_samples = full_buffer[start_sample:]
                new_audio_duration = len(new_audio_samples) / buffer_sample_rate
                new_audio = AudioData(
                    samples=new_audio_samples,
                    sample_rate=buffer_sample_rate,
                    duration=new_audio_duration,
                    source_format="raw"
                )

                new_segments = [
                    seg for seg in speaker_segments
                    if seg.end_time > new_audio_start
                ]

                adjusted_segments = []
                for seg in new_segments:
                    adjusted_start = max(0.0, seg.start_time - new_audio_start)
                    adjusted_end = seg.end_time - new_audio_start
                    from tenepal.speaker.diarizer import SpeakerSegment
                    adjusted_segments.append(SpeakerSegment(
                        speaker=seg.speaker,
                        start_time=adjusted_start,
                        end_time=adjusted_end
                    ))

                speaker_audio_pairs = slice_audio_by_speaker(new_audio, adjusted_segments)

                final_segments = []
                for spk_segment, spk_audio in speaker_audio_pairs:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as spk_tmp:
                        spk_tmp_path = Path(spk_tmp.name)
                    try:
                        save_wav(spk_audio, spk_tmp_path)
                        phonemes = recognize_phonemes(
                            spk_tmp_path,
                            backend=backend,
                            **_backend_kwargs(backend, model_size),
                        )

                        for p in phonemes:
                            p.start_time += new_audio_start + spk_segment.start_time

                        lang_segments = identify_language(
                            phonemes,
                            audio_data=(spk_audio.samples, spk_audio.sample_rate),
                        )

                        for seg in lang_segments:
                            seg.speaker = spk_segment.speaker

                        final_segments.extend(lang_segments)
                    finally:
                        if spk_tmp_path.exists():
                            spk_tmp_path.unlink()

                # Apply speaker-level language smoothing
                final_segments = smooth_by_speaker(final_segments)

                # Emit smoothed segments
                for seg in final_segments:
                    if seg.speaker and seg.speaker not in seen_speakers:
                        print(f"[new speaker detected]")
                        seen_speakers.add(seg.speaker)

                    _print_live_segment(
                        seg,
                        speaker_colors=speaker_colors,
                        pronounce=pronounce,
                    )
                    emitted_segments.append(seg)

            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        print()
        print("Live capture stopped.")
        print(f"Duration: {buffer_duration:.1f}s | Segments: {len(emitted_segments)}")

        # Print speaker stats
        if emitted_segments and enable_diarization:
            stats = format_speaker_stats(emitted_segments)
            print(f"\n{stats}")

        # Save SRT if requested
        if output_path is not None:
            write_srt(emitted_segments, output_path, pronounce=pronounce)
            print(f"Saved: {output_path}")


def _run_live_no_diarize(
    output_path: Optional[Path] = None,
    backend: str = "allosaurus",
    model_size: str = "300M",
    pronounce: Optional[str] = None,
) -> None:
    """Run live capture without diarization (v1 behavior).

    Uses sliding window phoneme buffering with language ID.
    """
    global _stop_capture

    # Max seconds of pending phonemes before force-emitting
    MAX_PENDING_SECONDS = 30.0

    # Phonemes not yet finalized into a language segment
    pending_phonemes = []
    # All emitted segments (for SRT output)
    emitted_segments = []
    total_duration = 0.0
    finalized_up_to = 0.0

    print("Live capture started (Ctrl+C to stop)")

    try:
        with AudioCapture() as capture:
            for audio_chunk in capture.chunks():
                # Check stop flag
                if _stop_capture:
                    break

                # Recognize phonemes from chunk
                chunk_phonemes = _recognize_from_audio(
                    audio_chunk,
                    backend=backend,
                    model_size=model_size,
                )

                # Adjust phoneme timestamps to be cumulative
                for p in chunk_phonemes:
                    p.start_time += total_duration

                pending_phonemes.extend(chunk_phonemes)
                total_duration += audio_chunk.duration

                if not pending_phonemes:
                    # Update status bar even with no phonemes
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    status = f"[LIVE] {total_duration:.0f}s | {len(emitted_segments)} segments | listening..."
                    sys.stdout.write(status)
                    sys.stdout.flush()
                    continue

                # Run language identification on pending phonemes
                segments = identify_language(pending_phonemes)

                # Decide what to emit
                pending_duration = total_duration - finalized_up_to
                force_emit = pending_duration > MAX_PENDING_SECONDS

                if force_emit:
                    # Buffer full: emit all but last (or all if single segment)
                    to_emit = segments[:-1] if len(segments) > 1 else segments
                else:
                    # Normal: emit all but last (keep trailing tentative)
                    to_emit = segments[:-1] if len(segments) > 1 else []

                if to_emit:
                    # Clear status bar line
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    sys.stdout.flush()

                    for seg in to_emit:
                        _print_live_segment(seg, pronounce=pronounce)  # No speaker colors
                        emitted_segments.append(seg)
                        finalized_up_to = seg.end_time

                    # Trim finalized phonemes from pending buffer
                    pending_phonemes = [
                        p for p in pending_phonemes
                        if p.start_time >= finalized_up_to
                    ]

                # Print status bar
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                status = f"[LIVE] {total_duration:.0f}s | {len(emitted_segments)} segments | {len(pending_phonemes)} pending"
                sys.stdout.write(status)
                sys.stdout.flush()

    finally:
        # Flush remaining pending phonemes
        if pending_phonemes:
            remaining = identify_language(pending_phonemes)
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()
            for seg in remaining:
                _print_live_segment(seg, pronounce=pronounce)
                emitted_segments.append(seg)

        print()
        print("Live capture stopped.")
        print(f"Duration: {total_duration:.1f}s | Segments: {len(emitted_segments)}")

        # Save SRT if requested
        if output_path is not None:
            write_srt(emitted_segments, output_path)
            print(f"Saved: {output_path}")


def process_batch(
    file_paths: List[Path],
    output_path: Optional[Path] = None,
    enable_diarization: bool = True,
    backend: str = "allosaurus",
    model_size: str = "300M",
    pronounce: Optional[str] = None,
    whisper_model: Optional[str] = None,
) -> int:
    """Process multiple audio files in batch mode.

    Args:
        file_paths: List of audio file paths to process
        output_path: Optional output override (ignored in batch mode with warning)
        enable_diarization: Whether to enable speaker diarization

    Returns:
        Exit code: 0 if all succeeded, 1 if any failed
    """
    # Warn if output override provided in batch mode
    if output_path is not None and len(file_paths) > 1:
        print("Warning: --output ignored in batch mode (each file gets its own .srt)")

    total_count = len(file_paths)
    success_count = 0
    total_duration = 0.0
    total_phonemes = 0
    errors: List[Tuple[str, str]] = []

    # Process each file
    for i, file_path in enumerate(file_paths, 1):
        print(f"\nProcessing [{i}/{total_count}]: {file_path.name}")

        try:
            stats = process_file(
                file_path,
                enable_diarization=enable_diarization,
                backend=backend,
                model_size=model_size,
                pronounce=pronounce,
                whisper_model=whisper_model,
            )
            success_count += 1
            total_duration += stats["duration"]
            total_phonemes += stats["phonemes"]
        except Exception as e:
            error_msg = str(e)
            errors.append((file_path.name, error_msg))
            print(f"Error processing {file_path.name}: {error_msg}", file=sys.stderr)

    # Print batch summary
    print("\n" + "=" * 40)
    print("Batch Summary")
    print("=" * 40)
    print(f"Files processed: {success_count}/{total_count}")
    print(f"Total audio duration: {total_duration:.1f}s")
    print(f"Total phonemes: {total_phonemes}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for filename, error_msg in errors:
            print(f"  - {filename}: {error_msg}")

    print("=" * 40)

    # Return exit code
    return 0 if success_count == total_count else 1


def setup_diarization() -> None:
    """Download pyannote.audio models for speaker diarization."""
    import os

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("Error: HUGGINGFACE_TOKEN environment variable not set.", file=sys.stderr)
        print("", file=sys.stderr)
        print("To set up speaker diarization:", file=sys.stderr)
        print("1. Create a HuggingFace account at https://huggingface.co", file=sys.stderr)
        print("2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1", file=sys.stderr)
        print("3. Create an access token at https://huggingface.co/settings/tokens", file=sys.stderr)
        print("4. Set: export HUGGINGFACE_TOKEN=hf_...", file=sys.stderr)
        sys.exit(1)

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("Error: pyannote.audio not installed.", file=sys.stderr)
        print("Install: pip install 'tenepal[diarization]'", file=sys.stderr)
        sys.exit(1)

    print("Downloading speaker diarization model...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token
        )
        print("Model downloaded successfully!")
        print("Speaker diarization is ready to use.")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


def setup_demucs() -> None:
    """Download Demucs htdemucs model for vocal isolation."""
    try:
        from demucs.api import Separator
    except ImportError:
        print("Error: demucs not installed.", file=sys.stderr)
        print("Install: pip install demucs", file=sys.stderr)
        sys.exit(1)

    print("Downloading Demucs htdemucs model...")
    try:
        # Instantiate separator to trigger model download
        Separator(model="htdemucs")
        print("Model downloaded successfully!")
        print("Demucs vocal isolation is ready to use.")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


def setup_whisper() -> None:
    """Download faster-whisper model weights."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster-whisper not installed.", file=sys.stderr)
        print("Install: pip install 'tenepal[transcription]'", file=sys.stderr)
        sys.exit(1)

    model_size = "base"  # Default model
    print(f"Downloading faster-whisper {model_size} model...")
    try:
        # Instantiate model to trigger download
        WhisperModel(model_size, device="cpu")
        print("Model downloaded successfully!")
        print("Whisper transcription is ready to use.")
        print("Use --whisper-model flag to enable text transcription.")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


def preprocess_command(args) -> None:
    """Run preprocessing pipeline on video file.

    Args:
        args: Parsed CLI arguments
    """
    from tenepal.preprocessing import preprocess_video

    # Validate exactly one file provided
    if not args.files or len(args.files) != 1:
        print("Error: preprocess command requires exactly one video file", file=sys.stderr)
        print("Usage: tenepal preprocess <video_file>", file=sys.stderr)
        sys.exit(1)

    video_path = Path(args.files[0])

    # Validate file exists
    if not video_path.exists():
        print(f"Error: File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Validate file extension
    extension = video_path.suffix.lower()
    if extension not in {".mkv", ".mp4"}:
        print(f"Error: Unsupported format: {extension}", file=sys.stderr)
        print("Supported formats: .mkv, .mp4", file=sys.stderr)
        sys.exit(1)

    # Get output directory (default: current directory)
    output_dir = args.output_dir if args.output_dir else Path.cwd()

    # Get preprocessing options
    skip_isolation = args.skip_isolation
    demucs_segment = args.demucs_segment

    print(f"Preprocessing: {video_path.name}")

    # Run preprocessing pipeline
    try:
        segments = preprocess_video(
            video_path,
            output_dir,
            skip_isolation=skip_isolation,
            demucs_segment=demucs_segment,
        )

        # Print summary
        print(f"\n{len(segments)} speech segments found")
        print(f"Segments manifest: {output_dir}/{video_path.stem}_segments.json")
        print(f"Intermediate files in: {output_dir}/")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def process_film_command(args) -> None:
    """Process film file end-to-end: video -> language-tagged SRT subtitles.

    Args:
        args: Parsed CLI arguments
    """
    from tenepal.orchestration import process_film

    # Validate exactly one file provided
    if not args.files or len(args.files) != 1:
        print("Error: process-film command requires exactly one video file", file=sys.stderr)
        print("Usage: tenepal process-film <video_file>", file=sys.stderr)
        sys.exit(1)

    video_path = Path(args.files[0])

    # Validate file exists
    if not video_path.exists():
        print(f"Error: File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Validate file extension (MKV/MP4 for video, WAV/MP3/FLAC also accepted)
    extension = video_path.suffix.lower()
    supported = {".mkv", ".mp4", ".wav", ".mp3", ".flac"}
    if extension not in supported:
        print(f"Error: Unsupported format: {extension}", file=sys.stderr)
        print(f"Supported formats: {', '.join(sorted(supported))}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_path = args.output if args.output else None

    try:
        srt_path = process_film(
            video_path=video_path,
            output_path=output_path,
            enable_diarization=not args.no_diarize,
            whisper_model=args.whisper_model,
            skip_isolation=args.skip_isolation,
            demucs_segment=args.demucs_segment,
            backend=args.backend,
            model_size=args.omnilingual_model,
            pronounce=args.pronounce,
            whisper_rescue=args.whisper_rescue,
        )
        print(f"\nFilm processing complete: {srt_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing film: {e}", file=sys.stderr)
        sys.exit(1)


def process_command(args) -> None:
    """Whisper-first processing pipeline.

    Runs Whisper with auto-detect first, then falls back to Allosaurus
    for low-confidence segments.

    Args:
        args: Parsed CLI arguments
    """
    if not args.files or len(args.files) != 1:
        print("Error: process command requires exactly one audio file", file=sys.stderr)
        print("Usage: tenepal process <audio_file>", file=sys.stderr)
        sys.exit(1)

    audio_path = Path(args.files[0])

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output if args.output else audio_path.with_suffix(".srt")
    whisper_model = args.whisper_model or "medium"
    confidence = args.confidence if args.confidence is not None else -0.5

    mode = "whisper-only" if args.whisper_only else "whisper-first"
    print(f"Processing ({mode}): {audio_path.name}")
    print(f"  Whisper model: {whisper_model}")
    if not args.whisper_only:
        print(f"  Confidence threshold: {confidence}")
    if args.whisper_only:
        print(f"  Spanish orthography: {'on' if args.spanish_orthography else 'off'}")

    try:
        if args.whisper_only:
            results = process_whisper_text_only(
                audio_path,
                whisper_model=whisper_model,
                output_path=output_path,
                enable_diarization=not args.no_diarize,
                spanish_orthography=args.spanish_orthography,
            )
        else:
            results = process_whisper_first(
                audio_path,
                whisper_model=whisper_model,
                confidence_threshold=confidence,
                output_path=output_path,
                enable_diarization=not args.no_diarize,
                whisper_rescue=args.whisper_rescue,
            )
        print(f"Wrote {len(results)} segments to {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for Tenepal CLI."""
    parser = argparse.ArgumentParser(
        description="Phonetic transcription tool for endangered languages",
        epilog="""supported formats: WAV, MP3, FLAC

examples:
  tenepal audio.wav                    Process single file
  tenepal *.wav                        Process all WAV files (batch mode)
  tenepal --live                       Capture system audio
  tenepal --live -o session.srt        Capture and save SRT
  tenepal analyze audio.wav            Analyze phoneme frequencies and detection accuracy
  tenepal preprocess film.mkv          Preprocess video for transcription
  tenepal preprocess film.mkv --skip-isolation   Skip vocal isolation
  tenepal process-film film.mkv        Process film end-to-end
  tenepal process-film film.mkv --whisper-model base  With Whisper text transcription
  tenepal process-film film.mkv --skip-isolation --no-diarize  Minimal processing
  tenepal process audio.wav            Whisper-first pipeline (auto-detect + fallback)
  tenepal process audio.wav --whisper-only --spanish-orthography
                                           Whisper-only text + orthography normalization
  tenepal process audio.wav --whisper-model medium --confidence -0.3
  tenepal setup-diarization            Download speaker diarization models
  tenepal setup-demucs                 Download Demucs models
  tenepal setup-docker                 Build GPU Docker image
  tenepal doctor                       Run GPU diagnostics

Live mode captures system audio (requires PulseAudio or PipeWire) and displays
streaming color-coded language-tagged phonemes. Press Ctrl+C to stop cleanly.
Batch mode auto-detects multiple files and shows progress.""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "command",
        nargs="?",
        help="Setup command to run",
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Audio files to process"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Live audio capture mode - capture system audio in real-time"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path (single file mode only; ignored in batch mode)"
    )

    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization"
    )

    parser.add_argument(
        "--omnilingual-model",
        choices=["300M", "7B"],
        default="300M",
        help="Omnilingual model size (default: 300M). 7B requires ~24GB VRAM.",
    )

    parser.add_argument(
        "--backend",
        choices=["allosaurus", "omnilingual", "dual"],
        default="allosaurus",
        help="ASR backend to use (default: allosaurus)",
    )

    parser.add_argument(
        "--pronounce",
        choices=["de", "es", "en"],
        default=None,
        help="Render IPA phonemes as human-readable spelling for locale (de/es/en). Display-only; does not affect recognition.",
    )

    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper model size for text transcription of known languages (default: disabled). "
             "Enables routing: known languages (SPA/ENG/DEU/FRA/ITA) get readable text via Whisper, "
             "Nahuatl gets IPA phonemes via Allosaurus.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Whisper avg_log_prob threshold for process command (default: -0.5). "
             "Segments below this threshold fall back to Allosaurus.",
    )

    parser.add_argument(
        "--whisper-rescue",
        action="store_true",
        help="Re-try unassigned speaker turns through Whisper with VAD disabled "
             "and vocabulary prompt. Recovers short utterances that Whisper's VAD "
             "filtered out in the main pass.",
    )

    parser.add_argument(
        "--whisper-only",
        action="store_true",
        help="Use Whisper-only segment transcription with text-marker language tagging. "
             "Skips Allosaurus/Omnilingual fallback passes.",
    )

    parser.add_argument(
        "--spanish-orthography",
        action="store_true",
        help="Normalize transcription text to Spanish-style orthography "
             "(useful for classical Nahuatl source normalization).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for preprocessing (default: current directory)",
    )

    parser.add_argument(
        "--skip-isolation",
        action="store_true",
        help="Skip Demucs vocal isolation (for clean audio without background music)",
    )

    parser.add_argument(
        "--demucs-segment",
        type=int,
        default=300,
        help="Demucs chunk size in seconds for long audio (default: 300)",
    )

    return parser


def validate_backend(backend_name: str) -> str:
    """Validate backend availability and apply fallback if needed."""
    from tenepal.phoneme.backend import get_backend
    from tenepal.phoneme.omnilingual_backend import OmnilingualBackend

    try:
        get_backend(backend_name)
    except (ValueError, RuntimeError) as exc:
        if backend_name != "allosaurus":
            print(
                f"Warning: Backend '{backend_name}' unavailable. Falling back to allosaurus.",
                file=sys.stderr,
            )
            return "allosaurus"
        raise exc

    if backend_name == "dual" and not OmnilingualBackend.is_available():
        print(
            "Warning: Omnilingual backend unavailable. Dual mode will use Allosaurus only.",
            file=sys.stderr,
        )
        return "allosaurus"

    return backend_name


def analyze_audio(args) -> None:
    """Analyze phoneme frequencies and detection accuracy for audio files.

    Args:
        args: CLI arguments with files list and backend setting
    """
    if not args.files:
        print("Error: No audio files specified", file=sys.stderr)
        print("Usage: tenepal analyze <audio_file> [<audio_file> ...]", file=sys.stderr)
        sys.exit(1)

    for file_path in args.files:
        audio_path = Path(file_path)

        if not audio_path.exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            continue

        print(f"\nAnalyzing: {audio_path.name}")
        print("=" * 80)

        # Load audio
        try:
            audio_data = load_audio(audio_path)
        except Exception as e:
            print(f"Error loading audio: {e}", file=sys.stderr)
            continue

        # Recognize phonemes
        try:
            phoneme_segments = recognize_phonemes(
                audio_path,
                backend=args.backend,
                **_backend_kwargs(args.backend, args.omnilingual_model),
            )
        except Exception as e:
            print(f"Error recognizing phonemes: {e}", file=sys.stderr)
            continue

        # Analyze phonemes
        analysis = analyze_phonemes(phoneme_segments)

        # Build confusion matrix
        confusion = build_confusion_matrix(analysis)

        # Format and print report
        report = format_analysis(analysis, confusion)
        print(report)


def setup_omnilingual() -> None:
    """Create Python 3.12 venv and install Omnilingual ASR dependencies."""
    from tenepal.phoneme.omnilingual_backend import OmnilingualBackend

    venv_path = OmnilingualBackend.VENV_PATH
    python312 = shutil.which("python3.12")
    if not python312:
        print(
            "Error: Python 3.12 is required. Install: sudo pacman -S python312",
            file=sys.stderr,
        )
        sys.exit(1)

    if venv_path.exists():
        print("Omnilingual environment already exists. Reinstalling dependencies...")
    else:
        print("Creating Python 3.12 virtual environment...")
        try:
            subprocess.run([python312, "-m", "venv", str(venv_path)], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Error creating venv: {exc}", file=sys.stderr)
            sys.exit(1)

    print("Installing omnilingual-asr (this may take several minutes)...")
    pip_path = venv_path / "bin" / "pip"
    try:
        subprocess.run([str(pip_path), "install", "omnilingual-asr"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error installing omnilingual-asr: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Omnilingual ASR setup complete! Use --backend omnilingual to transcribe.")


def setup_docker() -> None:
    """Build the tenepal GPU Docker image."""
    from tenepal.docker import is_docker_available, TENEPAL_IMAGE

    if not is_docker_available():
        print("Error: Docker is not available.", file=sys.stderr)
        print("Install Docker: https://docs.docker.com/engine/install/", file=sys.stderr)
        sys.exit(1)

    dockerfile_path = Path(__file__).parent.parent.parent / "docker" / "Dockerfile"
    if not dockerfile_path.exists():
        print(f"Error: Dockerfile not found at {dockerfile_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Building Docker image: {TENEPAL_IMAGE}")
    print("This may take several minutes on first build...")

    result = subprocess.run(
        [
            "docker",
            "build",
            "-t",
            TENEPAL_IMAGE,
            "-f",
            str(dockerfile_path),
            str(dockerfile_path.parent),
        ],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: Docker build failed (exit code {result.returncode})", file=sys.stderr)
        sys.exit(1)

    print(f"Docker image built successfully: {TENEPAL_IMAGE}")
    print("GPU inference is ready. Use --docker flag with process command.")


def doctor() -> None:
    """Run diagnostics on Docker and GPU setup."""
    from tenepal.docker import (
        is_docker_available,
        is_vulkan_available,
        is_image_built,
        TENEPAL_IMAGE,
        get_device_mounts,
    )

    print("Tenepal Doctor")
    print("=" * 40)

    checks = []
    docker_ok = is_docker_available()
    checks.append(
        (
            "Docker",
            docker_ok,
            "docker info" if docker_ok else "Install: https://docs.docker.com/engine/install/",
        )
    )

    vulkan_ok = is_vulkan_available()
    checks.append(
        (
            "Vulkan GPU (/dev/dri)",
            vulkan_ok,
            "Device found" if vulkan_ok else "No /dev/dri render nodes",
        )
    )

    image_ok = is_image_built() if docker_ok else False
    checks.append(
        (
            "Docker image",
            image_ok,
            TENEPAL_IMAGE if image_ok else "Run: tenepal setup-docker",
        )
    )

    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        marker = "+" if ok else "-"
        print(f"  [{marker}] {name}: {status} — {detail}")

    if docker_ok and image_ok:
        print()
        print("Container diagnostics:")
        device_mounts = get_device_mounts()
        cmd = ["docker", "run", "--rm"] + device_mounts + [
            TENEPAL_IMAGE,
            "python",
            "/app/healthcheck.py",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            try:
                import json

                health = json.loads(result.stdout)
                for check_name, check_data in health.get("checks", {}).items():
                    ok = check_data.get("ok", False)
                    marker = "+" if ok else "-"
                    detail = {k: v for k, v in check_data.items() if k != "ok"}
                    print(f"  [{marker}] {check_name}: {'PASS' if ok else 'FAIL'} — {detail}")
            except json.JSONDecodeError:
                print(f"  Raw output: {result.stdout[:500]}")
                if result.stderr:
                    print(f"  Errors: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print("  Container health check timed out (30s)")
        except Exception as exc:
            print(f"  Error running health check: {exc}")

    print()
    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        print("All host checks passed.")
    else:
        print("Some checks failed. Fix issues above and re-run: tenepal doctor")


def main():
    """Main entry point for Tenepal CLI."""
    parser = build_parser()
    args = parser.parse_args()

    commands = {"setup-diarization", "setup-omnilingual", "setup-docker", "setup-demucs", "setup-whisper", "doctor", "analyze", "preprocess", "process-film", "process"}
    if args.command not in commands:
        if args.command is not None:
            args.files = [args.command] + args.files
        args.command = None

    # Handle subcommands
    if args.command == "setup-diarization":
        setup_diarization()
        sys.exit(0)
    if args.command == "setup-omnilingual":
        setup_omnilingual()
        sys.exit(0)
    if args.command == "setup-docker":
        setup_docker()
        sys.exit(0)
    if args.command == "setup-demucs":
        setup_demucs()
        sys.exit(0)
    if args.command == "setup-whisper":
        setup_whisper()
        sys.exit(0)
    if args.command == "doctor":
        doctor()
        sys.exit(0)
    if args.command == "analyze":
        analyze_audio(args)
        sys.exit(0)
    if args.command == "preprocess":
        preprocess_command(args)
        sys.exit(0)
    if args.command == "process-film":
        args.backend = validate_backend(args.backend)
        process_film_command(args)
        sys.exit(0)
    if args.command == "process":
        process_command(args)
        sys.exit(0)

    # Validate backend availability with fallback
    args.backend = validate_backend(args.backend)

    # Handle no arguments
    if not args.files and not args.live:
        parser.print_help()
        sys.exit(1)

    # Handle live mode
    if args.live:
        if args.files:
            print("Error: --live cannot be used with file arguments", file=sys.stderr)
            sys.exit(1)
        run_live(
            output_path=args.output,
            enable_diarization=not args.no_diarize,
            backend=args.backend,
            model_size=args.omnilingual_model,
            pronounce=args.pronounce,
        )
        sys.exit(0)

    # Convert file paths to Path objects
    file_paths = [Path(f) for f in args.files]

    # Single file mode
    if len(file_paths) == 1:
        try:
            process_file(
                file_paths[0],
                args.output,
                enable_diarization=not args.no_diarize,
                backend=args.backend,
                model_size=args.omnilingual_model,
                pronounce=args.pronounce,
                whisper_model=args.whisper_model,
            )
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Batch mode
    exit_code = process_batch(
        file_paths,
        args.output,
        enable_diarization=not args.no_diarize,
        backend=args.backend,
        model_size=args.omnilingual_model,
        pronounce=args.pronounce,
        whisper_model=args.whisper_model,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

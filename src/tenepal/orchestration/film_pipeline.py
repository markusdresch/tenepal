"""Core orchestrator for end-to-end film processing.

process_film() chains all v4.0 components into a unified pipeline:
1. Preprocessing: extract → isolate → segment (Phase 22)
2. Audio path resolution
3. Language identification + diarization + transcription routing (Phases 23-24)
4. Terminal display of results
5. SRT generation

GPU memory is managed between stages to prevent OOM errors on long-form audio.
"""

import logging
from pathlib import Path

from tenepal.preprocessing import preprocess_video
from tenepal.pipeline import process_audio
from tenepal.language import print_language_segments
from tenepal.speaker import format_speaker_stats
from tenepal.subtitle import write_srt
from .lifecycle import cleanup_stage
from .progress import StageProgress

logger = logging.getLogger(__name__)


def process_film(
    video_path: Path,
    output_path: Path | None = None,
    enable_diarization: bool = True,
    whisper_model: str | None = None,
    skip_isolation: bool = False,
    demucs_segment: int = 300,
    backend: str = "allosaurus",
    model_size: str = "300M",
    pronounce: str | None = None,
    whisper_rescue: bool = False,
) -> Path:
    """Process full film through end-to-end pipeline with progress tracking.

    This is the core orchestrator for v4.0 - it chains preprocessing, language
    identification, transcription routing, and SRT generation into a unified
    pipeline with GPU memory management between stages.

    Pipeline stages:
    1. Preprocessing: Extract audio, isolate vocals (Demucs), segment speech (Silero-VAD)
    2. Audio path resolution: Determine which audio file to use downstream
    3. Language ID + transcription: Diarization, phoneme recognition, language ID, Whisper routing
    4. Display results: Print color-coded language segments and speaker stats
    5. SRT generation: Write final subtitle file

    Args:
        video_path: Path to input video file (MKV or MP4)
        output_path: Optional path for output SRT file (default: video_path with .srt extension)
        enable_diarization: Whether to attempt speaker diarization (default: True)
        whisper_model: Whisper model size for transcription routing (None=disabled, "base"=enabled)
        skip_isolation: If True, skip Demucs vocal isolation for clean audio (default: False)
        demucs_segment: Segment size in seconds for Demucs processing (default: 300)
        backend: Phoneme recognition backend ("allosaurus", "omnilingual", "dual")
        model_size: Model size for omnilingual backend ("300M", "1B")
        pronounce: Optional language code for pronunciation analysis

    Returns:
        Path to generated SRT file

    Raises:
        FileNotFoundError: If video file does not exist
        RuntimeError: If preprocessing fails (fatal error)

    Example:
        >>> from pathlib import Path
        >>> srt_path = process_film(
        ...     video_path=Path("film.mkv"),
        ...     enable_diarization=True,
        ...     whisper_model="base"
        ... )
        >>> print(f"Saved: {srt_path}")
    """
    video_path = Path(video_path)

    logger.info("Starting film processing pipeline: %s", video_path.name)

    with StageProgress(total_stages=5, description="Processing film") as progress:
        # Stage 1: Preprocessing (extract → isolate → segment)
        logger.info("Stage 1/5: Preprocessing (extract → isolate → segment)")
        try:
            segments = preprocess_video(
                video_path,
                output_dir=video_path.parent,
                skip_isolation=skip_isolation,
                demucs_segment=demucs_segment,
            )
            logger.info("Preprocessing complete: %d speech segments found", len(segments))
            progress.advance("Stage 1/5: Preprocessing complete")
        except Exception as exc:
            logger.error("Preprocessing failed: %s", exc)
            raise RuntimeError(f"Preprocessing stage failed: {exc}") from exc
        finally:
            # Free Demucs GPU memory
            cleanup_stage("preprocessing")

        # Stage 2: Determine audio path for downstream processing
        logger.info("Stage 2/5: Audio path resolution")
        if skip_isolation:
            audio_path = video_path.parent / f"{video_path.stem}_extracted.wav"
        else:
            audio_path = video_path.parent / f"{video_path.stem}_isolated.wav"

        # Validate audio file exists (fallback to extracted if isolated missing)
        if not audio_path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            fallback_path = video_path.parent / f"{video_path.stem}_extracted.wav"
            if fallback_path.exists():
                logger.info("Falling back to extracted audio: %s", fallback_path)
                audio_path = fallback_path
            else:
                raise FileNotFoundError(f"No audio file found: tried {audio_path} and {fallback_path}")

        logger.info("Audio path resolved: %s", audio_path.name)
        progress.advance("Stage 2/5: Audio path resolved")

        # Stage 3: Language identification + diarization via process_audio()
        logger.info("Stage 3/5: Language identification + transcription routing")
        try:
            language_segments = process_audio(
                audio_path,
                enable_diarization=enable_diarization,
                backend=backend,
                model_size=model_size,
                whisper_model=whisper_model,
            )
            logger.info("Language identification complete: %d segments", len(language_segments))
        except Exception as exc:
            # Graceful degradation: retry without diarization if it failed
            if enable_diarization:
                logger.warning("Diarization failed: %s. Retrying without diarization.", exc)
                try:
                    language_segments = process_audio(
                        audio_path,
                        enable_diarization=False,
                        backend=backend,
                        model_size=model_size,
                        whisper_model=whisper_model,
                    )
                    logger.info("Fallback successful: %d segments (no diarization)", len(language_segments))
                except Exception as exc2:
                    # Graceful degradation: retry without Whisper if it failed
                    if whisper_model:
                        logger.warning("Whisper failed: %s. Retrying phoneme-only mode.", exc2)
                        language_segments = process_audio(
                            audio_path,
                            enable_diarization=False,
                            backend=backend,
                            model_size=model_size,
                            whisper_model=None,
                        )
                        logger.info("Fallback successful: %d segments (phoneme-only)", len(language_segments))
                    else:
                        raise
            else:
                # Graceful degradation: retry without Whisper if diarization was already disabled
                if whisper_model:
                    logger.warning("Whisper failed: %s. Retrying phoneme-only mode.", exc)
                    language_segments = process_audio(
                        audio_path,
                        enable_diarization=False,
                        backend=backend,
                        model_size=model_size,
                        whisper_model=None,
                    )
                    logger.info("Fallback successful: %d segments (phoneme-only)", len(language_segments))
                else:
                    raise

        # Clean up GPU memory after transcription
        cleanup_stage("transcription")
        progress.advance("Stage 3/5: Language identification complete")

        # Stage 4: Display results in terminal
        logger.info("Stage 4/5: Displaying results")
        print_language_segments(language_segments, pronounce=pronounce)

        # Print speaker stats if diarization was active
        if enable_diarization and language_segments and language_segments[0].speaker:
            speaker_stats = format_speaker_stats(language_segments)
            print(speaker_stats)

        progress.advance("Stage 4/5: Results displayed")

        # Stage 5: SRT generation
        logger.info("Stage 5/5: SRT generation")
        if output_path is None:
            output_path = video_path.with_suffix(".srt")

        output_path = Path(output_path)
        write_srt(language_segments, output_path, pronounce=pronounce)
        logger.info("SRT file generated: %s", output_path)
        print(f"\nSaved: {output_path}")
        progress.advance("Stage 5/5: SRT generated")

    # Print final summary
    _print_summary(language_segments, output_path, enable_diarization, whisper_model)

    return output_path


def _print_summary(
    language_segments: list,
    output_path: Path,
    enable_diarization: bool,
    whisper_model: str | None,
) -> None:
    """Print final processing summary after pipeline completes.

    Args:
        language_segments: List of LanguageSegment from pipeline
        output_path: Path to generated SRT file
        enable_diarization: Whether diarization was active
        whisper_model: Whisper model name if transcription routing was active
    """
    # Calculate total duration
    total_duration = sum(seg.end_time - seg.start_time for seg in language_segments)

    # Get unique languages
    languages = sorted(set(seg.language for seg in language_segments))

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Segments: {len(language_segments)}")
    print(f"Duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)")
    print(f"Languages: {', '.join(languages)}")
    print(f"Diarization: {'Active' if enable_diarization else 'Disabled'}")
    print(f"Transcription: {'Active (' + whisper_model + ')' if whisper_model else 'Phoneme-only'}")
    print(f"Output: {output_path}")
    print("=" * 60)

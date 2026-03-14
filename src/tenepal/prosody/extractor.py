"""Prosodic feature extraction using Parselmouth (Praat).

Extracts pitch (F0), intensity, speech rate, and rhythm metrics from audio segments
for language classification via prosodic profiling.
"""

import logging
from dataclasses import dataclass

import numpy as np
import parselmouth
from scipy.signal import find_peaks

from .rhythm import compute_npvi

logger = logging.getLogger(__name__)


@dataclass
class ProsodyFeatures:
    """Prosodic features extracted from an audio segment.

    Attributes:
        f0_mean: Mean fundamental frequency in Hz (speaker-dependent baseline)
        f0_std: F0 standard deviation (pitch variability)
        f0_range: F0 range (max - min), speaker-normalized discriminator
        intensity_mean: Mean intensity in dB
        duration: Segment duration in seconds
        speech_rate: Estimated syllable rate (syllables/sec) from intensity peaks
        npvi_v: Normalized PVI for vocalic intervals (rhythm discriminator)
    """

    f0_mean: float
    f0_std: float
    f0_range: float
    intensity_mean: float
    duration: float
    speech_rate: float
    npvi_v: float


def extract_prosody(audio_data: np.ndarray, sample_rate: int) -> ProsodyFeatures | None:
    """Extract prosodic features from audio segment using Parselmouth.

    Args:
        audio_data: Audio samples as float32 numpy array (mono)
        sample_rate: Sample rate in Hz (typically 22050)

    Returns:
        ProsodyFeatures object if extraction succeeds, None otherwise

    Returns None when:
        - Duration < 1.0 second (insufficient for rhythm analysis)
        - No voiced frames detected (silent or unvoiced segment)
        - Parselmouth extraction fails (corrupt audio, etc.)
    """
    try:
        # Guard: check minimum duration
        duration = len(audio_data) / sample_rate
        if duration < 1.0:
            return None

        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)

        # Extract pitch (F0) contour
        # time_step=0.01 -> 10ms frame rate
        # pitch_floor=75 Hz (typical male low), pitch_ceiling=600 Hz (high female/child)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
        pitch_values = pitch.selected_array['frequency']

        # Filter unvoiced frames (pitch == 0 means unvoiced)
        voiced_pitch = pitch_values[pitch_values > 0]

        # Guard: no voiced frames -> silent or unvoiced segment
        if len(voiced_pitch) == 0:
            return None

        # Compute pitch statistics
        f0_mean = float(np.mean(voiced_pitch))
        f0_std = float(np.std(voiced_pitch))
        f0_range = float(np.max(voiced_pitch) - np.min(voiced_pitch))

        # Extract intensity contour
        intensity = sound.to_intensity(time_step=0.01)
        intensity_values = intensity.values.flatten()

        # Compute intensity statistics
        intensity_mean = float(np.mean(intensity_values))

        # Estimate syllable rate from intensity peaks
        # Peaks indicate energy bursts (approximate syllable nuclei)
        # height=mean_intensity: peaks above mean intensity
        # distance=5: minimum 50ms between peaks (max ~20 syllables/sec)
        peaks, _ = find_peaks(intensity_values, height=intensity_mean, distance=5)
        num_peaks = len(peaks)
        speech_rate = num_peaks / duration  # syllables per second

        # Compute vocalic interval durations from peak-to-peak distances
        if num_peaks >= 2:
            # Convert peak indices to time (10ms per frame)
            peak_times_ms = peaks * 10.0  # Convert frame index to milliseconds
            # Compute inter-peak intervals (vocalic durations)
            vocalic_durations = np.diff(peak_times_ms).tolist()
            # Compute nPVI from vocalic intervals
            npvi_v = compute_npvi(vocalic_durations)
        else:
            # Not enough peaks for rhythm analysis
            npvi_v = 0.0

        return ProsodyFeatures(
            f0_mean=f0_mean,
            f0_std=f0_std,
            f0_range=f0_range,
            intensity_mean=intensity_mean,
            duration=duration,
            speech_rate=speech_rate,
            npvi_v=npvi_v,
        )

    except Exception as exc:
        # Log warning and return None on any failure
        logger.warning("Prosody extraction failed: %s", exc)
        return None

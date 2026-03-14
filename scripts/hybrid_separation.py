#!/usr/bin/env python3
"""Hybrid voice separation: Pitch detection + Adaptive harmonic extraction.

Approach:
1. Detect F0 contours for male (60-180Hz) and female (150-400Hz)
2. Find overlap regions where both are active
3. Apply narrow-band harmonic extraction with adaptive bandwidth
4. Use higher harmonic count for better isolation

This should give cleaner separation than pure Wiener filtering.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft, istft, butter, sosfilt
from pathlib import Path
import argparse


def adaptive_harmonic_extraction(
    audio: np.ndarray,
    f0_trajectory: np.ndarray,
    sr: int = 16000,
    n_harmonics: int = 12,
    q_factor: float = 30.0,  # Higher Q = narrower bands
) -> np.ndarray:
    """Extract voice by tracking harmonics with narrow bandpass filters.

    Uses a bank of adaptive bandpass filters centered on F0 harmonics.
    """
    hop_length = len(audio) // len(f0_trajectory)

    # Interpolate F0 to sample rate
    f0_samples = np.interp(
        np.arange(len(audio)),
        np.arange(len(f0_trajectory)) * hop_length,
        f0_trajectory
    )

    # Process in overlapping frames for smooth transitions
    frame_size = 2048
    hop = frame_size // 4
    output = np.zeros_like(audio)
    window = np.hanning(frame_size)

    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start:start + frame_size] * window
        frame_f0 = np.mean(f0_samples[start:start + frame_size])

        if frame_f0 > 0:
            # Build harmonic comb filter for this frame
            filtered_frame = np.zeros_like(frame)

            for h in range(1, n_harmonics + 1):
                center_freq = frame_f0 * h
                if center_freq > sr * 0.45:  # Stay below Nyquist
                    break

                # Bandwidth based on Q factor
                bandwidth = center_freq / q_factor
                low = max(20, center_freq - bandwidth / 2)
                high = min(sr / 2 - 100, center_freq + bandwidth / 2)

                if low >= high:
                    continue

                # Design narrow bandpass
                try:
                    sos = butter(4, [low, high], btype='band', fs=sr, output='sos')
                    harmonic_component = sosfilt(sos, frame)
                    # Weight by 1/h to emphasize fundamentals
                    filtered_frame += harmonic_component / (h ** 0.5)
                except ValueError:
                    continue

            output[start:start + frame_size] += filtered_frame * window
        else:
            # Unvoiced: pass through with attenuation
            output[start:start + frame_size] += frame * 0.1 * window

    # Normalize overlap-add
    output /= (np.max(np.abs(output)) + 1e-9)
    return output * 0.9


def detect_overlap_regions(f0_male: np.ndarray, f0_female: np.ndarray,
                           hop_length: int, sr: int) -> list:
    """Find time regions where both speakers are active."""
    overlaps = []
    in_overlap = False
    start_frame = 0

    for i in range(len(f0_male)):
        both_active = f0_male[i] > 0 and f0_female[i] > 0
        if both_active and not in_overlap:
            in_overlap = True
            start_frame = i
        elif not both_active and in_overlap:
            in_overlap = False
            start_time = start_frame * hop_length / sr
            end_time = i * hop_length / sr
            overlaps.append((start_time, end_time))

    return overlaps


def hybrid_separation(audio_path: str, sr: int = 16000,
                      n_harmonics: int = 12, q_factor: float = 30.0):
    """Main hybrid separation pipeline."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    print(f"Loaded: {len(y)/sr:.2f}s @ {sr}Hz")

    # Detect F0 in male and female ranges
    print("Detecting F0 trajectories...")
    hop_length = 512

    f0_male, _, _ = librosa.pyin(
        y, fmin=60, fmax=180, sr=sr,
        frame_length=2048, hop_length=hop_length, fill_na=0.0
    )
    f0_female, _, _ = librosa.pyin(
        y, fmin=150, fmax=400, sr=sr,
        frame_length=2048, hop_length=hop_length, fill_na=0.0
    )

    # Stats
    male_active = np.sum(f0_male > 0) / len(f0_male) * 100
    female_active = np.sum(f0_female > 0) / len(f0_female) * 100
    male_median = np.nanmedian(f0_male[f0_male > 0]) if np.any(f0_male > 0) else 0
    female_median = np.nanmedian(f0_female[f0_female > 0]) if np.any(f0_female > 0) else 0

    print(f"  Male F0: median={male_median:.0f}Hz, active={male_active:.1f}%")
    print(f"  Female F0: median={female_median:.0f}Hz, active={female_active:.1f}%")

    # Find overlaps
    overlaps = detect_overlap_regions(f0_male, f0_female, hop_length, sr)
    overlap_duration = sum(e - s for s, e in overlaps)
    print(f"  Overlaps: {len(overlaps)} regions, {overlap_duration:.2f}s total")

    # Adaptive harmonic extraction
    print(f"Extracting harmonics (n={n_harmonics}, Q={q_factor})...")
    male_audio = adaptive_harmonic_extraction(y, f0_male, sr, n_harmonics, q_factor)
    female_audio = adaptive_harmonic_extraction(y, f0_female, sr, n_harmonics, q_factor)

    # Residual (what neither captured)
    residual = y - male_audio - female_audio
    residual = residual / (np.max(np.abs(residual)) + 1e-9) * 0.5

    stats = {
        'male_f0_median': male_median,
        'female_f0_median': female_median,
        'male_active_pct': male_active,
        'female_active_pct': female_active,
        'n_overlaps': len(overlaps),
        'overlap_duration_s': overlap_duration,
        'n_harmonics': n_harmonics,
        'q_factor': q_factor,
    }

    return male_audio, female_audio, residual, sr, stats


def main():
    parser = argparse.ArgumentParser(description='Hybrid voice separation')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('--output-dir', '-o', help='Output directory', default=None)
    parser.add_argument('--n-harmonics', '-n', type=int, default=12, help='Number of harmonics')
    parser.add_argument('--q-factor', '-q', type=float, default=30.0, help='Filter Q factor (higher=narrower)')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / 'hybrid_separated'
    output_dir.mkdir(exist_ok=True)

    # Separate
    male, female, residual, sr, stats = hybrid_separation(
        str(input_path),
        sr=args.sr,
        n_harmonics=args.n_harmonics,
        q_factor=args.q_factor
    )

    # Save outputs
    stem = input_path.stem
    sf.write(output_dir / f'{stem}_male.wav', male, sr)
    sf.write(output_dir / f'{stem}_female.wav', female, sr)
    sf.write(output_dir / f'{stem}_residual.wav', residual, sr)

    print(f"\nSaved to {output_dir}:")
    print(f"  {stem}_male.wav")
    print(f"  {stem}_female.wav")
    print(f"  {stem}_residual.wav")

    # Summary
    print(f"\n=== Hybrid Separation Stats ===")
    print(f"Male:   F0={stats['male_f0_median']:.0f}Hz, active={stats['male_active_pct']:.1f}%")
    print(f"Female: F0={stats['female_f0_median']:.0f}Hz, active={stats['female_active_pct']:.1f}%")
    print(f"Overlaps: {stats['n_overlaps']} regions, {stats['overlap_duration_s']:.2f}s")
    print(f"Params: {stats['n_harmonics']} harmonics, Q={stats['q_factor']}")

    return 0


if __name__ == '__main__':
    exit(main())

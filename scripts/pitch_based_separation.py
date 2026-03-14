#!/usr/bin/env python3
"""Pitch-based voice separation using PYIN + Wiener filtering.

Approach:
1. Detect multiple F0 candidates per frame using librosa.pyin
2. Build harmonic masks for each detected F0
3. Apply Wiener filtering in STFT domain
4. Reconstruct separated audio

Usage:
    python scripts/pitch_based_separation.py validation_video/separation_comparison/e03_20m10s/option_A/Hernán-1-3.vocals_1210-1242_SRC_00.wav
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft, istft
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def detect_dual_pitches(y: np.ndarray, sr: int = 16000,
                        fmin: float = 60, fmax: float = 400,
                        hop_length: int = 512) -> tuple:
    """Detect two simultaneous F0 trajectories using PYIN.

    Returns:
        (speaker1_f0, speaker2_f0, voiced_prob, times)
    """
    # PYIN returns multiple candidates per frame
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
        fill_na=0.0
    )

    # librosa.pyin only returns single f0, not candidates
    # We need a different approach: run pyin on multiple octave ranges

    # Strategy: detect primary F0, then look for secondary in different range
    # Male: 60-180Hz, Female: 150-400Hz

    f0_male, _, vp_male = librosa.pyin(
        y, fmin=60, fmax=180, sr=sr,
        frame_length=2048, hop_length=hop_length, fill_na=0.0
    )

    f0_female, _, vp_female = librosa.pyin(
        y, fmin=150, fmax=400, sr=sr,
        frame_length=2048, hop_length=hop_length, fill_na=0.0
    )

    times = librosa.frames_to_time(np.arange(len(f0_male)), sr=sr, hop_length=hop_length)

    return f0_male, f0_female, vp_male, vp_female, times


def build_harmonic_mask(f0_trajectory: np.ndarray,
                        freqs: np.ndarray,
                        n_time_frames: int,
                        sr: int,
                        n_harmonics: int = 8,
                        bandwidth_hz: float = 50) -> np.ndarray:
    """Build a spectral mask for harmonics of F0.

    Args:
        f0_trajectory: F0 per analysis frame (n_frames,)
        freqs: Frequency bins from STFT (n_freqs,)
        n_time_frames: Number of time frames in STFT
        sr: Sample rate
        n_harmonics: Number of harmonics to include
        bandwidth_hz: Width of each harmonic band

    Returns:
        mask: (n_freqs, n_time_frames) soft mask
    """
    mask = np.zeros((len(freqs), n_time_frames))

    # Interpolate F0 to STFT time grid
    f0_interp = np.interp(
        np.linspace(0, 1, n_time_frames),
        np.linspace(0, 1, len(f0_trajectory)),
        f0_trajectory
    )

    for t in range(n_time_frames):
        f0 = f0_interp[t]
        if f0 <= 0:
            continue

        for h in range(1, n_harmonics + 1):
            harmonic_freq = f0 * h
            if harmonic_freq > sr / 2:
                break

            # Gaussian window around harmonic
            sigma = bandwidth_hz / 2
            harmonic_mask = np.exp(-0.5 * ((freqs - harmonic_freq) / sigma) ** 2)
            mask[:, t] += harmonic_mask

    # Normalize per frame
    mask = np.clip(mask, 0, 1)
    return mask


def separate_voices_pitch(audio_path: str,
                          sr: int = 16000,
                          n_harmonics: int = 8,
                          bandwidth_hz: float = 40) -> tuple:
    """Separate two voices based on pitch.

    Returns:
        (speaker1_audio, speaker2_audio, residual, sr, stats)
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    print(f"Loaded: {len(y)/sr:.2f}s @ {sr}Hz")

    # Detect dual pitches
    print("Detecting F0 trajectories...")
    f0_male, f0_female, vp_male, vp_female, times = detect_dual_pitches(y, sr)

    # Stats
    male_active = np.sum(f0_male > 0) / len(f0_male) * 100
    female_active = np.sum(f0_female > 0) / len(f0_female) * 100
    male_median = np.nanmedian(f0_male[f0_male > 0]) if np.any(f0_male > 0) else 0
    female_median = np.nanmedian(f0_female[f0_female > 0]) if np.any(f0_female > 0) else 0

    print(f"  Male F0: median={male_median:.0f}Hz, active={male_active:.1f}%")
    print(f"  Female F0: median={female_median:.0f}Hz, active={female_active:.1f}%")

    # STFT
    print("Computing STFT...")
    nperseg = 2048
    noverlap = nperseg * 3 // 4
    f, t, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    spec_mag = np.abs(Zxx)
    spec_phase = np.angle(Zxx)

    # Build harmonic masks
    print("Building harmonic masks...")
    mask_male = build_harmonic_mask(f0_male, f, Zxx.shape[1], sr, n_harmonics, bandwidth_hz)
    mask_female = build_harmonic_mask(f0_female, f, Zxx.shape[1], sr, n_harmonics, bandwidth_hz)

    # Wiener filtering: normalize masks
    total_mask = mask_male + mask_female + 1e-8
    mask_male_norm = mask_male / total_mask
    mask_female_norm = mask_female / total_mask

    # Residual mask (what neither captured)
    mask_residual = 1.0 - (mask_male_norm + mask_female_norm)
    mask_residual = np.clip(mask_residual, 0, 1)

    # Apply masks
    print("Applying masks...")
    male_spec = Zxx * mask_male_norm
    female_spec = Zxx * mask_female_norm
    residual_spec = Zxx * mask_residual

    # Reconstruct
    _, male_audio = istft(male_spec, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, female_audio = istft(female_spec, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, residual_audio = istft(residual_spec, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # Normalize
    male_audio = male_audio / (np.max(np.abs(male_audio)) + 1e-9) * 0.9
    female_audio = female_audio / (np.max(np.abs(female_audio)) + 1e-9) * 0.9
    residual_audio = residual_audio / (np.max(np.abs(residual_audio)) + 1e-9) * 0.9

    stats = {
        'male_f0_median': male_median,
        'female_f0_median': female_median,
        'male_active_pct': male_active,
        'female_active_pct': female_active,
        'n_harmonics': n_harmonics,
        'bandwidth_hz': bandwidth_hz,
    }

    return male_audio, female_audio, residual_audio, sr, stats


def plot_separation(original: np.ndarray,
                    male: np.ndarray,
                    female: np.ndarray,
                    residual: np.ndarray,
                    sr: int,
                    output_path: str,
                    stats: dict):
    """Visualize separation results using spectrograms."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    def plot_spec(ax, audio, title):
        ax.specgram(audio, Fs=sr, cmap='magma', NFFT=2048, noverlap=1024)
        ax.set_title(title)
        ax.set_ylabel('Freq (Hz)')
        ax.set_ylim(0, 4000)

    plot_spec(axes[0], original, 'Original (Overlapping)')
    plot_spec(axes[1], male, f'Male Voice (F0≈{stats["male_f0_median"]:.0f}Hz, {stats["male_active_pct"]:.0f}% active)')
    plot_spec(axes[2], female, f'Female Voice (F0≈{stats["female_f0_median"]:.0f}Hz, {stats["female_active_pct"]:.0f}% active)')
    plot_spec(axes[3], residual, 'Residual (Non-harmonic)')
    axes[3].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Pitch-based voice separation')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('--output-dir', '-o', help='Output directory', default=None)
    parser.add_argument('--n-harmonics', '-n', type=int, default=8, help='Number of harmonics')
    parser.add_argument('--bandwidth', '-b', type=float, default=40, help='Harmonic bandwidth Hz')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / 'pitch_separated'
    output_dir.mkdir(exist_ok=True)

    # Separate
    male, female, residual, sr, stats = separate_voices_pitch(
        str(input_path),
        sr=args.sr,
        n_harmonics=args.n_harmonics,
        bandwidth_hz=args.bandwidth
    )

    # Load original for comparison
    original, _ = sf.read(str(input_path))
    if len(original) != len(male):
        # Trim to match
        min_len = min(len(original), len(male))
        original = original[:min_len]
        male = male[:min_len]
        female = female[:min_len]
        residual = residual[:min_len]

    # Save outputs
    stem = input_path.stem
    sf.write(output_dir / f'{stem}_male.wav', male, sr)
    sf.write(output_dir / f'{stem}_female.wav', female, sr)
    sf.write(output_dir / f'{stem}_residual.wav', residual, sr)

    print(f"\nSaved to {output_dir}:")
    print(f"  {stem}_male.wav")
    print(f"  {stem}_female.wav")
    print(f"  {stem}_residual.wav")

    # Plot
    plot_path = output_dir / f'{stem}_separation.png'
    plot_separation(original, male, female, residual, sr, str(plot_path), stats)

    # Summary
    print(f"\n=== Separation Stats ===")
    print(f"Male:   F0={stats['male_f0_median']:.0f}Hz, active={stats['male_active_pct']:.1f}%")
    print(f"Female: F0={stats['female_f0_median']:.0f}Hz, active={stats['female_active_pct']:.1f}%")
    print(f"Params: {stats['n_harmonics']} harmonics, {stats['bandwidth_hz']}Hz bandwidth")

    return 0


if __name__ == '__main__':
    exit(main())

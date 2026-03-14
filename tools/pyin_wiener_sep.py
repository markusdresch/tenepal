"""PYIN + Wiener voice separation — physics-based, no ML.

Approach:
1. Track F0 with PYIN (librosa) for two pitch ranges: male (60-180Hz), female (150-400Hz)
2. Build harmonic masks in STFT domain from tracked F0 trajectories
3. Wiener filtering to separate
4. Output two clean audio streams

Usage:
    python tools/pyin_wiener_sep.py <input.wav> [--out-dir /tmp] [--n-harmonics 8]
"""

import argparse
import numpy as np
import soundfile as sf
import librosa


def track_f0_pyin(audio, sr, fmin, fmax, hop_length=512):
    """Track F0 using PYIN within a pitch range."""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr,
        hop_length=hop_length, fill_na=0.0,
    )
    return f0, voiced_flag, voiced_prob


def build_harmonic_mask(f0_track, sr, n_fft, hop_length, n_harmonics=8,
                        harmonic_width_hz=25.0):
    """Build a soft harmonic mask in STFT domain from an F0 track.

    For each frame where F0 > 0, place Gaussian bumps at f0, 2*f0, ..., n*f0.
    """
    n_frames = len(f0_track)
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    mask = np.zeros((n_bins, n_frames), dtype=np.float32)

    sigma = harmonic_width_hz  # Gaussian width in Hz

    for t in range(n_frames):
        if f0_track[t] <= 0:
            continue
        for h in range(1, n_harmonics + 1):
            center = f0_track[t] * h
            if center > sr / 2:
                break
            # Gaussian bump
            mask[:, t] += np.exp(-0.5 * ((freqs - center) / sigma) ** 2)

    # Clip to [0, 1]
    mask = np.clip(mask, 0, 1)
    return mask


def wiener_separate(stft_mix, mask_a, mask_b, eps=1e-10):
    """Wiener filter: split STFT into two sources using soft masks.

    mask_final_a = mask_a^2 / (mask_a^2 + mask_b^2 + eps)
    """
    power_a = mask_a ** 2
    power_b = mask_b ** 2
    total = power_a + power_b + eps

    wiener_a = power_a / total
    wiener_b = power_b / total

    # Where neither mask claims energy, split 50/50
    unclaimed = (mask_a < 0.01) & (mask_b < 0.01)
    wiener_a[unclaimed] = 0.5
    wiener_b[unclaimed] = 0.5

    src_a = stft_mix * wiener_a
    src_b = stft_mix * wiener_b

    return src_a, src_b, wiener_a, wiener_b


def separate(audio, sr, n_fft=2048, hop_length=512, n_harmonics=8,
             male_range=(60, 180), female_range=(150, 400),
             harmonic_width_hz=25.0):
    """Separate audio into male/female streams using PYIN + Wiener.

    Returns:
        male_audio, female_audio, metadata dict
    """
    # 1. Track F0 for both ranges
    f0_male, voiced_male, prob_male = track_f0_pyin(
        audio, sr, fmin=male_range[0], fmax=male_range[1],
        hop_length=hop_length,
    )
    f0_female, voiced_female, prob_female = track_f0_pyin(
        audio, sr, fmin=female_range[0], fmax=female_range[1],
        hop_length=hop_length,
    )

    # Resolve ambiguous zone (150-180Hz) — assign to whichever has higher voicing prob
    for t in range(len(f0_male)):
        if f0_male[t] > 0 and f0_female[t] > 0:
            # Both tracking — check if in overlap zone
            if abs(f0_male[t] - f0_female[t]) < 40:
                # Too close, assign to higher-confidence tracker
                if prob_male[t] >= prob_female[t]:
                    f0_female[t] = 0
                else:
                    f0_male[t] = 0

    # 2. STFT of mix
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Align F0 track length to STFT frames
    n_stft_frames = stft.shape[1]
    f0_male = f0_male[:n_stft_frames]
    f0_female = f0_female[:n_stft_frames]
    if len(f0_male) < n_stft_frames:
        f0_male = np.pad(f0_male, (0, n_stft_frames - len(f0_male)))
    if len(f0_female) < n_stft_frames:
        f0_female = np.pad(f0_female, (0, n_stft_frames - len(f0_female)))

    # 3. Build harmonic masks
    mask_male = build_harmonic_mask(
        f0_male, sr, n_fft, hop_length,
        n_harmonics=n_harmonics, harmonic_width_hz=harmonic_width_hz,
    )
    mask_female = build_harmonic_mask(
        f0_female, sr, n_fft, hop_length,
        n_harmonics=n_harmonics, harmonic_width_hz=harmonic_width_hz,
    )

    # 4. Wiener filtering
    src_male_stft, src_female_stft, w_male, w_female = wiener_separate(
        stft, mask_male, mask_female,
    )

    # 5. ISTFT
    male_audio = librosa.istft(src_male_stft, hop_length=hop_length, length=len(audio))
    female_audio = librosa.istft(src_female_stft, hop_length=hop_length, length=len(audio))

    # Stats
    male_voiced = np.sum(f0_male > 0)
    female_voiced = np.sum(f0_female > 0)
    male_rms = float(np.sqrt(np.mean(male_audio ** 2)))
    female_rms = float(np.sqrt(np.mean(female_audio ** 2)))

    meta = {
        "male_f0_median": float(np.median(f0_male[f0_male > 0])) if male_voiced else 0,
        "female_f0_median": float(np.median(f0_female[f0_female > 0])) if female_voiced else 0,
        "male_voiced_frames": int(male_voiced),
        "female_voiced_frames": int(female_voiced),
        "total_frames": n_stft_frames,
        "male_rms": male_rms,
        "female_rms": female_rms,
    }

    return male_audio, female_audio, meta


def main():
    parser = argparse.ArgumentParser(description="PYIN+Wiener voice separation")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("--out-dir", default="/tmp", help="Output directory")
    parser.add_argument("--n-harmonics", type=int, default=8)
    parser.add_argument("--harmonic-width", type=float, default=25.0,
                        help="Gaussian width for harmonic bins (Hz)")
    parser.add_argument("--male-lo", type=float, default=60)
    parser.add_argument("--male-hi", type=float, default=180)
    parser.add_argument("--female-lo", type=float, default=150)
    parser.add_argument("--female-hi", type=float, default=400)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    audio, sr = sf.read(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    print(f"  {sr}Hz, {len(audio)/sr:.1f}s")

    print(f"Separating (PYIN+Wiener, {args.n_harmonics} harmonics, "
          f"width={args.harmonic_width}Hz)...")
    male, female, meta = separate(
        audio, sr,
        n_harmonics=args.n_harmonics,
        male_range=(args.male_lo, args.male_hi),
        female_range=(args.female_lo, args.female_hi),
        harmonic_width_hz=args.harmonic_width,
    )

    from pathlib import Path
    stem = Path(args.input).stem
    out_dir = Path(args.out_dir)

    male_path = out_dir / f"{stem}_male.wav"
    female_path = out_dir / f"{stem}_female.wav"
    sf.write(str(male_path), male, sr)
    sf.write(str(female_path), female, sr)

    print(f"\nResults:")
    print(f"  Male:   {male_path}")
    print(f"    F0 median: {meta['male_f0_median']:.0f}Hz, "
          f"voiced: {meta['male_voiced_frames']}/{meta['total_frames']} frames, "
          f"RMS: {meta['male_rms']:.4f}")
    print(f"  Female: {female_path}")
    print(f"    F0 median: {meta['female_f0_median']:.0f}Hz, "
          f"voiced: {meta['female_voiced_frames']}/{meta['total_frames']} frames, "
          f"RMS: {meta['female_rms']:.4f}")

    return male_path, female_path, meta


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visual diagnostics for SepFormer separation output.

Creates:
1) Waveform comparison (original clip vs source 1 vs source 2)
2) Spectrogram per track
3) F0/pitch tracking per track (Parselmouth)
4) RMS energy curves with overlap marking

Usage:
  venv/bin/python scripts/sepformer_visual_analysis.py
  venv/bin/python scripts/sepformer_visual_analysis.py --help
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import parselmouth


def _load_audio(
    path: Path,
    sr: int = 16000,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load mono audio with librosa."""
    y, out_sr = librosa.load(
        str(path),
        sr=sr,
        mono=True,
        offset=0.0 if start_s is None else max(0.0, start_s),
        duration=duration_s,
    )
    return y.astype(np.float32), out_sr


def _extract_f0(y: np.ndarray, sr: int, time_step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Extract pitch (Hz) using Praat/Parselmouth."""
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"].astype(np.float32)
    times = np.asarray([pitch.get_time_from_frame_number(i + 1) for i in range(len(f0))], dtype=np.float32)
    f0[f0 <= 0] = np.nan
    return times, f0


def _rms(y: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """RMS envelope + frame times."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=16000, hop_length=hop_length)
    return t.astype(np.float32), rms.astype(np.float32)


def _speech_mask(rms: np.ndarray, quantile: float = 0.60) -> tuple[np.ndarray, float]:
    """Simple per-track speech activity mask from energy."""
    thr = float(np.quantile(rms, quantile))
    return rms > thr, thr


def _save_html_report(
    html_path: Path,
    png_path: Path,
    original_path: Path,
    s1_path: Path,
    s2_path: Path,
    start_s: float,
    end_s: float,
    overlap_ratio: float,
    corr_s1_s2: float,
    thr_s1: float,
    thr_s2: float,
) -> None:
    """Write compact HTML report linking the generated figure."""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>SepFormer Visual Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #0b1220; color: #e5e7eb; }}
    .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    code {{ background: #1f2937; padding: 2px 6px; border-radius: 6px; }}
    img {{ width: 100%; border: 1px solid #1f2937; border-radius: 10px; }}
    .muted {{ color: #9ca3af; }}
  </style>
</head>
<body>
  <h1>SepFormer Visual Analysis</h1>
  <div class="card">
    <p><strong>Clip:</strong> {start_s:.3f}s - {end_s:.3f}s</p>
    <p><strong>Original:</strong> <code>{original_path}</code></p>
    <p><strong>Source 1:</strong> <code>{s1_path}</code></p>
    <p><strong>Source 2:</strong> <code>{s2_path}</code></p>
    <p><strong>Overlap ratio (energy-based):</strong> {overlap_ratio*100:.2f}%</p>
    <p><strong>Energy correlation S1/S2:</strong> {corr_s1_s2:.3f}</p>
    <p class="muted">RMS thresholds: S1={thr_s1:.6f}, S2={thr_s2:.6f}</p>
  </div>
  <div class="card">
    <img src="{png_path.name}" alt="SepFormer visual analysis figure" />
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual analysis for SepFormer output.")
    parser.add_argument(
        "--original",
        default="validation_video/Hernán-1-3.vocals.wav",
        help="Full original track path.",
    )
    parser.add_argument(
        "--source1",
        default="validation_video/sepformer_test/Hernán-1-3.vocals_1220-1240_SPEAKER_01.wav",
        help="Separated source 1 path.",
    )
    parser.add_argument(
        "--source2",
        default="validation_video/sepformer_test/Hernán-1-3.vocals_1220-1240_SPEAKER_02.wav",
        help="Separated source 2 path.",
    )
    parser.add_argument("--start-s", type=float, default=1220.0, help="Clip start in original.")
    parser.add_argument("--end-s", type=float, default=1240.0, help="Clip end in original.")
    parser.add_argument(
        "--out-dir",
        default="validation_video/sepformer_test/analysis",
        help="Output directory for PNG/HTML.",
    )
    args = parser.parse_args()

    original_path = Path(args.original)
    s1_path = Path(args.source1)
    s2_path = Path(args.source2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not original_path.exists():
        raise FileNotFoundError(f"Original file not found: {original_path}")
    if not s1_path.exists():
        raise FileNotFoundError(f"Source1 file not found: {s1_path}")
    if not s2_path.exists():
        raise FileNotFoundError(f"Source2 file not found: {s2_path}")

    duration = max(0.0, args.end_s - args.start_s)
    if duration <= 0:
        raise ValueError("end-s must be greater than start-s")

    y_orig, sr = _load_audio(original_path, sr=16000, start_s=args.start_s, duration_s=duration)
    y_s1, _ = _load_audio(s1_path, sr=16000)
    y_s2, _ = _load_audio(s2_path, sr=16000)

    n = min(len(y_orig), len(y_s1), len(y_s2))
    y_orig = y_orig[:n]
    y_s1 = y_s1[:n]
    y_s2 = y_s2[:n]
    t = np.arange(n, dtype=np.float32) / float(sr)

    f0_t_orig, f0_orig = _extract_f0(y_orig, sr)
    f0_t_s1, f0_s1 = _extract_f0(y_s1, sr)
    f0_t_s2, f0_s2 = _extract_f0(y_s2, sr)

    e_t, e_orig = _rms(y_orig)
    _, e_s1 = _rms(y_s1)
    _, e_s2 = _rms(y_s2)
    e_n = min(len(e_t), len(e_orig), len(e_s1), len(e_s2))
    e_t = e_t[:e_n]
    e_orig = e_orig[:e_n]
    e_s1 = e_s1[:e_n]
    e_s2 = e_s2[:e_n]

    mask_s1, thr_s1 = _speech_mask(e_s1, quantile=0.60)
    mask_s2, thr_s2 = _speech_mask(e_s2, quantile=0.60)
    overlap_mask = mask_s1 & mask_s2
    overlap_ratio = float(np.mean(overlap_mask)) if len(overlap_mask) else 0.0

    corr_s1_s2 = 0.0
    if np.std(e_s1) > 1e-9 and np.std(e_s2) > 1e-9:
        corr_s1_s2 = float(np.corrcoef(e_s1, e_s2)[0, 1])

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.22)
    titles = ["Original (clip)", "SPEAKER_01", "SPEAKER_02"]
    tracks = [y_orig, y_s1, y_s2]

    # Row 1: waveforms
    for i, (yy, ttl) in enumerate(zip(tracks, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(t, yy, linewidth=0.8)
        ax.set_title(f"Waveform: {ttl}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, duration)

    # Row 2: spectrograms
    for i, (yy, ttl) in enumerate(zip(tracks, titles)):
        ax = fig.add_subplot(gs[1, i])
        stft = librosa.stft(yy, n_fft=1024, hop_length=256, win_length=1024)
        db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        img = librosa.display.specshow(
            db,
            sr=sr,
            hop_length=256,
            x_axis="time",
            y_axis="hz",
            cmap="magma",
            ax=ax,
        )
        ax.set_title(f"Spectrogram: {ttl}")
        ax.set_ylim(0, 5000)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # Row 3: F0 tracking
    ax_f0 = fig.add_subplot(gs[2, :])
    ax_f0.plot(f0_t_orig, f0_orig, label="Original", linewidth=1.0, alpha=0.8)
    ax_f0.plot(f0_t_s1, f0_s1, label="SPEAKER_01", linewidth=1.2, alpha=0.9)
    ax_f0.plot(f0_t_s2, f0_s2, label="SPEAKER_02", linewidth=1.2, alpha=0.9)
    ax_f0.set_title("F0 / Pitch Tracking (Parselmouth)")
    ax_f0.set_xlabel("Time (s)")
    ax_f0.set_ylabel("F0 (Hz)")
    ax_f0.set_xlim(0, duration)
    ax_f0.set_ylim(60, 420)
    ax_f0.legend(loc="upper right")
    ax_f0.grid(alpha=0.25)

    # Row 4: energy + overlap
    ax_e = fig.add_subplot(gs[3, :])
    ax_e.plot(e_t, e_orig, label="Original RMS", alpha=0.7, linewidth=1.0)
    ax_e.plot(e_t, e_s1, label=f"SPEAKER_01 RMS (thr={thr_s1:.4f})", linewidth=1.1)
    ax_e.plot(e_t, e_s2, label=f"SPEAKER_02 RMS (thr={thr_s2:.4f})", linewidth=1.1)
    y_top = float(max(np.max(e_orig), np.max(e_s1), np.max(e_s2)) * 1.05)
    ax_e.fill_between(
        e_t,
        0,
        y_top,
        where=overlap_mask,
        color="crimson",
        alpha=0.16,
        label=f"Overlap marked ({overlap_ratio*100:.1f}%)",
    )
    ax_e.set_title("Energy Curves + Overlap Marking")
    ax_e.set_xlabel("Time (s)")
    ax_e.set_ylabel("RMS Energy")
    ax_e.set_xlim(0, duration)
    ax_e.set_ylim(0, y_top if y_top > 0 else 1.0)
    ax_e.grid(alpha=0.25)
    ax_e.legend(loc="upper right")

    fig.suptitle(
        "SepFormer Separation Diagnostics\n"
        f"Clip {args.start_s:.3f}s-{args.end_s:.3f}s | "
        f"Energy corr(S1,S2)={corr_s1_s2:.3f}",
        fontsize=14,
    )

    png_out = out_dir / "sepformer_visual_analysis_1220_1240.png"
    fig.savefig(png_out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    html_out = out_dir / "sepformer_visual_analysis_1220_1240.html"
    _save_html_report(
        html_path=html_out,
        png_path=png_out,
        original_path=original_path,
        s1_path=s1_path,
        s2_path=s2_path,
        start_s=args.start_s,
        end_s=args.end_s,
        overlap_ratio=overlap_ratio,
        corr_s1_s2=corr_s1_s2,
        thr_s1=thr_s1,
        thr_s2=thr_s2,
    )

    print(f"Saved PNG:  {png_out}")
    print(f"Saved HTML: {html_out}")
    print(f"Overlap ratio: {overlap_ratio*100:.2f}%")
    print(f"Energy correlation (S1/S2): {corr_s1_s2:.3f}")


if __name__ == "__main__":
    main()


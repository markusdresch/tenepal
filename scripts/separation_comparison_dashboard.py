#!/usr/bin/env python3
"""Build an HTML dashboard to compare separation options A/B/C.

Compares:
- Option A: SepFormer -> Pyannote
- Option B: Pyannote -> SepFormer (overlaps only)
- Option C: Cascaded (4 stems)

Per option:
- Waveform view
- Spectrogram (representative track)
- F0 tracking (Parselmouth)
- F0 stability score (higher = more stable)
- IPA transcription snippets from SRT (if available)

Designed to run even when files are missing (framework first).
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import parselmouth


@dataclass
class TrackInfo:
    path: Path
    name: str
    y: np.ndarray
    sr: int
    t: np.ndarray
    f0_t: np.ndarray
    f0: np.ndarray
    f0_stability: float


OPTION_META = {
    "A": {
        "title": "Option A",
        "subtitle": "SepFormer -> Pyannote",
        "hints": ("option_a", "a_", "sepformer_pyannote", "sepformer-to-pyannote"),
    },
    "B": {
        "title": "Option B",
        "subtitle": "Pyannote -> SepFormer (overlaps)",
        "hints": ("option_b", "b_", "pyannote_sepformer", "pyannote-to-sepformer"),
    },
    "C": {
        "title": "Option C",
        "subtitle": "Cascaded (4 stems)",
        "hints": ("option_c", "c_", "cascaded", "4stems", "4_stems"),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Separation comparison dashboard (A/B/C).")
    p.add_argument(
        "--input-dir",
        default="validation_video/separation_comparison",
        help="Directory filled by cc-opus with comparison WAV/SRT files.",
    )
    p.add_argument(
        "--original",
        default="validation_video/Hernán-1-3.vocals.wav",
        help="Original full vocals WAV path.",
    )
    p.add_argument("--start-s", type=float, default=1220.0, help="Clip start in original.")
    p.add_argument("--end-s", type=float, default=1240.0, help="Clip end in original.")
    p.add_argument(
        "--out-dir",
        default="validation_video/separation_comparison/analysis",
        help="Output directory for dashboard assets.",
    )
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    return p.parse_args()


def _normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _discover_option_files(input_dir: Path, option_key: str) -> tuple[list[Path], list[Path]]:
    """Discover WAV and SRT files for one option."""
    hints = OPTION_META[option_key]["hints"]
    wavs: list[Path] = []
    srts: list[Path] = []

    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        pl = _normalize_key(str(p.relative_to(input_dir)))
        if not any(h in pl for h in hints):
            continue
        if p.suffix.lower() == ".wav":
            wavs.append(p)
        elif p.suffix.lower() == ".srt":
            srts.append(p)

    # Fallback for subdirectory naming like input_dir/option_a/*.wav
    for sub in input_dir.iterdir() if input_dir.exists() else []:
        if not sub.is_dir():
            continue
        sl = _normalize_key(sub.name)
        if not any(h in sl for h in hints):
            continue
        wavs.extend([x for x in sub.rglob("*.wav") if x.is_file()])
        srts.extend([x for x in sub.rglob("*.srt") if x.is_file()])

    # Deduplicate + sort
    wavs = sorted(set(wavs))
    srts = sorted(set(srts))
    return wavs, srts


def _load_audio(path: Path, sr: int, offset: float = 0.0, duration: float | None = None) -> tuple[np.ndarray, int]:
    y, out_sr = librosa.load(str(path), sr=sr, mono=True, offset=max(0.0, offset), duration=duration)
    return y.astype(np.float32), out_sr


def _extract_f0(y: np.ndarray, sr: int, time_step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"].astype(np.float32)
    times = np.asarray([pitch.get_time_from_frame_number(i + 1) for i in range(len(f0))], dtype=np.float32)
    f0[f0 <= 0] = np.nan
    return times, f0


def _f0_stability(f0: np.ndarray) -> float:
    valid = f0[np.isfinite(f0) & (f0 > 0)]
    if valid.size < 5:
        return float("nan")
    median_f0 = float(np.median(valid))
    std_f0 = float(np.std(valid))
    if median_f0 <= 1e-9:
        return float("nan")
    return 1.0 - (std_f0 / median_f0)


def _parse_srt_segments(path: Path, limit: int = 20) -> list[dict]:
    """Very simple SRT parser with IPA extraction."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", txt.strip())
    out: list[dict] = []
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        tline = lines[1] if re.search(r"-->", lines[1]) else ""
        payload = " ".join(lines[2:]) if len(lines) > 2 else ""
        ipa = ""
        m = re.search(r"♫fused:\s*([^\n]+)", payload)
        if m:
            ipa = m.group(1).strip()
        text_clean = re.sub(r"♫[^\n]+", "", payload).strip()
        out.append({"time": tline, "text": text_clean, "ipa": ipa})
        if len(out) >= limit:
            break
    return out


def _build_track_infos(wav_paths: Iterable[Path], sr: int) -> list[TrackInfo]:
    tracks: list[TrackInfo] = []
    for p in wav_paths:
        y, out_sr = _load_audio(p, sr=sr)
        if y.size == 0:
            continue
        t = np.arange(len(y), dtype=np.float32) / float(out_sr)
        f0_t, f0 = _extract_f0(y, out_sr)
        st = _f0_stability(f0)
        tracks.append(
            TrackInfo(
                path=p,
                name=p.name,
                y=y,
                sr=out_sr,
                t=t,
                f0_t=f0_t,
                f0=f0,
                f0_stability=st,
            )
        )
    return tracks


def _plot_option_figure(
    option_key: str,
    option_title: str,
    option_subtitle: str,
    original_clip: TrackInfo | None,
    tracks: list[TrackInfo],
    out_png: Path,
) -> None:
    """Create one PNG figure for one option."""
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.30)

    # 1) Waveforms stacked
    ax_w = fig.add_subplot(gs[0, 0])
    offset = 0.0
    step = 1.2
    if original_clip is not None:
        y = original_clip.y / (np.max(np.abs(original_clip.y)) + 1e-9)
        ax_w.plot(original_clip.t, y + offset, linewidth=0.8, label="Original clip")
        ax_w.text(0.005, offset + 0.1, "Original", transform=ax_w.get_yaxis_transform())
        offset += step
    for tr in tracks:
        y = tr.y / (np.max(np.abs(tr.y)) + 1e-9)
        ax_w.plot(tr.t, y + offset, linewidth=0.8, label=tr.name)
        score = "nan" if not np.isfinite(tr.f0_stability) else f"{tr.f0_stability:.3f}"
        ax_w.text(0.005, offset + 0.1, f"{tr.name} | stability={score}", transform=ax_w.get_yaxis_transform())
        offset += step
    ax_w.set_title(f"{option_title} - Waveforms (stacked)")
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Norm amplitude + offset")
    ax_w.grid(alpha=0.2)

    # 2) Spectrogram representative (longest track)
    ax_s = fig.add_subplot(gs[1, 0])
    rep = max(tracks, key=lambda x: len(x.y), default=original_clip)
    if rep is not None:
        stft = librosa.stft(rep.y, n_fft=1024, hop_length=256, win_length=1024)
        db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        img = librosa.display.specshow(
            db,
            sr=rep.sr,
            hop_length=256,
            x_axis="time",
            y_axis="hz",
            cmap="magma",
            ax=ax_s,
        )
        ax_s.set_ylim(0, 5000)
        ax_s.set_title(f"{option_title} - Spectrogram ({rep.name})")
        fig.colorbar(img, ax=ax_s, format="%+2.0f dB")
    else:
        ax_s.text(0.5, 0.5, "No track available", ha="center", va="center")
        ax_s.set_title(f"{option_title} - Spectrogram")

    # 3) F0 track overlay
    ax_f0 = fig.add_subplot(gs[2, 0])
    if original_clip is not None:
        ax_f0.plot(original_clip.f0_t, original_clip.f0, linewidth=1.0, alpha=0.4, label="Original")
    for tr in tracks:
        score = "nan" if not np.isfinite(tr.f0_stability) else f"{tr.f0_stability:.3f}"
        ax_f0.plot(tr.f0_t, tr.f0, linewidth=1.1, alpha=0.9, label=f"{tr.name} ({score})")
    ax_f0.set_title(f"{option_title} - F0 / Pitch")
    ax_f0.set_xlabel("Time (s)")
    ax_f0.set_ylabel("F0 (Hz)")
    ax_f0.set_ylim(60, 420)
    ax_f0.grid(alpha=0.25)
    ax_f0.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{option_title}: {option_subtitle}", fontsize=14)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fmt_score(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.3f}"


def _build_html(
    out_html: Path,
    option_cards: list[dict],
    generated_files: list[Path],
    input_dir: Path,
) -> None:
    cards_html = []
    for c in option_cards:
        ipa_rows = "".join(
            f"<tr><td>{html.escape(seg['time'])}</td><td>{html.escape(seg['text'])}</td><td><code>{html.escape(seg['ipa'])}</code></td></tr>"
            for seg in c["ipa_segments"]
        )
        if not ipa_rows:
            ipa_rows = "<tr><td colspan='3'><em>No SRT/IPA segments found yet.</em></td></tr>"

        track_rows = "".join(
            f"<tr><td>{html.escape(t['name'])}</td><td>{_fmt_score(t['stability'])}</td></tr>"
            for t in c["track_scores"]
        )
        if not track_rows:
            track_rows = "<tr><td colspan='2'><em>No tracks found.</em></td></tr>"

        fig_html = (
            f"<img src='{html.escape(c['png_name'])}' alt='{html.escape(c['title'])} figure' />"
            if c["png_name"]
            else "<div class='warn'>No figure generated (missing audio files).</div>"
        )
        missing_html = ""
        if c["warnings"]:
            missing_html = "<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in c["warnings"]) + "</ul>"

        cards_html.append(
            f"""
            <section class="card">
              <h2>{html.escape(c['title'])}</h2>
              <p class="muted">{html.escape(c['subtitle'])}</p>
              {fig_html}
              <h3>F0 Stability</h3>
              <p><strong>Option mean:</strong> {_fmt_score(c['option_mean'])}</p>
              <table>
                <thead><tr><th>Track</th><th>Stability</th></tr></thead>
                <tbody>{track_rows}</tbody>
              </table>
              <h3>IPA Segments</h3>
              <table>
                <thead><tr><th>Time</th><th>Text</th><th>IPA</th></tr></thead>
                <tbody>{ipa_rows}</tbody>
              </table>
              {missing_html}
            </section>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Separation Comparison Dashboard</title>
  <style>
    body {{
      margin: 20px; background: #0b1220; color: #e5e7eb; font-family: Arial, sans-serif;
    }}
    h1 {{ margin-bottom: 6px; }}
    .muted {{ color: #9ca3af; }}
    .grid {{
      display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 16px; align-items: start;
    }}
    .card {{
      background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 14px;
    }}
    img {{ width: 100%; border-radius: 8px; border: 1px solid #1f2937; }}
    table {{
      width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px;
    }}
    th, td {{ border-bottom: 1px solid #1f2937; text-align: left; padding: 6px; vertical-align: top; }}
    code {{ background: #1f2937; padding: 2px 5px; border-radius: 5px; }}
    .warn {{ color: #fbbf24; padding: 8px 0; }}
    @media (max-width: 1200px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Separation Comparison Dashboard</h1>
  <p class="muted">Input directory: <code>{html.escape(str(input_dir))}</code></p>
  <p class="muted">Framework-ready: works before/after cc-opus WAV drop.</p>
  <div class="grid">
    {''.join(cards_html)}
  </div>
</body>
</html>
"""
    out_html.write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_path = Path(args.original)
    duration = max(0.0, args.end_s - args.start_s)
    if duration <= 0:
        raise ValueError("--end-s must be > --start-s")

    original_clip: TrackInfo | None = None
    if original_path.exists():
        y, sr = _load_audio(original_path, sr=args.sr, offset=args.start_s, duration=duration)
        t = np.arange(len(y), dtype=np.float32) / float(sr)
        f0_t, f0 = _extract_f0(y, sr)
        original_clip = TrackInfo(
            path=original_path,
            name=f"{original_path.name} [{args.start_s:.1f}-{args.end_s:.1f}]",
            y=y,
            sr=sr,
            t=t,
            f0_t=f0_t,
            f0=f0,
            f0_stability=_f0_stability(f0),
        )

    option_cards: list[dict] = []
    generated_pngs: list[Path] = []

    for opt in ("A", "B", "C"):
        meta = OPTION_META[opt]
        wavs, srts = _discover_option_files(input_dir, opt) if input_dir.exists() else ([], [])
        tracks = _build_track_infos(wavs, sr=args.sr)
        option_mean = float(np.nanmean([t.f0_stability for t in tracks])) if tracks else float("nan")
        warnings: list[str] = []
        if not input_dir.exists():
            warnings.append(f"Input directory not found: {input_dir}")
        if not wavs:
            warnings.append("No WAV tracks found for this option yet.")
        if not srts:
            warnings.append("No SRT files found for IPA segment display yet.")

        png_name = ""
        if tracks:
            out_png = out_dir / f"comparison_option_{opt}.png"
            _plot_option_figure(opt, meta["title"], meta["subtitle"], original_clip, tracks, out_png)
            generated_pngs.append(out_png)
            png_name = out_png.name

        ipa_segments: list[dict] = []
        for srt in srts[:3]:
            ipa_segments.extend(_parse_srt_segments(srt, limit=8))
        ipa_segments = ipa_segments[:24]

        option_cards.append(
            {
                "title": f"{meta['title']} ({opt})",
                "subtitle": meta["subtitle"],
                "png_name": png_name,
                "track_scores": [{"name": t.name, "stability": t.f0_stability} for t in tracks],
                "option_mean": option_mean,
                "ipa_segments": ipa_segments,
                "warnings": warnings,
            }
        )

    out_html = out_dir / "separation_comparison_dashboard.html"
    _build_html(out_html=out_html, option_cards=option_cards, generated_files=generated_pngs, input_dir=input_dir)

    print(f"Saved dashboard: {out_html}")
    for p in generated_pngs:
        print(f"Saved figure: {p}")


if __name__ == "__main__":
    main()


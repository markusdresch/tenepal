#!/usr/bin/env python3
"""Full Parselmouth + Allo CTC Debug for First 5 Minutes on Modal GPU.

Comprehensive analysis saving ALL raw data for architecture decisions.

Usage:
    modal run scripts/full_debug_5min.py
"""

import modal
import json
from pathlib import Path

app = modal.App("full-debug-5min")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install(
        "allosaurus",
        "soundfile",
        "numpy",
        "torch",
        "torchaudio",
        "praat-parselmouth",
        "transformers",
        "phonemizer",
    )
)

volume = modal.Volume.from_name("tenepal-models", create_if_missing=True)
CACHE_DIR = "/tenepal-models"


def classify_vowel(f1, f2):
    """Classify vowel from F1/F2 (Spanish 5-vowel system)."""
    if f1 is None or f2 is None:
        return None
    if f1 > 600 and 1000 < f2 < 1500:
        return "a"
    if 400 < f1 < 600 and 1600 < f2 < 2200:
        return "e"
    if f1 < 400 and f2 > 2000:
        return "i"
    if 400 < f1 < 600 and 700 < f2 < 1100:
        return "o"
    if f1 < 400 and f2 < 1000:
        return "u"
    return None


def classify_event(intensity, pitch, f1, f2, cog, voiced):
    """Classify acoustic event type."""
    if intensity is None or intensity < 40:
        return "silence"
    if not voiced:
        if cog and cog > 3000:
            return "fricative"  # /s/, /f/
        if intensity > 55:
            return "burst"  # stop release
        return "closure"  # stop closure
    else:
        if f1 and f1 > 200:
            return "vowel"
        if intensity < 55:
            return "nasal"
        return "voiced"


@app.function(
    image=image,
    gpu="T4",
    volumes={CACHE_DIR: volume},
    timeout=1800,  # 30 min
)
def run_full_analysis(audio_bytes: bytes, segments: list):
    """Run full Parselmouth + Allo CTC analysis on Modal GPU."""
    import numpy as np
    import soundfile as sf
    import torch
    import parselmouth
    import io
    import tempfile

    print("Loading audio...")
    audio_data, audio_sr = sf.read(io.BytesIO(audio_bytes))
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    print(f"Audio: {len(audio_data)/audio_sr:.1f}s, sr={audio_sr}")

    # Initialize Allosaurus
    print("\nInitializing Allosaurus...")
    import allosaurus
    from allosaurus.app import read_recognizer
    from allosaurus.audio import read_audio
    from allosaurus.am.utils import move_to_tensor
    from pathlib import Path as AlloPath

    allo_pkg_dir = AlloPath(allosaurus.__file__).parent / "pretrained"
    allo_cache = Path(CACHE_DIR) / "allosaurus" / "pretrained"
    if allo_cache.exists() and not allo_pkg_dir.exists():
        allo_pkg_dir.symlink_to(allo_cache)

    model = read_recognizer()
    phone_list = model.lm.inventory.unit.id_to_unit
    print(f"Allosaurus ready, {len(phone_list)} phones")

    # Initialize wav2vec2
    print("Initializing wav2vec2...")
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    import torchaudio

    w2v2_model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    w2v2_processor = Wav2Vec2Processor.from_pretrained(w2v2_model_id)
    w2v2_model = Wav2Vec2ForCTC.from_pretrained(w2v2_model_id).to("cuda")
    print("wav2vec2 ready")

    pm_results = []
    allo_results = []
    summary_lines = []

    print(f"\nProcessing {len(segments)} segments...")

    for seg_idx, seg in enumerate(segments):
        cue = seg["cue"]
        start = seg["start"]
        end = seg["end"]
        whisper_text = seg.get("text", "")

        print(f"\n[{seg_idx+1}/{len(segments)}] Cue {cue}: {start:.3f}-{end:.3f}s \"{whisper_text[:30]}...\"")

        # Extract audio chunk
        start_idx = int(start * audio_sr)
        end_idx = int(end * audio_sr)
        chunk = audio_data[start_idx:end_idx]
        duration_ms = len(chunk) / audio_sr * 1000

        if len(chunk) < audio_sr * 0.1:
            print(f"  Skipping: too short ({duration_ms:.0f}ms)")
            continue

        # === PARSELMOUTH ANALYSIS ===
        snd = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(audio_sr))

        # Get acoustic features
        try:
            intensity = snd.to_intensity(time_step=0.01)
            pitch = snd.to_pitch(time_step=0.01)
            formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5)
            spec = snd.to_spectrogram(window_length=0.005, time_step=0.01)
        except Exception as e:
            print(f"  Parselmouth error: {e}")
            continue

        pm_frames = []
        pm_phonemes = []
        prev_vowel = None

        for t in np.arange(0, snd.duration, 0.01):
            if t > snd.duration:
                break

            t_ms = round(t * 1000, 1)

            # Intensity
            int_val = intensity.get_value(t)
            int_db = round(float(int_val), 1) if not np.isnan(int_val) else None

            # Pitch
            pitch_val = pitch.get_value_at_time(t)
            voiced = not np.isnan(pitch_val) and pitch_val > 0
            pitch_hz = round(float(pitch_val), 1) if voiced else None

            # Formants
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            f3 = formants.get_value_at_time(3, t)
            f1_hz = round(float(f1), 1) if not np.isnan(f1) else None
            f2_hz = round(float(f2), 1) if not np.isnan(f2) else None
            f3_hz = round(float(f3), 1) if not np.isnan(f3) else None

            # Spectral CoG
            try:
                spec_slice = spec.to_spectrum_slice(t)
                cog = spec_slice.get_center_of_gravity()
                cog_hz = round(float(cog), 1) if not np.isnan(cog) else None
            except:
                cog_hz = None

            # Event classification
            event = classify_event(int_db, pitch_hz, f1_hz, f2_hz, cog_hz, voiced)

            # Vowel guess
            vowel_guess = classify_vowel(f1_hz, f2_hz) if event == "vowel" else None

            # Track phoneme sequence
            if event == "burst" and prev_vowel != "burst":
                pm_phonemes.append("C")  # Consonant marker
            if vowel_guess and vowel_guess != prev_vowel:
                pm_phonemes.append(vowel_guess)
            prev_vowel = vowel_guess if vowel_guess else event

            pm_frames.append({
                "t_ms": t_ms,
                "intensity_db": int_db,
                "pitch_hz": pitch_hz,
                "f1": f1_hz,
                "f2": f2_hz,
                "f3": f3_hz,
                "voiced": voiced,
                "cog_hz": cog_hz,
                "event": event,
                "vowel_guess": vowel_guess,
            })

        # === ALLOSAURUS CTC ANALYSIS ===
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, audio_sr)
            temp_path = f.name

        try:
            audio_allo = read_audio(temp_path)
            feat = model.pm.compute(audio_allo)
            feats = np.expand_dims(feat, 0)
            feat_len = np.array([feat.shape[0]], dtype=np.int32)

            tensor_feat, tensor_feat_len = move_to_tensor([feats, feat_len], model.config.device_id)

            with torch.no_grad():
                tensor_lprobs = model.am(tensor_feat, tensor_feat_len)

            if model.config.device_id >= 0:
                lprobs = tensor_lprobs.cpu().numpy()[0]
            else:
                lprobs = tensor_lprobs.numpy()[0]

            num_frames = lprobs.shape[0]
            frame_time_ms = duration_ms / num_frames

            # Get decoded output
            decoded = model.recognize(temp_path, lang_id='ipa', topk=1)
        except Exception as e:
            print(f"  Allo error: {e}")
            continue

        # Process CTC frames
        allo_frames = []
        blank_count = 0
        close_calls = 0
        wide_misses = 0

        for frame_idx in range(num_frames):
            frame_lprobs = lprobs[frame_idx]
            t_ms = round(frame_idx * frame_time_ms, 1)

            # Get top 5
            top_indices = np.argsort(frame_lprobs)[::-1][:5]
            top_logits = frame_lprobs[top_indices]
            top_phones = [phone_list.get(idx, f"?{idx}") for idx in top_indices]

            blank_logit = float(frame_lprobs[0])  # Index 0 is <blk>
            best_phone_logit = float(top_logits[0]) if top_phones[0] != "<blk>" else float(top_logits[1])
            margin = blank_logit - best_phone_logit
            blanked = top_phones[0] == "<blk>"

            if blanked:
                blank_count += 1
            if margin < 2.0:
                close_calls += 1
            if margin > 5.0:
                wide_misses += 1

            allo_frames.append({
                "t_ms": t_ms,
                "blank_logit": round(blank_logit, 2),
                "top5": [[top_phones[i], round(float(top_logits[i]), 2)] for i in range(5)],
                "margin": round(margin, 2),
                "blanked": blanked,
            })

        blank_pct = 100 * blank_count / num_frames if num_frames > 0 else 0

        # === WAV2VEC2 ===
        try:
            if audio_sr != 16000:
                chunk_t = torch.tensor(chunk).unsqueeze(0)
                chunk_t = torchaudio.functional.resample(chunk_t, audio_sr, 16000)
                chunk_16k = chunk_t.squeeze(0).numpy()
            else:
                chunk_16k = chunk

            inputs = w2v2_processor(chunk_16k, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                logits = w2v2_model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            w2v2_result = w2v2_processor.batch_decode(predicted_ids)[0]
            w2v2_result = w2v2_result.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
        except Exception as e:
            w2v2_result = f"ERROR:{e}"

        # Store results
        pm_entry = {
            "cue": cue,
            "start": start,
            "end": end,
            "whisper_text": whisper_text,
            "duration_ms": round(duration_ms, 1),
            "frames": pm_frames,
            "parselmouth_phonemes": " ".join(pm_phonemes),
            "w2v2_ipa": w2v2_result,
            "allo_ipa": decoded,
        }
        pm_results.append(pm_entry)

        allo_entry = {
            "cue": cue,
            "start": start,
            "end": end,
            "whisper_text": whisper_text,
            "total_frames": num_frames,
            "blank_frames": blank_count,
            "blank_pct": round(blank_pct, 1),
            "close_calls": close_calls,
            "wide_misses": wide_misses,
            "frames": allo_frames,
            "decoded": decoded,
        }
        allo_results.append(allo_entry)

        # Summary line
        summary_lines.append(f"Cue {cue}: blank={blank_pct:.1f}%, close={close_calls}, decoded=\"{decoded}\" vs w2v2=\"{w2v2_result}\"")

        print(f"  Blank: {blank_pct:.1f}%, Close calls: {close_calls}, Allo: \"{decoded}\"")

    # === CAPABILITY AUDIT ===
    print("\n" + "="*80)
    print("ALLO CAPABILITY AUDIT")
    print("="*80)

    total_segs = len(allo_results)
    avg_blank = sum(r["blank_pct"] for r in allo_results) / total_segs if total_segs else 0
    segs_gt5 = sum(1 for r in allo_results if len(r["decoded"].split()) > 5)
    segs_gt90 = sum(1 for r in allo_results if r["blank_pct"] > 90)

    best_seg = min(allo_results, key=lambda r: r["blank_pct"]) if allo_results else None
    worst_seg = max(allo_results, key=lambda r: r["blank_pct"]) if allo_results else None

    # Count phoneme occurrences
    all_decoded_phones = []
    for r in allo_results:
        all_decoded_phones.extend(r["decoded"].split())

    from collections import Counter
    phone_counts = Counter(all_decoded_phones)
    top_phones = phone_counts.most_common(20)

    audit = {
        "total_segments": total_segs,
        "avg_blank_pct": round(avg_blank, 1),
        "segs_decoded_gt5_phones": segs_gt5,
        "segs_blank_gt90_pct": segs_gt90,
        "best_segment": {"cue": best_seg["cue"], "blank_pct": best_seg["blank_pct"], "decoded": best_seg["decoded"]} if best_seg else None,
        "worst_segment": {"cue": worst_seg["cue"], "blank_pct": worst_seg["blank_pct"], "decoded": worst_seg["decoded"]} if worst_seg else None,
        "top_20_phones": top_phones,
        "unique_phones": list(phone_counts.keys()),
    }

    print(f"Segments analyzed: {total_segs}")
    print(f"Average blank %: {avg_blank:.1f}%")
    print(f"Segments with >5 decoded phones: {segs_gt5}")
    print(f"Segments with >90% blank: {segs_gt90}")
    if best_seg:
        print(f"Best (lowest blank): Cue {best_seg['cue']} - {best_seg['blank_pct']:.1f}%")
    if worst_seg:
        print(f"Worst (highest blank): Cue {worst_seg['cue']} - {worst_seg['blank_pct']:.1f}%")
    print(f"Top phones: {top_phones[:10]}")

    return {
        "pm_results": pm_results,
        "allo_results": allo_results,
        "summary_lines": summary_lines,
        "audit": audit,
    }


@app.local_entrypoint()
def main():
    """Run full analysis on Modal."""
    import re

    vocals_path = Path("validation_video/Hernán-1-1.vocals.wav")
    srt_path = Path("validation_video/Hernán-1-1.srt")

    if not vocals_path.exists():
        print(f"Vocals file not found: {vocals_path}")
        return

    # Parse SRT for first 5 minutes
    print("Parsing SRT for first 5 minutes...")
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r'\n\s*\n', text)

    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            cue = int(lines[0].strip())
        except:
            continue

        ts_match = re.match(r'(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)', lines[1])
        if not ts_match:
            continue

        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, ts_match.groups())
        start = h1*3600 + m1*60 + s1 + ms1/1000
        end = h2*3600 + m2*60 + s2 + ms2/1000

        if start > 300:  # First 5 minutes only
            break

        # Extract text
        content = lines[2] if len(lines) > 2 else ""
        text_match = re.search(r'\](.+)$', content)
        seg_text = text_match.group(1).strip() if text_match else content

        segments.append({
            "cue": cue,
            "start": start,
            "end": end,
            "text": seg_text,
        })

    print(f"Found {len(segments)} segments in first 5 minutes")

    # Load audio
    print(f"Loading vocals: {vocals_path}")
    audio_bytes = vocals_path.read_bytes()
    print(f"Uploading {len(audio_bytes)/1e6:.1f}MB to Modal...")

    # Run analysis
    results = run_full_analysis.remote(audio_bytes, segments)

    # Save outputs
    output_dir = Path("validation_video/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parselmouth JSONL
    pm_path = output_dir / "parselmouth_5min.jsonl"
    with open(pm_path, "w") as f:
        for entry in results["pm_results"]:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved: {pm_path}")

    # 2. Allo CTC JSONL
    allo_path = output_dir / "allo_ctc_5min.jsonl"
    with open(allo_path, "w") as f:
        for entry in results["allo_results"]:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved: {allo_path}")

    # 3. Human-readable summary
    summary_path = output_dir / "debug_5min_summary.txt"
    with open(summary_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("FULL DEBUG SUMMARY - FIRST 5 MINUTES\n")
        f.write("="*80 + "\n\n")

        for pm, allo in zip(results["pm_results"], results["allo_results"]):
            f.write(f"\n{'='*80}\n")
            f.write(f"Cue {pm['cue']}: \"{pm['whisper_text'][:50]}\" ({pm['start']:.3f}-{pm['end']:.3f}s)\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Duration: {pm['duration_ms']:.0f}ms\n")
            f.write(f"Allo decoded: {allo['decoded']}\n")
            f.write(f"w2v2 decoded: {pm['w2v2_ipa']}\n")
            f.write(f"PM phonemes: {pm['parselmouth_phonemes']}\n\n")

            f.write(f"Allo CTC: {allo['blank_pct']:.1f}% blank, {allo['close_calls']} close calls, {allo['wide_misses']} wide misses\n\n")

            # Allo frames (abbreviated)
            f.write("ALLO CTC FRAMES (first 20):\n")
            f.write(f"{'Time':>6} {'Blank':>8} {'Best':>12} {'Margin':>8} {'Close':>6}\n")
            f.write("-"*50 + "\n")
            for frame in allo['frames'][:20]:
                best = frame['top5'][0] if frame['top5'][0][0] != '<blk>' else frame['top5'][1]
                close = "YES" if frame['margin'] < 2.0 else ""
                f.write(f"{frame['t_ms']:>6.0f} {frame['blank_logit']:>8.1f} {best[0]:>12} {frame['margin']:>8.1f} {close:>6}\n")
            if len(allo['frames']) > 20:
                f.write(f"... ({len(allo['frames'])-20} more frames)\n")
            f.write("\n")

            # PM frames (abbreviated)
            f.write("PARSELMOUTH FRAMES (first 20):\n")
            f.write(f"{'Time':>6} {'Int':>6} {'Pitch':>6} {'F1':>6} {'F2':>6} {'Event':>10} {'Vowel':>6}\n")
            f.write("-"*60 + "\n")
            for frame in pm['frames'][:20]:
                int_v = frame['intensity_db'] if frame['intensity_db'] else 0
                pitch_v = frame['pitch_hz'] if frame['pitch_hz'] else 0
                f1_v = frame['f1'] if frame['f1'] else 0
                f2_v = frame['f2'] if frame['f2'] else 0
                vowel = frame['vowel_guess'] if frame['vowel_guess'] else "-"
                f.write(f"{frame['t_ms']:>6.0f} {int_v:>6.0f} {pitch_v:>6.0f} {f1_v:>6.0f} {f2_v:>6.0f} {frame['event']:>10} {vowel:>6}\n")
            if len(pm['frames']) > 20:
                f.write(f"... ({len(pm['frames'])-20} more frames)\n")
            f.write("\n")

    print(f"Saved: {summary_path}")

    # 4. Capability audit
    audit_path = output_dir / "allo_capability_audit.txt"
    audit = results["audit"]
    with open(audit_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("ALLOSAURUS CAPABILITY AUDIT - FIRST 5 MINUTES\n")
        f.write("="*80 + "\n\n")

        f.write(f"Segments analyzed: {audit['total_segments']}\n")
        f.write(f"Average blank %: {audit['avg_blank_pct']:.1f}%\n")
        f.write(f"Segments with >5 decoded phones: {audit['segs_decoded_gt5_phones']}\n")
        f.write(f"Segments with >90% blank: {audit['segs_blank_gt90_pct']}\n\n")

        if audit['best_segment']:
            f.write(f"Best segment (lowest blank):\n")
            f.write(f"  Cue {audit['best_segment']['cue']}: {audit['best_segment']['blank_pct']:.1f}%\n")
            f.write(f"  Decoded: {audit['best_segment']['decoded']}\n\n")

        if audit['worst_segment']:
            f.write(f"Worst segment (highest blank):\n")
            f.write(f"  Cue {audit['worst_segment']['cue']}: {audit['worst_segment']['blank_pct']:.1f}%\n")
            f.write(f"  Decoded: {audit['worst_segment']['decoded']}\n\n")

        f.write("Top 20 phonemes in Allo output:\n")
        for phone, count in audit['top_20_phones']:
            f.write(f"  {phone}: {count}\n")

        f.write(f"\nTotal unique phones decoded: {len(audit['unique_phones'])}\n")
        f.write(f"Phones: {' '.join(audit['unique_phones'][:50])}\n")

    print(f"Saved: {audit_path}")

    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Segments: {audit['total_segments']}")
    print(f"Avg blank: {audit['avg_blank_pct']:.1f}%")
    print(f">90% blank: {audit['segs_blank_gt90_pct']}")
    print(f"Files saved to: {output_dir}/")


if __name__ == "__main__":
    main()

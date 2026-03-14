#!/usr/bin/env python3
"""Extract prosodic features per annotated segment and compare NAH vs SPA.

Usage:
    python tools/prosody_analysis.py validation_video/Hernán-1-3.vocals.wav
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call


def extract_features(snd, start_s: float, end_s: float) -> dict | None:
    """Extract prosodic features from a time window."""
    dur = end_s - start_s
    if dur < 0.15:
        return None

    try:
        window = snd.extract_part(
            start_s, end_s, parselmouth.WindowShape.HAMMING, 1.0, False
        )
    except Exception:
        return None

    # --- F0 (Pitch) ---
    try:
        pitch = window.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0_values = []
        for t in pitch.xs():
            v = pitch.get_value_at_time(t)
            if not np.isnan(v) and v > 0:
                f0_values.append(v)
    except Exception:
        f0_values = []

    if len(f0_values) < 3:
        return None

    f0 = np.array(f0_values)
    f0_mean = float(np.mean(f0))
    f0_std = float(np.std(f0))
    f0_range = float(np.max(f0) - np.min(f0))
    f0_cv = f0_std / f0_mean if f0_mean > 0 else 0  # coefficient of variation

    # F0 slope (linear regression over time)
    t_norm = np.linspace(0, 1, len(f0))
    f0_slope = float(np.polyfit(t_norm, f0, 1)[0])  # semitones-like

    # --- Intensity ---
    try:
        intensity = window.to_intensity(minimum_pitch=75, time_step=0.01)
        int_values = []
        for t in intensity.xs():
            v = intensity.get_value(t)
            if not np.isnan(v):
                int_values.append(v)
    except Exception:
        int_values = []

    if len(int_values) < 3:
        return None

    int_arr = np.array(int_values)
    int_std = float(np.std(int_arr))
    int_range = float(np.max(int_arr) - np.min(int_arr))

    # --- nPVI (normalized Pairwise Variability Index) on intensity ---
    # Measures rhythm: higher = more stress-timed, lower = more syllable-timed
    npvi = 0.0
    if len(int_values) > 1:
        diffs = []
        for i in range(1, len(int_values)):
            a, b = int_values[i - 1], int_values[i]
            avg = (a + b) / 2
            if avg > 0:
                diffs.append(abs(a - b) / avg)
        npvi = float(np.mean(diffs) * 100) if diffs else 0.0

    # --- Voiced/Unvoiced ratio ---
    try:
        pp = window.to_pitch(time_step=0.01)
        n_frames = call(pp, "Get number of frames")
        voiced = 0
        total_frames = 0
        for fi in range(1, n_frames + 1):
            total_frames += 1
            v = call(pp, "Get value in frame", fi, "Hertz")
            if v > 0:
                voiced += 1
        vu_ratio = voiced / total_frames if total_frames > 0 else 0
    except Exception:
        vu_ratio = 0

    # --- HNR (Harmonics-to-Noise Ratio) ---
    try:
        hnr = call(window, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_mean = call(hnr, "Get mean", 0, 0)
        if np.isnan(hnr_mean):
            hnr_mean = 0.0
    except Exception:
        hnr_mean = 0.0

    # --- Speech rate proxy (acoustic events per second) ---
    # Count voiced↔unvoiced transitions
    states = []
    for t in intensity.xs():
        int_val = intensity.get_value(t)
        if np.isnan(int_val) or int_val < 50:
            states.append(0)  # silence
        else:
            pv = pitch.get_value_at_time(t)
            if np.isnan(pv) or pv <= 0:
                states.append(1)  # unvoiced
            else:
                states.append(2)  # voiced
    events = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
    speech_rate = events / dur if dur > 0 else 0

    # --- Jitter and Shimmer ---
    try:
        point_process = call(window, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call(
            [window, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        if np.isnan(jitter):
            jitter = 0.0
        if np.isnan(shimmer):
            shimmer = 0.0
    except Exception:
        jitter = 0.0
        shimmer = 0.0

    return {
        "duration": dur,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_range": f0_range,
        "f0_cv": f0_cv,
        "f0_slope": f0_slope,
        "int_std": int_std,
        "int_range": int_range,
        "npvi": npvi,
        "vu_ratio": vu_ratio,
        "hnr": hnr_mean,
        "speech_rate": speech_rate,
        "jitter": jitter,
        "shimmer": shimmer,
    }


def main(audio_path: str):
    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()
    snd = parselmouth.Sound(audio, sampling_frequency=float(sr))
    print(f"Audio: {len(audio)/sr:.1f}s, {sr}Hz")

    # Load annotations
    db = sqlite3.connect("tools/annotator/annotations.db")
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """
        SELECT cue_index, start_s, end_s, correct_lang, pipeline_lang
        FROM annotations
        WHERE media_file = 'Hernán-1-3.mp4'
        AND cue_index >= 0
        AND correct_lang IS NOT NULL AND correct_lang != ''
        AND segment_id NOT LIKE '_stale_%%'
        AND segment_id NOT LIKE '_sync_%%'
    """
    ).fetchall()
    print(f"Annotations: {len(rows)}")

    # Extract features per segment
    by_lang: dict[str, list[dict]] = {}
    skipped = 0
    for r in rows:
        lang = r["correct_lang"].upper()
        if lang not in ("NAH", "SPA"):
            continue
        feats = extract_features(snd, r["start_s"], r["end_s"])
        if feats is None:
            skipped += 1
            continue
        feats["cue"] = r["cue_index"]
        if lang not in by_lang:
            by_lang[lang] = []
        by_lang[lang].append(feats)

    print(f"Extracted: NAH={len(by_lang.get('NAH', []))}, SPA={len(by_lang.get('SPA', []))}, skipped={skipped}")
    print()

    # Compare distributions
    feature_names = [
        "f0_mean", "f0_std", "f0_range", "f0_cv", "f0_slope",
        "int_std", "int_range", "npvi", "vu_ratio", "hnr",
        "speech_rate", "jitter", "shimmer", "duration",
    ]

    print(f"{'Feature':<16} {'NAH mean':>10} {'NAH std':>10} {'SPA mean':>10} {'SPA std':>10} {'Δ/pooled':>10} {'Separable?':>12}")
    print("-" * 80)

    separable = []
    for fname in feature_names:
        nah_vals = np.array([f[fname] for f in by_lang.get("NAH", [])])
        spa_vals = np.array([f[fname] for f in by_lang.get("SPA", [])])

        if len(nah_vals) < 5 or len(spa_vals) < 5:
            continue

        nah_mean = np.mean(nah_vals)
        nah_std = np.std(nah_vals)
        spa_mean = np.mean(spa_vals)
        spa_std = np.std(spa_vals)

        # Cohen's d (effect size)
        pooled_std = np.sqrt((nah_std**2 + spa_std**2) / 2)
        d = abs(nah_mean - spa_mean) / pooled_std if pooled_std > 0 else 0

        # Interpretation
        if d >= 0.8:
            label = "*** LARGE"
        elif d >= 0.5:
            label = "** MEDIUM"
        elif d >= 0.3:
            label = "* SMALL"
        else:
            label = "negligible"

        separable.append((fname, d, label))
        print(
            f"{fname:<16} {nah_mean:>10.2f} {nah_std:>10.2f} "
            f"{spa_mean:>10.2f} {spa_std:>10.2f} {d:>10.3f} {label:>12}"
        )

    print()
    print("=== Ranking by separability (Cohen's d) ===")
    for fname, d, label in sorted(separable, key=lambda x: -x[1]):
        direction = ""
        nah_m = np.mean([f[fname] for f in by_lang.get("NAH", [])])
        spa_m = np.mean([f[fname] for f in by_lang.get("SPA", [])])
        if nah_m > spa_m:
            direction = "NAH > SPA"
        else:
            direction = "SPA > NAH"
        print(f"  {d:.3f}  {label:<15}  {fname:<16}  {direction}")

    # Percentile overlap analysis for top features
    print()
    print("=== Distribution overlap (10th-90th percentile) ===")
    for fname, d, label in sorted(separable, key=lambda x: -x[1])[:5]:
        nah_vals = np.array([f[fname] for f in by_lang.get("NAH", [])])
        spa_vals = np.array([f[fname] for f in by_lang.get("SPA", [])])
        nah_lo, nah_hi = np.percentile(nah_vals, [10, 90])
        spa_lo, spa_hi = np.percentile(spa_vals, [10, 90])
        # Overlap fraction
        overlap_lo = max(nah_lo, spa_lo)
        overlap_hi = min(nah_hi, spa_hi)
        total_range = max(nah_hi, spa_hi) - min(nah_lo, spa_lo)
        overlap = max(0, overlap_hi - overlap_lo) / total_range if total_range > 0 else 1
        print(f"  {fname:<16}  NAH [{nah_lo:.2f} - {nah_hi:.2f}]  SPA [{spa_lo:.2f} - {spa_hi:.2f}]  overlap={overlap:.0%}")


def classify(audio_path: str):
    """Train and evaluate classifiers on prosodic features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import warnings
    warnings.filterwarnings("ignore")

    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()
    snd = parselmouth.Sound(audio, sampling_frequency=float(sr))

    # Load annotations
    db = sqlite3.connect("tools/annotator/annotations.db")
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """
        SELECT cue_index, start_s, end_s, correct_lang, pipeline_lang
        FROM annotations
        WHERE media_file = 'Hernán-1-3.mp4'
        AND cue_index >= 0
        AND correct_lang IS NOT NULL AND correct_lang != ''
        AND segment_id NOT LIKE '_stale_%%'
        AND segment_id NOT LIKE '_sync_%%'
    """
    ).fetchall()

    # Extract features
    X_all, y_all, cues = [], [], []
    feature_names = [
        "f0_mean", "f0_std", "f0_range", "f0_cv", "f0_slope",
        "int_std", "int_range", "npvi", "vu_ratio", "hnr",
        "speech_rate", "jitter", "shimmer", "duration",
    ]
    skipped = 0
    for r in rows:
        lang = r["correct_lang"].upper()
        if lang not in ("NAH", "SPA"):
            continue
        feats = extract_features(snd, r["start_s"], r["end_s"])
        if feats is None:
            skipped += 1
            continue
        X_all.append([feats[f] for f in feature_names])
        y_all.append(1 if lang == "NAH" else 0)
        cues.append(r["cue_index"])

    X = np.array(X_all)
    y = np.array(y_all)
    print(f"Dataset: {len(X)} segments (NAH={sum(y)}, SPA={len(y)-sum(y)}, skipped={skipped})")
    print()

    # --- Cross-validation with multiple classifiers ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifiers = {
        "LogReg": LogisticRegression(max_iter=1000, C=1.0),
        "LogReg (top5)": LogisticRegression(max_iter=1000, C=1.0),
        "SVM-RBF": SVC(kernel="rbf", C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GBM": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    # Top 5 features by Cohen's d
    top5_idx = [feature_names.index(f) for f in ["hnr", "int_range", "npvi", "int_std", "shimmer"]]

    print(f"{'Classifier':<20} {'Accuracy':>10} {'F1-NAH':>10} {'F1-SPA':>10}")
    print("-" * 55)

    best_name, best_score = "", 0
    for name, clf in classifiers.items():
        if "top5" in name:
            X_used = X[:, top5_idx]
        else:
            X_used = X

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X_used, y, cv=cv, scoring="accuracy")

        # Also get per-class F1 via manual fold iteration
        f1_nah, f1_spa = [], []
        for train_idx, test_idx in cv.split(X_used, y):
            pipe_fold = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe_fold.fit(X_used[train_idx], y[train_idx])
            y_pred = pipe_fold.predict(X_used[test_idx])
            from sklearn.metrics import f1_score
            f1_nah.append(f1_score(y[test_idx], y_pred, pos_label=1))
            f1_spa.append(f1_score(y[test_idx], y_pred, pos_label=0))

        acc = np.mean(scores)
        print(f"{name:<20} {acc:>9.1%} {np.mean(f1_nah):>9.1%} {np.mean(f1_spa):>9.1%}")
        if acc > best_score:
            best_score = acc
            best_name = name

    # --- Feature importance from best model ---
    print()
    print(f"=== Feature importance (RandomForest) ===")
    pipe_rf = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])
    pipe_rf.fit(X, y)
    importances = pipe_rf.named_steps["clf"].feature_importances_
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 100)
        print(f"  {fname:<16} {imp:.3f}  {bar}")

    # --- Confusion matrix from best full model ---
    print()
    print(f"=== Confusion matrix (5-fold aggregated, RandomForest) ===")
    y_pred_all = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y):
        pipe_fold = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ])
        pipe_fold.fit(X[train_idx], y[train_idx])
        y_pred_all[test_idx] = pipe_fold.predict(X[test_idx])

    cm = confusion_matrix(y, y_pred_all)
    print(f"              Predicted")
    print(f"              SPA    NAH")
    print(f"  True SPA   {cm[0,0]:>4}   {cm[0,1]:>4}   (precision SPA: {cm[0,0]/(cm[0,0]+cm[1,0]):.0%})")
    print(f"  True NAH   {cm[1,0]:>4}   {cm[1,1]:>4}   (precision NAH: {cm[1,1]/(cm[0,1]+cm[1,1]):.0%})")
    print(f"  Recall SPA: {cm[0,0]/(cm[0,0]+cm[0,1]):.0%}, Recall NAH: {cm[1,1]/(cm[1,0]+cm[1,1]):.0%}")

    # --- Error analysis: which segments are hardest? ---
    print()
    print("=== Misclassified segments (prosody disagrees with annotation) ===")
    misclassified_nah = []  # True NAH predicted SPA
    misclassified_spa = []  # True SPA predicted NAH
    for i in range(len(y)):
        if y[i] != y_pred_all[i]:
            label = "NAH" if y[i] == 1 else "SPA"
            pred = "NAH" if y_pred_all[i] == 1 else "SPA"
            if y[i] == 1:
                misclassified_nah.append(cues[i])
            else:
                misclassified_spa.append(cues[i])
    print(f"  True NAH predicted SPA: {len(misclassified_nah)} segments")
    print(f"  True SPA predicted NAH: {len(misclassified_spa)} segments")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    audio = sys.argv[2] if len(sys.argv) > 2 else "validation_video/Hernán-1-3.vocals.wav"

    if mode == "classify":
        classify(audio)
    elif mode == "analyze":
        if len(sys.argv) > 1 and sys.argv[1].endswith(".wav"):
            main(sys.argv[1])
        else:
            main(audio)
    else:
        main(mode)  # treat as audio path

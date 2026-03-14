#!/usr/bin/env python3
"""
Prototype: Acoustic Ejective Detection for Maya Language ID

Three approaches compared:
1. Heuristic - VOT + burst intensity thresholds
2. Parselmouth + sklearn - Feature extraction + anomaly detection
3. wav2vec2 - Pretrained embeddings + clustering

Without ground truth labels, we use unsupervised methods to find
"ejective-like" stop consonants and compare detection patterns.
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

warnings.filterwarnings("ignore")

# ============================================================================
# Data structures
# ============================================================================

@dataclass
class StopCandidate:
    """A detected stop consonant candidate."""
    start_time: float
    end_time: float
    burst_time: float
    closure_duration: float  # ms
    vot: float  # voice onset time in ms
    burst_intensity: float  # dB
    burst_relative_intensity: float  # dB above context
    has_creak: bool
    f0_perturbation: float  # Hz drop after burst

    # Classification results
    heuristic_ejective: bool = False
    sklearn_ejective: bool = False
    w2v2_ejective: bool = False


@dataclass
class DetectionResult:
    """Results from one detection approach."""
    approach: str
    candidates: list[StopCandidate]
    ejective_count: int
    total_stops: int
    processing_time_ms: float

    @property
    def ejective_ratio(self) -> float:
        return self.ejective_count / max(1, self.total_stops)


# ============================================================================
# Approach 1: Heuristic Detection
# ============================================================================

class HeuristicDetector:
    """
    Simple threshold-based ejective detection.

    Ejective indicators:
    - Long closure duration (>80ms typical for ejectives)
    - High burst intensity (louder release)
    - VOT pattern (short positive VOT for voiceless ejectives)
    - F0 perturbation after release
    """

    # Thresholds tuned for ejective detection
    MIN_CLOSURE_MS = 60  # Ejectives tend to have longer closures
    MIN_BURST_RELATIVE_DB = 6  # Burst should be notably louder than context
    MAX_VOT_MS = 40  # Ejectives have short positive VOT
    MIN_F0_DROP_HZ = 10  # Pitch often drops after ejective

    def __init__(self, audio_path: str):
        self.sound = parselmouth.Sound(audio_path)
        self.intensity = self.sound.to_intensity(minimum_pitch=75)
        self.pitch = self.sound.to_pitch_cc(time_step=0.01)

    def find_stop_candidates(self) -> list[StopCandidate]:
        """Find potential stop consonants via derivative-based intensity analysis."""
        candidates = []

        # Get intensity contour with fine time resolution
        intensity = self.sound.to_intensity(minimum_pitch=50, time_step=0.005)
        times = intensity.xs()
        values = np.array([intensity.get_value(t) or 0 for t in times])

        # Compute derivative (rate of intensity change)
        dt = times[1] - times[0] if len(times) > 1 else 0.005
        derivative = np.gradient(values, dt)

        # Find rapid drops followed by rapid rises (closure → burst pattern)
        drop_threshold = -80  # dB/s (negative = drop)
        rise_threshold = 80   # dB/s (positive = rise)
        min_gap = 0.015       # Minimum 15ms between drop and rise
        max_gap = 0.120       # Maximum 120ms gap (closure duration)

        raw_candidates = []
        i = 0
        while i < len(derivative) - 5:
            # Look for significant drop
            if derivative[i] < drop_threshold:
                drop_time = times[i]
                drop_intensity = values[i]

                # Search forward for rise within closure window
                for j in range(i + 3, min(i + int(max_gap / dt), len(derivative))):
                    if derivative[j] > rise_threshold:
                        rise_time = times[j]
                        gap = rise_time - drop_time

                        if gap >= min_gap:
                            raw_candidates.append({
                                'drop_time': drop_time,
                                'rise_time': rise_time,
                                'drop_intensity': drop_intensity,
                                'rise_intensity': values[j],
                                'gap_ms': gap * 1000
                            })
                            i = j  # Skip past this candidate
                            break
                    # Stop if intensity drops too low (silence, not a stop)
                    if values[j] < 10:
                        break
            i += 1

        # Deduplicate: merge candidates within 50ms
        if not raw_candidates:
            return []

        deduped = [raw_candidates[0]]
        for c in raw_candidates[1:]:
            if c['drop_time'] - deduped[-1]['drop_time'] > 0.050:
                deduped.append(c)

        # Convert to StopCandidate objects
        for rc in deduped:
            candidate = self._analyze_candidate(
                rc['drop_time'],
                rc['drop_time'] + rc['gap_ms'] / 2000,  # Midpoint as closure end
                rc['rise_time'],
                values,
                times
            )
            if candidate:
                candidates.append(candidate)

        return candidates

    def _analyze_candidate(
        self,
        closure_start: float,
        closure_end: float,
        burst_time: float,
        intensity_values: np.ndarray,
        times: np.ndarray
    ) -> Optional[StopCandidate]:
        """Extract detailed features from a stop candidate."""

        closure_duration = (closure_end - closure_start) * 1000  # ms

        # Get burst intensity
        burst_intensity = self.intensity.get_value(burst_time)
        if burst_intensity is None:
            return None

        # Get context intensity (100ms window before closure)
        context_start = max(0, closure_start - 0.1)
        context_intensities = [
            self.intensity.get_value(t)
            for t in np.linspace(context_start, closure_start, 10)
        ]
        context_intensities = [v for v in context_intensities if v is not None]
        context_mean = np.mean(context_intensities) if context_intensities else 60

        burst_relative = burst_intensity - context_mean

        # Estimate VOT (time from burst to voicing onset)
        # Simplified: look for pitch onset after burst
        vot = 0
        for dt in np.arange(0.005, 0.060, 0.005):
            t = burst_time + dt
            if t < self.sound.xmax:
                f0 = self.pitch.get_value_at_time(t)
                if f0 is not None and f0 > 0:
                    vot = dt * 1000  # ms
                    break

        # Check for creak (irregular pitch periods) - simplified
        has_creak = False
        f0_values = []
        for dt in np.arange(-0.02, 0.02, 0.005):
            t = burst_time + dt
            if 0 < t < self.sound.xmax:
                f0 = self.pitch.get_value_at_time(t)
                if f0 is not None:
                    f0_values.append(f0)

        if len(f0_values) >= 3:
            f0_std = np.std(f0_values)
            has_creak = f0_std > 20  # High variability suggests creak

        # F0 perturbation (drop after burst)
        f0_before = self.pitch.get_value_at_time(closure_start)
        f0_after = self.pitch.get_value_at_time(burst_time + 0.03)
        f0_perturbation = 0
        if f0_before and f0_after:
            f0_perturbation = f0_before - f0_after

        return StopCandidate(
            start_time=closure_start,
            end_time=burst_time + 0.05,
            burst_time=burst_time,
            closure_duration=closure_duration,
            vot=vot,
            burst_intensity=burst_intensity,
            burst_relative_intensity=burst_relative,
            has_creak=has_creak,
            f0_perturbation=f0_perturbation
        )

    def classify(self, candidates: list[StopCandidate]) -> list[StopCandidate]:
        """Apply heuristic rules to classify ejectives."""
        for c in candidates:
            score = 0

            # Long closure
            if c.closure_duration >= self.MIN_CLOSURE_MS:
                score += 1

            # High burst intensity
            if c.burst_relative_intensity >= self.MIN_BURST_RELATIVE_DB:
                score += 1

            # Short VOT (typical for ejectives)
            if 0 < c.vot <= self.MAX_VOT_MS:
                score += 1

            # F0 perturbation
            if c.f0_perturbation >= self.MIN_F0_DROP_HZ:
                score += 1

            # Creak indicator
            if c.has_creak:
                score += 1

            # Need at least 3 indicators
            c.heuristic_ejective = score >= 3

        return candidates


# ============================================================================
# Approach 2: Parselmouth Features + sklearn
# ============================================================================

class SklearnDetector:
    """
    Extract acoustic features and use anomaly detection.

    Ejectives are "unusual" stops - we can find them via:
    - Isolation Forest (anomaly detection)
    - Or One-Class SVM

    No ground truth needed - just finds outlier stop consonants.
    """

    def __init__(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=0.15,  # Expect ~15% of stops to be ejectives
            random_state=42,
            n_estimators=100
        )

    def extract_features(self, candidates: list[StopCandidate]) -> np.ndarray:
        """Convert candidates to feature matrix."""
        features = []
        for c in candidates:
            features.append([
                c.closure_duration,
                c.vot,
                c.burst_intensity,
                c.burst_relative_intensity,
                1.0 if c.has_creak else 0.0,
                c.f0_perturbation
            ])
        return np.array(features)

    def classify(self, candidates: list[StopCandidate]) -> list[StopCandidate]:
        """Use isolation forest to find outlier (ejective-like) stops."""
        if len(candidates) < 5:
            # Not enough data for meaningful ML
            return candidates

        features = self.extract_features(candidates)

        # Normalize features
        features_scaled = self.scaler.fit_transform(features)

        # Fit and predict (-1 = outlier/anomaly = potential ejective)
        predictions = self.model.fit_predict(features_scaled)

        for i, c in enumerate(candidates):
            c.sklearn_ejective = predictions[i] == -1

        return candidates


# ============================================================================
# Approach 3: wav2vec2 Embeddings
# ============================================================================

class Wav2Vec2Detector:
    """
    Use pretrained wav2vec2 to get neural embeddings of stop regions,
    then cluster to find distinctive (ejective-like) patterns.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            import torch

            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            self.model.eval()
            self.torch = torch
            self.available = True
        except Exception as e:
            print(f"  [w2v2] Model load failed: {e}")
            self.available = False

    def extract_embeddings(
        self,
        audio: np.ndarray,
        sr: int,
        candidates: list[StopCandidate]
    ) -> Optional[np.ndarray]:
        """Extract wav2vec2 embeddings for each stop region."""
        if not self.available:
            return None

        embeddings = []

        for c in candidates:
            # Extract audio chunk around the stop (100ms window)
            start_sample = max(0, int((c.burst_time - 0.05) * sr))
            end_sample = min(len(audio), int((c.burst_time + 0.05) * sr))
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.02:  # Too short
                embeddings.append(np.zeros(768))
                continue

            # Process through wav2vec2
            inputs = self.processor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            with self.torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pool over time dimension
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)

        return np.array(embeddings)

    def classify(
        self,
        candidates: list[StopCandidate],
        audio: np.ndarray,
        sr: int
    ) -> list[StopCandidate]:
        """Use embeddings + clustering to find ejective-like stops."""
        if not self.available or len(candidates) < 5:
            return candidates

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        embeddings = self.extract_embeddings(audio, sr, candidates)
        if embeddings is None:
            return candidates

        # Reduce dimensionality and cluster
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # Use K-means with 2 clusters (ejective vs non-ejective)
        # The smaller cluster is likely ejectives
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_scaled)

        # Smaller cluster = ejectives (assuming ejectives are minority)
        cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
        ejective_cluster = 0 if cluster_sizes[0] < cluster_sizes[1] else 1

        for i, c in enumerate(candidates):
            c.w2v2_ejective = labels[i] == ejective_cluster

        return candidates


# ============================================================================
# Main comparison runner
# ============================================================================

def process_audio(audio_path: str, verbose: bool = True) -> dict:
    """Run all three approaches on an audio file."""
    import time

    results = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {Path(audio_path).name}")
        print(f"{'='*60}")

    # Load audio for wav2vec2
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Mono

    # ---- Approach 1: Heuristic ----
    if verbose:
        print("\n[1] Heuristic Detection...")
    t0 = time.time()

    heuristic = HeuristicDetector(audio_path)
    candidates = heuristic.find_stop_candidates()
    candidates = heuristic.classify(candidates)

    heuristic_time = (time.time() - t0) * 1000
    heuristic_ejectives = sum(1 for c in candidates if c.heuristic_ejective)

    results["heuristic"] = DetectionResult(
        approach="heuristic",
        candidates=candidates,
        ejective_count=heuristic_ejectives,
        total_stops=len(candidates),
        processing_time_ms=heuristic_time
    )

    if verbose:
        print(f"    Found {len(candidates)} stop candidates")
        print(f"    Classified {heuristic_ejectives} as ejective-like")
        print(f"    Time: {heuristic_time:.1f}ms")

    # ---- Approach 2: sklearn ----
    if verbose:
        print("\n[2] Sklearn Anomaly Detection...")
    t0 = time.time()

    sklearn_detector = SklearnDetector()
    candidates = sklearn_detector.classify(candidates)

    sklearn_time = (time.time() - t0) * 1000
    sklearn_ejectives = sum(1 for c in candidates if c.sklearn_ejective)

    results["sklearn"] = DetectionResult(
        approach="sklearn",
        candidates=candidates,
        ejective_count=sklearn_ejectives,
        total_stops=len(candidates),
        processing_time_ms=sklearn_time
    )

    if verbose:
        print(f"    Classified {sklearn_ejectives} as ejective-like (anomalies)")
        print(f"    Time: {sklearn_time:.1f}ms")

    # ---- Approach 3: wav2vec2 ----
    if verbose:
        print("\n[3] Wav2Vec2 Embedding Clustering...")
    t0 = time.time()

    w2v2_detector = Wav2Vec2Detector()
    candidates = w2v2_detector.classify(candidates, audio, sr)

    w2v2_time = (time.time() - t0) * 1000
    w2v2_ejectives = sum(1 for c in candidates if c.w2v2_ejective)

    results["wav2vec2"] = DetectionResult(
        approach="wav2vec2",
        candidates=candidates,
        ejective_count=w2v2_ejectives,
        total_stops=len(candidates),
        processing_time_ms=w2v2_time
    )

    if verbose:
        print(f"    Classified {w2v2_ejectives} as ejective-like (minority cluster)")
        print(f"    Time: {w2v2_time:.1f}ms")

    # ---- Agreement analysis ----
    if verbose and len(candidates) > 0:
        print("\n[Agreement Analysis]")

        all_three = sum(1 for c in candidates
                       if c.heuristic_ejective and c.sklearn_ejective and c.w2v2_ejective)
        at_least_two = sum(1 for c in candidates
                         if sum([c.heuristic_ejective, c.sklearn_ejective, c.w2v2_ejective]) >= 2)

        print(f"    All 3 agree (ejective): {all_three}")
        print(f"    At least 2 agree: {at_least_two}")

        # Show some example candidates
        print("\n[Top Candidates (all 3 agree or 2+ indicators)]")
        for i, c in enumerate(candidates[:10]):
            votes = sum([c.heuristic_ejective, c.sklearn_ejective, c.w2v2_ejective])
            if votes >= 2:
                print(f"    {c.burst_time:.2f}s: closure={c.closure_duration:.0f}ms, "
                      f"burst_rel={c.burst_relative_intensity:.1f}dB, "
                      f"VOT={c.vot:.0f}ms, votes={votes}/3")

    results["candidates"] = candidates
    return results


def run_comparison(sample_dir: str, use_vocals: bool = True):
    """Run comparison on all Maya samples."""

    sample_path = Path(sample_dir)

    # Find wav files
    if use_vocals:
        wav_files = sorted(sample_path.glob("*.vocals.wav"))
    else:
        wav_files = [f for f in sorted(sample_path.glob("*.wav"))
                     if ".vocals." not in f.name]

    if not wav_files:
        print(f"No wav files found in {sample_dir}")
        return

    print(f"\nFound {len(wav_files)} audio files")
    print(f"Using: {'vocals (isolated)' if use_vocals else 'original audio'}")

    all_results = {}

    for wav_file in wav_files:
        results = process_audio(str(wav_file))
        all_results[wav_file.name] = results

    # ---- Summary table ----
    print("\n" + "="*80)
    print("SUMMARY: Ejective Detection Comparison")
    print("="*80)

    print(f"\n{'File':<25} {'Stops':>6} {'Heur':>6} {'SKL':>6} {'W2V2':>6} {'2+':>6}")
    print("-"*60)

    totals = {"stops": 0, "heur": 0, "skl": 0, "w2v2": 0, "agree": 0}

    for filename, results in all_results.items():
        candidates = results["candidates"]
        heur = results["heuristic"].ejective_count
        skl = results["sklearn"].ejective_count
        w2v2 = results["wav2vec2"].ejective_count
        agree = sum(1 for c in candidates
                   if sum([c.heuristic_ejective, c.sklearn_ejective, c.w2v2_ejective]) >= 2)

        short_name = filename.replace(".vocals.wav", "").replace(".wav", "")
        print(f"{short_name:<25} {len(candidates):>6} {heur:>6} {skl:>6} {w2v2:>6} {agree:>6}")

        totals["stops"] += len(candidates)
        totals["heur"] += heur
        totals["skl"] += skl
        totals["w2v2"] += w2v2
        totals["agree"] += agree

    print("-"*60)
    print(f"{'TOTAL':<25} {totals['stops']:>6} {totals['heur']:>6} "
          f"{totals['skl']:>6} {totals['w2v2']:>6} {totals['agree']:>6}")

    # ---- Timing comparison ----
    print("\n" + "-"*60)
    print("Processing Time (ms per file, average)")
    print("-"*60)

    times = {"heur": [], "skl": [], "w2v2": []}
    for results in all_results.values():
        times["heur"].append(results["heuristic"].processing_time_ms)
        times["skl"].append(results["sklearn"].processing_time_ms)
        times["w2v2"].append(results["wav2vec2"].processing_time_ms)

    print(f"  Heuristic:  {np.mean(times['heur']):>8.1f}ms")
    print(f"  Sklearn:    {np.mean(times['skl']):>8.1f}ms")
    print(f"  Wav2Vec2:   {np.mean(times['w2v2']):>8.1f}ms")

    # ---- Recommendations ----
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print("""
Based on these results:

1. HEURISTIC: Fast, interpretable, but may over/under-detect based on thresholds.
   Best for: Quick baseline, real-time processing.

2. SKLEARN: Finds statistical outliers in acoustic feature space.
   Best for: When ejectives are truly "unusual" relative to other stops.

3. WAV2VEC2: Neural embeddings capture complex acoustic patterns.
   Best for: Highest accuracy potential, but needs GPU for speed.

RECOMMENDED APPROACH:
- Use heuristic as fast pre-filter (real-time capable)
- Use sklearn for batch validation
- Use wav2vec2 voting only when 2+ methods disagree

For Maya language ID integration:
- Count "2+ agreement" ejectives per segment
- Add as weighted feature to language profile (e.g., 3+ ejectives → strong Maya indicator)
""")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototype ejective detection comparison")
    parser.add_argument("--sample-dir", default="validation_video/maya_samples/",
                       help="Directory with Maya audio samples")
    parser.add_argument("--original", action="store_true",
                       help="Use original audio instead of vocal-isolated")
    parser.add_argument("--single", type=str,
                       help="Process single file only")

    args = parser.parse_args()

    if args.single:
        process_audio(args.single, verbose=True)
    else:
        run_comparison(args.sample_dir, use_vocals=not args.original)

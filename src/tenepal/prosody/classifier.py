"""Prosodic language profile classifier.

Scores ProsodyFeatures against language profiles to produce per-language
confidence scores, analogous to phoneme marker detection.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .extractor import ProsodyFeatures

logger = logging.getLogger(__name__)


@dataclass
class ProsodyProfile:
    """Prosodic profile for a language.

    Attributes:
        code: Language code (e.g., "spa", "eng")
        name: Language name
        rhythm_class: "syllable-timed", "stress-timed", or "stress-accent"
        f0_range: {"min": float, "max": float, "target": float} - F0 range in Hz
        f0_std: {"min": float, "max": float, "target": float} - F0 std deviation in Hz
        npvi_v: {"min": float, "max": float, "target": float} - Vocalic nPVI
        speech_rate: {"min": float, "max": float, "target": float} - Syllables/sec
        weight: Confidence weight (0-1), e.g., 0.5 for estimated profiles
    """

    code: str
    name: str
    rhythm_class: str
    f0_range: dict[str, float]
    f0_std: dict[str, float]
    npvi_v: dict[str, float]
    speech_rate: dict[str, float]
    weight: float


def load_prosody_profile(path: Path) -> ProsodyProfile:
    """Load a single ProsodyProfile from a JSON config file.

    Args:
        path: Path to the JSON profile file

    Returns:
        ProsodyProfile instance loaded from JSON

    Raises:
        FileNotFoundError: If the profile file doesn't exist
        json.JSONDecodeError: If the JSON is malformed
        KeyError: If required fields are missing
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ProsodyProfile(
        code=data["code"],
        name=data["name"],
        rhythm_class=data["rhythm_class"],
        f0_range=data["f0_range"],
        f0_std=data["f0_std"],
        npvi_v=data["npvi_v"],
        speech_rate=data["speech_rate"],
        weight=data["weight"],
    )


def load_prosody_profiles(directory: Path | None = None) -> list[ProsodyProfile]:
    """Load all prosodic profiles from the profiles directory.

    Args:
        directory: Path to profiles directory. If None, uses built-in default.

    Returns:
        List of ProsodyProfile objects, sorted by language code

    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    if directory is None:
        # Use the package's built-in profiles directory
        directory = Path(__file__).parent / "profiles"

    if not directory.exists():
        raise FileNotFoundError(f"Prosody profiles directory not found: {directory}")

    profiles = []
    for json_path in sorted(directory.glob("*.json")):
        profile = load_prosody_profile(json_path)
        profiles.append(profile)

    return sorted(profiles, key=lambda p: p.code)


def score_prosody_profile(
    features: ProsodyFeatures, profile: ProsodyProfile
) -> float:
    """Score ProsodyFeatures against a prosodic language profile.

    Args:
        features: Extracted prosodic features from audio segment
        profile: Target language prosodic profile

    Returns:
        Normalized score in [0, 1], weighted by profile.weight

    Algorithm:
        For each feature dimension (f0_range, f0_std, npvi_v, speech_rate):
        1. Compute normalized distance: abs(observed - target) / (max - min)
        2. Clamp distance to [0, 1]
        3. Mean distance across all dimensions
        4. Convert to similarity: score = 1.0 - mean_distance
        5. Multiply by profile.weight (reduces score for low-confidence profiles)
    """
    distances = []

    # F0 range distance
    f0_range_val = profile.f0_range["max"] - profile.f0_range["min"]
    if f0_range_val > 0:
        f0_range_distance = abs(features.f0_range - profile.f0_range["target"]) / f0_range_val
        distances.append(min(1.0, f0_range_distance))

    # F0 std distance
    f0_std_val = profile.f0_std["max"] - profile.f0_std["min"]
    if f0_std_val > 0:
        f0_std_distance = abs(features.f0_std - profile.f0_std["target"]) / f0_std_val
        distances.append(min(1.0, f0_std_distance))

    # nPVI distance
    npvi_range = profile.npvi_v["max"] - profile.npvi_v["min"]
    if npvi_range > 0:
        npvi_distance = abs(features.npvi_v - profile.npvi_v["target"]) / npvi_range
        distances.append(min(1.0, npvi_distance))

    # Speech rate distance
    rate_range = profile.speech_rate["max"] - profile.speech_rate["min"]
    if rate_range > 0:
        rate_distance = abs(features.speech_rate - profile.speech_rate["target"]) / rate_range
        distances.append(min(1.0, rate_distance))

    # Mean normalized distance
    if not distances:
        return 0.0

    mean_distance = sum(distances) / len(distances)

    # Convert distance to similarity score
    score = max(0.0, 1.0 - mean_distance)

    # Apply profile weight (e.g., NAH at 0.5 contributes less)
    return score * profile.weight


def score_all_profiles(
    features: ProsodyFeatures, profiles: list[ProsodyProfile] | None = None
) -> dict[str, float]:
    """Score ProsodyFeatures against all prosodic language profiles.

    Args:
        features: Extracted prosodic features from audio segment
        profiles: List of profiles to score against. If None, loads defaults.

    Returns:
        Dictionary mapping language codes to prosodic scores [0, 1]

    Example:
        >>> features = extract_prosody(audio_data, 22050)
        >>> scores = score_all_profiles(features)
        >>> scores
        {'deu': 0.45, 'eng': 0.52, 'fra': 0.31, ...}
    """
    if profiles is None:
        profiles = load_prosody_profiles()

    scores = {}
    for profile in profiles:
        score = score_prosody_profile(features, profile)
        scores[profile.code] = score

    return scores

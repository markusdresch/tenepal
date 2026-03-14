"""Profile loader for language detection.

Loads LanguageProfile objects from external JSON configuration files.
This enables semi-automated tuning of marker weights and thresholds.
"""

import json
from pathlib import Path

from tenepal.language.registry import LanguageProfile


def load_profile(path: Path) -> LanguageProfile:
    """Load a single LanguageProfile from a JSON config file.

    Args:
        path: Path to the JSON config file

    Returns:
        LanguageProfile instance loaded from the JSON config

    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the JSON is malformed
        KeyError: If required fields are missing
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert lists to sets
    marker_phonemes = set(data["marker_phonemes"])
    absent_phonemes = set(data["absent_phonemes"])

    # Convert marker_sequences from list of 2-element lists to set of tuples
    marker_sequences = {tuple(seq) for seq in data.get("marker_sequences", [])}

    # Load weights dicts
    marker_weights = data.get("marker_weights", {})

    # Convert sequence_weights keys from lists to tuples
    sequence_weights_raw = data.get("sequence_weights", {})
    sequence_weights = {}
    for key, value in sequence_weights_raw.items():
        # JSON keys are always strings, so we need to parse them
        # Format: "[phoneme1, phoneme2]" or tuple serialized as list
        if isinstance(key, str):
            # Parse string representation of list
            import ast
            parsed_key = ast.literal_eval(key)
            sequence_weights[tuple(parsed_key)] = value
        else:
            sequence_weights[tuple(key)] = value

    # Convert marker_trigrams from list of 3-element lists to set of tuples
    marker_trigrams = {tuple(tri) for tri in data.get("marker_trigrams", [])}

    # Convert trigram_weights keys from lists to tuples (same pattern as sequence_weights)
    trigram_weights_raw = data.get("trigram_weights", {})
    trigram_weights = {}
    for key, value in trigram_weights_raw.items():
        if isinstance(key, str):
            import ast
            parsed_key = ast.literal_eval(key)
            trigram_weights[tuple(parsed_key)] = value
        else:
            trigram_weights[tuple(key)] = value

    # Negative markers (reserved for future use)
    negative_markers = data.get("negative_markers", {})

    return LanguageProfile(
        code=data["code"],
        name=data["name"],
        family=data["family"],
        marker_phonemes=marker_phonemes,
        absent_phonemes=absent_phonemes,
        marker_sequences=marker_sequences,
        priority=data.get("priority", 0),
        marker_weights=marker_weights,
        sequence_weights=sequence_weights,
        threshold=data.get("threshold", 0.0),
        marker_trigrams=marker_trigrams,
        trigram_weights=trigram_weights,
        negative_markers=negative_markers,
    )


def load_profiles_from_directory(directory: Path | None = None) -> list[LanguageProfile]:
    """Load all .json profiles from the profiles directory.

    Args:
        directory: Path to profiles directory. If None, uses default location.

    Returns:
        List of LanguageProfile objects, sorted by language code

    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    if directory is None:
        # Use the package's built-in profiles directory
        directory = Path(__file__).parent / "profiles"

    if not directory.exists():
        raise FileNotFoundError(f"Profiles directory not found: {directory}")

    profiles = []
    for json_path in sorted(directory.glob("*.json")):
        profile = load_profile(json_path)
        profiles.append(profile)

    return sorted(profiles, key=lambda p: p.code)


def load_default_profiles() -> list[LanguageProfile]:
    """Load profiles from the package's built-in profiles directory.

    This is the standard way to load the default language profiles
    for tenepal language detection.

    Returns:
        List of LanguageProfile objects for all built-in languages
    """
    return load_profiles_from_directory()

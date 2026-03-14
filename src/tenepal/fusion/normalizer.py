"""Score normalization for phoneme channel.

Maps raw phonotactic scores (typically 0-150 range) to normalized [0, 1] range
for fusion with prosody scores.
"""


def normalize_phoneme_scores(
    scores: dict[str, float],
    min_val: float = 0.0,
    max_val: float = 150.0
) -> dict[str, float]:
    """Normalize phoneme scores from [min_val, max_val] to [0, 1] range.

    Uses min-max normalization: normalized = (score - min_val) / (max_val - min_val)
    Values outside the specified range are clamped to [0, 1].

    Args:
        scores: Dictionary mapping language codes to raw phoneme scores
        min_val: Minimum value of the input range (default: 0.0)
        max_val: Maximum value of the input range (default: 150.0)

    Returns:
        Dictionary with same keys, normalized values in [0, 1]

    Edge cases:
        - Empty dict returns empty dict
        - If max_val == min_val, returns 0.0 for all scores
        - Negative scores are clamped to 0.0
        - Scores above max_val are clamped to 1.0
    """
    if not scores:
        return {}

    # Edge case: no range to normalize
    if max_val == min_val:
        return {code: 0.0 for code in scores}

    range_val = max_val - min_val
    normalized = {}

    for code, score in scores.items():
        # Min-max normalization
        norm = (score - min_val) / range_val
        # Clamp to [0, 1]
        normalized[code] = max(0.0, min(1.0, norm))

    return normalized

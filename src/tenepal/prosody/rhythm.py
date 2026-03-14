"""Rhythm metric calculations for prosodic analysis.

Implements Pairwise Variability Index (PVI) metrics to distinguish
stress-timed (English, German) from syllable-timed (Spanish, French, Italian) languages.
"""

import numpy as np


def compute_npvi(durations: list[float]) -> float:
    """Compute Normalized Pairwise Variability Index (nPVI).

    nPVI quantifies rhythmic variability in speech by measuring the average
    difference between consecutive duration values, normalized by their mean.

    Formula: nPVI = 100 * (1/(m-1)) * SUM(|d[k] - d[k+1]| / ((d[k] + d[k+1])/2))

    Higher values indicate stress-timed rhythm (English ~57, German ~59).
    Lower values indicate syllable-timed rhythm (Spanish ~43, French ~44, Italian ~49).

    Args:
        durations: List of duration values (e.g., vocalic interval durations in ms)

    Returns:
        nPVI value (0-100+ scale)

    Edge cases:
        - Returns 0.0 for lists with fewer than 2 elements
        - Skips pairs where mean is 0 (would cause division by zero)
    """
    if len(durations) < 2:
        return 0.0

    durations_array = np.array(durations, dtype=np.float64)

    pairwise_sum = 0.0
    valid_pairs = 0

    for k in range(len(durations_array) - 1):
        d_k = durations_array[k]
        d_k_next = durations_array[k + 1]

        # Calculate mean of the pair
        pair_mean = (d_k + d_k_next) / 2.0

        # Skip pairs where mean is 0 to avoid division by zero
        if pair_mean == 0.0:
            continue

        # Normalized difference
        normalized_diff = abs(d_k - d_k_next) / pair_mean
        pairwise_sum += normalized_diff
        valid_pairs += 1

    # Return 0 if no valid pairs
    if valid_pairs == 0:
        return 0.0

    # nPVI formula
    npvi = 100.0 * (pairwise_sum / valid_pairs)
    return npvi


def compute_rpvi(durations: list[float]) -> float:
    """Compute Raw Pairwise Variability Index (rPVI).

    rPVI quantifies rhythmic variability without normalization, typically used
    for consonantal intervals where absolute differences are meaningful.

    Formula: rPVI = (1/(m-1)) * SUM(|d[k] - d[k+1]|)

    Args:
        durations: List of duration values (e.g., consonantal interval durations in ms)

    Returns:
        rPVI value (same units as input durations)

    Edge cases:
        - Returns 0.0 for lists with fewer than 2 elements
    """
    if len(durations) < 2:
        return 0.0

    durations_array = np.array(durations, dtype=np.float64)

    pairwise_sum = 0.0

    for k in range(len(durations_array) - 1):
        d_k = durations_array[k]
        d_k_next = durations_array[k + 1]

        # Raw difference (no normalization)
        raw_diff = abs(d_k - d_k_next)
        pairwise_sum += raw_diff

    # rPVI formula
    rpvi = pairwise_sum / (len(durations_array) - 1)
    return rpvi

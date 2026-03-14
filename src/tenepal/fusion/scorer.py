"""Adaptive weighted score fusion for multi-channel language identification.

Combines normalized phoneme scores and prosody scores using per-language-pair
fusion weights that prioritize the more discriminative channel.
"""

from dataclasses import dataclass

from .normalizer import normalize_phoneme_scores


@dataclass
class FusionWeights:
    """Fusion weights for combining phoneme and prosody channels.

    Attributes:
        alpha: Phoneme channel weight (typically 0.3-0.7)
        beta: Prosody channel weight (typically 0.3-0.7)

    Note: alpha + beta should typically equal 1.0 for interpretability,
    but this is not enforced.
    """
    alpha: float = 0.7  # phoneme channel
    beta: float = 0.3   # prosody channel


def default_fusion_weights() -> dict[frozenset[str], FusionWeights]:
    """Return default per-language-pair fusion weights.

    Returns dictionary mapping frozenset language pairs to FusionWeights.
    The weights determine how much to trust each channel for discriminating
    between specific language pairs.

    Strategy:
        - Phoneme-dominant (alpha=0.7, beta=0.3): Pairs where phoneme markers
          are highly discriminative (distant languages, strong markers)
        - Prosody-dominant (alpha=0.3, beta=0.7): Pairs where prosody is more
          discriminative (similar phoneme inventories, different rhythm)
        - Balanced (alpha=0.5, beta=0.5): Mixed cues

    Returns:
        Dictionary mapping language pair (as frozenset) to FusionWeights
    """
    weights = {}

    # Phoneme-dominant pairs (alpha=0.7, beta=0.3)
    # NAH is phonetically distant from Romance/Germanic languages
    phoneme_dominant = [
        ("nah", "spa"), ("nah", "eng"), ("nah", "deu"),
        ("nah", "fra"), ("nah", "ita"),
        # ENG/DEU: Strong phoneme markers like ʃtʁ/pfl discriminate well
        ("eng", "deu"),
    ]
    for pair in phoneme_dominant:
        weights[frozenset(pair)] = FusionWeights(alpha=0.7, beta=0.3)

    # Prosody-dominant pairs (alpha=0.3, beta=0.7)
    # Romance languages have similar phoneme inventories but different prosody
    prosody_dominant = [
        ("spa", "ita"),  # Similar phonemes, different rhythm (syllable-timed)
        ("spa", "fra"),  # Similar phonemes, different speech rate/rhythm
        ("fra", "ita"),  # Both Romance, prosody discriminates
    ]
    for pair in prosody_dominant:
        weights[frozenset(pair)] = FusionWeights(alpha=0.3, beta=0.7)

    # Balanced pairs (alpha=0.5, beta=0.5)
    # Mix of phoneme and prosodic cues
    balanced = [
        ("eng", "spa"), ("eng", "fra"), ("eng", "ita"),
        ("deu", "spa"), ("deu", "fra"), ("deu", "ita"),
    ]
    for pair in balanced:
        weights[frozenset(pair)] = FusionWeights(alpha=0.5, beta=0.5)

    return weights


def fuse_scores(
    phoneme_scores: dict[str, float],
    prosody_scores: dict[str, float] | None = None,
    weights: dict[frozenset[str], FusionWeights] | None = None
) -> dict[str, float]:
    """Fuse phoneme and prosody scores using adaptive per-pair weights.

    Args:
        phoneme_scores: Raw or normalized phoneme scores per language
        prosody_scores: Prosody scores [0,1] per language (optional)
        weights: Custom fusion weights dict (defaults to default_fusion_weights())

    Returns:
        Dictionary of fused scores per language [0, 1] range

    Fusion strategy:
        1. If prosody_scores is None/empty, return normalized phoneme scores
        2. Normalize phoneme_scores to [0,1] if needed (max > 1.0)
        3. For each language, find its closest competitor in phoneme scores
        4. Look up fusion weights for (language, competitor) pair
        5. Compute fused = alpha * phoneme_norm + beta * prosody_score
        6. If language not in prosody_scores, use phoneme_norm only

    Weight selection:
        - Closest competitor = language with highest phoneme score after current
        - Use pair-specific weights from weights dict
        - Default to alpha=0.7, beta=0.3 if pair not found
    """
    if not phoneme_scores:
        return {}

    # Fallback: prosody unavailable, return normalized phoneme scores
    if not prosody_scores:
        # Check if normalization needed (scores > 1.0 indicate raw scores)
        max_score = max(phoneme_scores.values())
        if max_score > 1.0:
            return normalize_phoneme_scores(phoneme_scores)
        return phoneme_scores.copy()

    # Load default weights if not provided
    if weights is None:
        weights = default_fusion_weights()

    # Normalize phoneme scores if needed
    max_score = max(phoneme_scores.values())
    if max_score > 1.0:
        phoneme_norm = normalize_phoneme_scores(phoneme_scores)
    else:
        phoneme_norm = phoneme_scores.copy()

    # Sort languages by phoneme score (descending) for competitor detection
    sorted_langs = sorted(phoneme_norm.items(), key=lambda x: -x[1])

    fused = {}

    for i, (lang, phon_score) in enumerate(sorted_langs):
        # Find closest competitor (next highest score)
        if i + 1 < len(sorted_langs):
            competitor = sorted_langs[i + 1][0]
        else:
            # Last place: compare to first place
            competitor = sorted_langs[0][0] if len(sorted_langs) > 1 else lang

        # Look up fusion weights for this pair
        pair_key = frozenset([lang, competitor])
        fusion_w = weights.get(pair_key, FusionWeights(alpha=0.7, beta=0.3))

        # Get prosody score (or fall back to phoneme-only)
        pros_score = prosody_scores.get(lang, None)

        if pros_score is not None:
            # Weighted fusion
            fused[lang] = fusion_w.alpha * phon_score + fusion_w.beta * pros_score
        else:
            # Missing prosody: use phoneme only
            fused[lang] = phon_score

    return fused

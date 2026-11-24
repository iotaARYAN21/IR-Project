"""Maximal Marginal Relevance (MMR): re-order results to reduce redundancy."""

from collections.abc import Sequence


def normalize(scores: Sequence[float]) -> list[float]:
    """Scale scores to 0..1 (min-max), so BM25, cosine and CrossEncoder scores mix
    on one scale. If every score is equal they all become 1.0."""
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def mmr_order(
    relevance: Sequence[float],
    similarity: Sequence[Sequence[float]],
    k: int,
    diversity: float,
) -> list[int]:
    """Indices of the k candidates chosen by MMR, in the order they were picked.

    `relevance[i]` is the score of candidate i (higher is better) and
    `similarity[i][j]` is how similar candidates i and j are (1 = identical).
    """
    count = len(relevance)
    if not 0 <= diversity <= 1:
        raise ValueError("diversity must be between 0 and 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(similarity) != count or any(len(row) != count for row in similarity):
        raise ValueError("similarity must be a square matrix matching relevance")
    if count == 0:
        return []

    scaled = normalize(relevance)
    remaining = list(range(count))
    chosen: list[int] = []
    # Highest similarity from each candidate to anything chosen so far.
    closest = [0.0] * count
    while remaining and len(chosen) < k:
        best = max(
            remaining,
            key=lambda i: (
                (1 - diversity) * scaled[i] - diversity * closest[i],
                -i,  # a tie keeps the original (better ranked) candidate
            ),
        )
        chosen.append(best)
        remaining.remove(best)
        for i in remaining:
            closest[i] = max(closest[i], similarity[i][best])
    return chosen

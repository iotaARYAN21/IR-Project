"""Combine ranked lists from several retrieval systems."""

from collections.abc import Sequence

# Defaults of the BM25 + dense hybrid, chosen on a dev split of the queries
# (see src/evaluation/tune_hybrid.py). BM25 has weight 1. The generic function
# below keeps the textbook default rrf_k = 60.
HYBRID_RRF_K = 10
HYBRID_DENSE_WEIGHT = 1.5


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """
    Fuse ranked lists of document IDs with reciprocal rank fusion.
    Returns (document ID, fused score) pairs, best first.
    """
    if rrf_k < 0:
        raise ValueError("rrf_k must not be negative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)

    return sorted(scores.items(), key=lambda item: -item[1])

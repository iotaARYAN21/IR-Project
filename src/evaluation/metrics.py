"""Ranking metrics used to evaluate information retrieval methods."""

import math
from collections.abc import Mapping, Sequence


def ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    relevance_by_doc: Mapping[str, float],
    k: int = 10,
) -> float:
    """Calculate NDCG@k using graded relevance labels."""
    _validate_k(k)
    gains = [
        _gain(relevance_by_doc.get(doc_id, 0.0))
        for doc_id in ranked_doc_ids[:k]
    ]
    ideal_gains = sorted(
        (_gain(relevance) for relevance in relevance_by_doc.values()),
        reverse=True,
    )[:k]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_dcg = sum(
        gain / math.log2(rank + 2)
        for rank, gain in enumerate(ideal_gains)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def mrr_at_k(
    ranked_doc_ids: Sequence[str],
    relevance_by_doc: Mapping[str, float],
    k: int = 10,
) -> float:
    """Calculate reciprocal rank of the first relevant result in the top k."""
    _validate_k(k)
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if relevance_by_doc.get(doc_id, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    ranked_doc_ids: Sequence[str],
    relevance_by_doc: Mapping[str, float],
    k: int = 10,
) -> float:
    """Fraction of the relevant documents that appear in the top k."""
    _validate_k(k)
    relevant = {
        doc_id for doc_id, relevance in relevance_by_doc.items() if relevance > 0
    }
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def per_query_scores(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Return NDCG@k, MRR@k and Recall@k for every query ID in qrels."""
    _validate_k(k)
    scores = {}
    for query_id, relevance_by_doc in qrels.items():
        ranked_doc_ids = rankings.get(query_id, ())
        scores[query_id] = {
            "ndcg": ndcg_at_k(ranked_doc_ids, relevance_by_doc, k),
            "mrr": mrr_at_k(ranked_doc_ids, relevance_by_doc, k),
            "recall": recall_at_k(ranked_doc_ids, relevance_by_doc, k),
        }
    return scores


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
) -> dict[str, float]:
    """Return mean NDCG@k, MRR@k and Recall@k over the query IDs in qrels."""
    _validate_k(k)
    if not qrels:
        raise ValueError("qrels must contain at least one query")

    per_query = per_query_scores(rankings, qrels, k)
    query_count = len(per_query)
    return {
        f"{name}@{k}": sum(scores[name] for scores in per_query.values())
        / query_count
        for name in ("ndcg", "mrr", "recall")
    }


def _gain(relevance: float) -> float:
    if relevance < 0:
        raise ValueError("relevance labels must be non-negative")
    return (2**relevance) - 1


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")

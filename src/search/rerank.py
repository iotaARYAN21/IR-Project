"""CrossEncoder reranking."""

import time
from collections.abc import Callable, Mapping, Sequence

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(model_name: str = RERANKER_MODEL):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512)


def rerank_scores(
    model, query: str, texts: Sequence[str], batch_size: int = 32
) -> list[float]:
    """Score each text against the query. Higher means more relevant."""
    if not texts:
        return []
    pairs = [(query, text) for text in texts]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(score) for score in scores]


def rerank_run_queries(
    query_ids: Sequence[str],
    query_texts: Mapping[str, str],
    run: Mapping[str, Sequence[tuple[str, float]]],
    doc_texts: Mapping[str, str],
    score_fn: Callable[[str, Sequence[str]], list[float]],
    candidates: int,
) -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
    """Rerank the top `candidates` documents of each query in a saved run.

    Returns the reranked lists, best first, as (document ID, reranker score)
    pairs, and the seconds spent in `score_fn` for each query. Only the model
    scoring is timed, not looking up the texts.
    """
    if candidates < 1:
        raise ValueError("candidates must be at least 1")

    reranked: dict[str, list[tuple[str, float]]] = {}
    seconds: list[float] = []
    for query_id in query_ids:
        doc_ids = [doc_id for doc_id, _ in run[query_id][:candidates]]
        missing = [doc_id for doc_id in doc_ids if doc_id not in doc_texts]
        if missing:
            raise ValueError(f"No text for documents {missing[:3]} (query {query_id})")
        texts = [doc_texts[doc_id] for doc_id in doc_ids]

        begin = time.perf_counter()
        scores = score_fn(query_texts[query_id], texts)
        seconds.append(time.perf_counter() - begin)

        if len(scores) != len(doc_ids):
            raise ValueError("score_fn must return one score per document")
        order = sorted(range(len(doc_ids)), key=lambda i: -scores[i])
        reranked[query_id] = [(doc_ids[i], float(scores[i])) for i in order]
    return reranked, seconds

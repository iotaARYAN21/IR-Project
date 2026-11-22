"""CrossEncoder reranking."""

from collections.abc import Sequence

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

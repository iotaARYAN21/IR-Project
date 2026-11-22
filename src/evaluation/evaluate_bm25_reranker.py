"""Evaluate BM25 retrieval followed by CrossEncoder reranking on NQ."""

import argparse
import time

from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.evaluation.evaluate_bm25 import QUERIES_FILE, load_qrels, load_queries
from src.evaluation.latency import format_latency, summarize_latencies
from src.evaluation.metrics import evaluate_rankings
from src.indexing.config import INDEX_NAME, create_es_client

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def search_bm25(client, index_name, query_text, top_n=100):
    """Fetch the initial candidate pool using BM25."""
    if not query_text.strip():
        return []

    response = client.search(
        index=index_name,
        size=top_n,
        query={
            "multi_match": {
                "query": query_text,
                "fields": ["title^2", "text"],
            }
        },
        source=["title", "text"],
    )
    return response["hits"]["hits"]


def rerank_hits(query_text, hits, model, batch_size=32):
    """Rerank BM25 hits by CrossEncoder score, highest first."""
    if not hits:
        return []

    pairs = []
    for hit in hits:
        doc = hit.get("_source", {})
        doc_text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
        pairs.append((query_text, doc_text))

    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    for hit, score in zip(hits, scores):
        hit["_rerank_score"] = float(score)

    return sorted(hits, key=lambda hit: hit["_rerank_score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Elasticsearch BM25 + CrossEncoder on the NQ test set."
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Rank cutoff for NDCG and MRR."
    )
    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=100,
        help="Number of BM25 candidates to fetch before reranking.",
    )
    parser.add_argument(
        "--index",
        default=INDEX_NAME,
        help=f"Elasticsearch index to search (default: {INDEX_NAME}).",
    )
    parser.add_argument(
        "--model",
        default=RERANKER_MODEL,
        help=f"CrossEncoder model name (default: {RERANKER_MODEL}).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N qrel queries (useful for quick checks).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Leading queries excluded from latency statistics (default: 5).",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.bm25_candidates < args.k:
        parser.error("--bm25-candidates must be at least --k")
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be at least 1")

    queries = load_queries()
    qrels = load_qrels()
    missing_queries = sorted(set(qrels) - set(queries))
    if missing_queries:
        raise ValueError(
            f"{len(missing_queries)} qrel query IDs are missing from "
            f"{QUERIES_FILE}: {', '.join(missing_queries[:5])}"
        )
    if args.max_queries is not None:
        qrels = dict(list(qrels.items())[: args.max_queries])

    print(f"Loading CrossEncoder model: {args.model}")
    reranker = CrossEncoder(args.model, max_length=512)

    client = create_es_client()
    rankings: dict[str, list[str]] = {}
    bm25_latencies: list[float] = []
    rerank_latencies: list[float] = []
    total_latencies: list[float] = []
    try:
        for query_id in tqdm(qrels, desc="Retrieving and reranking", unit="queries"):
            query_text = queries[query_id]
            start = time.perf_counter()
            hits = search_bm25(client, args.index, query_text, args.bm25_candidates)
            bm25_done = time.perf_counter()
            reranked = rerank_hits(query_text, hits, reranker)
            end = time.perf_counter()
            bm25_latencies.append(bm25_done - start)
            rerank_latencies.append(end - bm25_done)
            total_latencies.append(end - start)
            rankings[query_id] = [hit["_id"] for hit in reranked[: args.k]]
    finally:
        client.close()

    scores = evaluate_rankings(rankings, qrels, args.k)
    print(
        f"Evaluated BM25 + Reranker on {len(qrels)} queries "
        f"(top {args.k}, {args.bm25_candidates} BM25 candidates)"
    )
    print(f"NDCG@{args.k}: {scores[f'ndcg@{args.k}']:.4f}")
    print(f"MRR@{args.k}:  {scores[f'mrr@{args.k}']:.4f}")
    print(f"Recall@{args.k}: {scores[f'recall@{args.k}']:.4f}")
    for label, values in (
        ("BM25 stage latency", bm25_latencies),
        ("Rerank stage latency", rerank_latencies),
        ("Total latency", total_latencies),
    ):
        print(format_latency(label, summarize_latencies(values, args.warmup)))


if __name__ == "__main__":
    main()

"""Evaluate exact dense (semantic) retrieval against the NQ test qrels."""

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from src.evaluation.evaluate_bm25 import QUERIES_FILE, load_qrels, load_queries
from src.evaluation.latency import format_latency, summarize_latencies
from src.evaluation.metrics import evaluate_rankings, per_query_scores
from src.evaluation.results import default_output_path, save_results
from src.indexing.encode_corpus import DEFAULT_MODEL, EMBEDDINGS_ROOT
from src.search.dense import DenseIndex, encode_queries, load_query_encoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate exact dense retrieval on the NQ test set."
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Rank cutoff for NDCG, MRR and recall."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model that encoded the corpus."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Folder written by encode_corpus "
        "(default: data/nq/embeddings/<model name>).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N qrel queries (useful for quick checks).",
    )
    parser.add_argument(
        "--batch-queries",
        action="store_true",
        help="Search all queries in large batches. Much faster, but latency "
        "is not measured.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Leading queries excluded from latency statistics (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the results JSON (default: results/dense_k<k>.json).",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
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

    embeddings_dir = args.embeddings_dir or EMBEDDINGS_ROOT / args.model.split("/")[-1]
    print(f"Loading document vectors from {embeddings_dir}")
    start = time.perf_counter()
    index = DenseIndex.from_directory(embeddings_dir)
    print(f"Loaded {len(index)} vectors in {time.perf_counter() - start:.1f} s")
    print(f"Loading query encoder: {args.model}")
    encoder = load_query_encoder(args.model)

    rankings: dict[str, list[str]] = {}
    encode_times: list[float] = []
    search_times: list[float] = []
    total_times: list[float] = []
    query_ids = list(qrels)

    if args.batch_queries:
        texts = [queries[query_id] for query_id in query_ids]
        vectors = encode_queries(encoder, texts)
        hits = index.search_vectors(vectors, args.k)
        for query_id, query_hits in zip(query_ids, hits):
            rankings[query_id] = [doc_id for doc_id, _ in query_hits]
    else:
        for query_id in tqdm(query_ids, desc="Searching", unit="queries"):
            begin = time.perf_counter()
            vector = encode_queries(encoder, [queries[query_id]])
            encoded = time.perf_counter()
            hits = index.search_vectors(vector, args.k)[0]
            end = time.perf_counter()
            rankings[query_id] = [doc_id for doc_id, _ in hits]
            encode_times.append(encoded - begin)
            search_times.append(end - encoded)
            total_times.append(end - begin)

    scores = evaluate_rankings(rankings, qrels, args.k)
    print(f"Evaluated dense retrieval on {len(qrels)} queries (top {args.k})")
    print(f"NDCG@{args.k}: {scores[f'ndcg@{args.k}']:.4f}")
    print(f"MRR@{args.k}:  {scores[f'mrr@{args.k}']:.4f}")
    print(f"Recall@{args.k}: {scores[f'recall@{args.k}']:.4f}")

    latency = {}
    if args.batch_queries:
        print("Latency not measured in --batch-queries mode.")
    else:
        for key, label, values in (
            ("encode", "Query encoding latency", encode_times),
            ("search", "Vector search latency", search_times),
            ("total", "Total latency", total_times),
        ):
            latency[key] = summarize_latencies(values, args.warmup)
            print(format_latency(label, latency[key]))

    save_results(
        args.output or default_output_path("dense", args.k, args.max_queries),
        {
            "system": "dense",
            "k": args.k,
            "num_queries": len(qrels),
            "args": {
                "model": args.model,
                "embeddings_dir": str(embeddings_dir),
                "max_queries": args.max_queries,
                "batch_queries": args.batch_queries,
                "warmup": args.warmup,
            },
            "scores": scores,
            "latency": latency,
            "per_query": per_query_scores(rankings, qrels, args.k),
        },
    )


if __name__ == "__main__":
    main()

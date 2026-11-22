"""Evaluate BM25 + dense retrieval fused with reciprocal rank fusion.

This works from the run files that evaluate_bm25 and evaluate_dense save, so it
needs neither Elasticsearch nor the document vectors, and it takes seconds.
"""

import argparse
import time
from pathlib import Path

from src.evaluation.evaluate_bm25 import load_qrels
from src.evaluation.latency import (
    estimate_total_latency,
    format_latency,
    summarize_latencies,
)
from src.evaluation.metrics import evaluate_rankings, per_query_scores
from src.evaluation.results import default_output_path, save_results
from src.evaluation.runs import default_run_path, load_run, save_run
from src.search.hybrid import reciprocal_rank_fusion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 + dense retrieval with reciprocal rank fusion."
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Rank cutoff for NDCG, MRR and recall."
    )
    parser.add_argument(
        "--bm25-run", type=Path, default=default_run_path("bm25"), help="BM25 run file."
    )
    parser.add_argument(
        "--dense-run",
        type=Path,
        default=default_run_path("dense"),
        help="Dense run file.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=100,
        help="Top documents taken from each run before fusing (default: 100).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal rank fusion constant (default: 60).",
    )
    parser.add_argument(
        "--bm25-weight", type=float, default=1.0, help="Weight of the BM25 list."
    )
    parser.add_argument(
        "--dense-weight", type=float, default=1.0, help="Weight of the dense list."
    )
    parser.add_argument(
        "--run-depth",
        type=int,
        default=100,
        help="Fused documents saved per query for reranking (default: 100).",
    )
    parser.add_argument(
        "--run-output",
        type=Path,
        default=default_run_path("hybrid"),
        help="Where to save the fused run file (default: results/runs/hybrid.json).",
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
        help="Where to save the results JSON (default: results/hybrid_k<k>.json).",
    )
    args = parser.parse_args()
    for name in ("k", "candidates", "run_depth"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.rrf_k < 0:
        parser.error("--rrf-k must not be negative")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.bm25_weight < 0 or args.dense_weight < 0:
        parser.error("weights must not be negative")

    bm25_run = load_run(args.bm25_run)
    dense_run = load_run(args.dense_run)
    for run in (bm25_run, dense_run):
        if run["depth"] < args.candidates:
            parser.error(
                f"The {run['system']} run holds {run['depth']} documents per query, "
                f"fewer than --candidates {args.candidates}. Rerun that evaluator "
                "with a larger --run-depth."
            )

    qrels = load_qrels()
    shared = [
        query_id
        for query_id in qrels
        if query_id in bm25_run["queries"] and query_id in dense_run["queries"]
    ]
    if not shared:
        raise ValueError("The two run files have no qrel queries in common")
    if len(shared) < len(qrels):
        print(
            f"Warning: only {len(shared)} of {len(qrels)} qrel queries are in both "
            "run files. Scores cover those queries only."
        )
    qrels = {query_id: qrels[query_id] for query_id in shared}

    weights = [args.bm25_weight, args.dense_weight]
    fused_run: dict[str, list[tuple[str, float]]] = {}
    rankings: dict[str, list[str]] = {}
    fusion_times: list[float] = []
    for query_id in shared:
        bm25_ids = [doc_id for doc_id, _ in bm25_run["queries"][query_id]][
            : args.candidates
        ]
        dense_ids = [doc_id for doc_id, _ in dense_run["queries"][query_id]][
            : args.candidates
        ]
        start = time.perf_counter()
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids], args.rrf_k, weights)
        fusion_times.append(time.perf_counter() - start)
        fused_run[query_id] = fused[: args.run_depth]
        rankings[query_id] = [doc_id for doc_id, _ in fused]

    scores = evaluate_rankings(rankings, qrels, args.k)
    print(
        f"Evaluated hybrid (BM25 + dense, RRF k={args.rrf_k}) on {len(qrels)} "
        f"queries (top {args.k}, {args.candidates} candidates from each)"
    )
    print(f"NDCG@{args.k}: {scores[f'ndcg@{args.k}']:.4f}")
    print(f"MRR@{args.k}:  {scores[f'mrr@{args.k}']:.4f}")
    print(f"Recall@{args.k}: {scores[f'recall@{args.k}']:.4f}")

    fusion = summarize_latencies(fusion_times, args.warmup)
    print(format_latency("Fusion latency", fusion))
    estimate = estimate_total_latency(bm25_run, dense_run, fusion)
    if estimate:
        parts = estimate["components_mean_ms"]
        print(
            f"Estimated hybrid latency (mean, systems run one after the other): "
            f"{estimate['mean_ms']:.1f} ms = BM25 {parts['bm25']:.1f} + "
            f"dense {parts['dense']:.1f} + fusion {parts['fusion']:.3f}"
        )
    else:
        print(
            "Total latency not estimated: the run files have no latency "
            "(dense was run with --batch-queries?)."
        )

    save_run(
        args.run_output,
        "hybrid",
        fused_run,
        depth=args.run_depth,
        latency={"fusion": fusion},
    )
    save_results(
        args.output or default_output_path("hybrid", args.k),
        {
            "system": "hybrid",
            "k": args.k,
            "num_queries": len(qrels),
            "args": {
                "bm25_run": str(args.bm25_run),
                "dense_run": str(args.dense_run),
                "candidates": args.candidates,
                "rrf_k": args.rrf_k,
                "weights": {"bm25": args.bm25_weight, "dense": args.dense_weight},
                "run_depth": args.run_depth,
                "warmup": args.warmup,
            },
            "scores": scores,
            "latency": {"fusion": fusion, "estimated_total": estimate},
            "per_query": per_query_scores(rankings, qrels, args.k),
        },
    )


if __name__ == "__main__":
    main()

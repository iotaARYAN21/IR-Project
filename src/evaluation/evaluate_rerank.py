"""Rerank the candidates in a saved run file with a CrossEncoder and evaluate.

Works from the run files that evaluate_bm25, evaluate_dense and
evaluate_hybrid save, so it needs neither Elasticsearch nor the document
vectors, only data/nq/corpus.jsonl for the passage text. That also makes it easy
to run on a GPU machine.

    python3 -m src.evaluation.evaluate_rerank --run results/runs/hybrid.json
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.evaluation.evaluate_bm25 import QUERIES_FILE, load_qrels, load_queries
from src.evaluation.latency import format_latency, summarize_latencies
from src.evaluation.metrics import evaluate_rankings, per_query_scores
from src.evaluation.results import default_output_path, save_results
from src.evaluation.runs import default_run_path, load_run, save_run
from src.indexing.encode_corpus import CORPUS_FILE
from src.search.documents import load_corpus_texts
from src.search.rerank import (
    RERANKER_MODEL,
    load_reranker,
    rerank_run_queries,
    rerank_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank a saved run with a CrossEncoder and evaluate it."
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=default_run_path("hybrid"),
        help="Run file to rerank (default: results/runs/hybrid.json).",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Rank cutoff for NDCG, MRR and recall."
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=100,
        help="Top documents of each query to rerank (default: 100).",
    )
    parser.add_argument("--model", default=RERANKER_MODEL, help="CrossEncoder model.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE, help="corpus.jsonl.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Model batch size (default: 32)."
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
    parser.add_argument(
        "--run-output",
        type=Path,
        default=None,
        help="Where to save the reranked run "
        "(default: results/runs/<run system>_rerank[_n<N>].json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the results JSON "
        "(default: results/<run system>_rerank_k<k>[_n<N>].json).",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.candidates < args.k:
        parser.error("--candidates must be at least --k")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be at least 1")

    run = load_run(args.run)
    if run["depth"] < args.candidates:
        parser.error(
            f"The run holds {run['depth']} documents per query, fewer than "
            f"--candidates {args.candidates}."
        )
    system = f"{run['system']}_rerank"

    queries = load_queries()
    qrels = load_qrels()
    missing_queries = sorted(set(qrels) - set(queries))
    if missing_queries:
        raise ValueError(
            f"{len(missing_queries)} qrel query IDs are missing from "
            f"{QUERIES_FILE}: {', '.join(missing_queries[:5])}"
        )
    query_ids = [query_id for query_id in qrels if query_id in run["queries"]]
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]
    if not query_ids:
        raise ValueError("The run file has no qrel queries")
    if len(query_ids) < len(qrels) and args.max_queries is None:
        print(
            f"Warning: the run covers {len(query_ids)} of {len(qrels)} qrel "
            "queries. Scores cover those queries only."
        )
    qrels = {query_id: qrels[query_id] for query_id in query_ids}

    needed = {
        doc_id
        for query_id in query_ids
        for doc_id, _ in run["queries"][query_id][: args.candidates]
    }
    print(f"Reading the text of {len(needed)} documents from {args.corpus} ...")
    doc_texts = load_corpus_texts(args.corpus, needed)

    print(f"Loading CrossEncoder model: {args.model}")
    model = load_reranker(args.model)
    device = str(getattr(model, "device", "unknown"))
    print(f"Device: {device}")

    progress = tqdm(total=len(query_ids), desc="Reranking", unit="queries")

    def score(query_text, texts):
        scores = rerank_scores(model, query_text, texts, args.batch_size)
        progress.update(1)
        return scores

    reranked, seconds = rerank_run_queries(
        query_ids, queries, run["queries"], doc_texts, score, args.candidates
    )
    progress.close()
    rankings = {query_id: [doc_id for doc_id, _ in hits] for query_id, hits in reranked.items()}

    scores = evaluate_rankings(rankings, qrels, args.k)
    print(
        f"Evaluated {system} on {len(qrels)} queries "
        f"(top {args.k}, {args.candidates} candidates from the {run['system']} run)"
    )
    print(f"NDCG@{args.k}: {scores[f'ndcg@{args.k}']:.4f}")
    print(f"MRR@{args.k}:  {scores[f'mrr@{args.k}']:.4f}")
    print(f"Recall@{args.k}: {scores[f'recall@{args.k}']:.4f}")

    rerank_latency = summarize_latencies(seconds, args.warmup)
    print(format_latency(f"Rerank latency on {device}", rerank_latency))
    latency = {"rerank": rerank_latency}
    base = run.get("latency", {}).get("total")
    if base:
        estimated = base["mean_ms"] + rerank_latency["mean_ms"]
        latency["estimated_total"] = {
            "mean_ms": estimated,
            "components_mean_ms": {
                run["system"]: base["mean_ms"],
                "rerank": rerank_latency["mean_ms"],
            },
        }
        print(
            f"Estimated total latency (mean): {estimated:.1f} ms = "
            f"{run['system']} {base['mean_ms']:.1f} + rerank "
            f"{rerank_latency['mean_ms']:.1f}. Fetching the passage text is not included."
        )
    else:
        print(
            "Total latency not estimated: the run file has no total latency "
            "(dense was run with --batch-queries?)."
        )

    save_run(
        args.run_output or default_run_path(system, args.max_queries),
        system,
        reranked,
        depth=args.candidates,
        latency=latency,
    )
    save_results(
        args.output or default_output_path(system, args.k, args.max_queries),
        {
            "system": system,
            "k": args.k,
            "num_queries": len(qrels),
            "args": {
                "run": str(args.run),
                "model": args.model,
                "device": device,
                "candidates": args.candidates,
                "batch_size": args.batch_size,
                "max_queries": args.max_queries,
                "warmup": args.warmup,
            },
            "scores": scores,
            "latency": latency,
            "per_query": per_query_scores(rankings, qrels, args.k),
        },
    )


if __name__ == "__main__":
    main()

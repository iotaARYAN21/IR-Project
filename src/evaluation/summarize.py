"""Print the results of all systems as a Markdown table.

    python3 -m src.evaluation.summarize            # k = 10
    python3 -m src.evaluation.summarize --k 20
"""

import argparse
import json
from pathlib import Path

from src.evaluation.results import RESULTS_DIR, default_output_path

SYSTEMS = (
    ("bm25", "BM25"),
    ("dense", "Dense"),
    ("hybrid", "Hybrid (BM25 + dense, RRF)"),
    ("bm25_rerank", "BM25 + reranker"),
    ("hybrid_rerank", "Hybrid + reranker"),
)


def mean_latency_ms(results: dict) -> float | None:
    """Mean per-query latency of a results file, or None if it was not measured.

    Uses the estimated total for hybrid and reranked systems (which adds up the
    stages), and the total (or the single BM25 figure) otherwise.
    """
    latency = results.get("latency") or {}
    for path in (("estimated_total",), ("total",), ("bm25",)):
        entry = latency.get(path[0])
        if entry and "mean_ms" in entry:
            return float(entry["mean_ms"])
    return None


def build_table(results_by_system: dict[str, dict | None], k: int) -> str:
    """Markdown table with one row per system. Missing systems show dashes."""
    lines = [
        f"| System | NDCG@{k} | MRR@{k} | Recall@{k} | Mean latency (ms) | Queries |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for system, label in SYSTEMS:
        results = results_by_system.get(system)
        if not results:
            lines.append(f"| {label} | not run | | | | |")
            continue
        scores = results["scores"]
        latency = mean_latency_ms(results)
        lines.append(
            f"| {label} | {scores[f'ndcg@{k}']:.4f} | {scores[f'mrr@{k}']:.4f} "
            f"| {scores[f'recall@{k}']:.4f} "
            f"| {'' if latency is None else f'{latency:.1f}'} "
            f"| {results['num_queries']} |"
        )
    return "\n".join(lines)


def load_all(k: int, results_dir: Path = RESULTS_DIR) -> dict[str, dict | None]:
    loaded = {}
    for system, _ in SYSTEMS:
        path = results_dir / default_output_path(system, k).name
        loaded[system] = json.loads(path.read_text()) if path.is_file() else None
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise saved results.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff of the runs to show.")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    print(build_table(load_all(args.k), args.k))


if __name__ == "__main__":
    main()

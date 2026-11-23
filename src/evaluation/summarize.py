"""Print the results of all systems as Markdown tables.

    python3 -m src.evaluation.summarize            # k = 10
    python3 -m src.evaluation.summarize --k 20
    python3 -m src.evaluation.summarize --no-significance   # skip the bootstrap

Three tables are printed: quality and latency per system (with Recall@100 of
the candidate pool and significance against BM25), and a latency breakdown by
stage. Recall@100 is computed from the saved run files and the qrels.
"""

import argparse
import json
from pathlib import Path

from src.evaluation.compare import compare_runs
from src.evaluation.metrics import recall_at_k
from src.evaluation.results import RESULTS_DIR, default_output_path
from src.evaluation.runs import RUNS_DIR
from src.evaluation.tune_hybrid import TUNING_PATH

SYSTEMS = (
    ("bm25", "BM25"),
    ("dense", "Dense"),
    ("hybrid_equal", "Hybrid, equal weights (RRF k = 60)"),
    ("hybrid", "Hybrid, tuned (dense 1.5, k = 10) †"),
    ("bm25_rerank", "BM25 + reranker"),
    ("hybrid_equal_rerank", "Equal-weight hybrid + reranker"),
)
BASELINE = "bm25"
POOL_DEPTH = 100
SIGNIFICANCE_LEVEL = 0.05

# (label, system whose results file holds it, latency key)
STAGES = (
    ("BM25 search", "bm25", "bm25"),
    ("Dense: encode query", "dense", "encode"),
    ("Dense: exact search", "dense", "search"),
    ("Dense: total", "dense", "total"),
    ("RRF fusion", "hybrid", "fusion"),
    ("Rerank top-100 of BM25", "bm25_rerank", "rerank"),
    ("Rerank top-100 of equal-weight hybrid", "hybrid_equal_rerank", "rerank"),
)


def mean_latency_ms(results: dict) -> float | None:
    """Mean per-query latency of a results file, or None if it was not measured.

    Uses the estimated total for hybrid and reranked systems (which adds up the
    stages), and the total (or the single BM25 figure) otherwise.
    """
    latency = results.get("latency") or {}
    for key in ("estimated_total", "total", "bm25"):
        entry = latency.get(key)
        if entry and "mean_ms" in entry:
            return float(entry["mean_ms"])
    return None


def pool_recall(run: dict, qrels: dict, depth: int = POOL_DEPTH) -> float:
    """Mean Recall@depth of a loaded run over every query in the qrels."""
    scores = [
        recall_at_k(
            [doc_id for doc_id, _ in run["queries"].get(query_id, [])],
            relevant,
            depth,
        )
        for query_id, relevant in qrels.items()
    ]
    return sum(scores) / len(scores)


def significance_vs_baseline(
    results: dict, baseline: dict, resamples: int
) -> tuple[float, float]:
    """(NDCG difference from the baseline, two-sided bootstrap p-value)."""
    ndcg = compare_runs(baseline, results, resamples)["ndcg"]
    return ndcg["mean_diff"], ndcg["p_value"]


def format_versus(difference: float, p_value: float, resamples: int) -> str:
    p_text = f"p < {2 / resamples:.4f}" if p_value <= 0 else f"p = {p_value:.4f}"
    mark = "*" if p_value < SIGNIFICANCE_LEVEL else ""
    return f"{difference:+.4f}{mark} ({p_text})"


def build_table(
    results_by_system: dict[str, dict | None],
    k: int,
    recall_pool: dict[str, float] | None = None,
    versus: dict[str, tuple[float, float]] | None = None,
    resamples: int = 0,
) -> str:
    """Markdown table with one row per system. Missing systems show 'not run'.

    `recall_pool` maps a system to its Recall@100, and `versus` maps a system to
    (NDCG difference from BM25, p-value). Columns are left blank when not given.
    """
    recall_pool = recall_pool or {}
    versus = versus or {}
    lines = [
        f"| System | NDCG@{k} | MRR@{k} | Recall@{k} | Recall@{POOL_DEPTH} "
        f"| Mean latency (ms) | NDCG@{k} vs BM25 | Queries |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for system, label in SYSTEMS:
        results = results_by_system.get(system)
        if not results:
            lines.append(f"| {label} | not run | | | | | | |")
            continue
        scores = results["scores"]
        latency = mean_latency_ms(results)
        pool = recall_pool.get(system)
        if system == BASELINE:
            compared = "baseline"
        elif system in versus:
            compared = format_versus(*versus[system], resamples)
        else:
            compared = ""
        lines.append(
            f"| {label} | {scores[f'ndcg@{k}']:.4f} | {scores[f'mrr@{k}']:.4f} "
            f"| {scores[f'recall@{k}']:.4f} "
            f"| {'' if pool is None else f'{pool:.4f}'} "
            f"| {'' if latency is None else f'{latency:.1f}'} "
            f"| {compared} | {results['num_queries']} |"
        )
    return "\n".join(lines)


def build_latency_table(results_by_system: dict[str, dict | None]) -> str:
    """Per-stage latency (mean, median, p95, in ms) for the stages that were timed."""
    lines = [
        "| Stage | Mean (ms) | Median (ms) | p95 (ms) |",
        "| --- | --- | --- | --- |",
    ]
    for label, system, key in STAGES:
        results = results_by_system.get(system)
        entry = ((results or {}).get("latency") or {}).get(key)
        if not entry or "mean_ms" not in entry:
            continue
        lines.append(
            f"| {label} | {entry['mean_ms']:.1f} | {entry['median_ms']:.1f} "
            f"| {entry['p95_ms']:.1f} |"
        )
    return "\n".join(lines)


def load_all(k: int, results_dir: Path = RESULTS_DIR) -> dict[str, dict | None]:
    loaded = {}
    for system, _ in SYSTEMS:
        path = results_dir / default_output_path(system, k).name
        loaded[system] = json.loads(path.read_text()) if path.is_file() else None
    return loaded


def load_pool_recall(
    results_by_system: dict[str, dict | None], runs_dir: Path = RUNS_DIR
) -> dict[str, float]:
    """Recall@100 for every system that has results and a run file."""
    from src.evaluation.evaluate_bm25 import load_qrels
    from src.evaluation.runs import load_run

    qrels = load_qrels()
    recall = {}
    for system, results in results_by_system.items():
        path = runs_dir / f"{system}.json"
        if results and path.is_file():
            recall[system] = pool_recall(load_run(path), qrels)
    return recall


def tuning_note(path: Path = TUNING_PATH, k: int = 10) -> str | None:
    """Footnote for the tuned hybrid row, from the saved tuning results."""
    if not path.is_file():
        return None
    tuning = json.loads(path.read_text())
    key = f"ndcg@{tuning['k']}"
    held = tuning["held_out"]
    chosen = tuning["chosen"]
    return (
        f"† Dense weight {chosen['dense_weight']:g} (BM25 weight 1), RRF k "
        f"{chosen['rrf_k']}, chosen on {tuning['dev_queries']} dev queries. The row "
        f"covers all queries, including those dev queries, so it is slightly "
        f"optimistic. On the {tuning['held_out_queries']} held-out queries "
        f"NDCG@{tuning['k']} is {held['tuned'][key]:.4f}, against "
        f"{held['equal_weights'][key]:.4f} for the equal-weight hybrid."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise saved results.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff of the runs to show.")
    parser.add_argument(
        "--resamples", type=int, default=2000, help="Bootstrap resamples per system."
    )
    parser.add_argument(
        "--no-significance",
        action="store_true",
        help="Skip the bootstrap and the Recall@100 column (no qrels needed).",
    )
    args = parser.parse_args()
    if args.k < 1 or args.resamples < 1:
        parser.error("--k and --resamples must be at least 1")

    results = load_all(args.k)
    recall_pool, versus = {}, {}
    if not args.no_significance:
        recall_pool = load_pool_recall(results)
        baseline = results.get(BASELINE)
        if baseline:
            for system, system_results in results.items():
                if system != BASELINE and system_results:
                    versus[system] = significance_vs_baseline(
                        system_results, baseline, args.resamples
                    )
    print(build_table(results, args.k, recall_pool, versus, args.resamples))
    print()
    print("`*` = significant at p < 0.05 (paired bootstrap against BM25).")
    note = tuning_note()
    if note:
        print()
        print(note)
    print()
    print(build_latency_table(results))


if __name__ == "__main__":
    main()

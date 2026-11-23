"""Tune the hybrid fusion weights without overfitting the test queries.

NQ has one set of judged queries, so the queries are split once, with a fixed
seed, into a small *dev* part and a *held-out* part:

  1. every setting in a small grid is scored on the dev queries only;
  2. the best dev setting is chosen;
  3. the chosen setting and the equal-weight setting (RRF k = 60) are scored
     once on the held-out queries, with a paired bootstrap between them.

The held-out numbers are the ones to report. They are honest because those
queries played no part in choosing the setting. It works from the BM25 and
dense run files, so it needs no Elasticsearch and takes under a minute.

    python3 -m src.evaluation.tune_hybrid
    python3 -m src.evaluation.tune_hybrid --dev-fraction 0.3 --seed 1
"""

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

from src.evaluation.compare import compare_runs
from src.evaluation.metrics import evaluate_rankings, per_query_scores
from src.evaluation.results import RESULTS_DIR
from src.evaluation.runs import default_run_path, load_run
from src.search.hybrid import reciprocal_rank_fusion

DEFAULT_DENSE_WEIGHTS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
DEFAULT_RRF_KS = (10, 20, 30, 40, 50, 60)
EQUAL_WEIGHT_SETTING = (1.0, 60)  # (dense weight, RRF k), BM25 weight is always 1
TUNING_PATH = RESULTS_DIR / "tuning" / "hybrid_tuning.json"


def split_queries(
    query_ids: Sequence[str], dev_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split query IDs into (dev, held_out). The same seed gives the same split."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    ids = sorted(query_ids)
    if len(ids) < 2:
        raise ValueError("Need at least two queries to split")
    random.Random(seed).shuffle(ids)
    cut = min(max(1, round(len(ids) * dev_fraction)), len(ids) - 1)
    return sorted(ids[:cut]), sorted(ids[cut:])


def fuse_rankings(
    bm25_run: dict,
    dense_run: dict,
    query_ids: Sequence[str],
    dense_weight: float,
    rrf_k: int,
    candidates: int,
) -> dict[str, list[str]]:
    """Fused ranking of document IDs for each query (BM25 weight fixed at 1)."""
    rankings = {}
    for query_id in query_ids:
        bm25_ids = [d for d, _ in bm25_run["queries"][query_id]][:candidates]
        dense_ids = [d for d, _ in dense_run["queries"][query_id]][:candidates]
        fused = reciprocal_rank_fusion(
            [bm25_ids, dense_ids], rrf_k, [1.0, dense_weight]
        )
        rankings[query_id] = [doc_id for doc_id, _ in fused]
    return rankings


def grid_search(
    bm25_run: dict,
    dense_run: dict,
    qrels: dict,
    dev_ids: Sequence[str],
    dense_weights: Sequence[float],
    rrf_ks: Sequence[int],
    k: int,
    candidates: int,
) -> list[dict]:
    """Score every (dense weight, RRF k) pair on the dev queries only."""
    dev_qrels = {query_id: qrels[query_id] for query_id in dev_ids}
    results = []
    for rrf_k in rrf_ks:
        for weight in dense_weights:
            rankings = fuse_rankings(
                bm25_run, dense_run, dev_ids, weight, rrf_k, candidates
            )
            scores = evaluate_rankings(rankings, dev_qrels, k)
            results.append(
                {"dense_weight": weight, "rrf_k": rrf_k, "dev": scores}
            )
    return results


def choose_best(grid: list[dict], k: int) -> dict:
    """Best dev NDCG. Ties go to the setting closest to equal weights, k = 60."""

    def distance(entry: dict) -> float:
        return abs(entry["dense_weight"] - EQUAL_WEIGHT_SETTING[0]) + abs(
            entry["rrf_k"] - EQUAL_WEIGHT_SETTING[1]
        ) / 100

    return max(grid, key=lambda e: (e["dev"][f"ndcg@{k}"], -distance(e)))


def held_out_run(
    bm25_run: dict,
    dense_run: dict,
    qrels: dict,
    held_ids: Sequence[str],
    dense_weight: float,
    rrf_k: int,
    k: int,
    candidates: int,
    name: str,
) -> dict:
    """Held-out scores in the shape compare_runs expects."""
    held_qrels = {query_id: qrels[query_id] for query_id in held_ids}
    rankings = fuse_rankings(
        bm25_run, dense_run, held_ids, dense_weight, rrf_k, candidates
    )
    return {
        "system": name,
        "k": k,
        "scores": evaluate_rankings(rankings, held_qrels, k),
        "per_query": per_query_scores(rankings, held_qrels, k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune hybrid fusion on a dev split and report on held-out queries."
    )
    parser.add_argument("--k", type=int, default=10, help="Cutoff for the metrics.")
    parser.add_argument("--bm25-run", type=Path, default=default_run_path("bm25"))
    parser.add_argument("--dense-run", type=Path, default=default_run_path("dense"))
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument(
        "--dev-fraction", type=float, default=0.3, help="Share of queries for tuning."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the split.")
    parser.add_argument(
        "--dense-weights", type=float, nargs="+", default=list(DEFAULT_DENSE_WEIGHTS)
    )
    parser.add_argument("--rrf-ks", type=int, nargs="+", default=list(DEFAULT_RRF_KS))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=TUNING_PATH)
    args = parser.parse_args()
    if args.k < 1 or args.candidates < 1 or args.resamples < 1:
        parser.error("--k, --candidates and --resamples must be at least 1")
    if min(args.dense_weights) < 0 or min(args.rrf_ks) < 0:
        parser.error("weights and RRF constants must not be negative")

    from src.evaluation.evaluate_bm25 import load_qrels

    bm25_run = load_run(args.bm25_run)
    dense_run = load_run(args.dense_run)
    qrels = load_qrels()
    shared = [
        q for q in qrels if q in bm25_run["queries"] and q in dense_run["queries"]
    ]
    dev_ids, held_ids = split_queries(shared, args.dev_fraction, args.seed)
    print(
        f"{len(shared)} judged queries: {len(dev_ids)} dev (tuning), "
        f"{len(held_ids)} held-out (reporting), seed {args.seed}"
    )

    grid = grid_search(
        bm25_run, dense_run, qrels, dev_ids,
        args.dense_weights, args.rrf_ks, args.k, args.candidates,
    )
    best = choose_best(grid, args.k)
    key = f"ndcg@{args.k}"

    print(f"\nDev NDCG@{args.k} (BM25 weight fixed at 1):\n")
    print("| RRF k \\ dense weight | " + " | ".join(f"{w:g}" for w in args.dense_weights) + " |")
    print("| --- |" + " --- |" * len(args.dense_weights))
    for rrf_k in args.rrf_ks:
        cells = []
        for weight in args.dense_weights:
            entry = next(
                e for e in grid if e["rrf_k"] == rrf_k and e["dense_weight"] == weight
            )
            text = f"{entry['dev'][key]:.4f}"
            cells.append(f"**{text}**" if entry is best else text)
        print(f"| {rrf_k} | " + " | ".join(cells) + " |")
    print(
        f"\nChosen on dev: dense weight {best['dense_weight']:g}, "
        f"RRF k {best['rrf_k']}"
    )

    equal = held_out_run(
        bm25_run, dense_run, qrels, held_ids, *EQUAL_WEIGHT_SETTING, args.k,
        args.candidates, "hybrid_equal",
    )
    tuned = held_out_run(
        bm25_run, dense_run, qrels, held_ids, best["dense_weight"], best["rrf_k"],
        args.k, args.candidates, "hybrid_tuned",
    )
    dense = {
        "system": "dense",
        "k": args.k,
        "per_query": per_query_scores(
            {q: [d for d, _ in dense_run["queries"][q]] for q in held_ids},
            {q: qrels[q] for q in held_ids},
            args.k,
        ),
    }
    print(f"\nHeld-out ({len(held_ids)} queries) NDCG@{args.k}:")
    for run in (equal, tuned):
        print(f"  {run['system']:<15} {run['scores'][key]:.4f}")
    versus = {
        "tuned_vs_equal": compare_runs(equal, tuned, args.resamples)["ndcg"],
        "tuned_vs_dense": compare_runs(dense, tuned, args.resamples)["ndcg"],
    }
    for name, stats in versus.items():
        p = stats["p_value"]
        print(
            f"  {name}: diff {stats['mean_diff']:+.4f} "
            f"(95% CI {stats['ci_low']:+.4f} to {stats['ci_high']:+.4f}), "
            f"p = {p:.4f}" if p > 0 else
            f"  {name}: diff {stats['mean_diff']:+.4f} "
            f"(95% CI {stats['ci_low']:+.4f} to {stats['ci_high']:+.4f}), "
            f"p < {2 / args.resamples:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "k": args.k,
                "seed": args.seed,
                "dev_fraction": args.dev_fraction,
                "dev_queries": len(dev_ids),
                "held_out_queries": len(held_ids),
                "grid": grid,
                "chosen": {"dense_weight": best["dense_weight"], "rrf_k": best["rrf_k"]},
                "held_out": {
                    "equal_weights": equal["scores"],
                    "tuned": tuned["scores"],
                    "versus": versus,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved tuning results to {args.output}")
    print(
        "To produce the run over all queries (saved as results/runs/hybrid.json), "
        "use:\n"
        f"  python3 -m src.evaluation.evaluate_hybrid --dense-weight "
        f"{best['dense_weight']:g} --rrf-k {best['rrf_k']}"
    )


if __name__ == "__main__":
    main()

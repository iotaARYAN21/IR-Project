"""Compare two saved evaluation runs with a paired bootstrap test.

Example:
    python3 -m src.evaluation.compare results/bm25_k10.json \
        results/bm25_reranker_k10.json
"""

import argparse
import random
from collections.abc import Sequence

from src.evaluation.latency import percentile
from src.evaluation.results import load_results

METRICS = ("ndcg", "mrr", "recall")
TIE_TOLERANCE = 1e-12


def paired_bootstrap(
    differences: Sequence[float], resamples: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Bootstrap the mean of per-query score differences (B minus A).

    Returns the mean difference, a 95% confidence interval and a two-sided
    p-value. The p-value is twice the smaller share of resampled means on
    either side of zero, so it can only be as small as about 2 / resamples.
    """
    if not differences:
        raise ValueError("differences must not be empty")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")

    rng = random.Random(seed)
    count = len(differences)
    means = [sum(rng.choices(differences, k=count)) / count for _ in range(resamples)]
    at_most_zero = sum(mean <= 0 for mean in means) / resamples
    at_least_zero = sum(mean >= 0 for mean in means) / resamples
    return {
        "mean_diff": sum(differences) / count,
        "ci_low": percentile(means, 2.5),
        "ci_high": percentile(means, 97.5),
        "p_value": min(1.0, 2 * min(at_most_zero, at_least_zero)),
    }


def win_loss_tie(differences: Sequence[float]) -> dict[str, int]:
    """Count queries where B beat A, lost to A, or tied."""
    wins = sum(diff > TIE_TOLERANCE for diff in differences)
    losses = sum(diff < -TIE_TOLERANCE for diff in differences)
    return {"wins": wins, "losses": losses, "ties": len(differences) - wins - losses}


def compare_runs(
    run_a: dict, run_b: dict, resamples: int = 10000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Compare two loaded results files on the queries they share."""
    if run_a["k"] != run_b["k"]:
        raise ValueError(
            f"Runs use different cutoffs (k={run_a['k']} vs k={run_b['k']})"
        )

    shared = sorted(set(run_a["per_query"]) & set(run_b["per_query"]))
    if not shared:
        raise ValueError("The two runs have no queries in common")

    comparison = {}
    for metric in METRICS:
        scores_a = [run_a["per_query"][query_id][metric] for query_id in shared]
        scores_b = [run_b["per_query"][query_id][metric] for query_id in shared]
        differences = [b - a for a, b in zip(scores_a, scores_b)]
        comparison[metric] = {
            "mean_a": sum(scores_a) / len(shared),
            "mean_b": sum(scores_b) / len(shared),
            "queries": len(shared),
            "resamples": resamples,
            **paired_bootstrap(differences, resamples, seed),
            **win_loss_tie(differences),
        }
    return comparison


def format_comparison(
    name_a: str, name_b: str, k: int, comparison: dict[str, dict[str, float]]
) -> str:
    lines = [f"A = {name_a}", f"B = {name_b}", f"Cutoff k = {k}", ""]
    for metric, stats in comparison.items():
        p_value = stats["p_value"]
        p_text = f"{p_value:.4f}" if p_value > 0 else f"< {2 / stats['resamples']:.4f}"
        lines.append(
            f"{metric.upper()}@{k}: A {stats['mean_a']:.4f}  B {stats['mean_b']:.4f}  "
            f"diff {stats['mean_diff']:+.4f} "
            f"(95% CI {stats['ci_low']:+.4f} to {stats['ci_high']:+.4f}), "
            f"p = {p_text}"
        )
        lines.append(
            f"    B better on {stats['wins']} queries, worse on "
            f"{stats['losses']}, tied on {stats['ties']} "
            f"(of {int(stats['queries'])})"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap comparison of two saved evaluation runs."
    )
    parser.add_argument("run_a", help="Results JSON of the baseline (A).")
    parser.add_argument("run_b", help="Results JSON of the system to test (B).")
    parser.add_argument(
        "--resamples", type=int, default=10000, help="Bootstrap resamples."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be at least 1")

    run_a = load_results(args.run_a)
    run_b = load_results(args.run_b)
    comparison = compare_runs(run_a, run_b, args.resamples, args.seed)
    print(format_comparison(run_a["system"], run_b["system"], run_a["k"], comparison))


if __name__ == "__main__":
    main()

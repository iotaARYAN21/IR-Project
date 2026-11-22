"""Helpers for summarising per-query latency measurements."""

import statistics
from collections.abc import Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    """Return the pct-th percentile (0-100) using linear interpolation."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")

    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latencies(
    seconds: Sequence[float], warmup: int = 0
) -> dict[str, float]:
    """Summarise per-query latencies given in seconds."""
    if warmup < 0:
        raise ValueError("warmup must not be negative")
    if not seconds:
        raise ValueError("seconds must not be empty")

    measured = list(seconds[warmup:]) if warmup < len(seconds) else list(seconds)
    total = sum(measured)
    return {
        "count": float(len(measured)),
        "mean_ms": statistics.fmean(measured) * 1000,
        "median_ms": statistics.median(measured) * 1000,
        "p95_ms": percentile(measured, 95) * 1000,
        "p99_ms": percentile(measured, 99) * 1000,
        "total_s": total,
        "qps": len(measured) / total if total else float("inf"),
    }


def format_latency(label: str, stats: dict[str, float]) -> str:
    """One printable line for a summarize_latencies result."""
    return (
        f"{label}: median {stats['median_ms']:.1f} ms, "
        f"mean {stats['mean_ms']:.1f} ms, "
        f"p95 {stats['p95_ms']:.1f} ms, "
        f"p99 {stats['p99_ms']:.1f} ms "
        f"({int(stats['count'])} queries, {stats['qps']:.1f} queries/s)"
    )


def estimate_total_latency(bm25_run: dict, dense_run: dict, fusion: dict):
    """Mean latency if BM25, dense search and fusion run one after the other."""
    parts = {}
    for name, run in (("bm25", bm25_run), ("dense", dense_run)):
        total = run.get("latency", {}).get("total")
        if not total:
            return None
        parts[name] = total["mean_ms"]
    parts["fusion"] = fusion["mean_ms"]
    return {"mean_ms": sum(parts.values()), "components_mean_ms": parts}

"""Save and load run files: the ranked documents each system returned.

A run file keeps, for every query, a list of (document ID, score) pairs. Hybrid
fusion and reranking read these files, so a slow system only has to be run once.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.evaluation.results import RESULTS_DIR

RUNS_DIR = RESULTS_DIR / "runs"


def default_run_path(system: str, max_queries: int | None = None) -> Path:
    """Default run file, for example results/runs/bm25.json."""
    name = system if max_queries is None else f"{system}_n{max_queries}"
    return RUNS_DIR / f"{name}.json"


def save_run(
    path: Path,
    system: str,
    queries: Mapping[str, Sequence[tuple[str, float]]],
    depth: int,
    latency: dict | None = None,
) -> None:
    """Write a run file. `latency` holds latency summaries, if they were measured."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": system,
        "depth": depth,
        "latency": latency or {},
        "queries": {
            query_id: [[doc_id, score] for doc_id, score in hits]
            for query_id, hits in queries.items()
        },
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file)
        file.write("\n")
    print(f"Saved run to {path}")


def load_run(path: Path) -> dict:
    """Read a run file. Hits are returned as (document ID, score) tuples."""
    with Path(path).open("r", encoding="utf-8") as file:
        run = json.load(file)
    for key in ("system", "depth", "queries"):
        if key not in run:
            raise ValueError(f"{path} is not a run file (missing '{key}')")
    run["queries"] = {
        query_id: [(doc_id, float(score)) for doc_id, score in hits]
        for query_id, hits in run["queries"].items()
    }
    return run

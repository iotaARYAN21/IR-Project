"""Save and load evaluation results as JSON files."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def default_output_path(system: str, k: int, max_queries: int | None = None) -> Path:
    """Default results file, for example results/bm25_k10.json."""
    name = f"{system}_k{k}"
    if max_queries is not None:
        name += f"_n{max_queries}"
    return RESULTS_DIR / f"{name}.json"


def save_results(path: Path, results: dict) -> None:
    """Write a results dictionary to path as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
    print(f"Saved results to {path}")


def load_results(path: Path) -> dict:
    """Read a results file written by save_results."""
    with Path(path).open("r", encoding="utf-8") as file:
        results = json.load(file)
    for key in ("system", "k", "per_query"):
        if key not in results:
            raise ValueError(f"{path} is not a results file (missing '{key}')")
    return results

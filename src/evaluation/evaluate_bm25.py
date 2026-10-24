"""Evaluate the BM25 baseline against the NQ test qrels."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.indexing.config import DATASET_DIR, INDEX_NAME, create_es_client
from src.search.bm25 import search
from src.evaluation.metrics import evaluate_rankings

QUERIES_FILE = DATASET_DIR / "queries.jsonl"
QRELS_FILE = DATASET_DIR / "qrels" / "test.tsv"


def load_queries(path: Path = QUERIES_FILE) -> dict[str, str]:
    """Read BEIR-format queries from a JSONL file."""
    queries = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}"
                ) from exc

            query_id = record.get("_id")
            text = record.get("text")
            if query_id is None or not isinstance(text, str):
                raise ValueError(
                    f"Expected '_id' and string 'text' in {path} "
                    f"on line {line_number}"
                )
            query_id = str(query_id)
            if query_id in queries:
                raise ValueError(f"Duplicate query ID '{query_id}' in {path}")
            queries[query_id] = text

    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def load_qrels(path: Path = QRELS_FILE) -> dict[str, dict[str, float]]:
    """Read tab-separated BEIR qrels with query-id, corpus-id, and score."""
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        expected_columns = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not expected_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Expected TSV columns {sorted(expected_columns)} in {path}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                query_id = row["query-id"]
                doc_id = row["corpus-id"]
                relevance = float(row["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid qrel in {path} on line {line_number}"
                ) from exc

            if not query_id or not doc_id or not math.isfinite(relevance):
                raise ValueError(
                    f"Invalid qrel in {path} on line {line_number}"
                )
            if relevance < 0:
                raise ValueError(
                    f"Negative relevance in {path} on line {line_number}"
                )
            if doc_id in qrels[query_id]:
                raise ValueError(
                    f"Duplicate qrel for query '{query_id}', document "
                    f"'{doc_id}' in {path}"
                )
            qrels[query_id][doc_id] = relevance

    if not qrels:
        raise ValueError(f"No qrels found in {path}")
    return dict(qrels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Elasticsearch BM25 on the NQ test set."
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Rank cutoff for NDCG and MRR."
    )
    parser.add_argument(
        "--index",
        default=INDEX_NAME,
        help=f"Elasticsearch index to search (default: {INDEX_NAME}).",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    queries = load_queries()
    qrels = load_qrels()
    missing_queries = sorted(set(qrels) - set(queries))
    if missing_queries:
        raise ValueError(
            f"{len(missing_queries)} qrel query IDs are missing from "
            f"{QUERIES_FILE}: {', '.join(missing_queries[:5])}"
        )

    client = create_es_client()
    try:
        rankings = {
            query_id: search(client, queries[query_id], args.k, args.index)
            for query_id in qrels
        }
    finally:
        client.close()

    scores = evaluate_rankings(rankings, qrels, args.k)
    print(f"Evaluated BM25 on {len(qrels)} queries (top {args.k})")
    print(f"NDCG@{args.k}: {scores[f'ndcg@{args.k}']:.4f}")
    print(f"MRR@{args.k}:  {scores[f'mrr@{args.k}']:.4f}")


if __name__ == "__main__":
    main()

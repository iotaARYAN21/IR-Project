import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.summarize import (
    build_latency_table,
    build_table,
    format_versus,
    load_all,
    mean_latency_ms,
    pool_recall,
)


def fake_results(ndcg, latency=None, queries=3452):
    return {
        "scores": {"ndcg@10": ndcg, "mrr@10": ndcg + 0.1, "recall@10": ndcg + 0.2},
        "num_queries": queries,
        "latency": latency or {},
    }


class MeanLatencyTests(unittest.TestCase):
    def test_prefers_the_estimated_total(self):
        latency = {"estimated_total": {"mean_ms": 400.0}, "fusion": {"mean_ms": 0.1}}
        self.assertEqual(mean_latency_ms({"latency": latency}), 400.0)

    def test_uses_total_then_bm25_and_none_when_not_measured(self):
        self.assertEqual(mean_latency_ms({"latency": {"total": {"mean_ms": 5.0}}}), 5.0)
        self.assertEqual(mean_latency_ms({"latency": {"bm25": {"mean_ms": 3.0}}}), 3.0)
        self.assertIsNone(mean_latency_ms({"latency": {}}))
        self.assertIsNone(mean_latency_ms({"latency": {"estimated_total": None}}))


class TableTests(unittest.TestCase):
    def test_rows_show_scores_and_dashes_for_missing_systems(self):
        table = build_table(
            {"bm25": fake_results(0.3, {"bm25": {"mean_ms": 4.0}}), "dense": None}, 10
        )
        lines = table.split("\n")
        self.assertIn("NDCG@10", lines[0])
        self.assertIn(
            "| BM25 | 0.3000 | 0.4000 | 0.5000 |  | 4.0 | baseline | 3452 |", table
        )
        self.assertIn("| Dense | not run |", table)
        self.assertIn("| Equal-weight hybrid + reranker | not run |", table)

    def test_table_shows_pool_recall_and_significance(self):
        table = build_table(
            {"bm25": fake_results(0.3), "dense": fake_results(0.4)},
            10,
            recall_pool={"bm25": 0.6, "dense": 0.9},
            versus={"dense": (0.1, 0.0)},
            resamples=1000,
        )
        self.assertIn("| 0.6000 |", table)
        self.assertIn("+0.1000* (p < 0.0020)", table)

    def test_load_all_reads_files_by_default_name(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "bm25_k10.json").write_text(json.dumps(fake_results(0.3)))
            loaded = load_all(10, directory)
        self.assertEqual(loaded["bm25"]["scores"]["ndcg@10"], 0.3)
        self.assertIsNone(loaded["dense"])


class PoolRecallTests(unittest.TestCase):
    def test_averages_recall_over_qrel_queries_and_counts_missing_as_zero(self):
        run = {"queries": {"q1": [("a", 1.0), ("b", 0.5)]}}
        qrels = {"q1": {"b": 1.0}, "q2": {"x": 1.0}}
        self.assertEqual(pool_recall(run, qrels, depth=2), 0.5)
        self.assertEqual(pool_recall(run, qrels, depth=1), 0.0)


class FormatTests(unittest.TestCase):
    def test_marks_only_significant_differences(self):
        self.assertEqual(format_versus(0.05, 0.2, 1000), "+0.0500 (p = 0.2000)")
        self.assertEqual(format_versus(-0.05, 0.01, 1000), "-0.0500* (p = 0.0100)")

    def test_latency_table_lists_only_timed_stages(self):
        stats = {"mean_ms": 2.0, "median_ms": 1.5, "p95_ms": 4.0}
        table = build_latency_table({"bm25": {"latency": {"bm25": stats}}})
        self.assertIn("| BM25 search | 2.0 | 1.5 | 4.0 |", table)
        self.assertNotIn("Dense", table)


if __name__ == "__main__":
    unittest.main()

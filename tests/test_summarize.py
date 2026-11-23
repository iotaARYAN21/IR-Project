import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.summarize import build_table, load_all, mean_latency_ms


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
        self.assertIn("| BM25 | 0.3000 | 0.4000 | 0.5000 | 4.0 | 3452 |", table)
        self.assertIn("| Dense | not run |", table)
        self.assertIn("| Hybrid + reranker | not run |", table)

    def test_load_all_reads_files_by_default_name(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "bm25_k10.json").write_text(json.dumps(fake_results(0.3)))
            loaded = load_all(10, directory)
        self.assertEqual(loaded["bm25"]["scores"]["ndcg@10"], 0.3)
        self.assertIsNone(loaded["dense"])


if __name__ == "__main__":
    unittest.main()

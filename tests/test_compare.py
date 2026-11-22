import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.compare import compare_runs, paired_bootstrap, win_loss_tie
from src.evaluation.results import default_output_path, load_results, save_results


def make_run(system, ndcg_by_query, k=10):
    return {
        "system": system,
        "k": k,
        "per_query": {
            query_id: {"ndcg": value, "mrr": value, "recall": value}
            for query_id, value in ndcg_by_query.items()
        },
    }


class PairedBootstrapTests(unittest.TestCase):
    def test_clear_improvement_is_significant(self):
        stats = paired_bootstrap([0.2] * 50 + [0.1] * 50, resamples=500)
        self.assertAlmostEqual(stats["mean_diff"], 0.15)
        self.assertGreater(stats["ci_low"], 0)
        self.assertLess(stats["p_value"], 0.05)

    def test_no_difference_has_p_value_one(self):
        stats = paired_bootstrap([0.0] * 20, resamples=200)
        self.assertEqual(stats["mean_diff"], 0.0)
        self.assertEqual(stats["p_value"], 1.0)

    def test_symmetric_noise_is_not_significant(self):
        stats = paired_bootstrap([0.5, -0.5] * 50, resamples=500)
        self.assertGreater(stats["p_value"], 0.05)
        self.assertLess(stats["ci_low"], 0)
        self.assertGreater(stats["ci_high"], 0)

    def test_same_seed_gives_same_result(self):
        diffs = [0.3, -0.1, 0.0, 0.2, 0.1]
        self.assertEqual(
            paired_bootstrap(diffs, resamples=100, seed=7),
            paired_bootstrap(diffs, resamples=100, seed=7),
        )

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([])


class WinLossTieTests(unittest.TestCase):
    def test_counts_each_outcome(self):
        self.assertEqual(
            win_loss_tie([0.5, -0.2, 0.0, 0.1]),
            {"wins": 2, "losses": 1, "ties": 1},
        )


class CompareRunsTests(unittest.TestCase):
    def test_compares_only_shared_queries(self):
        run_a = make_run("a", {"q1": 0.0, "q2": 0.5, "only-a": 1.0})
        run_b = make_run("b", {"q1": 1.0, "q2": 0.5, "only-b": 0.0})
        result = compare_runs(run_a, run_b, resamples=50)
        ndcg = result["ndcg"]
        self.assertEqual(ndcg["queries"], 2)
        self.assertAlmostEqual(ndcg["mean_a"], 0.25)
        self.assertAlmostEqual(ndcg["mean_b"], 0.75)
        self.assertEqual((ndcg["wins"], ndcg["losses"], ndcg["ties"]), (1, 0, 1))

    def test_rejects_different_cutoffs(self):
        with self.assertRaises(ValueError):
            compare_runs(
                make_run("a", {"q1": 1.0}, k=10),
                make_run("b", {"q1": 1.0}, k=20),
            )

    def test_rejects_runs_with_no_shared_queries(self):
        with self.assertRaises(ValueError):
            compare_runs(make_run("a", {"q1": 1.0}), make_run("b", {"q2": 1.0}))


class ResultsFileTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        run = make_run("bm25", {"q1": 0.5})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "run.json"
            save_results(path, run)
            self.assertEqual(load_results(path), run)

    def test_load_rejects_files_that_are_not_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.json"
            path.write_text(json.dumps({"hello": "world"}))
            with self.assertRaises(ValueError):
                load_results(path)

    def test_default_output_path_separates_limited_runs(self):
        full = default_output_path("bm25_reranker", 10)
        quick = default_output_path("bm25_reranker", 10, max_queries=50)
        self.assertEqual(full.name, "bm25_reranker_k10.json")
        self.assertEqual(quick.name, "bm25_reranker_k10_n50.json")


if __name__ == "__main__":
    unittest.main()

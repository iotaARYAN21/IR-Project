import tempfile
import unittest
from pathlib import Path

from src.evaluation.latency import estimate_total_latency
from src.evaluation.runs import default_run_path, load_run, save_run
from src.search.hybrid import reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_scores_follow_the_formula(self):
        fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], rrf_k=60))
        self.assertAlmostEqual(fused["a"], 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused["c"], 1 / 62)

    def test_document_in_both_lists_beats_documents_in_only_one(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
        self.assertEqual([doc_id for doc_id, _ in fused][0], "b")

    def test_weights_change_the_order(self):
        rankings = [["a"], ["b"]]
        self.assertEqual(reciprocal_rank_fusion(rankings, weights=[2, 1])[0][0], "a")
        self.assertEqual(reciprocal_rank_fusion(rankings, weights=[1, 2])[0][0], "b")

    def test_ties_keep_the_order_of_the_earlier_list(self):
        fused = reciprocal_rank_fusion([["a"], ["b"]], rrf_k=0, weights=[1, 1])
        # a is rank 1 in list one, b is rank 1 in list two: equal scores
        self.assertEqual([doc_id for doc_id, _ in fused], ["a", "b"])

    def test_duplicates_within_a_list_count_once(self):
        fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], rrf_k=60))
        self.assertAlmostEqual(fused["a"], 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 63)

    def test_empty_inputs(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"]], rrf_k=-1)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


class RunFileTests(unittest.TestCase):
    def test_round_trip_keeps_ids_scores_and_order(self):
        queries = {"q1": [("d2", 3.5), ("d1", 1.25)], "q2": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs" / "bm25.json"
            save_run(path, "bm25", queries, depth=2, latency={"total": {"mean_ms": 4.0}})
            run = load_run(path)
        self.assertEqual(run["system"], "bm25")
        self.assertEqual(run["depth"], 2)
        self.assertEqual(run["queries"], queries)
        self.assertEqual(run["latency"]["total"]["mean_ms"], 4.0)

    def test_rejects_files_that_are_not_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.json"
            path.write_text('{"hello": 1}')
            with self.assertRaises(ValueError):
                load_run(path)

    def test_default_run_path_separates_limited_runs(self):
        self.assertEqual(default_run_path("dense").name, "dense.json")
        self.assertEqual(default_run_path("dense", 50).name, "dense_n50.json")


class LatencyEstimateTests(unittest.TestCase):
    def test_sums_the_mean_latencies(self):
        bm25 = {"latency": {"total": {"mean_ms": 5.0}}}
        dense = {"latency": {"total": {"mean_ms": 400.0}}}
        estimate = estimate_total_latency(bm25, dense, {"mean_ms": 0.1})
        self.assertAlmostEqual(estimate["mean_ms"], 405.1)

    def test_none_when_a_run_has_no_latency(self):
        bm25 = {"latency": {"total": {"mean_ms": 5.0}}}
        self.assertIsNone(estimate_total_latency(bm25, {"latency": {}}, {"mean_ms": 0.1}))


if __name__ == "__main__":
    unittest.main()

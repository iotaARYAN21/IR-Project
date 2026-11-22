import math
import unittest

from src.evaluation.metrics import (
    evaluate_rankings,
    mrr_at_k,
    ndcg_at_k,
    per_query_scores,
    recall_at_k,
)


class RankingMetricsTests(unittest.TestCase):
    def test_mrr_uses_rank_of_first_relevant_result(self):
        qrels = {"doc-2": 1.0, "doc-3": 1.0}
        self.assertEqual(mrr_at_k(["doc-1", "doc-2", "doc-3"], qrels), 0.5)

    def test_mrr_returns_zero_when_no_relevant_result_is_retrieved(self):
        self.assertEqual(mrr_at_k(["doc-1"], {"doc-2": 1.0}), 0.0)

    def test_ndcg_is_one_for_ideal_ranking(self):
        qrels = {"doc-1": 3.0, "doc-2": 2.0, "doc-3": 1.0}
        self.assertEqual(ndcg_at_k(["doc-1", "doc-2", "doc-3"], qrels), 1.0)

    def test_ndcg_uses_graded_relevance_and_rank_discount(self):
        qrels = {"doc-1": 3.0, "doc-2": 2.0}
        ideal_dcg = 7.0 + 3.0 / math.log2(3)
        actual_dcg = 3.0 + 7.0 / math.log2(3)
        self.assertAlmostEqual(
            ndcg_at_k(["doc-2", "doc-1"], qrels),
            actual_dcg / ideal_dcg,
        )

    def test_averages_scores_over_all_qrel_queries(self):
        scores = evaluate_rankings(
            {"q1": ["relevant"]},
            {"q1": {"relevant": 1.0}, "q2": {"other": 1.0}},
            k=10,
        )
        self.assertEqual(
            scores, {"ndcg@10": 0.5, "mrr@10": 0.5, "recall@10": 0.5}
        )

    def test_rejects_invalid_cutoff(self):
        with self.assertRaises(ValueError):
            mrr_at_k([], {}, k=0)
        with self.assertRaises(ValueError):
            recall_at_k([], {}, k=0)

    def test_recall_is_one_when_single_relevant_doc_is_first(self):
        self.assertEqual(recall_at_k(["doc-1", "doc-2"], {"doc-1": 1.0}), 1.0)

    def test_recall_is_fraction_of_relevant_docs_in_top_k(self):
        qrels = {"doc-1": 1.0, "doc-3": 1.0}
        self.assertEqual(recall_at_k(["doc-1", "doc-2", "doc-3"], qrels, k=2), 0.5)
        self.assertEqual(recall_at_k(["doc-1", "doc-2", "doc-3"], qrels, k=3), 1.0)

    def test_recall_ignores_zero_relevance_and_handles_no_relevant_docs(self):
        self.assertEqual(recall_at_k(["doc-1"], {"doc-1": 0.0}), 0.0)
        self.assertEqual(recall_at_k(["doc-1"], {"doc-1": 1.0, "doc-2": 0.0}), 1.0)

    def test_per_query_scores_gives_one_entry_per_qrel_query(self):
        scores = per_query_scores(
            {"q1": ["relevant"]},
            {"q1": {"relevant": 1.0}, "q2": {"other": 1.0}},
            k=10,
        )
        self.assertEqual(
            scores,
            {
                "q1": {"ndcg": 1.0, "mrr": 1.0, "recall": 1.0},
                "q2": {"ndcg": 0.0, "mrr": 0.0, "recall": 0.0},
            },
        )

    def test_evaluate_rankings_reports_recall_at_the_same_k(self):
        rankings = {"q1": ["a", "b", "relevant"]}
        qrels = {"q1": {"relevant": 1.0}}
        self.assertEqual(evaluate_rankings(rankings, qrels, k=2)["recall@2"], 0.0)
        self.assertEqual(evaluate_rankings(rankings, qrels, k=3)["recall@3"], 1.0)


if __name__ == "__main__":
    unittest.main()

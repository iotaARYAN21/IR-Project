import json
import tempfile
import unittest
from pathlib import Path

from src.search.cli import Settings, apply_command, format_response, format_timings
from src.search.documents import fetch_documents, load_corpus_texts
from src.search.rerank import rerank_run_queries
from src.search.hybrid import (
    HYBRID_DENSE_WEIGHT,
    HYBRID_RRF_K,
    reciprocal_rank_fusion,
)
from src.search.pipeline import SearchEngine

DOCS = {
    doc_id: {"title": f"Title {doc_id}", "text": f"Text of {doc_id}"}
    for doc_id in "abcdefg"
}


def make_engine(rerank=None):
    calls = {"bm25": [], "dense": [], "fetch": []}

    def bm25(query, limit):
        calls["bm25"].append(limit)
        return [("a", 9.0), ("b", 8.0), ("c", 7.0), ("d", 6.0)][:limit]

    def dense(query, limit):
        calls["dense"].append(limit)
        return [("c", 0.9), ("e", 0.8), ("b", 0.7), ("f", 0.6)][:limit]

    def fetch(ids):
        calls["fetch"].append(list(ids))
        return {doc_id: DOCS[doc_id] for doc_id in ids if doc_id in DOCS}

    return SearchEngine(bm25, dense, fetch, rerank), calls


class SearchEngineTests(unittest.TestCase):
    def test_bm25_keeps_retrieval_order_and_scores(self):
        engine, _ = make_engine()
        response = engine.search("q", method="bm25", k=3)
        self.assertEqual([h.doc_id for h in response.hits], ["a", "b", "c"])
        self.assertEqual(response.hits[0].score, 9.0)
        self.assertEqual(response.hits[0].title, "Title a")

    def test_dense_keeps_retrieval_order(self):
        engine, _ = make_engine()
        response = engine.search("q", method="dense", k=2)
        self.assertEqual([h.doc_id for h in response.hits], ["c", "e"])

    def test_hybrid_matches_reciprocal_rank_fusion(self):
        engine, _ = make_engine()
        response = engine.search("q", method="hybrid", k=5, depth=4)
        expected = reciprocal_rank_fusion(
            [["a", "b", "c", "d"], ["c", "e", "b", "f"]],
            HYBRID_RRF_K,
            [1.0, HYBRID_DENSE_WEIGHT],
        )
        self.assertEqual(
            [h.doc_id for h in response.hits], [d for d, _ in expected][:5]
        )
        self.assertIn("fusion", response.timings_ms)

    def test_hybrid_weights_change_the_result(self):
        engine, _ = make_engine()
        # A weight of zero switches a list off, so the other list decides.
        bm25_only = engine.search("q", method="hybrid", k=1, depth=4, weights=(1, 0))
        dense_only = engine.search("q", method="hybrid", k=1, depth=4, weights=(0, 1))
        self.assertEqual(bm25_only.hits[0].doc_id, "a")
        self.assertEqual(dense_only.hits[0].doc_id, "c")

    def test_rerank_reorders_candidates_and_returns_top_k(self):
        def prefer_later(query, texts):
            return [float(i) for i in range(len(texts))]  # last candidate scores best

        engine, calls = make_engine(rerank=prefer_later)
        response = engine.search("q", method="bm25", k=2, rerank=True, candidates=4)
        self.assertEqual(calls["bm25"], [4])
        self.assertEqual([h.doc_id for h in response.hits], ["d", "c"])
        self.assertEqual(response.hits[0].score, 3.0)
        self.assertTrue(response.reranked)
        self.assertIn("rerank", response.timings_ms)

    def test_rerank_gets_title_and_text_of_each_candidate(self):
        seen = {}

        def capture(query, texts):
            seen["texts"] = list(texts)
            return [0.0] * len(texts)

        engine, _ = make_engine(rerank=capture)
        engine.search("q", method="bm25", k=1, rerank=True, candidates=2)
        self.assertEqual(seen["texts"], ["Title a Text of a", "Title b Text of b"])

    def test_without_rerank_only_k_documents_are_fetched(self):
        engine, calls = make_engine()
        engine.search("q", method="bm25", k=2)
        self.assertEqual(calls["fetch"], [["a", "b"]])

    def test_missing_documents_are_skipped(self):
        engine = SearchEngine(
            bm25=lambda q, n: [("a", 2.0), ("zzz", 1.0), ("b", 0.5)],
            fetch=lambda ids: {i: DOCS[i] for i in ids if i in DOCS},
        )
        response = engine.search("q", method="bm25", k=3)
        self.assertEqual([h.doc_id for h in response.hits], ["a", "b"])

    def test_blank_query_returns_nothing_and_calls_nothing(self):
        engine, calls = make_engine()
        self.assertEqual(engine.search("   ", method="hybrid").hits, [])
        self.assertEqual(calls["bm25"], [])

    def test_timings_include_total(self):
        engine, _ = make_engine()
        timings = engine.search("q", method="bm25").timings_ms
        self.assertIn("bm25", timings)
        self.assertIn("fetch", timings)
        self.assertGreaterEqual(timings["total"], timings["bm25"])

    def test_rejects_bad_arguments_and_missing_components(self):
        engine, _ = make_engine()
        with self.assertRaises(ValueError):
            engine.search("q", method="nope")
        with self.assertRaises(ValueError):
            engine.search("q", k=0)
        with self.assertRaises(RuntimeError):
            engine.search("q", method="bm25", rerank=True)  # no reranker configured


class FetchDocumentsTests(unittest.TestCase):
    class FakeClient:
        def __init__(self):
            self.request = None

        def mget(self, **kwargs):
            self.request = kwargs
            return {
                "docs": [
                    {"_id": "a", "found": True, "_source": {"title": "T", "text": "X"}},
                    {"_id": "b", "found": False},
                    {"_id": "c", "found": True, "_source": {"title": None}},
                ]
            }

    def test_returns_found_documents_with_defaults(self):
        client = self.FakeClient()
        documents = fetch_documents(client, ["a", "b", "c"], "idx")
        self.assertEqual(documents["a"], {"title": "T", "text": "X"})
        self.assertEqual(documents["c"], {"title": "", "text": ""})
        self.assertNotIn("b", documents)
        self.assertEqual(client.request["index"], "idx")
        self.assertEqual(client.request["ids"], ["a", "b", "c"])

    def test_no_ids_makes_no_request(self):
        client = self.FakeClient()
        self.assertEqual(fetch_documents(client, [], "idx"), {})
        self.assertIsNone(client.request)


class LoadCorpusTextsTests(unittest.TestCase):
    def _corpus(self, directory):
        path = Path(directory) / "corpus.jsonl"
        rows = [
            {"_id": "d1", "title": "One", "text": "first"},
            {"_id": "d2", "title": "", "text": "second"},
            {"_id": "d3", "title": "Three", "text": "third"},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_returns_title_and_text_for_requested_ids_only(self):
        with tempfile.TemporaryDirectory() as directory:
            texts = load_corpus_texts(self._corpus(directory), ["d1", "d2", "missing"])
        self.assertEqual(texts, {"d1": "One first", "d2": "second"})

    def test_no_ids_reads_nothing(self):
        self.assertEqual(load_corpus_texts(Path("/no/such/file"), []), {})


class RerankRunQueriesTests(unittest.TestCase):
    RUN = {"q1": [("a", 3.0), ("b", 2.0), ("c", 1.0)], "q2": [("c", 2.0), ("a", 1.0)]}
    DOCS = {"a": "text a", "b": "text b", "c": "text c"}
    QUERIES = {"q1": "first query", "q2": "second query"}

    def test_reorders_by_score_and_reports_one_time_per_query(self):
        def prefer_c_then_a(query, texts):
            order = {"text c": 2.0, "text a": 1.0, "text b": 0.0}
            return [order[t] for t in texts]

        reranked, seconds = rerank_run_queries(
            ["q1", "q2"], self.QUERIES, self.RUN, self.DOCS, prefer_c_then_a, 3
        )
        self.assertEqual([d for d, _ in reranked["q1"]], ["c", "a", "b"])
        self.assertEqual([d for d, _ in reranked["q2"]], ["c", "a"])
        self.assertEqual(reranked["q1"][0], ("c", 2.0))
        self.assertEqual(len(seconds), 2)

    def test_only_the_top_candidates_are_scored(self):
        seen = []
        rerank_run_queries(
            ["q1"], self.QUERIES, self.RUN, self.DOCS,
            lambda q, texts: seen.append((q, list(texts))) or [0.0] * len(texts), 2,
        )
        self.assertEqual(seen, [("first query", ["text a", "text b"])])

    def test_equal_scores_keep_the_original_order(self):
        reranked, _ = rerank_run_queries(
            ["q1"], self.QUERIES, self.RUN, self.DOCS, lambda q, t: [1.0] * len(t), 3
        )
        self.assertEqual([d for d, _ in reranked["q1"]], ["a", "b", "c"])

    def test_missing_text_and_bad_scorer_are_errors(self):
        with self.assertRaises(ValueError):
            rerank_run_queries(["q1"], self.QUERIES, self.RUN, {"a": "x"}, lambda q, t: [0.0] * len(t), 3)
        with self.assertRaises(ValueError):
            rerank_run_queries(["q1"], self.QUERIES, self.RUN, self.DOCS, lambda q, t: [0.0], 3)
        with self.assertRaises(ValueError):
            rerank_run_queries(["q1"], self.QUERIES, self.RUN, self.DOCS, lambda q, t: [], 0)


class CommandTests(unittest.TestCase):
    def test_commands_change_settings(self):
        settings = Settings()
        apply_command(settings, ":method dense")
        apply_command(settings, ":rerank on")
        apply_command(settings, ":k 5")
        apply_command(settings, ":candidates 20")
        self.assertEqual(
            (settings.method, settings.rerank, settings.k, settings.candidates),
            ("dense", True, 5, 20),
        )
        apply_command(settings, ":rerank off")
        self.assertFalse(settings.rerank)

    def test_bad_commands_raise_and_leave_settings_alone(self):
        settings = Settings()
        for line in (":method nope", ":rerank maybe", ":k 0", ":k abc", ":bogus", ":"):
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    apply_command(settings, line)
        self.assertEqual(settings, Settings())


class FormatTests(unittest.TestCase):
    def test_response_shows_rank_title_id_score_and_text(self):
        engine, _ = make_engine()
        response = engine.search("q", method="bm25", k=2)
        text = format_response(response)
        self.assertIn(" 1. Title a", text)
        self.assertIn("[a, BM25 score 9.000]", text)
        self.assertIn("Text of b", text)

    def test_empty_response_says_so(self):
        engine, _ = make_engine()
        self.assertEqual(format_response(engine.search("", method="bm25")), "No results.")

    def test_timings_are_listed_with_total_last(self):
        line = format_timings({"bm25": 4.0, "fetch": 6.5, "total": 11.0})
        self.assertEqual(line, "bm25 4.0 ms | fetch 6.5 ms | total 11.0 ms")


if __name__ == "__main__":
    unittest.main()

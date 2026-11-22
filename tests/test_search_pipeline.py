import unittest

from src.search.cli import Settings, apply_command, format_response, format_timings
from src.search.documents import fetch_documents
from src.search.hybrid import reciprocal_rank_fusion
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
            [["a", "b", "c", "d"], ["c", "e", "b", "f"]], 60, [1.0, 1.0]
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

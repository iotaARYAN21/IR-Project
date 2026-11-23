import unittest

from src.search.bm25 import search_proximity_with_scores


class ProximitySearchTests(unittest.TestCase):
    class FakeClient:
        def search(self, **kwargs):
            self.request = kwargs
            return {"hits": {"hits": [{"_id": "doc1", "_score": 1.5}]}}

    def test_builds_phrase_query_with_fields_and_slop(self):
        client = self.FakeClient()
        results = search_proximity_with_scores(
            client, "machine learning", limit=7, slop=3, index_name="custom"
        )
        self.assertEqual(results, [("doc1", 1.5)])
        self.assertEqual(client.request["index"], "custom")
        self.assertEqual(client.request["size"], 7)
        self.assertEqual(client.request["query"]["multi_match"], {
            "query": "machine learning",
            "type": "phrase",
            "fields": ["title^2", "text"],
            "slop": 3,
        })

    def test_rejects_negative_slop(self):
        with self.assertRaises(ValueError):
            search_proximity_with_scores(self.FakeClient(), "q", slop=-1)


if __name__ == "__main__":
    unittest.main()

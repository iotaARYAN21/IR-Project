import unittest

from src.search.bm25 import search_with_scores


class ProximitySearchTests(unittest.TestCase):
    class FakeClient:
        def search(self, **kwargs):
            self.request = kwargs
            return {"hits": {"hits": [{"_id": "doc1", "_score": 1.5}]}}

    def test_plain_search_is_unchanged_without_proximity(self):
        client = self.FakeClient()
        search_with_scores(client, "machine learning", limit=7, index_name="custom")
        self.assertEqual(client.request["index"], "custom")
        self.assertEqual(client.request["size"], 7)
        self.assertEqual(
            client.request["query"],
            {"multi_match": {"query": "machine learning", "fields": ["title^2", "text"]}},
        )

    def test_proximity_adds_a_phrase_boost_without_filtering(self):
        client = self.FakeClient()
        results = search_with_scores(client, "machine learning", proximity_slop=3)
        self.assertEqual(results, [("doc1", 1.5)])
        query = client.request["query"]["bool"]
        self.assertEqual(
            query["must"],
            [{"multi_match": {"query": "machine learning", "fields": ["title^2", "text"]}}],
        )
        self.assertEqual(
            query["should"],
            [
                {
                    "multi_match": {
                        "query": "machine learning",
                        "type": "phrase",
                        "fields": ["title^2", "text"],
                        "slop": 3,
                    }
                }
            ],
        )

    def test_slop_zero_is_a_valid_boost(self):
        client = self.FakeClient()
        search_with_scores(client, "q", proximity_slop=0)
        self.assertEqual(client.request["query"]["bool"]["should"][0]["multi_match"]["slop"], 0)

    def test_exact_phrase_makes_the_boost_redundant(self):
        client = self.FakeClient()
        search_with_scores(client, "q", exact_phrase=True, proximity_slop=2)
        self.assertEqual(client.request["query"]["multi_match"]["type"], "phrase")
        self.assertNotIn("slop", client.request["query"]["multi_match"])

    def test_rejects_negative_slop(self):
        with self.assertRaises(ValueError):
            search_with_scores(self.FakeClient(), "q", proximity_slop=-1)

    def test_builds_custom_boolean_search(self):
        client = self.FakeClient()
        search_with_scores(
            client, "climate change", title_only=True,
            must_include="policy", must_exclude="sports", exact_phrase=True,
        )
        self.assertEqual(client.request["query"], {
            "bool": {
                "must": [
                    {"match_phrase": {"title": "climate change"}},
                    {"multi_match": {"query": "policy", "fields": ["title^2"]}},
                ],
                "should": [],
                "must_not": [
                    {"multi_match": {"query": "sports", "fields": ["title^2"]}},
                ],
            }
        })


if __name__ == "__main__":
    unittest.main()

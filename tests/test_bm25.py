import unittest

from src.search.bm25 import search_with_scores, suggest_titles


def ALL_WORDS(text):
    return {
        "multi_match": {
            "query": text,
            "type": "cross_fields",
            "operator": "and",
            "fields": ["title^2", "text"],
        }
    }


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

    def test_must_include_and_exclude_need_all_their_words(self):
        client = self.FakeClient()
        search_with_scores(client, "q", must_include="climate policy", must_exclude="sports football")
        query = client.request["query"]["bool"]
        self.assertEqual(query["must"][1], ALL_WORDS("climate policy"))
        self.assertEqual(query["must_not"], [ALL_WORDS("sports football")])

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
                    ALL_WORDS("policy"),
                ],
                "should": [],
                "must_not": [
                    ALL_WORDS("sports"),
                ],
            }
        })


class SuggestTitlesTests(unittest.TestCase):
    class FakeClient:
        def search(self, **kwargs):
            self.request = kwargs
            return {
                "aggregations": {
                    "titles": {
                        "buckets": [
                            {"key": "Jaguar", "doc_count": 40},
                            {"key": "Jaguar Cars", "doc_count": 12},
                        ]
                    }
                }
            }

    def test_asks_for_titles_starting_with_the_prefix_ignoring_case(self):
        client = self.FakeClient()
        titles = suggest_titles(client, " jag ", limit=5, index_name="custom")
        self.assertEqual(titles, [("Jaguar", 40), ("Jaguar Cars", 12)])
        self.assertEqual(client.request["index"], "custom")
        self.assertEqual(client.request["size"], 0)
        self.assertEqual(
            client.request["query"],
            {"prefix": {"title.raw": {"value": "jag", "case_insensitive": True}}},
        )
        self.assertEqual(
            client.request["aggs"],
            {"titles": {"terms": {"field": "title.raw", "size": 5}}},
        )

    def test_blank_prefix_returns_nothing_without_a_query(self):
        client = self.FakeClient()
        self.assertEqual(suggest_titles(client, "   "), [])
        self.assertFalse(hasattr(client, "request"))

    def test_rejects_bad_limit(self):
        with self.assertRaises(ValueError):
            suggest_titles(self.FakeClient(), "jag", limit=0)


if __name__ == "__main__":
    unittest.main()

import unittest

from src.evaluation.tune_hybrid import choose_best, fuse_rankings, split_queries


class SplitTests(unittest.TestCase):
    def test_split_is_disjoint_complete_and_reproducible(self):
        ids = [f"q{i}" for i in range(100)]
        dev, held = split_queries(ids, 0.3, seed=1)
        self.assertEqual(len(dev), 30)
        self.assertEqual(set(dev) | set(held), set(ids))
        self.assertFalse(set(dev) & set(held))
        self.assertEqual((dev, held), split_queries(reversed(list(ids)), 0.3, seed=1))
        self.assertNotEqual(dev, split_queries(ids, 0.3, seed=2)[0])

    def test_rejects_bad_fraction_and_tiny_input(self):
        with self.assertRaises(ValueError):
            split_queries(["a", "b"], 1.0, 0)
        with self.assertRaises(ValueError):
            split_queries(["a"], 0.5, 0)


class ChooseBestTests(unittest.TestCase):
    def entry(self, weight, rrf_k, ndcg):
        return {"dense_weight": weight, "rrf_k": rrf_k, "dev": {"ndcg@10": ndcg}}

    def test_picks_highest_dev_ndcg(self):
        grid = [self.entry(1, 60, 0.4), self.entry(2, 60, 0.5)]
        self.assertEqual(choose_best(grid, 10)["dense_weight"], 2)

    def test_tie_goes_to_setting_closest_to_default(self):
        grid = [self.entry(4, 60, 0.5), self.entry(1.5, 60, 0.5)]
        self.assertEqual(choose_best(grid, 10)["dense_weight"], 1.5)


class FuseTests(unittest.TestCase):
    def test_dense_weight_changes_the_order(self):
        bm25 = {"queries": {"q": [("a", 1.0), ("b", 0.5)]}}
        dense = {"queries": {"q": [("b", 1.0), ("a", 0.5)]}}
        equal = fuse_rankings(bm25, dense, ["q"], 1.0, 60, 100)["q"]
        heavy = fuse_rankings(bm25, dense, ["q"], 3.0, 60, 100)["q"]
        self.assertEqual(equal[0], "a")  # tie on score, first inserted wins
        self.assertEqual(heavy[0], "b")


if __name__ == "__main__":
    unittest.main()

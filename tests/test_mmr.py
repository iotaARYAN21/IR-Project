import unittest

from src.search.mmr import jaccard, mmr_order, normalize, passage_similarity, word_set

# Candidates 0 and 1 are near-duplicates; 2 is different but a bit less relevant.
RELEVANCE = [1.0, 0.9, 0.7]
SIMILARITY = [
    [1.0, 0.95, 0.1],
    [0.95, 1.0, 0.1],
    [0.1, 0.1, 1.0],
]


class NormalizeTests(unittest.TestCase):
    def test_scales_to_zero_one(self):
        self.assertEqual(normalize([2.0, 4.0, 6.0]), [0.0, 0.5, 1.0])

    def test_equal_scores_all_become_one(self):
        self.assertEqual(normalize([3.0, 3.0]), [1.0, 1.0])


class SimilarityTests(unittest.TestCase):
    def test_word_set_lowercases_and_drops_common_words(self):
        self.assertEqual(word_set("The Cat, and the HAT!"), frozenset({"cat", "hat"}))

    def test_jaccard_is_shared_over_total(self):
        self.assertEqual(jaccard(frozenset("abc"), frozenset("bcd")), 0.5)
        self.assertEqual(jaccard(frozenset(), frozenset()), 0.0)
        self.assertEqual(jaccard(frozenset("ab"), frozenset("ab")), 1.0)

    def test_same_title_counts_as_a_repeat_whatever_the_text(self):
        matrix = passage_similarity(
            ["Nvidia", " nvidia ", "Apple"],
            ["graphics cards", "founded in 1993", "fruit trees"],
        )
        self.assertEqual(matrix[0][1], 1.0)
        self.assertEqual(matrix[0][2], 0.0)

    def test_different_titles_use_word_overlap(self):
        matrix = passage_similarity(
            ["GPU", "Graphics card"],
            ["a chip for rendering graphics", "a chip for rendering graphics fast"],
        )
        self.assertEqual(matrix[0][1], 0.5)  # 3 shared words of 6
        self.assertLess(matrix[0][1], 1.0)

    def test_matrix_is_symmetric_with_ones_on_the_diagonal(self):
        matrix = passage_similarity(["a", "b", "c"], ["x y", "y z", "z w"])
        for i in range(3):
            self.assertEqual(matrix[i][i], 1.0)
            for j in range(3):
                self.assertEqual(matrix[i][j], matrix[j][i])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            passage_similarity(["a"], [])


class MmrTests(unittest.TestCase):
    def test_zero_diversity_keeps_the_relevance_order(self):
        self.assertEqual(mmr_order(RELEVANCE, SIMILARITY, 3, 0.0), [0, 1, 2])

    def test_diversity_pushes_the_near_duplicate_down(self):
        self.assertEqual(mmr_order(RELEVANCE, SIMILARITY, 3, 0.5), [0, 2, 1])

    def test_first_pick_is_always_the_most_relevant(self):
        for diversity in (0.0, 0.5, 1.0):
            self.assertEqual(mmr_order(RELEVANCE, SIMILARITY, 3, diversity)[0], 0)

    def test_returns_at_most_k_and_never_repeats(self):
        order = mmr_order(RELEVANCE, SIMILARITY, 2, 0.5)
        self.assertEqual(len(order), 2)
        self.assertEqual(len(set(order)), 2)
        self.assertEqual(len(mmr_order(RELEVANCE, SIMILARITY, 10, 0.5)), 3)

    def test_ties_keep_the_original_order(self):
        identical = [[1.0] * 3 for _ in range(3)]
        self.assertEqual(mmr_order([1.0, 1.0, 1.0], identical, 3, 0.5), [0, 1, 2])

    def test_empty_input(self):
        self.assertEqual(mmr_order([], [], 3, 0.5), [])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            mmr_order(RELEVANCE, SIMILARITY, 3, 1.5)
        with self.assertRaises(ValueError):
            mmr_order(RELEVANCE, SIMILARITY, 0, 0.5)
        with self.assertRaises(ValueError):
            mmr_order(RELEVANCE, SIMILARITY[:2], 3, 0.5)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from src.indexing.encode_corpus import save_chunk
    from src.search.dense import DenseIndex, load_embeddings


def unit_vectors(count, dimension, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dimension)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def reference_top_k(vectors, query, k):
    scores = vectors @ query
    return list(np.argsort(-scores)[:k])


@unittest.skipIf(np is None, "numpy is not installed")
class DenseIndexTests(unittest.TestCase):
    def setUp(self):
        self.vectors = unit_vectors(250, 16)
        self.doc_ids = [f"doc{i}" for i in range(250)]
        self.queries = unit_vectors(7, 16, seed=1)

    def test_matches_a_full_sort_across_chunk_boundaries(self):
        # chunk_rows=60 forces several chunks, the last one smaller than the rest
        index = DenseIndex(self.doc_ids, self.vectors, chunk_rows=60)
        results = index.search_vectors(self.queries, k=10, query_block=3)
        for query, hits in zip(self.queries, results):
            expected = [self.doc_ids[i] for i in reference_top_k(self.vectors, query, 10)]
            self.assertEqual([doc_id for doc_id, _ in hits], expected)

    def test_scores_are_descending_and_match_dot_products(self):
        index = DenseIndex(self.doc_ids, self.vectors, chunk_rows=100)
        hits = index.search_vectors(self.queries[:1], k=5)[0]
        scores = [score for _, score in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        best_id, best_score = hits[0]
        row = self.doc_ids.index(best_id)
        self.assertAlmostEqual(best_score, float(self.vectors[row] @ self.queries[0]), 5)

    def test_a_document_finds_itself_first(self):
        index = DenseIndex(self.doc_ids, self.vectors, chunk_rows=70)
        self.assertEqual(index.search(self.vectors[123], k=3)[0], "doc123")

    def test_k_larger_than_the_index_returns_every_document(self):
        index = DenseIndex(self.doc_ids[:4], self.vectors[:4], chunk_rows=3)
        self.assertEqual(len(index.search(self.queries[0], k=50)), 4)

    def test_vectors_for_returns_rows_in_the_requested_order(self):
        index = DenseIndex(self.doc_ids, self.vectors)
        found = index.vectors_for(["doc7", "doc2"])
        np.testing.assert_array_equal(found, self.vectors[[7, 2]])
        with self.assertRaises(KeyError):
            index.vectors_for(["missing"])

    def test_rejects_bad_input(self):
        index = DenseIndex(self.doc_ids, self.vectors)
        with self.assertRaises(ValueError):
            index.search_vectors(self.queries, k=0)
        with self.assertRaises(ValueError):
            index.search_vectors(np.zeros((1, 3), np.float32), k=1)
        with self.assertRaises(ValueError):
            DenseIndex(self.doc_ids[:3], self.vectors)


@unittest.skipIf(np is None, "numpy is not installed")
class LoadEmbeddingsTests(unittest.TestCase):
    def _write(self, directory, complete=True):
        vectors = unit_vectors(5, 4).astype(np.float16)
        save_chunk(directory, 0, ["a", "b", "c"], vectors[:3])
        save_chunk(directory, 1, ["d", "e"], vectors[3:])
        meta = {"documents": 5, "dimension": 4, "chunk_size": 3, "complete": complete}
        (directory / "meta.json").write_text(json.dumps(meta))
        return vectors

    def test_loads_all_chunks_in_order_as_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            vectors = self._write(directory)
            doc_ids, loaded = load_embeddings(directory)
            self.assertEqual(doc_ids, ["a", "b", "c", "d", "e"])
            self.assertEqual(loaded.dtype, np.float32)
            np.testing.assert_array_equal(loaded, vectors.astype(np.float32))

    def test_refuses_an_incomplete_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self._write(directory, complete=False)
            with self.assertRaises(ValueError):
                load_embeddings(directory)

    def test_missing_chunk_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self._write(directory)
            (directory / "vectors_00001.npy").unlink()
            with self.assertRaises(ValueError):
                load_embeddings(directory)


if __name__ == "__main__":
    unittest.main()

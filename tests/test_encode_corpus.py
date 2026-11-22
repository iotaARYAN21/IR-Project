import json
import tempfile
import unittest
from pathlib import Path

from src.indexing.encode_corpus import (
    chunk_is_complete,
    chunk_paths,
    chunked,
    document_text,
    parse_shard,
    read_corpus,
    save_chunk,
    verify_encoding,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class TextAndChunkingTests(unittest.TestCase):
    def test_document_text_joins_title_and_text(self):
        self.assertEqual(
            document_text({"title": "Cats", "text": "Cats purr."}), "Cats Cats purr."
        )

    def test_document_text_handles_missing_fields(self):
        self.assertEqual(document_text({"text": "Only text."}), "Only text.")
        self.assertEqual(document_text({"title": None, "text": None}), "")

    def test_chunked_groups_items_and_keeps_the_remainder(self):
        self.assertEqual(list(chunked(range(5), 2)), [[0, 1], [2, 3], [4]])

    def test_chunked_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            list(chunked([1], 0))

    def test_read_corpus_yields_ids_and_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(
                json.dumps({"_id": "doc0", "title": "A", "text": "b"})
                + "\n\n"
                + json.dumps({"_id": 7, "title": "", "text": "c"})
                + "\n"
            )
            self.assertEqual(list(read_corpus(path)), [("doc0", "A b"), ("7", "c")])


class ShardTests(unittest.TestCase):
    def test_parses_index_and_count(self):
        self.assertEqual(parse_shard("2/4"), (2, 4))
        self.assertEqual(parse_shard("0/1"), (0, 1))

    def test_rejects_bad_values(self):
        for value in ("4/4", "-1/4", "1", "a/b", "0/0", "1/2/3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_shard(value)


@unittest.skipIf(np is None, "numpy is not installed")
class ChunkFileTests(unittest.TestCase):
    def test_saved_chunk_is_complete_only_with_matching_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            vectors = np.zeros((3, 4), dtype=np.float16)
            save_chunk(output_dir, 0, ["a", "b", "c"], vectors)
            self.assertTrue(chunk_is_complete(output_dir, 0, 3))
            self.assertFalse(chunk_is_complete(output_dir, 0, 4))
            self.assertFalse(chunk_is_complete(output_dir, 1, 3))

    def test_chunk_without_ids_file_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_chunk(output_dir, 0, ["a"], np.zeros((1, 4), dtype=np.float16))
            chunk_paths(output_dir, 0)[1].unlink()
            self.assertFalse(chunk_is_complete(output_dir, 0, 1))

    def _write_corpus(self, directory, count):
        path = Path(directory) / "corpus.jsonl"
        path.write_text(
            "".join(
                json.dumps({"_id": f"doc{i}", "title": "t", "text": "x"}) + "\n"
                for i in range(count)
            )
        )
        return path

    def test_verify_accepts_a_complete_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._write_corpus(directory, 5)
            output_dir = Path(directory) / "emb"
            output_dir.mkdir()
            (output_dir / "meta.json").write_text(json.dumps({"chunk_size": 2}))
            for index, ids in enumerate([["doc0", "doc1"], ["doc2", "doc3"], ["doc4"]]):
                save_chunk(output_dir, index, ids, np.zeros((len(ids), 4), np.float16))
            self.assertEqual(verify_encoding(output_dir, corpus)[:2], ([], 5))

    def test_verify_reports_missing_and_mismatched_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._write_corpus(directory, 5)
            output_dir = Path(directory) / "emb"
            output_dir.mkdir()
            (output_dir / "meta.json").write_text(json.dumps({"chunk_size": 2}))
            save_chunk(output_dir, 0, ["doc0", "doc1"], np.zeros((2, 4), np.float16))
            # Chunk 1 has the right size but the wrong documents.
            save_chunk(output_dir, 1, ["x", "y"], np.zeros((2, 4), np.float16))
            # Chunk 2 is missing.
            self.assertEqual(verify_encoding(output_dir, corpus)[0], [1, 2])


if __name__ == "__main__":
    unittest.main()

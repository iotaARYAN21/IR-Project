"""Exact (brute-force) dense retrieval over precomputed document vectors.

The vectors come from src.indexing.encode_corpus. They are unit length, so the
dot product of a query vector and a document vector is their cosine similarity.
Every document is scored for every query, so there is no approximation error.
"""

import json
from pathlib import Path

import numpy as np

from src.indexing.encode_corpus import chunk_paths

DEFAULT_CHUNK_ROWS = 262144
DEFAULT_QUERY_BLOCK = 128


def load_embeddings(directory: Path) -> tuple[list[str], np.ndarray]:
    """Load every chunk written by encode_corpus into one float32 matrix.

    Returns (doc_ids, vectors) with vectors of shape (documents, dimension).
    Needs about 4 bytes x documents x dimension of RAM (about 4 GB for NQ).
    """
    directory = Path(directory)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No meta.json in {directory}")
    meta = json.loads(meta_path.read_text())
    if not meta.get("complete"):
        raise ValueError(
            f"{directory} is not marked complete. Run "
            "`python3 -m src.indexing.encode_corpus --verify` first."
        )

    total, dimension = meta["documents"], meta["dimension"]
    vectors = np.empty((total, dimension), dtype=np.float32)
    doc_ids: list[str] = []
    row = 0
    index = 0
    while chunk_paths(directory, index)[0].is_file():
        vectors_path, ids_path = chunk_paths(directory, index)
        chunk = np.load(vectors_path)
        chunk_ids = json.loads(ids_path.read_text())
        if chunk.shape[0] != len(chunk_ids) or chunk.shape[1] != dimension:
            raise ValueError(f"Chunk {index} in {directory} is inconsistent")
        if row + len(chunk_ids) > total:
            raise ValueError(f"{directory} holds more documents than meta.json says")
        vectors[row : row + len(chunk_ids)] = chunk
        doc_ids.extend(chunk_ids)
        row += len(chunk_ids)
        index += 1

    if row != total:
        raise ValueError(f"Expected {total} documents in {directory}, found {row}")
    return doc_ids, vectors


def _keep_top(scores: np.ndarray, indices: np.ndarray, k: int):
    """Keep the k highest scores in each row (unordered)."""
    if scores.shape[1] <= k:
        return scores, indices
    top = np.argpartition(scores, -k, axis=1)[:, -k:]
    return (
        np.take_along_axis(scores, top, axis=1),
        np.take_along_axis(indices, top, axis=1),
    )


class DenseIndex:
    """All document vectors in memory, searched exhaustively."""

    def __init__(
        self,
        doc_ids: list[str],
        vectors: np.ndarray,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
    ):
        if vectors.ndim != 2 or len(doc_ids) != vectors.shape[0]:
            raise ValueError("doc_ids and vectors must have the same length")
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be at least 1")
        self.doc_ids = doc_ids
        self.vectors = vectors
        self.chunk_rows = chunk_rows
        self._row_of: dict[str, int] | None = None

    @classmethod
    def from_directory(cls, directory: Path) -> "DenseIndex":
        doc_ids, vectors = load_embeddings(directory)
        return cls(doc_ids, vectors)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def vectors_for(self, doc_ids) -> np.ndarray:
        """Vectors of the given documents, one row each, in the given order."""
        if self._row_of is None:  # built on first use: about a second for NQ
            self._row_of = {doc_id: row for row, doc_id in enumerate(self.doc_ids)}
        try:
            rows = [self._row_of[doc_id] for doc_id in doc_ids]
        except KeyError as exc:
            raise KeyError(f"No vector for document {exc.args[0]}") from exc
        return self.vectors[rows]

    def search_vectors(
        self,
        queries: np.ndarray,
        k: int = 10,
        query_block: int = DEFAULT_QUERY_BLOCK,
    ) -> list[list[tuple[str, float]]]:
        """Return the k best (doc_id, score) pairs for every query vector."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if query_block < 1:
            raise ValueError("query_block must be at least 1")
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        if queries.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"Query vectors have {queries.shape[1]} dimensions, "
                f"the index has {self.vectors.shape[1]}"
            )

        k = min(k, len(self))
        results: list[list[tuple[str, float]]] = []
        for start in range(0, len(queries), query_block):
            results.extend(self._search_block(queries[start : start + query_block], k))
        return results

    def _search_block(self, queries: np.ndarray, k: int):
        best_scores = best_indices = None
        for start in range(0, len(self), self.chunk_rows):
            scores = queries @ self.vectors[start : start + self.chunk_rows].T
            indices = np.broadcast_to(
                np.arange(start, start + scores.shape[1]), scores.shape
            )
            scores, indices = _keep_top(scores, indices, k)
            if best_scores is not None:
                scores = np.concatenate([best_scores, scores], axis=1)
                indices = np.concatenate([best_indices, indices], axis=1)
                scores, indices = _keep_top(scores, indices, k)
            best_scores, best_indices = scores, indices

        order = np.argsort(-best_scores, axis=1)
        best_scores = np.take_along_axis(best_scores, order, axis=1)
        best_indices = np.take_along_axis(best_indices, order, axis=1)
        return [
            [(self.doc_ids[i], float(s)) for i, s in zip(row_i, row_s)]
            for row_i, row_s in zip(best_indices, best_scores)
        ]

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[str]:
        """Return the ids of the k closest documents to one query vector."""
        return [doc_id for doc_id, _ in self.search_vectors(query_vector, k)[0]]


def load_query_encoder(model_name: str):
    """Load the same sentence-transformers model used to encode the corpus."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_queries(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode query texts into normalised float32 vectors."""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

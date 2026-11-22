"""One search entry point over BM25, dense, hybrid and reranked retrieval.

The engine takes plain functions for each component, so it can be tested
without Elasticsearch, vectors or models, and so heavy parts can be loaded
only when a method needs them.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.search.hybrid import reciprocal_rank_fusion

METHODS = ("bm25", "dense", "hybrid")

SearchFn = Callable[[str, int], list[tuple[str, float]]]
FetchFn = Callable[[Sequence[str]], dict[str, dict]]
RerankFn = Callable[[str, Sequence[str]], list[float]]


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    score: float


@dataclass
class SearchResponse:
    query: str
    method: str
    reranked: bool
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class SearchEngine:
    def __init__(
        self,
        bm25: SearchFn | None = None,
        dense: SearchFn | None = None,
        fetch: FetchFn | None = None,
        rerank: RerankFn | None = None,
    ):
        self._bm25 = bm25
        self._dense = dense
        self._fetch = fetch
        self._rerank = rerank

    def search(
        self,
        query: str,
        method: str = "hybrid",
        k: int = 10,
        rerank: bool = False,
        candidates: int = 30,
        depth: int = 100,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> SearchResponse:
        """Search and return the top k hits with per-stage timings.

        With rerank=True the best `candidates` documents from the chosen
        method are rescored by the CrossEncoder and the top k of those are
        returned. For hybrid search, `depth` documents are taken from each of
        BM25 and dense search before fusing.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, not {method!r}")
        if k < 1 or candidates < 1 or depth < 1:
            raise ValueError("k, candidates and depth must be at least 1")

        timings: dict[str, float] = {}
        started = time.perf_counter()

        def timed(name, function, *args):
            begin = time.perf_counter()
            result = function(*args)
            timings[name] = (
                timings.get(name, 0.0) + (time.perf_counter() - begin) * 1000
            )
            return result

        if not query.strip():
            return SearchResponse(query, method, rerank, [], timings)

        pool = max(k, candidates) if rerank else k
        if method == "bm25":
            ranked = timed("bm25", self._require(self._bm25, "BM25"), query, pool)
        elif method == "dense":
            ranked = timed("dense", self._require(self._dense, "dense"), query, pool)
        else:
            bm25_hits = timed(
                "bm25", self._require(self._bm25, "BM25"), query, max(depth, pool)
            )
            dense_hits = timed(
                "dense", self._require(self._dense, "dense"), query, max(depth, pool)
            )
            lists = [[d for d, _ in bm25_hits], [d for d, _ in dense_hits]]
            ranked = timed(
                "fusion", reciprocal_rank_fusion, lists, rrf_k, list(weights)
            )[:pool]

        documents = timed(
            "fetch",
            self._require(self._fetch, "document fetch"),
            [d for d, _ in ranked],
        )
        entries = [
            (doc_id, score, documents[doc_id])
            for doc_id, score in ranked
            if doc_id in documents
        ]

        if rerank and entries:
            texts = [f"{doc['title']} {doc['text']}".strip() for _, _, doc in entries]
            new_scores = timed(
                "rerank", self._require(self._rerank, "reranker"), query, texts
            )
            entries = [
                (doc_id, new_score, doc)
                for (doc_id, _, doc), new_score in zip(entries, new_scores)
            ]
            entries.sort(key=lambda entry: -entry[1])

        hits = [
            Hit(doc_id, doc["title"], doc["text"], score)
            for doc_id, score, doc in entries[:k]
        ]
        timings["total"] = (time.perf_counter() - started) * 1000
        return SearchResponse(query, method, rerank, hits, timings)

    @staticmethod
    def _require(function, name: str):
        if function is None:
            raise RuntimeError(f"No {name} component is configured")
        return function

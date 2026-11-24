"""One search entry point over BM25, dense, hybrid and reranked retrieval.

The engine takes plain functions for each component, so it can be tested
without Elasticsearch, vectors or models, and so heavy parts can be loaded
only when a method needs them.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.search.mmr import mmr_order, passage_similarity
from src.search.hybrid import (
    HYBRID_DENSE_WEIGHT,
    HYBRID_RRF_K,
    reciprocal_rank_fusion,
)

METHODS = ("bm25", "dense", "hybrid")
# Largest proximity distance accepted. Any whole number from 0 up to this works;
# the limit only guards against typing a huge value by mistake.
MAX_SLOP = 100
MIN_SUGGEST_CHARS = 2

SearchFn = Callable[[str, int], list[tuple[str, float]]]
FetchFn = Callable[[Sequence[str]], dict[str, dict]]
RerankFn = Callable[[str, Sequence[str]], list[float]]
SuggestFn = Callable[[str, int], list[tuple[str, int]]]


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
        suggest: SuggestFn | None = None,
    ):
        self._bm25 = bm25
        self._dense = dense
        self._fetch = fetch
        self._rerank = rerank
        self._suggest = suggest

    def suggest(self, prefix: str, limit: int = 8) -> list[str]:
        """Article titles that start with what the user has typed so far.

        Needs at least MIN_SUGGEST_CHARS characters; shorter input gives nothing,
        because a one-letter prefix matches far too many titles to be useful.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        prefix = prefix.strip()
        if len(prefix) < MIN_SUGGEST_CHARS:
            return []
        titles = self._require(self._suggest, "title suggestions")(prefix, limit)
        return [title for title, _ in titles]

    def search(
        self,
        query: str,
        method: str = "hybrid",
        k: int = 10,
        rerank: bool = False,
        candidates: int = 30,
        depth: int = 100,
        slop: int | None = None,
        diversity: float | None = None,
        title_only: bool = False,
        must_include: str | None = None,
        must_exclude: str | None = None,
        exact_phrase: bool = False,
        rrf_k: int = HYBRID_RRF_K,
        weights: tuple[float, float] = (1.0, HYBRID_DENSE_WEIGHT),
    ) -> SearchResponse:
        """Search and return the top k hits with per-stage timings.

        With rerank=True the best `candidates` documents from the chosen
        method are rescored by the CrossEncoder and the top k of those are
        returned. For hybrid search, `depth` documents are taken from each of
        BM25 and dense search before fusing. `diversity` (0 to 1) re-orders the
        candidates with MMR to reduce near-duplicate results; None or 0 is off.
        `slop` turns on the proximity boost
        in BM25 (also in the BM25 half of hybrid search); None leaves it off.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, not {method!r}")
        if k < 1 or candidates < 1 or depth < 1:
            raise ValueError("k, candidates and depth must be at least 1")
        if slop is not None and not 0 <= slop <= MAX_SLOP:
            raise ValueError(f"slop must be between 0 and {MAX_SLOP}")
        if diversity is not None and not 0 <= diversity <= 1:
            raise ValueError("diversity must be between 0 and 1")

        timings: dict[str, float] = {}
        started = time.perf_counter()

        def timed(name, function, *args, **kwargs):
            begin = time.perf_counter()
            result = function(*args, **kwargs)
            timings[name] = (
                timings.get(name, 0.0) + (time.perf_counter() - begin) * 1000
            )
            return result

        if not query.strip():
            return SearchResponse(query, method, rerank, [], timings)

        pool = max(k, candidates) if rerank or diversity else k
        if method == "bm25":
            options = {
                "title_only": title_only,
                "must_include": must_include,
                "must_exclude": must_exclude,
                "exact_phrase": exact_phrase,
                "proximity_slop": slop,
            }
            # Pass only the options that are set, so a plain BM25 function works.
            options = {
                name: value
                for name, value in options.items()
                if value is not None and value is not False and value != ""
            }
            ranked = timed("bm25", self._require(self._bm25, "BM25"), query, pool, **options)
        elif method == "dense":
            ranked = timed("dense", self._require(self._dense, "dense"), query, pool)
        else:
            bm25_hits = timed(
                "bm25",
                self._require(self._bm25, "BM25"),
                query,
                max(depth, pool),
                **({} if slop is None else {"proximity_slop": slop}),
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

        if diversity and len(entries) > 1:
            # MMR picks from the whole candidate pool, using the current scores
            # (the CrossEncoder's if reranked) as relevance.
            similarity = timed(
                "similarity",
                passage_similarity,
                [doc["title"] for _, _, doc in entries],
                [doc["text"] for _, _, doc in entries],
            )
            order = timed(
                "diversify",
                mmr_order,
                [score for _, score, _ in entries],
                similarity,
                k,
                diversity,
            )
            entries = [entries[i] for i in order]

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

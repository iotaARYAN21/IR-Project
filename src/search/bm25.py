"""BM25 retrieval using the Elasticsearch index."""

from elasticsearch import Elasticsearch

from src.indexing.config import INDEX_NAME


def search_with_scores(
    client: Elasticsearch,
    query: str,
    limit: int = 10,
    index_name: str = INDEX_NAME,
    title_only: bool = False,
    must_include: str | None = None,
    must_exclude: str | None = None,
    exact_phrase: bool = False,
    proximity_slop: int | None = None,
) -> list[tuple[str, float]]:
    """Return (document ID, BM25 score) pairs, best first."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if proximity_slop is not None and proximity_slop < 0:
        raise ValueError("proximity_slop cannot be negative")

    if not query.strip():
        return []

    # title_only restricts only the main query to the title. The must include and
    # must exclude terms are filters, so they look in the title and the text.
    fields = ["title^2"] if title_only else ["title^2", "text"]
    filter_fields = ["title^2", "text"]

    def all_words(text: str) -> dict:
        return {
            "multi_match": {
                "query": text,
                "type": "cross_fields",
                "operator": "and",
                "fields": filter_fields,
            }
        }

    main_query = (
        {"match_phrase": {"title": query}} if title_only and exact_phrase
        else {"multi_match": {"query": query, "type": "phrase", "fields": fields}}
        if exact_phrase
        else {"multi_match": {"query": query, "fields": fields}}
    )
    must = [main_query]
    should = []
    must_not = []
    if must_include and must_include.strip():
        must.append(all_words(must_include))
    if must_exclude and must_exclude.strip():
        must_not.append(all_words(must_exclude))
    if proximity_slop is not None and not exact_phrase:
        should.append(
            {
                "multi_match": {
                    "query": query,
                    "type": "phrase",
                    "fields": fields,
                    "slop": proximity_slop,
                }
            }
        )
    query_clause = (
        {"bool": {"must": must, "should": should, "must_not": must_not}}
        if len(must) > 1 or should or must_not
        else main_query
    )
    response = client.search(
        index=index_name,
        size=limit,
        query=query_clause,
    )
    return [(hit["_id"], float(hit["_score"])) for hit in response["hits"]["hits"]]


def search(
    client: Elasticsearch,
    query: str,
    limit: int = 10,
    index_name: str = INDEX_NAME,
) -> list[str]:
    """Return document IDs ranked by BM25 for a query."""
    return [
        doc_id for doc_id, _ in search_with_scores(client, query, limit, index_name)
    ]


def suggest_titles(
    client: Elasticsearch,
    prefix: str,
    limit: int = 8,
    index_name: str = INDEX_NAME,
) -> list[tuple[str, int]]:
    """Article titles that start with `prefix` (case ignored), as (title, passages).

    Uses the exact-string `title.raw` field. The titles are grouped, and the ones
    with the most passages come first (big articles have many passages), so the
    list is a rough popularity ranking.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    prefix = prefix.strip()
    if not prefix:
        return []

    response = client.search(
        index=index_name,
        size=0,
        query={"prefix": {"title.raw": {"value": prefix, "case_insensitive": True}}},
        aggs={"titles": {"terms": {"field": "title.raw", "size": limit}}},
    )
    buckets = response["aggregations"]["titles"]["buckets"]
    return [(bucket["key"], int(bucket["doc_count"])) for bucket in buckets]

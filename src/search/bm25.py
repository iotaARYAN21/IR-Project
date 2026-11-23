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
) -> list[tuple[str, float]]:
    """Return (document ID, BM25 score) pairs, best first."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    if not query.strip():
        return []

    fields = ["title^2"] if title_only else ["title^2", "text"]
    main_query = (
        {"match_phrase": {"title": query}} if title_only and exact_phrase
        else {"multi_match": {"query": query, "type": "phrase", "fields": fields}}
        if exact_phrase
        else {"multi_match": {"query": query, "fields": fields}}
    )
    must = [main_query]
    must_not = []
    if must_include and must_include.strip():
        must.append({"multi_match": {"query": must_include, "fields": fields}})
    if must_exclude and must_exclude.strip():
        must_not.append({"multi_match": {"query": must_exclude, "fields": fields}})
    query_clause = (
        {"bool": {"must": must, "must_not": must_not}}
        if len(must) > 1 or must_not
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

def search_proximity_with_scores(
    client: Elasticsearch,
    query: str,
    limit: int = 10,
    slop: int = 2,
    index_name: str = INDEX_NAME,
) -> list[tuple[str, float]]:
    """Return documents matching query terms within the specified proximity."""

    if limit < 1:
        raise ValueError("limit must be at least 1")

    if slop < 0:
        raise ValueError("slop cannot be negative")

    if not query.strip():
        return []

    response = client.search(
        index=index_name,
        size=limit,
        query={
            "multi_match": {
                "query": query,
                "type": "phrase",
                "fields": ["title^2", "text"],
                "slop": slop,
            }
        },
    )

    return [
        (hit["_id"], float(hit["_score"]))
        for hit in response["hits"]["hits"]
    ]

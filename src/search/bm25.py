"""BM25 retrieval using the Elasticsearch index."""

from elasticsearch import Elasticsearch

from src.indexing.config import INDEX_NAME


def search(
    client: Elasticsearch,
    query: str,
    limit: int = 10,
    index_name: str = INDEX_NAME,
) -> list[str]:
    """Return document IDs ranked by BM25 for a query."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    if not query.strip():
        return []

    response = client.search(
        index=index_name,
        size=limit,
        query={
            "multi_match": {
                "query": query,
                "fields": ["title^2", "text"],
            }
        },
    )
    return [hit["_id"] for hit in response["hits"]["hits"]]

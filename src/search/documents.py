"""Fetch document text from Elasticsearch by ID."""

from collections.abc import Sequence


def fetch_documents(client, ids: Sequence[str], index_name: str) -> dict[str, dict]:
    """Return {doc_id: {"title": ..., "text": ...}} for the IDs that exist."""
    if not ids:
        return {}
    response = client.mget(
        index=index_name, ids=list(ids), source_includes=["title", "text"]
    )
    documents = {}
    for doc in response["docs"]:
        if not doc.get("found"):
            continue
        source = doc.get("_source", {})
        documents[doc["_id"]] = {
            "title": source.get("title") or "",
            "text": source.get("text") or "",
        }
    return documents

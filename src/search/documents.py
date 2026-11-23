"""Fetch document text, from Elasticsearch or straight from the corpus file."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path


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


def load_corpus_texts(path: Path, ids: Iterable[str]) -> dict[str, str]:
    """Read "title text" for the given IDs from a BEIR corpus.jsonl file.

    Streams the file once and stops as soon as every ID has been found, so no
    Elasticsearch is needed. Unknown IDs are simply missing from the result.
    """
    wanted = set(ids)
    found: dict[str, str] = {}
    if not wanted:
        return found
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = str(doc.get("_id") or doc.get("doc_id"))
            if doc_id in wanted:
                found[doc_id] = f"{doc.get('title') or ''} {doc.get('text') or ''}".strip()
                if len(found) == len(wanted):
                    break
    return found

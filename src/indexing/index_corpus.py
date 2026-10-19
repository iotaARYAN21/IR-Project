import json
import argparse

from elasticsearch import helpers
from tqdm import tqdm

from src.indexing.config import (
    CORPUS_FILE,
    INDEX_NAME,
    BULK_BATCH_SIZE,
    BM25_K1,
    BM25_B,
    create_es_client,
)


def create_index(client, recreate=False):
    if client.indices.exists(index=INDEX_NAME):
        if not recreate:
            raise RuntimeError(
                f"Index '{INDEX_NAME}' already exists. "
                "Use --recreate to delete and rebuild it."
            )

        print(f"Deleting existing index: {INDEX_NAME}")
        client.indices.delete(index=INDEX_NAME)

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": BM25_K1,
                    "b": BM25_B,
                }
            },
        },
        "mappings": {
            "properties": {
                "doc_id": {
                    "type": "keyword"
                },
                "title": {
                    "type": "text",
                    "analyzer": "standard", # elastic search analyzer automatically perfroms tokenization,lowercasing,token normalization
                    "index_options": "offsets",
                    "similarity": "custom_bm25",
                    "fields": {
                        "raw": {
                            "type": "keyword",
                            "ignore_above": 256,
                        }
                    },
                },
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                    "index_options": "offsets",
                    "similarity": "custom_bm25",
                },
                "url": {
                    "type": "keyword",
                    "ignore_above": 2048,
                },
            }
        },
    }

    client.indices.create(
        index=INDEX_NAME,
        **mapping,
    )

    print(f"Created index: {INDEX_NAME}")


def read_documents():
    if not CORPUS_FILE.is_file():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_FILE}\n"
            "Place the BEIR NQ corpus at data/nq/corpus.jsonl."
        )

    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            # BEIR corpus normally uses "_id", "title", and "text".
            doc_id = doc.get("_id") or doc.get("doc_id")

            if doc_id is None:
                raise ValueError(
                    f"Missing document ID on line {line_number}"
                )

            source = {
                "doc_id": str(doc_id),
                "title": doc.get("title") or "",
                "text": doc.get("text") or "",
            }

            if doc.get("url"):
                source["url"] = doc["url"]

            yield {
                "_op_type": "index",
                "_index": INDEX_NAME,
                "_id": str(doc_id),
                "_source": source,
            }


def index_corpus(client):
    batch = []
    indexed_count = 0

    with tqdm(desc="Indexing documents", unit="docs") as progress:
        for action in read_documents():
            batch.append(action)

            if len(batch) >= BULK_BATCH_SIZE:
                success, _ = helpers.bulk(
                    client,
                    batch,
                    raise_on_error=True,
                    request_timeout=120,
                )
                indexed_count += success
                progress.update(success)
                batch.clear()

        # Index the final partial batch.
        if batch:
            success, _ = helpers.bulk(
                client,
                batch,
                raise_on_error=True,
                request_timeout=120,
            )
            indexed_count += success
            progress.update(success)

    client.indices.refresh(index=INDEX_NAME)

    actual_count = client.count(index=INDEX_NAME)["count"]

    print(f"\nDocuments indexed this run: {indexed_count}")
    print(f"Documents in Elasticsearch: {actual_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the existing index and rebuild it.",
    )
    args = parser.parse_args()

    client = create_es_client()

    try:
        create_index(client, recreate=args.recreate)
        index_corpus(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()

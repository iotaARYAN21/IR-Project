import getpass
import os
from pathlib import Path

from elasticsearch import Elasticsearch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "nq"
CORPUS_FILE = DATASET_DIR / "corpus.jsonl"

# Elasticsearch connection
ES_URL = os.environ.get("ELASTICSEARCH_URL", "https://localhost:9200")
ES_USERNAME = os.environ.get("ELASTICSEARCH_USERNAME", "elastic")
ES_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")
CA_CERT_PATH = os.environ.get(
    "ELASTICSEARCH_CA_CERT",
    os.environ.get("ES_CA_CERT"),
)
ES_CA_CERT = (
    Path(CA_CERT_PATH)
    if CA_CERT_PATH
    else None
)

# Index configuration
INDEX_NAME = os.environ.get("ELASTICSEARCH_INDEX", "nq_documents")
BULK_BATCH_SIZE = 500

# Changing BM25 parameters requires rebuilding the index.
BM25_K1 = 1.2
BM25_B = 0.75


def create_es_client():
    if ES_CA_CERT is not None and not ES_CA_CERT.is_file():
        raise FileNotFoundError(
            f"Elasticsearch CA certificate not found: {ES_CA_CERT}"
        )

    password = ES_PASSWORD or getpass.getpass(
        f"Password for Elasticsearch user '{ES_USERNAME}': "
    )

    client_options = {
        "basic_auth": (ES_USERNAME, password),
        "request_timeout": 120,
    }
    if ES_CA_CERT is not None:
        client_options["ca_certs"] = str(ES_CA_CERT)

    client = Elasticsearch(ES_URL, **client_options)

    # Verify the connection and credentials.
    client.info()
    print("Connected to Elasticsearch successfully.")

    return client

import getpass
import os
from pathlib import Path

from elasticsearch import Elasticsearch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "nq"
CORPUS_FILE = DATASET_DIR / "corpus.jsonl"

# Elasticsearch connection
ES_URL = "https://localhost:9200"
ES_USERNAME = "elastic"

# Override this environment variable if your certificate is elsewhere.
ES_CA_CERT = Path(
    os.environ.get(
        "ES_CA_CERT",
        r"D:\IR\elasticsearch-9.5.5-windows-x86_64"
        r"\elasticsearch-9.5.5\config\certs\http_ca.crt",
    )
)

# Index configuration
INDEX_NAME = "nq_documents"
BULK_BATCH_SIZE = 500

# BM25 parameters -> changing them requires re-indexing . placed here so that changing the values in easy ,instead of finding them params in diff files
BM25_K1 = 1.2
BM25_B = 0.75


def create_es_client():
    if not ES_CA_CERT.is_file():
        raise FileNotFoundError(
            f"Elasticsearch CA certificate not found: {ES_CA_CERT}"
        )

    password = getpass.getpass(
        f"Password for Elasticsearch user '{ES_USERNAME}': "
    )

    client = Elasticsearch(
        ES_URL,
        basic_auth=(ES_USERNAME, password),
        ca_certs=str(ES_CA_CERT),
        request_timeout=120,
    )

    # Verify the connection and credentials.
    client.info()
    print("Connected to Elasticsearch successfully.")

    return client

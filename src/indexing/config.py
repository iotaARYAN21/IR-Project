import getpass
import os
from pathlib import Path

from elasticsearch import Elasticsearch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path) -> None:
    """Set variables from a .env file, without overriding ones already set.

    Lines look like KEY=value or export KEY="value". Blank lines and lines
    starting with # are ignored. Variables already in the environment win, so a
    value exported in the terminal overrides the file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        key = key.strip()
        if key:
            os.environ.setdefault(key, value)


load_env_file(PROJECT_ROOT / ".env")
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

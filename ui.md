# UI setup and run commands

The UI is written with vanilla JavaScript, HTML, and CSS. There is no `.env` file and no JavaScript framework dependency.

## 1. Start Elasticsearch

Start your local Elasticsearch service and make sure the NQ index exists:

```bash
python3 -m src.indexing.index_corpus
```

Use `--recreate` only when you intentionally want to rebuild the index:

```bash
python3 -m src.indexing.index_corpus --recreate
```

## 2. Set Elasticsearch connection variables
Bu
Linux/macOS:

```bash
export ELASTICSEARCH_URL="https://localhost:9200"
export ELASTICSEARCH_USERNAME="elastic"
export ELASTICSEARCH_PASSWORD="your-password"
export ELASTICSEARCH_CA_CERT="/path/to/http_ca.crt"
export ELASTICSEARCH_INDEX="nq_documents"
```

`ELASTICSEARCH_PASSWORD` is required for the server process. `ELASTICSEARCH_CA_CERT` is optional if your system already trusts the Elasticsearch certificate.

## 3. Start the UI server

Activate the project environment and install dependencies:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
node ui/server.js
```

Open:

```text
http://127.0.0.1:3000
```

To use another host or port:

```bash
UI_HOST=127.0.0.1 UI_PORT=8080 node ui/server.js
```

## 4. Choose settings in the UI

- **BM25:** Fast sparse retrieval. Elasticsearch must be running and the index must be built.
- **Dense:** Semantic retrieval. It requires `data/nq/corpus.jsonl` and verified embeddings in `data/nq/embeddings/multi-qa-MiniLM-L6-cos-v1/`.
- **Hybrid:** Combines BM25 and dense retrieval. It requires Elasticsearch and dense embeddings.
- **Rerank:** Applies the CrossEncoder to the selected number of candidates. It downloads the model on first use and uses more CPU time.
- **Compare methods:** Turn this on to choose a second method for a side-by-side comparison with the selected method.
- **Results:** Enter the number of result cards to show (1–1000).
- **Candidates:** Enter how many documents reranking should score (1–1000).

For the lowest memory usage, choose **BM25**, leave **Rerank** off, and do not enable dense or hybrid search.

## 5. Stop the server

Press `Ctrl+C` in the terminal running `node ui/server.js`.

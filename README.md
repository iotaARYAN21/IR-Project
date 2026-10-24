# Information Retrieval Project

This is a course project about building and comparing search methods using
the Natural Questions (NQ) dataset. The aim is to start with a BM25 baseline,
then add other retrieval methods and search features, and compare their
results using standard ranking metrics.

## What we have done so far

- Added an Elasticsearch index builder for the NQ document collection. The
  index uses Elasticsearch's standard analyzer and BM25 similarity.
- Added BM25 search over document titles and text, with a small boost for
  titles.
- Added evaluation using NDCG@k and MRR@k against the NQ test qrels.
- Added unit tests for the ranking metric calculations.
- Added setup and run instructions below.

The NQ query and qrels files are included. The document corpus is not
included, so it needs to be downloaded separately before building the index.

## What we plan to do

The remaining work is planned in stages:

- Compare BM25 with BM25 plus a reranker.
- Add semantic search, then evaluate BM25 plus semantic search and a
  combination of BM25, semantic search, and reranking.
- Implement and compare a retrieval method that does not use BM25, together
  with semantic search.
- Explore query expansion and proximity search, autocomplete, spelling
  correction ("Did you mean?"), and smart snippets.
- Improve and document preprocessing and the index as needed, including
  tokenization, lowercasing, stemming or lemmatization, stopword handling,
  and term positions or frequencies.

The reranking approach, semantic model, and non-BM25 retrieval method have not
been chosen yet. The list above describes future work; it is not all
implemented in the current code.

## Data and setup

The repository includes `data/nq/queries.jsonl` and
`data/nq/qrels/test.tsv`. Place the BEIR-format NQ `corpus.jsonl` file at
`data/nq/corpus.jsonl` before building the index.

Use Python 3.10 or newer. Install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Start Elasticsearch locally, then set its connection details. For a local
Elasticsearch installation that uses its generated HTTPS certificate:

```bash
export ELASTICSEARCH_URL="https://localhost:9200"
export ELASTICSEARCH_USERNAME="elastic"
export ELASTICSEARCH_PASSWORD="your-password"
export ELASTICSEARCH_CA_CERT="/path/to/http_ca.crt"
```

`ELASTICSEARCH_CA_CERT` is optional when the server certificate is already
trusted by your system. The index name defaults to `nq_documents`; override it
with `ELASTICSEARCH_INDEX` if needed.

## Build and evaluate

Build the index after the corpus and Elasticsearch are ready:

```bash
python3 -m src.indexing.index_corpus
```

To delete and rebuild the existing index, run:

```bash
python3 -m src.indexing.index_corpus --recreate
```

Run the baseline evaluation on the test qrels:

```bash
python3 -m src.evaluation.evaluate_bm25
```

The default output reports NDCG@10 and MRR@10. Choose a different rank cutoff
with `--k`, for example:

```bash
python3 -m src.evaluation.evaluate_bm25 --k 20
```

The evaluator expects the index to have already been built and uses the same
Elasticsearch connection settings as the indexer.

## Run the metric tests

```bash
python3 -m unittest discover -s tests -v
```

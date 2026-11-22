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
- Added a BM25 + CrossEncoder reranker evaluation (see "BM25 + reranker"
  below).
- Added unit tests for the ranking metric calculations.
- Added setup and run instructions below.

The NQ query and qrels files are included. The document corpus is not
included, so it needs to be downloaded separately before building the index.

## What we plan to do

The remaining work is planned in stages:

- Run the full BM25 vs. BM25 plus reranker comparison and record the results.
- Add semantic search, then evaluate BM25 plus semantic search and a
  combination of BM25, semantic search, and reranking.
- Implement and compare a retrieval method that does not use BM25, together
  with semantic search.
- Explore query expansion and proximity search, autocomplete, spelling
  correction ("Did you mean?"), and smart snippets.
- Improve and document preprocessing and the index as needed, including
  tokenization, lowercasing, stemming or lemmatization, stopword handling,
  and term positions or frequencies.

The semantic model and non-BM25 retrieval method have not been chosen yet. The list above describes future work; it is not all
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

## Run the unit tests

From the project root, run all unit tests with:

```bash
python3 -m unittest discover -s tests
```

Add `-v` to list each test. To run a single file:

```bash
python3 -m unittest tests.test_metrics
```

The tests do not need Elasticsearch or the corpus.

## BM25 + reranker

`src/evaluation/evaluate_bm25_reranker.py` adds a second ranking stage on top
of BM25 and evaluates it with the same NDCG@k and MRR@k metrics, so its output
can be compared directly with the BM25 baseline.

### Required libraries

The reranker needs everything the BM25 baseline needs, plus a CrossEncoder
model:

| Library | Why it is needed |
| --- | --- |
| `elasticsearch` | Fetches the BM25 candidate documents from the index. |
| `tqdm` | Shows a progress bar while queries are processed. |
| `sentence-transformers` | Loads and runs the CrossEncoder reranking model. |
| `torch` | Runs the model. It is installed automatically by `sentence-transformers`. |
| `transformers`, `huggingface_hub` | Load the model and download its weights. Both are installed automatically by `sentence-transformers`. |

All of these are covered by `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

The first run downloads the model `cross-encoder/ms-marco-MiniLM-L-6-v2`
(about 90 MB) from Hugging Face, so it needs internet access once. Later runs
use the cached copy. A GPU is not required; the script also runs on CPU, but
more slowly.

### How the reranker works

The reranker uses a two-stage retrieve-then-rerank pipeline:

1. **BM25 retrieval.** For each test query, Elasticsearch returns the top
   `--bm25-candidates` documents (100 by default). It uses the same
   `multi_match` query as the baseline, over `title^2` and `text`.
2. **CrossEncoder scoring.** Each candidate is joined into one string,
   `title + " " + text`, and paired with the query. The CrossEncoder
   `cross-encoder/ms-marco-MiniLM-L-6-v2` reads each (query, document) pair
   together and outputs a relevance score. The model was trained on MS MARCO
   passage ranking. Input is cut to 512 tokens.
3. **Reordering.** The candidates are sorted by CrossEncoder score, highest
   first. The BM25 score is not used after this point. The top `k` document
   IDs become the final ranking.
4. **Evaluation.** The rankings are scored against `data/nq/qrels/test.tsv`
   with NDCG@k and MRR@k, over the same queries as the BM25 evaluator.

The reranker can only reorder what BM25 returns. A relevant document that is
not among the BM25 candidates cannot be recovered, so `--bm25-candidates`
sets the upper limit on recall. The CrossEncoder processes every
query-document pair separately, so it is much slower than BM25. Running time
grows with (number of queries × number of candidates).

### How to run it

Start Elasticsearch, build the index (see above), and set the same connection
variables as for the BM25 evaluator. On Linux or macOS:

```bash
export ELASTICSEARCH_URL="https://localhost:9200"
export ELASTICSEARCH_USERNAME="elastic"
export ELASTICSEARCH_PASSWORD="your-password"
export ELASTICSEARCH_CA_CERT="/path/to/http_ca.crt"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
$env:ELASTICSEARCH_URL = "https://localhost:9200"
$env:ELASTICSEARCH_USERNAME = "elastic"
$env:ELASTICSEARCH_PASSWORD = "your-password"
$env:ELASTICSEARCH_CA_CERT = "C:\path\to\elasticsearch\config\certs\http_ca.crt"
```

If `ELASTICSEARCH_PASSWORD` is not set, the script asks for it.

First run a quick check on a few queries:

```bash
python3 -m src.evaluation.evaluate_bm25_reranker --max-queries 50
```

Then run the full evaluation:

```bash
python3 -m src.evaluation.evaluate_bm25_reranker
```

On Windows, use `python` instead of `python3`. The full run reranks 100
candidates for each of the 3,452 test queries and can take an hour or more on
CPU.

Available options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--k` | `10` | Rank cutoff for NDCG and MRR. |
| `--bm25-candidates` | `100` | Number of BM25 documents to rerank per query. Must be at least `--k`. |
| `--index` | `nq_documents` | Elasticsearch index to search. |
| `--model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model to use for reranking. |
| `--max-queries` | all | Evaluate only the first N test queries. |

For example, to rerank 50 candidates and report metrics at 20:

```bash
python3 -m src.evaluation.evaluate_bm25_reranker --bm25-candidates 50 --k 20
```

To compare with the baseline, run both evaluators with the same `--k`:

```bash
python3 -m src.evaluation.evaluate_bm25 --k 10
python3 -m src.evaluation.evaluate_bm25_reranker --k 10
```

The output looks like this:

```
Evaluated BM25 + Reranker on 3452 queries (top 10, 100 BM25 candidates)
NDCG@10: <score>
MRR@10:  <score>
```


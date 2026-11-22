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
- Added evaluation using NDCG@k, MRR@k and Recall@k against the NQ test
  qrels, plus per-query latency timing.
- Added a BM25 + CrossEncoder reranker evaluation (see "BM25 + reranker"
  below).
- Added unit tests for the ranking metric and latency calculations.
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

The default output reports NDCG@10, MRR@10 and Recall@10, followed by a
latency summary. Choose a different rank cutoff with `--k`. It applies to all
three metrics, for example:

```bash
python3 -m src.evaluation.evaluate_bm25 --k 20
```

### Metrics

- **NDCG@k** rewards relevant documents that appear near the top.
- **MRR@k** is the reciprocal rank of the first relevant document in the top
  k, or 0 if there is none.
- **Recall@k** is the fraction of a query's relevant documents found in the
  top k. Order inside the top k does not matter. Use `--k 100` to see how many
  relevant documents appear in the top 100, which is also the ceiling for a
  reranker over 100 BM25 candidates. Note that `--k` changes the cutoff of
  all three metrics at once.

### Latency

The evaluator times each query's search call and prints a summary line:

```
BM25 latency: median <ms> ms, mean <ms> ms, p95 <ms> ms, p99 <ms> ms (<n> queries, <q> queries/s)
```

The median is the typical query. The mean is pulled up by slow outliers. The
p95 and p99 values show the slow tail, and queries/s is the number of queries
divided by the time they took when run one after another. The first queries
are slow because of cold caches, so the first 5 are left out of the statistics
by default. Change this with `--warmup N`. Time spent connecting or loading
models is not counted. Latency depends on your hardware, so note the machine
when you report it.

The evaluator expects the index to have already been built and uses the same
Elasticsearch connection settings as the indexer.

## Dense (semantic) retrieval

`src/search/dense.py` searches precomputed document vectors exhaustively, so
the result is exact. The vectors are made by `src/indexing/encode_corpus.py`
with `multi-qa-MiniLM-L6-cos-v1` (a GPU is much faster for this step, see
`cluster/`). Check the vectors, then evaluate:

```bash
python3 -m src.indexing.encode_corpus --verify
python3 -m src.evaluation.evaluate_dense --max-queries 50     # quick check
python3 -m src.evaluation.evaluate_dense                      # full run, measures latency
python3 -m src.evaluation.evaluate_dense --batch-queries      # much faster, no latency
```

It needs about 4 GB of free RAM, because all vectors are held as float32.
Elasticsearch is not needed. Results are saved to `results/dense_k10.json`
and can be compared with the other systems using `src.evaluation.compare`.

## Saved results and significance testing

Each evaluator saves its results to a JSON file: `results/bm25_k10.json` for
the baseline and `results/bm25_reranker_k10.json` for the reranker (the `k`
in the name follows `--k`). A run limited with `--max-queries N` is saved
separately, as `..._n<N>.json`, so a quick check never overwrites a full run.
Use `--output PATH` to choose another file. Each file holds the settings, the
mean scores, the latency summary and the NDCG, MRR and recall of every query.

Compare two saved runs on the queries they share, using a paired bootstrap
over queries:

```bash
python3 -m src.evaluation.compare results/bm25_k10.json results/bm25_reranker_k10.json
```

For each metric it prints both means, the difference (B minus A) with a 95%
confidence interval, a two-sided p-value, and on how many queries B did
better, worse or the same. If the confidence interval excludes 0, the
difference is unlikely to come from chance. Both runs must use the same
`--k`. Options are `--resamples` (default 10000) and `--seed`.

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
of BM25 and evaluates it with the same NDCG@k, MRR@k and Recall@k metrics, so
its output can be compared directly with the BM25 baseline. It also reports
latency separately for the BM25 stage, the rerank stage and the total.

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
   with NDCG@k, MRR@k and Recall@k, over the same queries as the BM25
   evaluator.

The reranker can only reorder what BM25 returns. A relevant document that is
not among the BM25 candidates cannot be recovered, so `--bm25-candidates`
sets the upper limit on recall. Reranking only reorders the candidates, so
Recall@`--bm25-candidates` is the same for BM25 and BM25 + reranker. The
improvement shows up in NDCG and MRR. The CrossEncoder processes every
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
| `--k` | `10` | Rank cutoff for NDCG, MRR and Recall. |
| `--bm25-candidates` | `100` | Number of BM25 documents to rerank per query. Must be at least `--k`. |
| `--index` | `nq_documents` | Elasticsearch index to search. |
| `--model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model to use for reranking. |
| `--max-queries` | all | Evaluate only the first N test queries. |
| `--warmup` | `5` | Leading queries left out of the latency statistics. |

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
Recall@10: <score>
BM25 stage latency: median <ms> ms, mean <ms> ms, p95 <ms> ms, p99 <ms> ms (...)
Rerank stage latency: median <ms> ms, mean <ms> ms, p95 <ms> ms, p99 <ms> ms (...)
Total latency: median <ms> ms, mean <ms> ms, p95 <ms> ms, p99 <ms> ms (...)
```

The rerank stage dominates the total time, because the CrossEncoder scores
every candidate. To report recall over all candidates, use `--k 100`.


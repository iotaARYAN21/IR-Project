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
  qrels, with per-query latency timing. Results and per-query scores are saved
  to JSON, and `src.evaluation.compare` runs a paired bootstrap significance
  test between two runs.
- Added a BM25 + CrossEncoder reranker evaluation (see "BM25 + reranker"
  below).
- Added semantic (dense) retrieval: the corpus is encoded with
  `multi-qa-MiniLM-L6-cos-v1`, and `src/search/dense.py` searches the vectors
  exactly. See "Document embeddings" and "Dense (semantic) retrieval".
- Added hybrid retrieval that fuses BM25 and dense results with reciprocal
  rank fusion (see "Hybrid retrieval").
- Added a command-line search tool for BM25, dense, hybrid and proximity
  retrieval, with optional reranking and per-stage timings (see "Live search").
- Added a browser UI with proximity distance controls and BM25 Advanced Search
  options for exact phrases, required or excluded terms, and title-only search.
- Added unit tests for the metrics, latency, fusion, dense search, encoder and
  search pipeline code.
- Added setup and run instructions below.

The code for all of the above is written and tested, and the full evaluations
have been run (see Results below).

The NQ query and qrels files are included. The document corpus and the
embeddings are not, so they need to be downloaded separately (see "Data and
setup" and "Document embeddings").

## Results

Full test set (3,452 queries), cutoff k = 10, run on a laptop CPU (i7-1255U,
no GPU). The tables come from the saved `results/*.json` files;
`python3 -m src.evaluation.summarize` prints them again.

Recall@100 is the recall of the 100 candidates each system returns, which is
the ceiling for the reranker. Mean latency of hybrid and reranked systems is the
sum of their stage means. The last column is the NDCG difference from BM25 with
a paired bootstrap p-value.

| System | NDCG@10 | MRR@10 | Recall@10 | Recall@100 | Mean latency (ms) | NDCG@10 vs BM25 | Queries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.2802 | 0.2547 | 0.3900 | 0.6355 | 15.0 | baseline | 3452 |
| Dense | 0.4464 | 0.3930 | 0.6559 | 0.8874 | 324.8 | +0.1662* (p < 0.0010) | 3452 |
| Hybrid, equal weights (RRF k = 60) | 0.4247 | 0.3752 | 0.6167 | 0.8788 | 339.9 | +0.1446* (p < 0.0010) | 3452 |
| Hybrid, tuned (dense 1.5, k = 10) † | 0.4779 | 0.4265 | 0.6829 | 0.8858 | 339.9 | +0.1977* (p < 0.0010) | 3452 |
| BM25 + reranker | 0.4726 | 0.4441 | 0.5950 | 0.6355 | 3479.8 | +0.1924* (p < 0.0010) | 3452 |
| Equal-weight hybrid + reranker | 0.5737 | 0.5264 | 0.7649 | 0.8788 | 3795.4 | +0.2935* (p < 0.0010) | 3452 |

`*` = significant at p < 0.05 (paired bootstrap against BM25).

† Dense weight 1.5 (BM25 weight 1), RRF k 10, chosen on 1036 dev queries. The row covers all queries, including those dev queries, so it is slightly optimistic. On the 2416 held-out queries NDCG@10 is 0.4776, against 0.4276 for the equal-weight hybrid.

Latency by stage (per query):

| Stage | Mean (ms) | Median (ms) | p95 (ms) |
| --- | --- | --- | --- |
| BM25 search | 15.0 | 13.1 | 27.3 |
| Dense: encode query | 116.8 | 107.9 | 130.8 |
| Dense: exact search | 208.0 | 218.1 | 268.8 |
| Dense: total | 324.8 | 302.7 | 399.3 |
| RRF fusion | 0.0 | 0.0 | 0.1 |
| Rerank top-100 of BM25 | 3464.8 | 3366.3 | 4803.0 |
| Rerank top-100 of equal-weight hybrid | 3455.5 | 3357.7 | 4978.4 |

### Tuning the hybrid without overfitting

`python3 -m src.evaluation.tune_hybrid` splits the judged queries once (seed 0)
into 30% dev and 70% held-out. It tries a small grid of dense weights
(0.5 to 4, BM25 fixed at 1) and RRF constants (10, 20, 30, 40, 50, 60) on the dev queries
only, picks the best by dev NDCG@10, then scores that choice and the default
(equal weights, k = 60) on the held-out queries with a paired bootstrap. Only the
held-out numbers should be reported. Results are saved to
`results/tuning/hybrid_tuning.json`.

## What remains to be done

Roughly in the order we plan to do it:

1. **Pairwise significance tests.** The table compares every system with
   BM25. Run `src.evaluation.compare` on the other pairs the report needs
   (for example dense vs hybrid, and hybrid vs hybrid + reranker).
2. **CrossEncoder score caching.** The BM25 and hybrid candidate lists overlap
   heavily, so cached scores would save most of the reranking time. Not done.
3. **Hybrid defaults.** The tuned setting (dense weight 1.5, RRF k = 10) is now
   the default in the evaluator, pipeline and CLI. The reranker was run on the
   equal-weight hybrid (`hybrid_equal_rerank`); reranking the tuned hybrid is not
   needed to compare fusion settings.
4. **A retrieval method that does not use BM25**, combined with semantic
   search. The method is not chosen yet. The simplest candidate is another
   Elasticsearch similarity (for example language-model Dirichlet scoring),
   which needs a second index.
5. **Web UI improvements.** The UI now supports BM25, dense, hybrid and
   proximity methods, reranking, side-by-side comparisons, highlighted terms,
   proximity distance, and BM25 Advanced Search. Further work could add query
   expansion, autocomplete and spelling correction ("Did you mean?"). The
   dense vectors take 4 GB of RAM, so memory use on the demo machine matters.
6. **Search features:** query expansion, autocomplete, spelling correction
   ("Did you mean?"), and smart snippets. The index already stores term offsets
   for highlighting.
7. **Preprocessing and the index:** improve and document tokenization,
   lowercasing, stemming or lemmatization, stopword handling, and term
   positions or frequencies. This needs a rebuilt index and about 10 GB of
   disk, and `src/indexing/preprocess.py` is still empty.
8. **Optional: Elasticsearch vector index** (kNN) for realistic dense latency,
   and so that dense search runs inside the same server as BM25.
9. **Analysis and report:** error analysis with example queries, the effect of
   sparse relevance judgments, and the final comparison and latency discussion.

## Data and setup

The repository includes `data/nq/queries.jsonl` and
`data/nq/qrels/test.tsv`. Place the BEIR-format NQ `corpus.jsonl` file at
`data/nq/corpus.jsonl` before building the index. Dense and hybrid search also
need the precomputed document embeddings, see "Document embeddings" below.

Use Python 3.10 or newer. Install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Start Elasticsearch locally, then set its connection details. The simplest way
is a `.env` file in the project root, which the code reads automatically (it is
ignored by git, so the password is not committed):

```bash
cp .env.example .env     # then edit .env and fill in your password and certificate path
```

```bash
ELASTICSEARCH_URL="https://localhost:9200"
ELASTICSEARCH_USERNAME="elastic"
ELASTICSEARCH_PASSWORD="your-password"
ELASTICSEARCH_CA_CERT="/path/to/http_ca.crt"
```

Variables already set in the terminal (`export NAME=...`) take priority over
the file. The commands below that run in the terminal and not through the
Python code (for example `curl`) need the file loaded into the shell with
`set -a; source .env; set +a`.

`ELASTICSEARCH_CA_CERT` is optional when the server certificate is already
trusted by your system. The index name defaults to `nq_documents`; override it
with `ELASTICSEARCH_INDEX` if needed.

## Document embeddings

Dense (semantic) search and the hybrid search need a vector for every passage
in the corpus. They are made with `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`
(384 dimensions, stored as float16, about 2 GB for the 2,681,468 passages). The
BM25 evaluation and the BM25 + reranker evaluation do not need them.

### Get the embeddings

Download the embeddings from: **<https://drive.google.com/drive/folders/1bI5l_9bk_raA9Huak051WUnMeno9CXkw?usp=sharing>**

Download it (Drive may split a large folder into several zip files) and unzip all parts into `data/nq/embeddings/`, so that this layout results:

```
data/nq/embeddings/multi-qa-MiniLM-L6-cos-v1/
    meta.json
    vectors_00000.npy ... vectors_00053.npy    (54 files)
    ids_00000.json    ... ids_00053.json       (54 files)
```

`meta.json` and the 54 `vectors_*.npy` and 54 `ids_*.json` files must sit
directly in that folder, not in a subfolder. The folder is ignored by git.
`data/nq/corpus.jsonl` must also be in place, because the check below compares
the embeddings against it.

### Check them

```bash
python3 -m src.indexing.encode_corpus --verify
```

This checks every chunk against the corpus and should print
`OK: all chunks present, 2681468 documents`. If it lists missing or wrong
chunks, download them again. The search code refuses to load embeddings that
have not passed this check.

Loading the embeddings takes about 4 GB of RAM, because the vectors are held
as float32.

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
with `multi-qa-MiniLM-L6-cos-v1` (a GPU is much faster for this step). Check
the vectors, then evaluate:

```bash
python3 -m src.indexing.encode_corpus --verify
python3 -m src.evaluation.evaluate_dense --max-queries 50     # quick check
python3 -m src.evaluation.evaluate_dense                      # full run, measures latency
python3 -m src.evaluation.evaluate_dense --batch-queries      # much faster, no latency
```

It needs about 4 GB of free RAM, because all vectors are held as float32.
Elasticsearch is not needed. Results are saved to `results/dense_k10.json`
and can be compared with the other systems using `src.evaluation.compare`.

## Hybrid retrieval (BM25 + dense)

The BM25 and dense evaluators also save a **run file**: the top documents and
scores they returned for every query (`results/runs/bm25.json`,
`results/runs/dense.json`). `--run-depth` (default 100) sets how many documents
are kept per query, and each evaluator fetches at least that many, so its
latency is measured at that depth. Fusion and later reranking read these files,
so each slow system only has to run once.

```bash
python3 -m src.evaluation.evaluate_bm25        # needs Elasticsearch running
python3 -m src.evaluation.evaluate_dense       # needs the document vectors
python3 -m src.evaluation.evaluate_hybrid      # needs neither, takes seconds
```

`evaluate_hybrid` merges the two ranked lists with **reciprocal rank fusion**
(RRF): a document at rank r in a list gets `1 / (rrf_k + r)` from it, and the
scores from both lists are added. Only ranks are used, so the very different
BM25 and cosine score scales need no normalising. Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--k` | `10` | Rank cutoff for NDCG, MRR and Recall. |
| `--candidates` | `100` | Top documents taken from each run before fusing. |
| `--rrf-k` | `10` | RRF constant. Larger values flatten the difference between ranks. |
| `--bm25-weight` | `1.0` | Weight of the BM25 list. |
| `--dense-weight` | `1.5` | Weight of the dense list. |
| `--run-depth` | `100` | Fused documents saved to `results/runs/hybrid.json`. |

The defaults (dense weight 1.5, RRF k = 10) were tuned on a dev split of the
queries (see "Tuning the hybrid"). The equal-weight hybrid in the results
table is `--dense-weight 1 --rrf-k 60`; its files are `hybrid_equal_k10.json`
and `results/runs/hybrid_equal.json`, while `hybrid_k10.json` and
`results/runs/hybrid.json` hold the tuned default. The fused run is saved so it
can be reranked later. The hybrid latency is
estimated as BM25 + dense + fusion (mean latencies, systems run one after the
other), because the two searches are measured in their own runs. If dense was
run with `--batch-queries`, no latency is stored and no estimate is printed.

## Reranking a saved run

`src/evaluation/evaluate_rerank.py` takes any saved run file (BM25, dense or
hybrid), rescoring its top candidates with the CrossEncoder and evaluating the
new order. It needs neither Elasticsearch nor the document vectors, only
`data/nq/corpus.jsonl` for the passage text, so it can also run on a GPU
machine.

```bash
python3 -m src.evaluation.evaluate_rerank --run results/runs/hybrid.json --max-queries 50   # quick check
python3 -m src.evaluation.evaluate_rerank --run results/runs/hybrid.json                    # hybrid + reranker
python3 -m src.evaluation.evaluate_rerank --run results/runs/bm25.json                      # BM25 + reranker
```

Options: `--candidates` (default 100, top documents to rerank, at most the run's
depth), `--k`, `--model`, `--batch-size`, `--max-queries`, `--warmup`,
`--output` and `--run-output`. Results are saved as `results/<run system>_rerank_k10.json`
(for example `bm25_rerank`), so `src.evaluation.compare` works on them.

**Timing.** It measures the time the model takes to score each query's
candidates (the rerank stage only, not reading the text) and prints the median,
mean, p95 and p99. The device is printed and saved with the results, because
the numbers depend heavily on it. If the run file stored a latency for its own
search, it also prints the estimated total: the run's mean latency plus the
mean rerank time. Fetching the passage text is not included. Measure on a
machine with nothing else running, since other processes slow the model down
and distort the latency.

## Live search (command line)

`src/search/cli.py` searches the passages interactively with BM25, dense,
hybrid or proximity retrieval, with optional CrossEncoder reranking. It needs
Elasticsearch running (it supplies the passage text) and the same connection
settings as the evaluators. Dense and hybrid search also need the document
vectors.

```bash
python3 -m src.search.cli                                   # interactive
python3 -m src.search.cli --method hybrid --query "who wrote hamlet"
python3 -m src.search.cli --method bm25 --rerank --candidates 30 --k 5 --query "..."
python3 -m src.search.cli --method proximity --slop 0 --query "climate change"
```

For proximity search, `--slop` sets the phrase distance from 0 (exact phrase)
to 20 (more flexible). The default is 2. In interactive mode, type a query or
use `:method bm25|dense|hybrid|proximity`, `:slop N`, `:rerank on|off`, `:k N`,
`:candidates N`, `:settings`, `:help` or `:quit`. Each answer lists the results
with a short snippet and the time spent in every stage. Components load the
first time they are used: the dense vectors take about 4 GB of RAM and some
seconds to load, and the reranker model is downloaded on first use. Reranking
30 candidates on a CPU takes roughly half a second, and 100 candidates close to
two seconds.

## Web search UI

The browser UI uses the same search pipeline. It supports method selection,
reranking, method comparison and proximity distance. When BM25 is selected,
expand **Advanced search** to request an exact phrase, require or exclude terms,
or search titles only. These BM25 options apply to the BM25 result column when
comparing methods. Follow [ui.md](ui.md) to start the UI and configure
Elasticsearch.

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
variables as for the BM25 evaluator. The easiest way is the `.env` file described under "Data and setup". Otherwise, on Linux or macOS:

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


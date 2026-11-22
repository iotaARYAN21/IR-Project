"""Command-line search over the NQ passages.

One query and exit:
    python3 -m src.search.cli --method hybrid --query "who wrote hamlet"

Interactive (type a query, or a command that starts with ':'):
    python3 -m src.search.cli
"""

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

from src.search.pipeline import METHODS, SearchEngine, SearchResponse

SCORE_LABELS = {
    ("bm25", False): "BM25 score",
    ("dense", False): "cosine similarity",
    ("hybrid", False): "RRF score",
}
HELP = """Type a query and press Enter. Commands:
  :method bm25|dense|hybrid   choose the retrieval method
  :rerank on|off              rescore the best candidates with the CrossEncoder
  :k N                        number of results to show
  :candidates N               documents the reranker rescores
  :settings                   show the current settings
  :help                       show this help
  :quit                       leave"""


@dataclass
class Settings:
    method: str = "hybrid"
    k: int = 10
    rerank: bool = False
    candidates: int = 30


def apply_command(settings: Settings, line: str) -> str:
    """Apply a ':command' to the settings and return a message to print.

    Raises ValueError for unknown commands or bad values.
    """
    parts = line.strip().lstrip(":").split()
    if not parts:
        raise ValueError("Empty command. Type :help.")
    name, args = parts[0].lower(), parts[1:]

    if name == "help":
        return HELP
    if name == "settings":
        return describe(settings)
    if name == "method":
        if len(args) != 1 or args[0] not in METHODS:
            raise ValueError(f"Usage: :method {'|'.join(METHODS)}")
        settings.method = args[0]
        return describe(settings)
    if name == "rerank":
        if len(args) != 1 or args[0] not in ("on", "off"):
            raise ValueError("Usage: :rerank on|off")
        settings.rerank = args[0] == "on"
        return describe(settings)
    if name in ("k", "candidates"):
        if len(args) != 1 or not args[0].isdigit() or int(args[0]) < 1:
            raise ValueError(f"Usage: :{name} N (a whole number, at least 1)")
        setattr(settings, name, int(args[0]))
        return describe(settings)
    raise ValueError(f"Unknown command ':{name}'. Type :help.")


def describe(settings: Settings) -> str:
    rerank = f"on ({settings.candidates} candidates)" if settings.rerank else "off"
    return f"method={settings.method}, rerank={rerank}, results={settings.k}"


def snippet(text: str, width: int = 240) -> str:
    return textwrap.shorten(text, width=width, placeholder=" ...")


def format_response(response: SearchResponse, width: int = 100) -> str:
    """Render a response as plain text."""
    if not response.hits:
        return "No results."
    label = "CrossEncoder score" if response.reranked else SCORE_LABELS[
        (response.method, False)
    ]
    lines = []
    for rank, hit in enumerate(response.hits, start=1):
        lines.append(f"{rank:>2}. {hit.title or '(no title)'}   [{hit.doc_id}, {label} {hit.score:.3f}]")
        lines.append(
            textwrap.fill(
                snippet(hit.text),
                width=width,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )
    return "\n".join(lines)


def format_timings(timings_ms: dict[str, float]) -> str:
    parts = [f"{name} {value:.1f} ms" for name, value in timings_ms.items() if name != "total"]
    total = timings_ms.get("total")
    suffix = f" | total {total:.1f} ms" if total is not None else ""
    return " | ".join(parts) + suffix


class Lazy:
    """Build an expensive component the first time it is used."""

    def __init__(self, label: str, loader):
        self.label = label
        self.loader = loader
        self.value = None

    def get(self):
        if self.value is None:
            print(f"Loading {self.label} ...", flush=True)
            self.value = self.loader()
            print(f"Loaded {self.label}.", flush=True)
        return self.value


def build_engine(args) -> SearchEngine:
    """Create the engine. Elasticsearch, vectors and models load on first use."""
    from src.indexing.config import create_es_client
    from src.indexing.encode_corpus import DEFAULT_MODEL, EMBEDDINGS_ROOT
    from src.search.bm25 import search_with_scores
    from src.search.dense import DenseIndex, encode_queries, load_query_encoder
    from src.search.documents import fetch_documents
    from src.search.rerank import load_reranker, rerank_scores

    embeddings_dir = args.embeddings_dir or EMBEDDINGS_ROOT / DEFAULT_MODEL.split("/")[-1]
    client = Lazy("Elasticsearch connection", create_es_client)
    dense = Lazy(
        "dense index and query encoder",
        lambda: (DenseIndex.from_directory(embeddings_dir), load_query_encoder(DEFAULT_MODEL)),
    )
    reranker = Lazy("reranker model", load_reranker)

    def bm25_search(query, limit):
        return search_with_scores(client.get(), query, limit, args.index)

    def dense_search(query, limit):
        index, encoder = dense.get()
        return index.search_vectors(encode_queries(encoder, [query]), limit)[0]

    def fetch(ids):
        return fetch_documents(client.get(), ids, args.index)

    def rerank(query, texts):
        return rerank_scores(reranker.get(), query, texts)

    return SearchEngine(bm25_search, dense_search, fetch, rerank)


def run_query(engine: SearchEngine, settings: Settings, query: str, args) -> None:
    response = engine.search(
        query,
        method=settings.method,
        k=settings.k,
        rerank=settings.rerank,
        candidates=settings.candidates,
        depth=args.depth,
        rrf_k=args.rrf_k,
        weights=(args.bm25_weight, args.dense_weight),
    )
    print(format_response(response))
    if response.timings_ms:
        print(f"\n{describe(settings)}\n{format_timings(response.timings_ms)}")


def main() -> None:
    from src.indexing.config import INDEX_NAME

    parser = argparse.ArgumentParser(description="Search the NQ passages.")
    parser.add_argument("--query", help="Run this one query and exit.")
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--rerank", action="store_true", help="Rerank with the CrossEncoder.")
    parser.add_argument("--k", type=int, default=10, help="Results to show (default: 10).")
    parser.add_argument(
        "--candidates",
        type=int,
        default=30,
        help="Documents the reranker rescores (default: 30).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=100,
        help="Documents taken from each of BM25 and dense search before fusing (default: 100).",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant (default: 60).")
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--index", default=INDEX_NAME, help="Elasticsearch index.")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Folder with the document vectors (default: data/nq/embeddings/<model>).",
    )
    args = parser.parse_args()
    for name in ("k", "candidates", "depth"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")

    settings = Settings(args.method, args.k, args.rerank, args.candidates)
    engine = build_engine(args)

    if args.query is not None:
        run_query(engine, settings, args.query, args)
        return

    try:
        import readline  # noqa: F401  (gives arrow-key history to input())
    except ImportError:
        pass
    print(f"{describe(settings)}\nType a query, or :help for commands.")
    while True:
        try:
            line = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in (":quit", ":q", ":exit"):
            break
        try:
            if line.startswith(":"):
                print(apply_command(settings, line))
            else:
                run_query(engine, settings, line, args)
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()

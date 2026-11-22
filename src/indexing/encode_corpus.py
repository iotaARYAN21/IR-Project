"""Encode the NQ corpus into dense vectors with a sentence-transformers model.

The vectors are written in chunks, so a long run can be stopped and resumed:
chunks that are already complete are skipped when the script is started again.

Usage:
    python3 -m src.indexing.encode_corpus --benchmark 2000   # speed test, writes nothing
    python3 -m src.indexing.encode_corpus                    # full corpus
"""

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILE = PROJECT_ROOT / "data" / "nq" / "corpus.jsonl"
EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "nq" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def document_text(doc: dict) -> str:
    """Text that is encoded for a corpus record: title followed by body."""
    return f"{doc.get('title') or ''} {doc.get('text') or ''}".strip()


def read_corpus(path: Path = CORPUS_FILE) -> Iterator[tuple[str, str]]:
    """Yield (doc_id, text) for every record in a BEIR corpus file."""
    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc.get("_id") or doc.get("doc_id")
            if doc_id is None:
                raise ValueError(f"Missing document ID on line {line_number}")
            yield str(doc_id), document_text(doc)


def chunked(items: Iterable, size: int) -> Iterator[list]:
    """Group items into lists of at most `size`."""
    if size < 1:
        raise ValueError("size must be at least 1")
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def chunk_paths(output_dir: Path, index: int) -> tuple[Path, Path]:
    return (
        output_dir / f"vectors_{index:05d}.npy",
        output_dir / f"ids_{index:05d}.json",
    )


def chunk_is_complete(output_dir: Path, index: int, expected_rows: int) -> bool:
    """True if a chunk was fully written with the expected number of rows."""
    import numpy as np

    vectors_path, ids_path = chunk_paths(output_dir, index)
    if not (vectors_path.is_file() and ids_path.is_file()):
        return False
    try:
        rows = np.load(vectors_path, mmap_mode="r").shape[0]
        ids = json.loads(ids_path.read_text())
    except (OSError, ValueError):
        return False
    return rows == expected_rows and len(ids) == expected_rows


def save_chunk(output_dir: Path, index: int, doc_ids: list[str], vectors) -> None:
    """Write a chunk. The ids file goes last, so it marks the chunk complete."""
    import numpy as np

    vectors_path, ids_path = chunk_paths(output_dir, index)
    temp_vectors = vectors_path.with_suffix(".tmp")
    with temp_vectors.open("wb") as file:
        np.save(file, vectors)
    temp_vectors.replace(vectors_path)

    temp_ids = ids_path.with_suffix(".tmp")
    temp_ids.write_text(json.dumps(doc_ids))
    temp_ids.replace(ids_path)


def encode_texts(model, texts: list[str], batch_size: int, slice_size: int, progress):
    """Encode texts into normalised float16 vectors, updating a progress bar."""
    import numpy as np

    parts = []
    for start in range(0, len(texts), slice_size):
        part = texts[start : start + slice_size]
        parts.append(
            model.encode(
                part,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
        progress.update(len(part))
    return np.concatenate(parts).astype(np.float16)


def count_documents(path: Path = CORPUS_FILE) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def load_model(model_name: str, max_seq_length: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    print(
        f"Model: {model_name} ({model.get_sentence_embedding_dimension()} dimensions)"
    )
    print(f"Device: {model.device}, max sequence length: {model.max_seq_length}")
    return model


def run_benchmark(model, count: int, batch_size: int) -> None:
    """Time the encoding of the first `count` documents and estimate the total."""
    from tqdm import tqdm

    texts = [text for _, text in zip(range(count), (t for _, t in read_corpus()))]
    total_documents = count_documents()
    with tqdm(total=len(texts), desc="Benchmark", unit="docs") as progress:
        start = time.perf_counter()
        encode_texts(model, texts, batch_size, 1024, progress)
        elapsed = time.perf_counter() - start

    rate = len(texts) / elapsed
    print(f"\nEncoded {len(texts)} documents in {elapsed:.1f} s ({rate:.0f} docs/s)")
    print(
        f"Estimated time for {total_documents} documents: "
        f"{total_documents / rate / 3600:.1f} hours"
    )


def parse_shard(value: str) -> tuple[int, int]:
    """Parse 'I/N' into (index, count), for example '2/4'."""
    try:
        index_text, count_text = value.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError:
        raise ValueError(
            f"Shard must look like 'I/N', for example '0/4', not {value!r}"
        ) from None
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"Shard index must be in [0, {count}), got {index}/{count}")
    return index, count


def prepare_meta(model, model_name: str, output_dir: Path, chunk_size: int) -> dict:
    """Create meta.json on the first run, or check that an existing one matches."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.json"
    meta = {
        "model": model_name,
        "dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "chunk_size": chunk_size,
        "dtype": "float16",
        "normalized": True,
    }
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text())
        for key in ("model", "max_seq_length", "chunk_size"):
            if existing.get(key) != meta[key]:
                raise ValueError(
                    f"{output_dir} was created with {key}={existing.get(key)!r}, "
                    f"not {meta[key]!r}. Use another --output-dir to start fresh."
                )
        return existing

    temp_path = meta_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(meta, indent=2))
    temp_path.replace(meta_path)
    return meta


def run_encoding(
    model,
    model_name: str,
    output_dir: Path,
    chunk_size: int,
    batch_size: int,
    shard: tuple[int, int] = (0, 1),
) -> None:
    """Encode the corpus chunk by chunk.

    With shard=(i, n) only chunks whose index modulo n equals i are encoded,
    so n jobs can work on the same output directory at the same time.
    """
    from tqdm import tqdm

    shard_index, shard_count = shard
    sharded = shard_count > 1
    meta = prepare_meta(model, model_name, output_dir, chunk_size)

    total_documents = None if sharded else count_documents()
    done = 0
    with tqdm(total=total_documents, desc="Encoding", unit="docs") as progress:
        for index, chunk in enumerate(chunked(read_corpus(), chunk_size)):
            if index % shard_count != shard_index:
                continue
            doc_ids = [doc_id for doc_id, _ in chunk]
            if chunk_is_complete(output_dir, index, len(chunk)):
                progress.update(len(chunk))
                done += len(chunk)
                continue

            vectors = encode_texts(
                model, [text for _, text in chunk], batch_size, 2048, progress
            )
            save_chunk(output_dir, index, doc_ids, vectors)
            done += len(chunk)

    if sharded:
        print(
            f"\nShard {shard_index}/{shard_count} finished: {done} documents. "
            "Run with --verify once every shard has finished."
        )
    else:
        meta.update({"documents": done, "complete": True})
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\nEncoded {done} documents into {output_dir}")


def verify_encoding(
    output_dir: Path, corpus_path: Path = CORPUS_FILE
) -> tuple[list[int], int, dict]:
    """Check every chunk against the corpus.

    Returns the indexes of missing or mismatched chunks, the number of corpus
    documents, and the contents of meta.json.
    """
    meta = json.loads((output_dir / "meta.json").read_text())
    missing = []
    total = 0
    for index, chunk in enumerate(chunked(read_corpus(corpus_path), meta["chunk_size"])):
        total += len(chunk)
        doc_ids = [doc_id for doc_id, _ in chunk]
        _, ids_path = chunk_paths(output_dir, index)
        if not chunk_is_complete(output_dir, index, len(chunk)) or (
            json.loads(ids_path.read_text()) != doc_ids
        ):
            missing.append(index)
    return missing, total, meta


def run_verify(output_dir: Path) -> bool:
    missing, total, meta = verify_encoding(output_dir)
    if missing:
        print(f"{len(missing)} chunk(s) missing or wrong: {missing}")
        print("Run the encoder again (it skips finished chunks) and re-verify.")
        return False

    meta.update({"documents": total, "complete": True})
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"OK: all chunks present, {total} documents in {output_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode the NQ corpus with a sentence-transformers model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to encode with.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write chunks (default: data/nq/embeddings/<model name>).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Model batch size (default: 64)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Documents per saved chunk (default: 50000).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Tokens kept per document (default: 256).",
    )
    parser.add_argument(
        "--shard",
        default="0/1",
        metavar="I/N",
        help="Encode only chunks with index %% N == I, for parallel jobs "
        "(default: 0/1, everything).",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Stop immediately if no GPU is available.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check that every chunk exists and matches the corpus. No model needed.",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=None,
        metavar="N",
        help="Only time the encoding of the first N documents. Writes nothing.",
    )
    args = parser.parse_args()
    for name in ("batch_size", "chunk_size", "max_seq_length"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.benchmark is not None and args.benchmark < 1:
        parser.error("--benchmark must be at least 1")
    try:
        shard = parse_shard(args.shard)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or EMBEDDINGS_ROOT / args.model.split("/")[-1]

    if args.verify:
        raise SystemExit(0 if run_verify(output_dir) else 1)

    model = load_model(args.model, args.max_seq_length)
    if args.require_gpu and model.device.type == "cpu":
        raise SystemExit("--require-gpu was set, but the model is running on the CPU.")

    if args.benchmark is not None:
        run_benchmark(model, args.benchmark, args.batch_size)
    else:
        start = time.perf_counter()
        run_encoding(
            model, args.model, output_dir, args.chunk_size, args.batch_size, shard
        )
        print(f"Elapsed: {(time.perf_counter() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()

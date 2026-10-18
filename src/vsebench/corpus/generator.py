"""Corpus generation — download and embed a public dataset to produce a 10M-vector corpus."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = structlog.get_logger()

DEFAULT_DATASET = "wikipedia"
DEFAULT_CONFIG = "20220301.en"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384
CHUNK_SIZE = 100_000
QUERY_COUNT = 1_000
TARGET_VECTORS = 10_000_000


@dataclass
class CorpusConfig:
    dataset_name: str = DEFAULT_DATASET
    dataset_config: str = DEFAULT_CONFIG
    embedding_model: str = DEFAULT_MODEL
    dimension: int = DIMENSION
    target_vectors: int = TARGET_VECTORS
    query_count: int = QUERY_COUNT
    chunk_size: int = CHUNK_SIZE
    output_dir: Path = field(default_factory=lambda: Path("data/corpus"))
    seed: int = 42


def build_corpus(config: CorpusConfig | None = None) -> dict:
    if config is None:
        config = CorpusConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("loading dataset", dataset=config.dataset_name, config=config.dataset_config)

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split="train",
        streaming=True,
    )

    model = SentenceTransformer(config.embedding_model)
    texts: list[str] = []
    embeddings: list[np.ndarray] = []
    metadata: list[dict] = []
    idx = 0

    for row in tqdm(dataset, total=config.target_vectors, desc="Embedding corpus"):
        text = _extract_text(row)
        if not text or len(text) < 50:
            continue

        texts.append(text)
        embedding = model.encode(text, normalize_embeddings=True)
        embeddings.append(embedding)
        metadata.append({"id": idx, "source": config.dataset_name, "char_length": len(text)})
        idx += 1

        if len(texts) >= config.chunk_size:
            _flush_chunk(texts, embeddings, metadata, config.output_dir, idx // config.chunk_size)
            texts.clear()
            embeddings.clear()
            metadata.clear()

        if idx >= config.target_vectors:
            break

    if texts:
        _flush_chunk(texts, embeddings, metadata, config.output_dir, (idx // config.chunk_size) + 1)

    _sample_queries(dataset, model, config)

    manifest = {
        "total_vectors": idx,
        "dimension": config.dimension,
        "embedding_model": config.embedding_model,
        "dataset": config.dataset_name,
        "query_count": min(config.query_count, idx // 100),
    }
    with open(config.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("corpus generation complete", **manifest)
    return manifest


def _extract_text(row: dict) -> str:
    for key in ("text", "content", "passage", "title"):
        value = row.get(key, "")
        if value and isinstance(value, str) and len(value.strip()) > 20:
            return value.strip()

    title = row.get("title", "")
    text = row.get("text", "")
    if title and text:
        return f"{title}\n\n{text}"
    return ""


def _flush_chunk(
    texts: list[str],
    embeddings: list[np.ndarray],
    metadata: list[dict],
    output_dir: Path,
    chunk_idx: int,
) -> None:
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    emb_array = np.stack(embeddings, axis=0).astype(np.float32)
    table = pa.table(
        {
            "id": [m["id"] for m in metadata],
            "text": pa.array(texts, type=pa.large_string()),
            "embedding": pa.array(
                [emb_array[i].tobytes() for i in range(len(texts))],
                type=pa.binary(),
            ),
            "source": [m["source"] for m in metadata],
            "char_length": [m["char_length"] for m in metadata],
        }
    )
    pq.write_table(table, chunk_dir / f"corpus_{chunk_idx:05d}.parquet")
    logger.debug("flushed chunk", chunk=chunk_idx, rows=len(texts))


def _sample_queries(
    dataset: Iterator,
    model: SentenceTransformer,
    config: CorpusConfig,
) -> None:
    query_texts: list[str] = []
    seen = set()

    for row in dataset:
        text = _extract_text(row)
        if not text or text in seen:
            continue
        seen.add(text)
        query_texts.append(text[:200])
        if len(query_texts) >= config.query_count:
            break

    query_embeddings = model.encode(query_texts, normalize_embeddings=True)

    queries_dir = config.output_dir / "queries"
    queries_dir.mkdir(exist_ok=True)

    table = pa.table(
        {
            "query_id": list(range(len(query_texts))),
            "query_text": pa.array(query_texts, type=pa.large_string()),
            "query_embedding": pa.array(
                [query_embeddings[i].tobytes() for i in range(len(query_texts))],
                type=pa.binary(),
            ),
        }
    )
    pq.write_table(table, queries_dir / "queries.parquet")

    np.save(queries_dir / "query_vectors.npy", query_embeddings.astype(np.float32))

    logger.info("sampled queries", count=len(query_texts))


def load_corpus_chunks(corpus_dir: Path) -> Iterator[pa.Table]:
    chunk_dir = Path(corpus_dir) / "chunks"
    for parquet_file in sorted(chunk_dir.glob("corpus_*.parquet")):
        yield pq.read_table(parquet_file)


def load_queries(corpus_dir: Path) -> tuple[list[str], np.ndarray]:
    queries_dir = Path(corpus_dir) / "queries"
    table = pq.read_table(queries_dir / "queries.parquet")
    texts = table.column("query_text").to_pylist()
    vectors = np.load(queries_dir / "query_vectors.npy")
    return texts, vectors

"""Ingestion harness — identical interface across pgvector, Qdrant, and Weaviate."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import structlog
from tqdm import tqdm

logger = structlog.get_logger()


@dataclass
class IngestionMetrics:
    store: str
    total_vectors: int = 0
    total_time_seconds: float = 0.0
    vectors_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    index_build_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class VectorStore(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None: ...

    @abstractmethod
    def insert_batch(self, ids: list[int], vectors: np.ndarray, payloads: list[dict]) -> None: ...

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, top_k: int, filter_expr: dict | None = None
    ) -> list[dict]: ...

    @abstractmethod
    def build_index(self) -> float: ...

    @abstractmethod
    def collection_stats(self) -> dict: ...

    @abstractmethod
    def delete_collection(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


def ingest_corpus(
    store: VectorStore,
    collection_name: str,
    dimension: int,
    chunks: list[pa.Table],
    batch_size: int = 1000,
) -> IngestionMetrics:
    metrics = IngestionMetrics(store=store.__class__.__name__)

    try:
        store.connect()
        store.create_collection(collection_name, dimension)

        t0 = time.perf_counter()
        total = 0

        for chunk in tqdm(chunks, desc=f"Ingesting into {metrics.store}"):
            ids = chunk.column("id").to_pylist()
            texts = chunk.column("text").to_pylist()
            emb_bytes = chunk.column("embedding").to_pylist()

            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_embs = np.array(
                    [np.frombuffer(emb_bytes[j], dtype=np.float32) for j in range(i, min(i + batch_size, len(ids)))],
                    dtype=np.float32,
                )
                batch_payloads = [
                    {"text": texts[j], "source": "wikipedia"}
                    for j in range(i, min(i + batch_size, len(ids)))
                ]
                store.insert_batch(batch_ids, batch_embs, batch_payloads)
                total += len(batch_ids)

        metrics.total_vectors = total
        metrics.total_time_seconds = time.perf_counter() - t0
        metrics.vectors_per_second = total / metrics.total_time_seconds if metrics.total_time_seconds > 0 else 0.0

        t_index = store.build_index()
        metrics.index_build_time_seconds = t_index

        stats = store.collection_stats()
        logger.info("ingestion complete", store=metrics.store, **stats)

    except Exception as e:
        logger.error("ingestion failed", store=metrics.store, error=str(e))
        metrics.errors.append(str(e))
    finally:
        store.close()

    return metrics

"""Query suite — exact search ground truth, Recall@k computation, and query runner."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from tqdm import tqdm

from ..ingestion.harness import VectorStore

logger = structlog.get_logger()


@dataclass
class QueryResult:
    store: str
    query_id: int
    query_text: str
    top_k: int
    results: list[dict]
    latency_ms: float
    recall: float | None = None


@dataclass
class GroundTruth:
    query_id: int
    top_k: int
    result_ids: list[int]


def compute_ground_truth(
    corpus_chunks: list[pa.Table],
    query_vectors: np.ndarray,
    top_k: int = 10,
    batch_size: int = 1000,
) -> list[GroundTruth]:
    all_embeddings: list[np.ndarray] = []
    all_ids: list[int] = []

    for chunk in corpus_chunks:
        emb_bytes = chunk.column("embedding").to_pylist()
        ids = chunk.column("id").to_pylist()
        for b, i in zip(emb_bytes, ids):
            all_embeddings.append(np.frombuffer(b, dtype=np.float32))
            all_ids.append(i)

    corpus_matrix = np.stack(all_embeddings, axis=0)
    corpus_norm = corpus_matrix / np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
    query_norm = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    ground_truths: list[GroundTruth] = []

    for q_idx in tqdm(range(len(query_vectors)), desc="Computing ground truth"):
        for batch_start in range(0, len(corpus_matrix), batch_size):
            batch_end = min(batch_start + batch_size, len(corpus_matrix))
            batch_corpus = corpus_norm[batch_start:batch_end]
            scores = batch_corpus @ query_norm[q_idx]
            if batch_start == 0:
                all_scores = scores
            else:
                all_scores = np.concatenate([all_scores, scores])

        top_indices = np.argsort(all_scores)[-top_k:][::-1]
        top_ids = [all_ids[int(i)] for i in top_indices]
        ground_truths.append(GroundTruth(query_id=q_idx, top_k=top_k, result_ids=top_ids))

    return ground_truths


def save_ground_truth(ground_truths: list[GroundTruth], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"query_id": gt.query_id, "top_k": gt.top_k, "result_ids": gt.result_ids}
        for gt in ground_truths
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info("ground truth saved", path=str(path), queries=len(ground_truths))


def load_ground_truth(path: Path) -> list[GroundTruth]:
    with open(path) as f:
        data = json.load(f)
    return [GroundTruth(**item) for item in data]


def compute_recall(result: QueryResult, ground_truth: list[GroundTruth]) -> float:
    gt = ground_truth[result.query_id]
    result_ids = set(r["id"] for r in result.results)
    gt_ids = set(gt.result_ids[: result.top_k])
    if not gt_ids:
        return 1.0
    return len(result_ids & gt_ids) / len(gt_ids)


def run_query_suite(
    store: VectorStore,
    collection_name: str,
    query_texts: list[str],
    query_vectors: np.ndarray,
    ground_truth: list[GroundTruth],
    top_k: int = 10,
    warmup: int = 10,
) -> list[QueryResult]:
    store.connect()
    results: list[QueryResult] = []

    warmup_indices = list(range(min(warmup, len(query_vectors))))
    for q_idx in warmup_indices:
        store.search(query_vectors[q_idx], top_k)

    for q_idx in tqdm(range(len(query_vectors)), desc=f"Querying {store.__class__.__name__}"):
        t0 = time.perf_counter()
        hits = store.search(query_vectors[q_idx], top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = QueryResult(
            store=store.__class__.__name__,
            query_id=q_idx,
            query_text=query_texts[q_idx],
            top_k=top_k,
            results=hits,
            latency_ms=elapsed_ms,
        )
        result.recall = compute_recall(result, ground_truth)
        results.append(result)

    store.close()
    return results


def run_query_suite_with_filters(
    store: VectorStore,
    collection_name: str,
    query_vectors: np.ndarray,
    ground_truth: list[GroundTruth],
    filter_field: str = "source",
    filter_value: str = "wikipedia",
    top_k: int = 10,
) -> list[QueryResult]:
    store.connect()
    results: list[QueryResult] = []

    for q_idx in tqdm(range(len(query_vectors)), desc=f"Filtered query {store.__class__.__name__}"):
        t0 = time.perf_counter()
        hits = store.search(
            query_vectors[q_idx],
            top_k,
            filter_expr={filter_field: filter_value},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = QueryResult(
            store=store.__class__.__name__,
            query_id=q_idx,
            query_text="",
            top_k=top_k,
            results=hits,
            latency_ms=elapsed_ms,
        )
        results.append(result)

    store.close()
    return results

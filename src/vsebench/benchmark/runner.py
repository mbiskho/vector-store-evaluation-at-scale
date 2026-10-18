"""Benchmark runner — orchestrates ingestion, queries, and metric collection."""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ..corpus.generator import load_corpus_chunks, load_queries
from ..ingestion.harness import VectorStore, ingest_corpus, IngestionMetrics
from ..ingestion.pgvector_store import PgVectorStore
from ..ingestion.qdrant_store import QdrantVectorStore
from ..ingestion.weaviate_store import WeaviateVectorStore
from ..query.suite import (
    GroundTruth,
    QueryResult,
    compute_ground_truth,
    load_ground_truth,
    run_query_suite,
    save_ground_truth,
)

logger = structlog.get_logger()


@dataclass
class StoreBenchmark:
    store_name: str
    ingestion: IngestionMetrics | None = None
    query_results: list[QueryResult] = field(default_factory=list)
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    total_vectors: int = 0
    peak_memory_mb: float = 0.0
    index_build_time_s: float = 0.0


@dataclass
class BenchmarkRun:
    run_id: str
    stores: list[StoreBenchmark] = field(default_factory=list)
    ground_truth_path: str = ""
    corpus_path: str = ""
    query_count: int = 0
    top_k: int = 10
    dimension: int = 384


STORE_FACTORIES = {
    "pgvector": lambda cfg: PgVectorStore(cfg.get("pgvector", {}).get("conn_string", "postgresql://postgres:postgres@localhost:5432/vectordb")),
    "qdrant": lambda cfg: QdrantVectorStore(cfg.get("qdrant", {}).get("url", "http://localhost:6333")),
    "weaviate": lambda cfg: WeaviateVectorStore(cfg.get("weaviate", {}).get("url", "http://localhost:8080")),
}


def run_benchmark(
    corpus_dir: Path,
    stores: list[str],
    top_k: int = 10,
    collection_name: str = "benchmark_collection",
    store_configs: dict | None = None,
    skip_ingestion: bool = False,
    ground_truth_path: Path | None = None,
) -> BenchmarkRun:
    if store_configs is None:
        store_configs = {}

    import uuid

    run = BenchmarkRun(
        run_id=uuid.uuid4().hex[:8],
        corpus_path=str(corpus_dir),
        top_k=top_k,
    )

    corpus_chunks = list(load_corpus_chunks(corpus_dir))
    query_texts, query_vectors = load_queries(corpus_dir)
    run.query_count = len(query_texts)

    gt_path = ground_truth_path or corpus_dir / "ground_truth.json"
    if gt_path.exists():
        logger.info("loading existing ground truth", path=str(gt_path))
        ground_truth = load_ground_truth(gt_path)
    else:
        logger.info("computing ground truth")
        ground_truth = compute_ground_truth(corpus_chunks, query_vectors, top_k=top_k)
        save_ground_truth(ground_truth, gt_path)

    run.ground_truth_path = str(gt_path)

    for store_name in stores:
        logger.info("benchmarking store", store=store_name)
        factory = STORE_FACTORIES.get(store_name)
        if factory is None:
            logger.error("unknown store", store=store_name)
            continue

        store = factory(store_configs)
        bench = StoreBenchmark(store_name=store_name)

        if not skip_ingestion:
            bench.ingestion = ingest_corpus(store, collection_name, run.dimension, corpus_chunks)
            bench.total_vectors = bench.ingestion.total_vectors
            bench.peak_memory_mb = bench.ingestion.peak_memory_mb
            bench.index_build_time_s = bench.ingestion.index_build_time_seconds
        else:
            store.connect()

        bench.query_results = run_query_suite(
            store, collection_name, query_texts, query_vectors, ground_truth, top_k=top_k
        )

        latencies = [r.latency_ms for r in bench.query_results]
        latencies.sort()
        bench.p50_latency_ms = float(np.percentile(latencies, 50))
        bench.p95_latency_ms = float(np.percentile(latencies, 95))
        bench.p99_latency_ms = float(np.percentile(latencies, 99))
        bench.mean_latency_ms = float(np.mean(latencies))
        bench.recall_at_10 = float(np.mean([r.recall for r in bench.query_results if r.recall is not None]))

        run.stores.append(bench)
        store.delete_collection()

    return run


def save_benchmark_results(run: BenchmarkRun, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run.run_id,
        "corpus_path": run.corpus_path,
        "query_count": run.query_count,
        "top_k": run.top_k,
        "stores": {},
    }

    for bench in run.stores:
        summary["stores"][bench.store_name] = {
            "total_vectors": bench.total_vectors,
            "recall_at_10": bench.recall_at_10,
            "p50_latency_ms": bench.p50_latency_ms,
            "p95_latency_ms": bench.p95_latency_ms,
            "p99_latency_ms": bench.p99_latency_ms,
            "mean_latency_ms": bench.mean_latency_ms,
            "index_build_time_s": bench.index_build_time_s,
            "ingestion_vectors_per_second": bench.ingestion.vectors_per_second if bench.ingestion else 0,
            "ingestion_total_time_s": bench.ingestion.total_time_seconds if bench.ingestion else 0,
        }

        latencies = [r.latency_ms for r in bench.query_results]
        summary["stores"][bench.store_name]["latency_distribution"] = {
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "std": float(np.std(latencies)),
        }

    with open(output_dir / f"summary_{run.run_id}.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / f"full_results_{run.run_id}.json", "w") as f:
        json.dump({"summary": summary, "stores": {b.store_name: [asdict(r) for r in b.query_results] for b in run.stores}}, f, indent=2, default=str)

    logger.info("benchmark results saved", path=str(output_dir))

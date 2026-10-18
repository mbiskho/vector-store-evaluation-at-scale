import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vsebench.analysis.reporter import (
    compute_monthly_cost,
    build_comparison_table,
    build_decision_matrix,
    CostModel,
    load_benchmark_summary,
)


@pytest.fixture
def sample_results():
    return {
        "run_id": "test123",
        "corpus_path": "/data/corpus",
        "query_count": 1000,
        "top_k": 10,
        "stores": {
            "PgVectorStore": {
                "total_vectors": 10_000_000,
                "recall_at_10": 0.982,
                "p50_latency_ms": 5.2,
                "p95_latency_ms": 18.7,
                "p99_latency_ms": 45.3,
                "mean_latency_ms": 8.1,
                "index_build_time_s": 120.0,
                "ingestion_vectors_per_second": 3500,
                "ingestion_total_time_s": 2857.0,
                "latency_distribution": {"min": 1.2, "max": 120.0, "std": 8.5},
            },
            "QdrantVectorStore": {
                "total_vectors": 10_000_000,
                "recall_at_10": 0.991,
                "p50_latency_ms": 4.1,
                "p95_latency_ms": 12.3,
                "p99_latency_ms": 28.7,
                "mean_latency_ms": 6.2,
                "index_build_time_s": 80.0,
                "ingestion_vectors_per_second": 8200,
                "ingestion_total_time_s": 1219.0,
                "latency_distribution": {"min": 0.8, "max": 80.0, "std": 5.2},
            },
            "WeaviateVectorStore": {
                "total_vectors": 10_000_000,
                "recall_at_10": 0.976,
                "p50_latency_ms": 6.8,
                "p95_latency_ms": 22.1,
                "p99_latency_ms": 52.4,
                "mean_latency_ms": 9.5,
                "index_build_time_s": 0.0,
                "ingestion_vectors_per_second": 5200,
                "ingestion_total_time_s": 1923.0,
                "latency_distribution": {"min": 1.5, "max": 150.0, "std": 11.2},
            },
        },
    }


class TestCostModel:
    def test_pgvector_monthly_cost(self):
        cost = compute_monthly_cost("PgVectorStore", 10_000_000, 384)
        assert cost["total_monthly"] > 0
        assert cost["storage_cost"] > 0
        assert cost["compute_cost"] > 0
        assert cost["ops_cost"] > 0

    def test_cost_scales_with_vectors(self):
        cost_1m = compute_monthly_cost("PgVectorStore", 1_000_000, 384)
        cost_10m = compute_monthly_cost("PgVectorStore", 10_000_000, 384)
        assert cost_10m["storage_cost"] > cost_1m["storage_cost"]

    def test_custom_cost_model(self):
        model = CostModel(store_name="custom", compute_hourly=1.0, storage_gb_monthly=0.05, managed_service_premium=0, ops_hours_per_month=4)
        cost = compute_monthly_cost("custom", 10_000_000, 384, model)
        assert cost["compute_cost"] == 1.0 * 24 * 30


class TestComparisonTable:
    def test_builds_table(self, sample_results):
        df = build_comparison_table(sample_results)
        assert len(df) == 3
        assert "Store" in df.columns
        assert "Recall@10" in df.columns
        assert "p95 (ms)" in df.columns


class TestDecisionMatrix:
    def test_builds_matrix(self, sample_results):
        matrix = build_decision_matrix(sample_results)
        assert len(matrix) == 5
        for entry in matrix:
            assert "scenario" in entry
            assert "recommendation" in entry


class TestLoadSummary:
    def test_loads_json(self, sample_results):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_results, f)
            path = f.name

        loaded = load_benchmark_summary(Path(path))
        assert loaded["run_id"] == "test123"
        assert len(loaded["stores"]) == 3

        Path(path).unlink()

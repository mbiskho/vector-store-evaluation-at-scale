import json
import tempfile
from pathlib import Path

import pytest

from vsebench.benchmark.runner import (
    BenchmarkRun,
    StoreBenchmark,
    save_benchmark_results,
)


class TestBenchmarkRun:
    def test_creates_run(self):
        run = BenchmarkRun(run_id="abc123")
        assert run.run_id == "abc123"
        assert run.stores == []

    def test_adds_store_benchmark(self):
        run = BenchmarkRun(run_id="test")
        bench = StoreBenchmark(store_name="pgvector", recall_at_10=0.95)
        run.stores.append(bench)
        assert len(run.stores) == 1
        assert run.stores[0].store_name == "pgvector"


class TestSaveResults:
    def test_saves_summary(self):
        run = BenchmarkRun(run_id="save_test")
        bench = StoreBenchmark(
            store_name="pgvector",
            recall_at_10=0.95,
            p50_latency_ms=5.0,
            p95_latency_ms=15.0,
            p99_latency_ms=30.0,
            mean_latency_ms=8.0,
            total_vectors=10000,
        )
        run.stores.append(bench)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_benchmark_results(run, Path(tmpdir))
            summary_files = list(Path(tmpdir).glob("summary_*.json"))
            assert len(summary_files) == 1

            with open(summary_files[0]) as f:
                data = json.load(f)
            assert data["run_id"] == "save_test"
            assert "pgvector" in data["stores"]
            assert data["stores"]["pgvector"]["recall_at_10"] == 0.95

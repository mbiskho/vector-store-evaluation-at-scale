import numpy as np
import pytest

from vsebench.ingestion.harness import IngestionMetrics
from vsebench.ingestion.pgvector_store import PgVectorStore
from vsebench.ingestion.qdrant_store import QdrantVectorStore
from vsebench.ingestion.weaviate_store import WeaviateVectorStore


class TestIngestionMetrics:
    def test_default_metrics(self):
        metrics = IngestionMetrics(store="test")
        assert metrics.total_vectors == 0
        assert metrics.total_time_seconds == 0.0
        assert metrics.errors == []

    def test_metrics_accumulate(self):
        metrics = IngestionMetrics(store="test", total_vectors=1000, total_time_seconds=10.0)
        assert metrics.total_vectors == 1000
        assert metrics.vectors_per_second == 100.0


class TestPgVectorStore:
    def test_constructor(self):
        store = PgVectorStore(conn_string="postgresql://test:test@localhost/testdb")
        assert store.conn_string == "postgresql://test:test@localhost/testdb"
        assert store.conn is None

    def test_search_result_shape(self):
        store = PgVectorStore()
        assert store.search.__name__ == "search"


class TestQdrantStore:
    def test_constructor(self):
        store = QdrantVectorStore(url="http://localhost:6333")
        assert store.url == "http://localhost:6333"
        assert store.prefer_grpc is True

    def test_custom_url(self):
        store = QdrantVectorStore(url="http://custom:9999", prefer_grpc=False)
        assert store.url == "http://custom:9999"
        assert store.prefer_grpc is False


class TestWeaviateStore:
    def test_constructor(self):
        store = WeaviateVectorStore(url="http://localhost:8080")
        assert store.url == "http://localhost:8080"
        assert store.grpc_port == 50051

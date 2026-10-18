import time

import numpy as np
import pyarrow as pa
import pytest

from vsebench.ingestion.harness import ingest_corpus
from vsebench.ingestion.pgvector_store import PgVectorStore
from vsebench.query.suite import GroundTruth, compute_recall, run_query_suite


@pytest.fixture
def sample_chunks():
    np.random.seed(42)
    dim = 384
    n = 1000

    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    emb_bytes = [v.tobytes() for v in vectors]
    texts = [f"Document {i} about vector databases and similarity search." for i in range(n)]

    table = pa.table(
        {
            "id": list(range(n)),
            "text": pa.array(texts, type=pa.large_string()),
            "embedding": pa.array(emb_bytes, type=pa.binary()),
            "source": ["wikipedia"] * n,
            "char_length": [len(t) for t in texts],
        }
    )
    return [table], vectors, texts


@pytest.fixture
def query_vectors():
    np.random.seed(99)
    n_queries = 20
    dim = 384
    vectors = np.random.randn(n_queries, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


@pytest.mark.integration
class TestPgVectorIntegration:
    def test_ingest_and_search(self, sample_chunks, query_vectors):
        chunks, corpus_vectors, texts = sample_chunks

        store = PgVectorStore(conn_string="postgresql://postgres:postgres@localhost:5432/vectordb")
        metrics = ingest_corpus(store, "test_integration", 384, chunks, batch_size=500)

        assert metrics.total_vectors == 1000
        assert metrics.total_time_seconds > 0
        assert metrics.vectors_per_second > 0

        store.connect()
        results = store.search(query_vectors[0], top_k=10)
        assert len(results) == 10
        assert "id" in results[0]
        assert "score" in results[0]

        stats = store.collection_stats()
        assert stats["vectors"] == 1000

        store.delete_collection()
        store.close()

    def test_recall_computation(self, sample_chunks, query_vectors):
        chunks, corpus_vectors, texts = sample_chunks

        gt_ids = []
        q_norm = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        c_norm = corpus_vectors / np.linalg.norm(corpus_vectors, axis=1, keepdims=True)

        for q in q_norm:
            scores = c_norm @ q
            top10 = np.argsort(scores)[-10:][::-1]
            gt_ids.append(top10.tolist())

        ground_truth = [GroundTruth(query_id=i, top_k=10, result_ids=ids) for i, ids in enumerate(gt_ids)]

        store = PgVectorStore(conn_string="postgresql://postgres:postgres@localhost:5432/vectordb")
        metrics = ingest_corpus(store, "test_recall", 384, chunks, batch_size=500)
        assert metrics.errors == []

        results = run_query_suite(
            store,
            "test_recall",
            texts[: len(query_vectors)],
            query_vectors,
            ground_truth,
            top_k=10,
            warmup=2,
        )

        recalls = [r.recall for r in results if r.recall is not None]
        mean_recall = np.mean(recalls) if recalls else 0.0
        assert mean_recall > 0.7

        store.delete_collection()
        store.close()

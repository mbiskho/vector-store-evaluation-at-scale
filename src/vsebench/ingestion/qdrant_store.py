"""Qdrant store implementation."""

import time
from typing import Any

import numpy as np
import structlog
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from .harness import VectorStore

logger = structlog.get_logger()


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str = "http://localhost:6333", prefer_grpc: bool = True):
        self.url = url
        self.prefer_grpc = prefer_grpc
        self.client: QdrantClient | None = None
        self.collection_name: str = ""

    def connect(self) -> None:
        self.client = QdrantClient(url=self.url, prefer_grpc=self.prefer_grpc)

    def create_collection(self, name: str, dimension: int) -> None:
        self.collection_name = name
        self.client.delete_collection(collection_name=name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=4),
        )

    def insert_batch(self, ids: list[int], vectors: np.ndarray, payloads: list[dict]) -> None:
        points = [
            models.PointStruct(id=int(id_), vector=vec.tolist(), payload=pl)
            for id_, vec, pl in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=False)

    def search(
        self, query_vector: np.ndarray, top_k: int, filter_expr: dict | None = None
    ) -> list[dict]:
        qfilter = None
        if filter_expr:
            conditions = [
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in filter_expr.items()
            ]
            qfilter = models.Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [
            {"id": r.id, "text": r.payload.get("text", ""), "score": r.score}
            for r in results
        ]

    def build_index(self) -> float:
        t0 = time.perf_counter()
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
        )
        elapsed = time.perf_counter() - t0
        logger.info("qdrant index built", seconds=elapsed)
        return elapsed

    def collection_stats(self) -> dict:
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "vectors": info.points_count,
            "segments": info.segments_count,
            "vectors": info.points_count,
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(collection_name=self.collection_name)

    def close(self) -> None:
        if self.client:
            self.client.close()

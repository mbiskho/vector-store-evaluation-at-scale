"""Weaviate store implementation."""

import time
from typing import Any

import numpy as np
import structlog
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.collections.classes.config import VectorDistances

from .harness import VectorStore

logger = structlog.get_logger()


class WeaviateVectorStore(VectorStore):
    def __init__(self, url: str = "http://localhost:8080", grpc_port: int = 50051):
        self.url = url
        self.grpc_port = grpc_port
        self.client: weaviate.WeaviateClient | None = None
        self.collection: Any = None
        self.collection_name: str = ""

    def connect(self) -> None:
        self.client = weaviate.connect_to_local(
            host=self.url.replace("http://", "").split(":")[0],
            grpc_port=self.grpc_port,
        )

    def create_collection(self, name: str, dimension: int) -> None:
        self.collection_name = name
        try:
            self.client.collections.delete(name)
        except Exception:
            pass

        self.collection = self.client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=200,
                max_connections=16,
            ),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ],
        )

    def insert_batch(self, ids: list[int], vectors: np.ndarray, payloads: list[dict]) -> None:
        with self.collection.batch.dynamic() as batch:
            for id_, vec, pl in zip(ids, vectors, payloads):
                batch.add_object(
                    uuid=id_,
                    vector=vec.tolist(),
                    properties={"text": pl["text"], "source": pl["source"]},
                )

    def search(
        self, query_vector: np.ndarray, top_k: int, filter_expr: dict | None = None
    ) -> list[dict]:
        kwargs = {"limit": top_k}
        if filter_expr:
            kwargs["filters"] = filter_expr

        response = self.collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=top_k,
            return_metadata=["distance", "certainty"],
        )
        results = []
        for obj in response.objects:
            results.append({
                "id": obj.uuid.int,
                "text": obj.properties.get("text", ""),
                "score": obj.metadata.certainty or 1.0 - obj.metadata.distance,
            })
        return results

    def build_index(self) -> float:
        return 0.0

    def collection_stats(self) -> dict:
        agg = self.collection.aggregate.over_all(total_count=True)
        return {"vectors": agg.total_count}

    def delete_collection(self) -> None:
        self.client.collections.delete(self.collection_name)

    def close(self) -> None:
        if self.client:
            self.client.close()

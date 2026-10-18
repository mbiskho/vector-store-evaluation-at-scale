"""Locust load testing — concurrent query load against each store."""

import json
import random
import time
from pathlib import Path

import numpy as np
from locust import HttpUser, between, task

QUERY_VECTORS: np.ndarray | None = None
TOP_K = 10


def load_vectors(path: str) -> np.ndarray:
    return np.load(path)


class PgVectorUser(HttpUser):
    wait_time = between(0.01, 0.1)
    host = "http://localhost:8000"

    @task
    def search(self):
        idx = random.randint(0, len(QUERY_VECTORS) - 1)
        self.client.post(
            "/search",
            json={"vector": QUERY_VECTORS[idx].tolist(), "top_k": TOP_K},
            name="/search",
        )


class QdrantUser(HttpUser):
    wait_time = between(0.01, 0.1)
    host = "http://localhost:6333"

    @task
    def search(self):
        idx = random.randint(0, len(QUERY_VECTORS) - 1)
        self.client.post(
            "/collections/benchmark/points/search",
            json={"vector": QUERY_VECTORS[idx].tolist(), "limit": TOP_K},
            name="/search",
        )


class WeaviateUser(HttpUser):
    wait_time = between(0.01, 0.1)
    host = "http://localhost:8080"

    @task
    def search(self):
        idx = random.randint(0, len(QUERY_VECTORS) - 1)
        self.client.get(
            f"/v1/objects?nearVector={json.dumps({'vector': QUERY_VECTORS[idx].tolist()})}&limit={TOP_K}",
            name="/search",
        )

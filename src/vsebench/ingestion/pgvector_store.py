"""pgvector store implementation."""

import time
from typing import Any

import numpy as np
import psycopg
import structlog
from pgvector.psycopg import register_vector

from .harness import VectorStore

logger = structlog.get_logger()


class PgVectorStore(VectorStore):
    def __init__(self, conn_string: str = "postgresql://postgres:postgres@localhost:5432/vectordb"):
        self.conn_string = conn_string
        self.conn: Any = None
        self.collection_name: str = ""

    def connect(self) -> None:
        self.conn = psycopg.connect(self.conn_string)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def create_collection(self, name: str, dimension: int) -> None:
        self.collection_name = name
        self.conn.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
        self.conn.execute(
            f"CREATE TABLE {name} (id INTEGER PRIMARY KEY, embedding vector({dimension}), text TEXT, source TEXT)"
        )

    def insert_batch(self, ids: list[int], vectors: np.ndarray, payloads: list[dict]) -> None:
        with self.conn.cursor() as cur:
            with cur.copy(f"COPY {self.collection_name} (id, embedding, text, source) FROM STDIN") as copy:
                for i, vec, payload in zip(ids, vectors, payloads):
                    copy.write_row((int(i), vec.tolist(), payload["text"], payload["source"]))

    def search(
        self, query_vector: np.ndarray, top_k: int, filter_expr: dict | None = None
    ) -> list[dict]:
        if filter_expr:
            where_clause = " AND ".join(f"{k} = %s" for k in filter_expr)
            params = list(filter_expr.values()) + [query_vector.tolist(), top_k]
            rows = self.conn.execute(
                f"SELECT id, text, 1 - (embedding <=> %s) AS score "
                f"FROM {self.collection_name} WHERE {where_clause} "
                f"ORDER BY embedding <=> %s LIMIT %s",
                params,
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"SELECT id, text, 1 - (embedding <=> %s) AS score "
                f"FROM {self.collection_name} "
                f"ORDER BY embedding <=> %s LIMIT %s",
                [query_vector.tolist(), top_k],
            ).fetchall()

        return [{"id": r[0], "text": r[1], "score": float(r[2])} for r in rows]

    def build_index(self) -> float:
        t0 = time.perf_counter()
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.collection_name}_embedding_idx "
            f"ON {self.collection_name} USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {max(100, self.collection_stats()['vectors'] // 1000)})"
        )
        elapsed = time.perf_counter() - t0
        logger.info("pgvector index built", seconds=elapsed)
        return elapsed

    def collection_stats(self) -> dict:
        row = self.conn.execute(f"SELECT COUNT(*), pg_size_pretty(pg_total_relation_size(%s))", [self.collection_name]).fetchone()
        return {"vectors": row[0], "disk_size": row[1]}

    def delete_collection(self) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {self.collection_name} CASCADE")

    def close(self) -> None:
        if self.conn:
            self.conn.close()

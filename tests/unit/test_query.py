import numpy as np
import pytest

from vsebench.query.suite import GroundTruth, QueryResult, compute_recall


class TestGroundTruth:
    def test_ground_truth_creation(self):
        gt = GroundTruth(query_id=0, top_k=10, result_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert gt.query_id == 0
        assert len(gt.result_ids) == 10

    def test_ground_truth_serializable(self):
        import json
        from dataclasses import asdict

        gt = GroundTruth(query_id=1, top_k=10, result_ids=[1, 2, 3])
        data = asdict(gt)
        assert json.dumps(data)


class TestRecallComputation:
    def test_perfect_recall(self):
        result = QueryResult(
            store="test",
            query_id=0,
            query_text="test",
            top_k=10,
            results=[{"id": i} for i in range(10)],
            latency_ms=5.0,
        )
        ground_truth = [GroundTruth(query_id=0, top_k=10, result_ids=list(range(10)))]
        recall = compute_recall(result, ground_truth)
        assert recall == 1.0

    def test_zero_recall(self):
        result = QueryResult(
            store="test",
            query_id=0,
            query_text="test",
            top_k=10,
            results=[{"id": i} for i in range(100, 110)],
            latency_ms=5.0,
        )
        ground_truth = [GroundTruth(query_id=0, top_k=10, result_ids=list(range(10)))]
        recall = compute_recall(result, ground_truth)
        assert recall == 0.0

    def test_partial_recall(self):
        result = QueryResult(
            store="test",
            query_id=0,
            query_text="test",
            top_k=10,
            results=[{"id": i} for i in [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]],
            latency_ms=5.0,
        )
        ground_truth = [GroundTruth(query_id=0, top_k=10, result_ids=list(range(10)))]
        recall = compute_recall(result, ground_truth)
        assert recall == 0.5

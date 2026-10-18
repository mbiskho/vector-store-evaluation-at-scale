"""Results analysis — charts, decision matrix, and comparison tables."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog

logger = structlog.get_logger()


@dataclass
class CostModel:
    store_name: str
    compute_hourly: float
    storage_gb_monthly: float
    managed_service_premium: float
    ops_hours_per_month: float
    ops_hourly_rate: float = 75.0


DEFAULT_COST_MODELS = {
    "pgvector": CostModel(
        store_name="pgvector",
        compute_hourly=0.80,
        storage_gb_monthly=0.10,
        managed_service_premium=0.0,
        ops_hours_per_month=8,
    ),
    "qdrant": CostModel(
        store_name="Qdrant",
        compute_hourly=0.80,
        storage_gb_monthly=0.10,
        managed_service_premium=0.0,
        ops_hours_per_month=6,
    ),
    "weaviate": CostModel(
        store_name="Weaviate",
        compute_hourly=0.80,
        storage_gb_monthly=0.10,
        managed_service_premium=0.0,
        ops_hours_per_month=6,
    ),
}


def load_benchmark_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for store_name, metrics in results.get("stores", {}).items():
        rows.append(
            {
                "Store": store_name,
                "Recall@10": f"{metrics['recall_at_10']:.3f}",
                "p50 (ms)": f"{metrics['p50_latency_ms']:.1f}",
                "p95 (ms)": f"{metrics['p95_latency_ms']:.1f}",
                "p99 (ms)": f"{metrics['p99_latency_ms']:.1f}",
                "Mean (ms)": f"{metrics['mean_latency_ms']:.1f}",
                "Vectors/s (ingest)": f"{metrics.get('ingestion_vectors_per_second', 0):.0f}",
                "Index Build (s)": f"{metrics.get('index_build_time_s', 0):.1f}",
                "Total Vectors": f"{metrics['total_vectors']:,}",
            }
        )
    return pd.DataFrame(rows)


def compute_monthly_cost(
    store_name: str,
    vectors: int,
    dimension: int,
    cost_model: CostModel | None = None,
) -> dict[str, float]:
    if cost_model is None:
        cost_model = DEFAULT_COST_MODELS.get(store_name, DEFAULT_COST_MODELS["pgvector"])

    vector_size_bytes = dimension * 4
    storage_gb = (vectors * vector_size_bytes) / (1024**3)
    storage_gb_with_overhead = storage_gb * 1.3

    storage_cost = storage_gb_with_overhead * cost_model.storage_gb_monthly
    compute_cost = cost_model.compute_hourly * 24 * 30
    ops_cost = cost_model.ops_hours_per_month * cost_model.ops_hourly_rate
    managed_cost = cost_model.managed_service_premium * 30

    total = storage_cost + compute_cost + ops_cost + managed_cost

    return {
        "storage_cost": round(storage_cost, 2),
        "compute_cost": round(compute_cost, 2),
        "ops_cost": round(ops_cost, 2),
        "total_monthly": round(total, 2),
    }


def build_decision_matrix(results: dict) -> list[dict]:
    scenarios = [
        {"name": "Low latency, high recall", "constraints": {"p95_max_ms": 20, "recall_min": 0.95}},
        {"name": "Best recall at any cost", "constraints": {"recall_min": 0.99}},
        {"name": "Cheapest to operate", "constraints": {"cost_sensitive": True}},
        {"name": "Fastest ingestion", "constraints": {"ingest_priority": True}},
        {"name": "Filtered search required", "constraints": {"filters": True}},
    ]

    matrix = []
    stores = results.get("stores", {})

    for scenario in scenarios:
        best_store = None
        best_score = -1.0

        for store_name, metrics in stores.items():
            score = 0.0

            if "p95_max_ms" in scenario["constraints"]:
                if metrics["p95_latency_ms"] <= scenario["constraints"]["p95_max_ms"]:
                    score += 3.0
            if "recall_min" in scenario["constraints"]:
                if metrics["recall_at_10"] >= scenario["constraints"]["recall_min"]:
                    score += 3.0
            if "cost_sensitive" in scenario["constraints"]:
                cost = compute_monthly_cost(store_name, metrics["total_vectors"], 384)
                score += 100.0 / (cost["total_monthly"] + 1)
            if "ingest_priority" in scenario["constraints"]:
                ingest_rate = metrics.get("ingestion_vectors_per_second", 0)
                score += ingest_rate / 1000

            if score > best_score:
                best_score = score
                best_store = store_name

        matrix.append({"scenario": scenario["name"], "recommendation": best_store, "score": round(best_score, 1)})

    return matrix


def create_latency_chart(results: dict, output_path: Path | None = None) -> go.Figure:
    stores = results.get("stores", {})
    store_names = list(stores.keys())
    p50 = [stores[s]["p50_latency_ms"] for s in store_names]
    p95 = [stores[s]["p95_latency_ms"] for s in store_names]
    p99 = [stores[s]["p99_latency_ms"] for s in store_names]

    fig = go.Figure()
    x_positions = list(range(len(store_names)))

    fig.add_trace(go.Bar(name="p50", x=store_names, y=p50, marker_color="#4CAF50"))
    fig.add_trace(go.Bar(name="p95", x=store_names, y=p95, marker_color="#FF9800"))
    fig.add_trace(go.Bar(name="p99", x=store_names, y=p99, marker_color="#F44336"))

    fig.update_layout(
        title="Query Latency by Percentile (ms)",
        barmode="group",
        xaxis_title="Vector Store",
        yaxis_title="Latency (ms)",
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))
        fig.write_image(str(output_path).replace(".html", ".png"))

    return fig


def create_recall_chart(results: dict, output_path: Path | None = None) -> go.Figure:
    stores = results.get("stores", {})
    store_names = list(stores.keys())
    recalls = [stores[s]["recall_at_10"] * 100 for s in store_names]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=store_names,
            y=recalls,
            marker_color=["#2196F3", "#9C27B0", "#00BCD4"],
            text=[f"{r:.1f}%" for r in recalls],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"Recall@{results.get('top_k', 10)}",
        xaxis_title="Vector Store",
        yaxis_title="Recall (%)",
        template="plotly_white",
        yaxis_range=[0, 100],
    )

    if output_path:
        fig.write_html(str(output_path))
        fig.write_image(str(output_path).replace(".html", ".png"))

    return fig


def create_cost_chart(
    results: dict,
    cost_models: dict[str, CostModel] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    stores = results.get("stores", {})
    if cost_models is None:
        cost_models = DEFAULT_COST_MODELS

    store_names = list(stores.keys())
    storage_costs = []
    compute_costs = []
    ops_costs = []

    for s in store_names:
        vectors = stores[s]["total_vectors"]
        cost = compute_monthly_cost(s, vectors, 384, cost_models.get(s))
        storage_costs.append(cost["storage_cost"])
        compute_costs.append(cost["compute_cost"])
        ops_costs.append(cost["ops_cost"])

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Storage", x=store_names, y=storage_costs, marker_color="#4CAF50"))
    fig.add_trace(go.Bar(name="Compute", x=store_names, y=compute_costs, marker_color="#2196F3"))
    fig.add_trace(go.Bar(name="Ops Labour", x=store_names, y=ops_costs, marker_color="#FF9800"))

    fig.update_layout(
        title="Estimated Monthly Cost at 10M Vectors (USD)",
        barmode="stack",
        xaxis_title="Vector Store",
        yaxis_title="Monthly Cost (USD)",
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))
        fig.write_image(str(output_path).replace(".html", ".png"))

    return fig


def create_frontier_chart(results: dict, output_path: Path | None = None) -> go.Figure:
    stores = results.get("stores", {})

    fig = go.Figure()

    for store_name, metrics in stores.items():
        cost = compute_monthly_cost(store_name, metrics["total_vectors"], 384)
        fig.add_trace(
            go.Scatter(
                x=[metrics["p95_latency_ms"]],
                y=[metrics["recall_at_10"]],
                mode="markers+text",
                name=store_name,
                text=[store_name],
                textposition="top center",
                marker=dict(
                    size=cost["total_monthly"] / 5,
                    sizemode="area",
                    sizeref=2.0 * max(1, cost["total_monthly"]) / (40**2),
                    sizemin=20,
                ),
                hovertemplate=f"{store_name}<br>p95: {metrics['p95_latency_ms']:.1f}ms<br>Recall: {metrics['recall_at_10']:.3f}<br>Cost: ${cost['total_monthly']}/mo",
            )
        )

    fig.update_layout(
        title="Cost/Performance Frontier (bubble size = monthly cost)",
        xaxis_title="p95 Latency (ms)",
        yaxis_title="Recall@10",
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))
        fig.write_image(str(output_path).replace(".html", ".png"))

    return fig


def generate_full_report(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison_table(results)
    comparison.to_csv(output_dir / "comparison_table.csv", index=False)
    comparison.to_markdown(output_dir / "comparison_table.md", index=False)

    decision = build_decision_matrix(results)
    with open(output_dir / "decision_matrix.json", "w") as f:
        json.dump(decision, f, indent=2)

    create_latency_chart(results, output_dir / "latency_chart.html")
    create_recall_chart(results, output_dir / "recall_chart.html")
    create_cost_chart(results, output_dir / "cost_chart.html")
    create_frontier_chart(results, output_dir / "frontier_chart.html")

    with open(output_dir / "cost_breakdown.json", "w") as f:
        costs = {}
        for store_name, metrics in results.get("stores", {}).items():
            costs[store_name] = compute_monthly_cost(store_name, metrics["total_vectors"], 384)
        json.dump(costs, f, indent=2)

    logger.info("full report generated", path=str(output_dir))
    return output_dir

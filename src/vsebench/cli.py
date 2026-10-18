"""CLI — vsebench command-line interface."""

import json
from pathlib import Path

import click
import structlog

from .analysis.reporter import generate_full_report, load_benchmark_summary
from .benchmark.runner import run_benchmark
from .corpus.generator import CorpusConfig, build_corpus

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@click.group()
@click.version_option()
def main():
    """Vector Store Evaluation at Scale — benchmark pgvector, Qdrant, and Weaviate at 10M vectors."""


@main.command()
@click.option("--dataset", default="wikipedia", help="HuggingFace dataset name")
@click.option("--config", "dataset_config", default="20220301.en", help="Dataset config")
@click.option("--model", default="all-MiniLM-L6-v2", help="Sentence-transformers model")
@click.option("--target", default=10_000_000, help="Target vector count")
@click.option("--queries", default=1_000, help="Number of query vectors to sample")
@click.option("--output", default="data/corpus", help="Output directory")
def generate(dataset, dataset_config, model, target, queries, output):
    """Generate the 10M-vector corpus from a public dataset."""
    cfg = CorpusConfig(
        dataset_name=dataset,
        dataset_config=dataset_config,
        embedding_model=model,
        target_vectors=target,
        query_count=queries,
        output_dir=Path(output),
    )
    manifest = build_corpus(cfg)
    click.echo(f"Corpus generated: {manifest['total_vectors']:,} vectors in {output}")


@main.command()
@click.option("--corpus", "corpus_dir", default="data/corpus", help="Corpus directory")
@click.option("--stores", default="pgvector,qdrant,weaviate", help="Comma-separated store names")
@click.option("--top-k", default=10, help="Number of nearest neighbors")
@click.option("--collection", default="benchmark_collection", help="Collection/table name")
@click.option("--config-file", default=None, help="JSON config file for store connections")
@click.option("--skip-ingestion", is_flag=True, help="Skip ingestion, assume data already loaded")
@click.option("--output", default="results", help="Results output directory")
def run(corpus_dir, stores, top_k, collection, config_file, skip_ingestion, output):
    """Run the full benchmark across selected stores."""
    store_list = [s.strip() for s in stores.split(",")]
    store_configs = {}

    if config_file:
        with open(config_file) as f:
            store_configs = json.load(f)

    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        click.echo(f"Corpus directory not found: {corpus_dir}. Run 'vsebench generate' first.", err=True)
        raise SystemExit(1)

    result = run_benchmark(
        corpus_dir=corpus_path,
        stores=store_list,
        top_k=top_k,
        collection_name=collection,
        store_configs=store_configs,
        skip_ingestion=skip_ingestion,
    )

    output_path = Path(output)
    from .benchmark.runner import save_benchmark_results

    save_benchmark_results(result, output_path)

    click.echo(f"\nBenchmark complete. Results saved to {output_path}")
    for bench in result.stores:
        click.echo(
            f"  {bench.store_name}: "
            f"Recall@{top_k}={bench.recall_at_10:.3f}, "
            f"p50={bench.p50_latency_ms:.1f}ms, "
            f"p95={bench.p95_latency_ms:.1f}ms, "
            f"p99={bench.p99_latency_ms:.1f}ms"
        )


@main.command()
@click.option("--results", "results_file", required=True, help="Path to benchmark summary JSON")
@click.option("--output", default="results/report", help="Report output directory")
def report(results_file, output):
    """Generate charts, comparison tables, and decision matrix from benchmark results."""
    data = load_benchmark_summary(Path(results_file))
    output_dir = Path(output)
    generate_full_report(data, output_dir)

    click.echo(f"Report generated in {output_dir}")
    click.echo("Files:")
    for f in sorted(output_dir.iterdir()):
        click.echo(f"  {f.name}")


@main.command()
@click.option("--corpus", "corpus_dir", default="data/corpus", help="Corpus directory")
@click.option("--top-k", default=10, help="Number of nearest neighbors")
def ground_truth(corpus_dir, top_k):
    """Pre-compute exact-search ground truth for the corpus."""
    from .corpus.generator import load_corpus_chunks, load_queries
    from .query.suite import compute_ground_truth, save_ground_truth

    corpus_path = Path(corpus_dir)
    chunks = list(load_corpus_chunks(corpus_path))
    _, query_vectors = load_queries(corpus_path)

    gt = compute_ground_truth(chunks, query_vectors, top_k=top_k)
    save_ground_truth(gt, corpus_path / "ground_truth.json")
    click.echo(f"Ground truth computed: {len(gt)} queries, top_k={top_k}")


@main.command()
@click.option("--stores", default="pgvector,qdrant,weaviate", help="Stores to load test")
@click.option("--users", default=100, help="Number of concurrent users")
@click.option("--spawn-rate", default=10, help="Users spawned per second")
@click.option("--run-time", default="5m", help="Test duration (e.g., 5m, 1h)")
@click.option("--host", default="http://localhost:8089", help="Locust web UI host")
def loadtest(stores, users, spawn_rate, run_time, host):
    """Start Locust load testing against running stores."""
    import subprocess
    import sys

    store_list = [s.strip() for s in stores.split(",")]
    click.echo(f"Load testing stores: {store_list}")
    click.echo("Ensure stores are running and the Locust proxy is in place.")

    locustfile = Path(__file__).parent / "loadtest" / "locustfile.py"
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(locustfile),
        "--headless",
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", run_time,
        "--host", host,
        "--html", "results/loadtest_report.html",
        "--csv", "results/loadtest",
    ]

    click.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

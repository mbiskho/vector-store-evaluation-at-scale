.PHONY: help generate corpus run report loadtest clean docker-up docker-down docker-benchmark terraform-init terraform-apply terraform-destroy

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

generate: ## Generate 10M-vector corpus
	python -m vsebench.cli generate --target 10000000 --output data/corpus

corpus-small: ## Generate small corpus for testing (100K vectors)
	python -m vsebench.cli generate --target 100000 --output data/corpus_small

ground-truth: ## Pre-compute exact search ground truth
	python -m vsebench.cli ground-truth --corpus data/corpus

run: ## Run full benchmark against all stores
	python -m vsebench.cli run --corpus data/corpus --output results

run-small: ## Run benchmark on small corpus
	python -m vsebench.cli run --corpus data/corpus_small --output results

run-pgvector: ## Run benchmark against pgvector only
	python -m vsebench.cli run --stores pgvector --corpus data/corpus --output results

report: ## Generate charts and report from results
	python -m vsebench.cli report --results results/summary_*.json --output results/report

loadtest: ## Run Locust load test
	python -m vsebench.cli loadtest --users 100 --run-time 5m

docker-up: ## Start vector store containers
	docker compose -f docker/docker-compose.yml up -d pgvector qdrant weaviate

docker-down: ## Stop and remove containers
	docker compose -f docker/docker-compose.yml down -v

docker-benchmark: ## Run benchmark in Docker
	docker compose -f docker/docker-compose.yml --profile benchmark up --abort-on-container-exit benchmark

docker-monitoring: ## Start monitoring stack
	docker compose -f docker/docker-compose.yml --profile monitoring up -d

terraform-init: ## Initialize Terraform
	cd terraform && terraform init

terraform-plan: ## Plan Terraform changes
	cd terraform && terraform plan

terraform-apply: ## Apply Terraform (creates EKS cluster and deploys stores)
	cd terraform && terraform apply

terraform-destroy: ## Destroy infrastructure
	cd terraform && terraform destroy

clean: ## Remove generated data and results
	rm -rf data/corpus data/corpus_small results/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

install: ## Install package in dev mode
	pip install -e ".[dev]"

test: ## Run tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --cov=src/vsebench --cov-report=term-missing

lint: ## Lint code
	python -m ruff check src/ tests/

format: ## Format code
	python -m ruff format src/ tests/

typecheck: ## Run type checker
	python -m mypy src/

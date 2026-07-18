.DEFAULT_GOAL := help
.PHONY: help install up down migrate ingest ingest-url refresh-urls run-api run-bot \
        test test-unit test-integration lint format typecheck check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the environment (core + dev)
	uv sync

up: ## Start Postgres via docker compose
	docker compose up -d db

down: ## Stop docker compose services
	docker compose down

migrate: ## Apply forward-only SQL migrations
	uv run python scripts/migrate.py

ingest: ## Ingest files: make ingest FILES="a.pdf b.md"
	uv run python -m kb.cli.ingest $(FILES)

ingest-url: ## Ingest a URL: make ingest-url URL="https://..."
	uv run python -m kb.cli.ingest url $(URL)

refresh-urls: ## Re-fetch all url-sourced documents
	uv run python -m kb.cli.ingest refresh-urls

run-api: ## Run the FastAPI backend
	uv run uvicorn kb.api.app:app --host 0.0.0.0 --port 8000

run-bot: ## Run the Telegram bot (long polling)
	uv run python -m kb.channels.telegram

test: ## Run the full test suite
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest -m "not integration"

test-integration: ## Run integration tests only (needs Docker)
	uv run pytest -m integration

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format with ruff
	uv run ruff format .

typecheck: ## Type-check with mypy (strict)
	uv run mypy

check: lint typecheck test-unit ## Lint + typecheck + unit tests

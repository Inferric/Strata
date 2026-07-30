.PHONY: bootstrap services first-run test lint report

bootstrap:
	./scripts/bootstrap.sh

services:
	docker compose up -d postgres mlflow prefect api web

first-run:
	./scripts/first-real-run.sh

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run mypy src
	npm --prefix web run lint

report:
	uv run strata-report --summary artifacts/latest/summary.json --output reports/generated/latest

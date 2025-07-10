.PHONY: help setup test clean lint format all

help:
	@echo "Available targets:"
	@echo "  setup   - Install dependencies and setup pre-commit"
	@echo "  test    - Run tests"
	@echo "  lint    - Run pre-commit hooks on all files"
	@echo "  format  - Run formatting tools (ruff + isort)"
	@echo "  clean   - Clean up temp files"
	@echo "  all     - Setup and test"

setup:
	uv sync --group dev
	uv run pre-commit install

test:
	uv run pytest -v

lint:
	uv run pre-commit run --all-files

format:
	uv run ruff format .
	uv run isort .

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__/ .pytest_cache/
	rm -rf mlruns/ mlflow_runs/
	rm -f sentiment_pipeline.pkl mlflow_server.*
	rm -rf category_indexer/
	@echo "Done!"

all: setup test

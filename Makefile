.PHONY: help setup test test-examples test-src clean lint format all

help:
	@echo "Available targets:"
	@echo "  setup        - Install dependencies and setup pre-commit"
	@echo "  test         - Run all tests (examples + src)"
	@echo "  test-examples - Run example tests only"
	@echo "  test-src     - Run src tests only"
	@echo "  lint         - Run pre-commit hooks on all files"
	@echo "  format       - Run formatting tools (ruff + isort)"
	@echo "  clean        - Clean up temp files"
	@echo "  all          - Setup and test"

setup:
	uv sync --group dev
	uv run pre-commit install

test: test-examples test-src

test-examples:
	uv run pytest examples/tests/ -v

test-src:
	@echo "Running src tests..."
	@if [ -d "src" ] && [ -d "tests" ]; then \
		uv run pytest tests/ -v; \
	else \
		echo "No src tests found yet"; \
	fi

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

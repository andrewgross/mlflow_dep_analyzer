.PHONY: help setup test test-examples test-src clean lint format build all

help:
	@echo "Available targets:"
	@echo "  setup        - Install dependencies and setup pre-commit"
	@echo "  test         - Run all tests (examples + src)"
	@echo "  test-examples - Run example tests only"
	@echo "  test-src     - Run src tests only"
	@echo "  lint         - Run pre-commit hooks on all files"
	@echo "  format       - Run formatting tools (ruff + isort)"
	@echo "  build        - Build the package"
	@echo "  clean        - Clean up temp files and build artifacts"
	@echo "  all          - Setup and test"

setup:
	uv sync --group dev
	uv run pre-commit install

test: test-examples test-src

test-examples:
	uv run pytest examples/tests/ -v

test-src:
	uv run pytest tests/ -v

test-reset: clean test

format:
	uv run pre-commit run --all-files


build:
	@echo "Building package..."
	uv build
	@echo "Build complete!"

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__/ .pytest_cache/
	rm -rf category_indexer/
	rm -rf dist/ build/
	find . -name "*.pyc" -delete
	find . -name "*.pkl" -delete
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Done!"

all: setup test

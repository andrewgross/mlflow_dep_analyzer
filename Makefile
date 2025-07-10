.PHONY: help setup test clean all

help:
	@echo "Available targets:"
	@echo "  setup  - Install dependencies"
	@echo "  test   - Run tests"
	@echo "  clean  - Clean up temp files"
	@echo "  all    - Setup and test"

setup:
	uv sync

test:
	uv run pytest -v

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__/ .pytest_cache/ 
	rm -rf mlruns/ mlflow_runs/
	rm -f sentiment_pipeline.pkl mlflow_server.*
	rm -rf category_indexer/
	@echo "Done!"

all: setup test
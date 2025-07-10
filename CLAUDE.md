# Claude Development Guide

## Package Management
- Use `uv` for Python package management
- Prefix Python commands with `uv run` to use the managed environment
- Example: `uv run python script.py`, `uv run pytest`, `uv run mypy`

## Git Workflow
- Run tests before each commit
- Commit early and often as we fix things
- Write clear, descriptive commit messages
- If pre-commit hooks fail, fix the issues but keep the original commit message when re-committing

## Common Commands
- Install dependencies: `uv sync`
- Run tests: `uv run pytest tests/` (includes MLflow server management via fixtures)
- Run tests verbosely: `uv run pytest -v`
- Run specific test: `uv run pytest tests/test_integration.py::TestMLflowIntegration::test_model_save_with_code_paths`
- Run linting: `uv run ruff check`
- Run type checking: `uv run mypy`

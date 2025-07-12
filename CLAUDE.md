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

## Code Quality Guidelines

### Avoid Hardcoding Project-Specific Names
- **DO NOT** hardcode directory names specific to this project (like "examples", "projects")
- **DO NOT** hardcode lists of modules, packages, or other values that can be discovered programmatically
- **DO** use generalizable solutions that work across different Python projects
- **DO** leverage Python's built-in capabilities for discovery (e.g., `sys.stdlib_module_names`)

#### Examples:
❌ **Bad**: Hardcoded stdlib modules
```python
STDLIB_MODULES = {"os", "sys", "json", "datetime", ...}  # Long hardcoded list
```

✅ **Good**: Use Python's built-in discovery
```python
# Use sys.stdlib_module_names (Python 3.10+) with fallback
if hasattr(sys, 'stdlib_module_names'):
    return set(sys.stdlib_module_names)
```

❌ **Bad**: Project-specific directory names
```python
search_dirs = ["examples", "projects", "my_app"]  # Specific to this codebase
```

✅ **Good**: Common Python project patterns
```python
search_dirs = ["src", "lib", "packages"]  # Universal Python patterns
```

### Prefer Discovery Over Hardcoding
- Use Python's introspection capabilities (`inspect`, `importlib`, `sys` module attributes)
- Search for patterns rather than maintaining static lists
- Make code that works across different Python environments and project structures

# Claude Development Guide

## Package Management
- Use `uv` for Python package management
- Prefix Python commands with `uv run` to use the managed environment
- Example: `uv run python script.py`, `uv run pytest`, `uv run mypy`
- **Minimum Python Version**: 3.11+ (no need to support older versions)

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
- **DO NOT** hardcode specific package names or priority rules (like "mlflow", "tensorflow", etc.)
- **DO** use generalizable solutions that work across different Python projects
- **DO** leverage Python's built-in capabilities for discovery (e.g., `sys.stdlib_module_names`)
- **DO** use heuristics and patterns rather than explicit package lists

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

❌ **Bad**: Hardcoded package priorities
```python
priority_rules = {
    'mlflow': ['mlflow', 'mlflow-skinny'],  # Hardcoded specific packages
    'tensorflow': ['tensorflow', 'tensorflow-gpu']
}
```

✅ **Good**: Pattern-based heuristics
```python
# Prefer packages without variant suffixes
main_packages = [pkg for pkg in candidates
                if not any(suffix in pkg.lower()
                          for suffix in ['-skinny', '-headless', '-contrib'])]
```

### Prefer Discovery Over Hardcoding
- Use Python's introspection capabilities (`inspect`, `importlib`, `sys` module attributes)
- Search for patterns rather than maintaining static lists
- Make code that works across different Python environments and project structures

## Testing Guidelines

### Write Exact Tests Without Wiggle Room
- **DO NOT** use >= or > comparisons for counting dependencies unless truly necessary
- **DO** write tests that expect exact matches for deterministic outcomes
- **DO** use set operations and exact equality checks when possible

#### Examples:
❌ **Bad**: Over-permissive assertions
```python
assert len(found_deps) >= 3  # Accepts any number 3 or higher
```

✅ **Good**: Exact expectations
```python
expected_deps = {"pandas", "numpy", "scikit-learn"}
assert expected_deps.issubset(found_deps), f"Missing: {expected_deps - found_deps}"
```

❌ **Bad**: Testing raw requirements with versions
```python
requirements = ["pandas==2.3.1", "numpy==2.3.1"]
assert "pandas" in requirements  # Fails due to versioning
```

✅ **Good**: Extract package names first
```python
requirements = ["pandas==2.3.1", "numpy==2.3.1"]
package_names = {req.split("==")[0] for req in requirements}
assert "pandas" in package_names
```

## Task Management

### Use Multiple Parallel Tasks When Possible
- **DO** leverage the ability to run multiple tasks concurrently for maximum efficiency
- **DO** batch independent searches, file reads, and analysis tasks together
- **DO** break complex work into parallel subtasks when they don't depend on each other

#### Examples:
✅ **Good**: Parallel task usage
```python
# Run multiple searches concurrently
- Search for hardcoded lists
- Analyze test failures
- Examine module logic
- Review package discovery code
```

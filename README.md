# MLflow Packaging Test Repository

This repository demonstrates the problem and solution for packaging MLflow models with shared code dependencies, specifically for Databricks environments.

## Problem Statement

When training models in Databricks that inherit from shared base classes, the models fail to load outside the original workspace due to missing imports. This happens because:

1. Models depend on shared utilities (e.g., `projects.shared_utils.base_model.BaseModelV3`)
2. These imports work during training (running in the workspace)
3. But fail during inference when the model is loaded elsewhere

## Solution

Use MLflow's `code_paths` parameter to bundle the entire `projects/` directory with the model, making it self-contained and portable.

## Repository Structure

```
projects/
├── shared_utils/
│   ├── __init__.py
│   ├── base_model.py          # BaseModelV3 class
│   └── databricks/
│       ├── __init__.py
│       └── helpers.py         # Enhanced log_model functions
├── my_model/
│   ├── __init__.py
│   └── sentiment_model.py     # Example model subclass
notebooks/
├── train_model.py             # Training simulation
tests/
├── test_model_loading.py      # Testing model loading scenarios
```

## Key Files

### `projects/shared_utils/base_model.py`
- Contains `BaseModelV3` base class with artifact management
- Handles serialization of complex objects like StringIndexerModels

### `projects/shared_utils/databricks/helpers.py`
- `log_model()` - Standard logging function
- `log_model_with_code_paths()` - Enhanced version that includes source code
- `get_projects_source_path()` - Helper to get correct path for code_paths

### `projects/my_model/sentiment_model.py`
- Example model that inherits from BaseModelV3
- Demonstrates real-world usage patterns

## Usage

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Integration Tests
```bash
uv run pytest
```

The test suite automatically:
- Starts an MLflow server using session fixtures
- Creates PySpark session for Spark artifacts
- Tests both approaches (with/without code_paths)
- Simulates isolated environments
- Cleans up automatically

### 3. Run Specific Tests
```bash
# Test basic functionality
uv run pytest tests/test_integration.py::TestMLflowIntegration::test_model_training_and_basic_prediction

# Test code_paths solution
uv run pytest tests/test_integration.py::TestMLflowIntegration::test_model_save_with_code_paths

# Test isolated environment loading
uv run pytest tests/test_integration.py::TestMLflowIntegration::test_model_loading_in_isolated_environment
```

## The Solution in Action

### Without code_paths (problematic):
```python
log_model(
    model=model,
    artifact_path="sentiment_model"
)
```

### With code_paths (solution):
```python
log_model_with_code_paths(
    model=model,
    artifact_path="sentiment_model"
)
```

The enhanced function automatically includes the `projects/` directory, making the model self-contained.

## Benefits

1. **Portability** - Models work anywhere, not just in the original workspace
2. **Reproducibility** - Each model version includes exact code used at training time
3. **No wheel building** - Keeps current workflow intact
4. **Version isolation** - Different model versions can use different code versions

## Testing Scenarios

The pytest integration suite demonstrates:
1. **Basic model training and prediction** - Verifies core functionality
2. **Save/load without code_paths** - Works in same environment only
3. **Save/load with code_paths** - Works everywhere (portable solution)
4. **Isolated environment testing** - Simulates different Databricks workspaces
5. **Spark artifacts handling** - Tests StringIndexer and other Spark components
6. **Multiple model versions** - Ensures code isolation between versions

## Test Fixtures

- **`mlflow_server`** - Session-scoped MLflow server with automatic cleanup
- **`spark_session`** - Session-scoped PySpark context
- **`isolated_env`** - Context manager for testing in clean environments
- **`trained_model`** - Pre-trained sentiment model for testing
- **`sample_data`** - Consistent training data across tests

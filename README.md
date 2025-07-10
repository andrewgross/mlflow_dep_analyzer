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

### 2. Train and Log Models
```bash
uv run python notebooks/train_model.py
```

This will create two model versions:
- One without `code_paths` (problematic)
- One with `code_paths` (solution)

### 3. Test Model Loading
```bash
uv run python tests/test_model_loading.py
```

This demonstrates loading models in different environments to show the difference.

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

The test suite demonstrates:
1. Loading models in the same environment
2. Loading models in clean environments (no repo access)
3. Loading models from different directories
4. Comparing behavior with/without code_paths
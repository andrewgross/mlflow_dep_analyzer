# MLflow Dependency Analyzer

**Smart dependency analysis and minimal requirements generation for MLflow models.**

Automatically detect and generate minimal `code_paths` and `requirements` lists for your MLflow models using AST-based analysis, ensuring portable and reproducible model deployments without dependency bloat.

## Installation

```bash
pip install mlflow-dep-analyzer
```

## Quick Start

Log an MLflow model with minimal dependencies:

```python
import mlflow
import mlflow.sklearn
from mlflow_dep_analyzer import analyze_code_dependencies, analyze_code_paths
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Train your model
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Find minimal requirements
requirements = analyze_code_dependencies(
    code_paths=["my_model.py"],
    repo_root="."
)

# Find code dependencies
code_paths = analyze_code_paths(
    entry_files=["my_model.py"],
    repo_root="."
)

# Log model with minimal dependencies
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "random_forest",
        code_paths=code_paths,
        pip_requirements=requirements
    )
```

## Development

This repo uses `uv` for environment management. For the simplest setup, make sure it is installed.

```bash
git clone https://github.com/andrewgross/mlflow-dep-analyzer
cd mlflow-dep-analyzer
make setup
make test
```

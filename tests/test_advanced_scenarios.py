"""
Advanced test scenarios for comprehensive dependency analysis.

This module tests the most challenging scenarios including lazy imports,
modern ML frameworks, and advanced dependency patterns that are critical
for real-world usage.
"""

import pytest

from mlflow_dep_analyzer.unified_analyzer import UnifiedDependencyAnalyzer
from tests.fixtures.lazy_imports import create_lazy_imports_fixtures
from tests.fixtures.ml_frameworks import create_ml_frameworks_fixtures


class TestMLFrameworkDependencies:
    """Test dependency analysis for modern ML frameworks."""

    @pytest.fixture
    def ml_frameworks(self, tmp_path):
        """Create ML framework fixtures."""
        return create_ml_frameworks_fixtures(tmp_path)

    def test_tensorflow_project_dependencies(self, ml_frameworks):
        """Test comprehensive TensorFlow project dependency detection."""
        analyzer = UnifiedDependencyAnalyzer(str(ml_frameworks["tensorflow"]))

        train_file = ml_frameworks["tensorflow"] / "train.py"
        serve_file = ml_frameworks["tensorflow"] / "serve.py"

        result = analyzer.analyze_dependencies([str(train_file), str(serve_file)])

        # Verify structure
        assert "requirements" in result
        assert "code_paths" in result

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Core TensorFlow ecosystem
        expected_tf_deps = {
            "tensorflow",
            "tensorboard",
            "tensorflow-probability",
            "tensorflow-datasets",
            "tensorflow-addons",
            "tensorflow-hub",
            "tensorflow-model-optimization",
            "tensorflowjs",
        }

        # Scientific computing stack
        expected_sci_deps = {"numpy", "pandas", "matplotlib", "seaborn", "scikit-learn"}

        # MLflow integration
        expected_mlflow_deps = {"mlflow"}

        # Check key dependencies are found
        key_deps = expected_tf_deps | expected_sci_deps | expected_mlflow_deps
        found_key_deps = key_deps.intersection(package_names)
        assert len(found_key_deps) >= 8, f"Expected TensorFlow deps, found: {found_key_deps}"

        # Should not include stdlib modules
        stdlib_modules = {"os", "sys", "json", "time", "pathlib", "logging"}
        found_stdlib = stdlib_modules.intersection(package_names)
        assert len(found_stdlib) == 0, f"Found stdlib modules: {found_stdlib}"

    def test_pytorch_project_dependencies(self, ml_frameworks):
        """Test comprehensive PyTorch project dependency detection."""
        analyzer = UnifiedDependencyAnalyzer(str(ml_frameworks["pytorch"]))

        train_file = ml_frameworks["pytorch"] / "train.py"

        result = analyzer.analyze_dependencies([str(train_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Core PyTorch ecosystem
        expected_torch_deps = {
            "torch",
            "torchvision",
            "pytorch-lightning",
            "torchmetrics",
            "timm",
            "efficientnet-pytorch",
            "segmentation-models-pytorch",
        }

        # Data processing
        expected_data_deps = {"albumentations", "numpy", "pandas"}

        # Check key dependencies by category
        found_torch_deps = expected_torch_deps.intersection(package_names)
        found_data_deps = expected_data_deps.intersection(package_names)

        # Expect at least some PyTorch deps and some data processing deps
        assert len(found_torch_deps) >= 3, f"Expected PyTorch deps, found: {found_torch_deps}"
        assert len(found_data_deps) >= 2, f"Expected data deps, found: {found_data_deps}"

    def test_huggingface_project_dependencies(self, ml_frameworks):
        """Test Hugging Face Transformers project dependency detection."""
        analyzer = UnifiedDependencyAnalyzer(str(ml_frameworks["huggingface"]))

        train_file = ml_frameworks["huggingface"] / "train_nlp.py"

        result = analyzer.analyze_dependencies([str(train_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Core transformers ecosystem
        expected_hf_deps = {
            "transformers",
            "datasets",
            "evaluate",
            "peft",
            "accelerate",
            "deepspeed",
            "optimum",
            "sentencepiece",
            "tokenizers",
        }

        # Supporting libraries
        expected_support_deps = {"wandb", "mlflow", "torch"}

        # Check key dependencies
        key_deps = expected_hf_deps | expected_support_deps
        found_key_deps = key_deps.intersection(package_names)
        assert len(found_key_deps) >= 8, f"Expected HuggingFace deps, found: {found_key_deps}"

    def test_scientific_computing_project_dependencies(self, ml_frameworks):
        """Test scientific computing project dependency detection."""
        analyzer = UnifiedDependencyAnalyzer(str(ml_frameworks["scientific"]))

        analysis_file = ml_frameworks["scientific"] / "analysis.py"

        result = analyzer.analyze_dependencies([str(analysis_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Core scientific stack
        expected_sci_deps = {
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "altair",
        }

        # Advanced scientific libraries
        expected_advanced_deps = {
            "numba",
            "dask",
            "xarray",
            "polars",
            "astropy",
            "networkx",
            "statsmodels",
            "pymc3",
            "sympy",
            "cvxpy",
        }

        # Check comprehensive scientific ecosystem
        key_deps = expected_sci_deps | expected_advanced_deps
        found_key_deps = key_deps.intersection(package_names)
        assert len(found_key_deps) >= 15, f"Expected scientific deps, found: {found_key_deps}"


class TestLazyImportDetection:
    """Test detection of lazy imports - the critical missing capability."""

    @pytest.fixture
    def lazy_imports(self, tmp_path):
        """Create lazy import fixtures."""
        return create_lazy_imports_fixtures(tmp_path)

    def test_function_level_imports_detection(self, lazy_imports):
        """Test detection of imports inside functions."""
        analyzer = UnifiedDependencyAnalyzer(str(lazy_imports))

        func_imports_file = lazy_imports / "function_imports" / "basic_function_imports.py"

        result = analyzer.analyze_dependencies([str(func_imports_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # These imports are INSIDE functions and should be detected
        # This test currently FAILS because AST analysis doesn't detect
        # imports inside function bodies
        expected_lazy_imports = {
            "pandas",
            "numpy",
            "scikit-learn",
            "torch",
            "tensorflow",
            "matplotlib",
            "seaborn",
            "plotly",
            "requests",
            "yaml",
            "xgboost",
            "lightgbm",
            "catboost",
        }

        found_lazy_imports = expected_lazy_imports.intersection(package_names)

        # This assertion will fail with current implementation
        # demonstrating the critical gap in lazy import detection
        assert (
            len(found_lazy_imports) >= 8
        ), f"CRITICAL GAP: Lazy imports not detected. Expected: {expected_lazy_imports}, Found: {found_lazy_imports}"

    def test_conditional_imports_detection(self, lazy_imports):
        """Test detection of conditional imports."""
        analyzer = UnifiedDependencyAnalyzer(str(lazy_imports))

        conditional_file = lazy_imports / "conditional_imports" / "environment_conditional.py"

        result = analyzer.analyze_dependencies([str(conditional_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Conditional imports that should be detected
        expected_conditional = {
            "prometheus_client",
            "statsd",
            "sentry_sdk",
            "redis",
            "sqlalchemy",
            "psycopg2",
            "pymongo",
            "celery",
            "pika",
        }

        found_conditional = expected_conditional.intersection(package_names)

        # This test shows limited conditional import detection
        assert len(found_conditional) >= 3, f"Limited conditional import detection. Found: {found_conditional}"

    def test_class_method_imports_detection(self, lazy_imports):
        """Test detection of imports inside class methods."""
        analyzer = UnifiedDependencyAnalyzer(str(lazy_imports))

        class_file = lazy_imports / "class_method_imports" / "class_imports.py"

        result = analyzer.analyze_dependencies([str(class_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Method-level imports
        expected_method_imports = {
            "scikit-learn",
            "torch",
            "transformers",
            "pandas",
            "numpy",
            "matplotlib",
            "plotly",
            "bokeh",
            "altair",
        }

        found_method_imports = expected_method_imports.intersection(package_names)

        # Another critical gap - method-level imports not detected
        assert len(found_method_imports) >= 4, f"Method-level imports not fully detected. Found: {found_method_imports}"

    def test_property_imports_detection(self, lazy_imports):
        """Test detection of imports inside properties."""
        analyzer = UnifiedDependencyAnalyzer(str(lazy_imports))

        property_file = lazy_imports / "property_imports" / "property_imports.py"

        result = analyzer.analyze_dependencies([str(property_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Property-level imports
        expected_property_imports = {
            "pandas",
            "numpy",
            "scikit-learn",
            "torch",
            "tensorflow",
            "matplotlib",
            "seaborn",
            "plotly",
            "polars",
            "dask",
        }

        found_property_imports = expected_property_imports.intersection(package_names)

        # Property imports are also not detected by current AST analysis
        assert len(found_property_imports) >= 3, f"Property-level imports not detected. Found: {found_property_imports}"

    def test_decorator_imports_detection(self, lazy_imports):
        """Test detection of imports inside decorators."""
        analyzer = UnifiedDependencyAnalyzer(str(lazy_imports))

        decorator_file = lazy_imports / "decorator_imports" / "decorator_imports.py"

        result = analyzer.analyze_dependencies([str(decorator_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Decorator-level imports
        expected_decorator_imports = {
            "pandas",
            "numpy",
            "scikit-learn",
            "torch",
            "tensorflow",
            "redis",
            "prometheus_client",
            "statsd",
            "pydantic",
            "cerberus",
        }

        found_decorator_imports = expected_decorator_imports.intersection(package_names)

        # Decorator imports also not fully detected
        assert (
            len(found_decorator_imports) >= 2
        ), f"Decorator-level imports not detected. Found: {found_decorator_imports}"

    def test_runtime_imports_detection(self, lazy_imports):
        """Test detection of runtime imports using importlib."""
        analyzer = UnifiedDependencyAnalyzer(str(lazy_imports))

        runtime_file = lazy_imports / "runtime_imports" / "runtime_imports.py"

        result = analyzer.analyze_dependencies([str(runtime_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Runtime imports are nearly impossible to detect statically
        # but we should at least find importlib usage
        assert "importlib" not in requirements, "importlib is stdlib and should be filtered"

        # Some imports might still be detectable
        possible_runtime_imports = {"scikit-learn", "torch", "tensorflow", "pandas", "numpy"}

        found_runtime = possible_runtime_imports.intersection(package_names)

        # Runtime imports are the hardest to detect statically
        # This demonstrates the fundamental limitation of static analysis
        print(f"Runtime imports found: {found_runtime} (limited by static analysis)")


class TestAdvancedMLflowIntegration:
    """Test MLflow-specific integration scenarios."""

    def test_mlflow_model_flavors_dependencies(self, tmp_path):
        """Test MLflow model flavor dependency detection."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create MLflow flavor usage scenarios
        mlflow_sklearn_file = tmp_path / "mlflow_sklearn.py"
        mlflow_sklearn_file.write_text('''
"""MLflow sklearn flavor usage."""

import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow.tracking import MlflowClient

import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


def train_and_log_sklearn_model():
    """Train and log sklearn model with MLflow."""

    # Generate sample data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log metrics
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        return model


def load_and_serve_model(model_uri: str):
    """Load and serve MLflow model."""

    # Load model
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Create prediction function
    def predict(data):
        return loaded_model.predict(data)

    return predict
''')

        result = analyzer.analyze_dependencies([str(mlflow_sklearn_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # MLflow sklearn flavor dependencies
        expected_mlflow_sklearn = {"mlflow", "scikit-learn", "pandas", "numpy"}
        found_mlflow_sklearn = expected_mlflow_sklearn.intersection(package_names)
        not_found_mlflow_sklearn = expected_mlflow_sklearn.difference(package_names)

        assert len(found_mlflow_sklearn) >= 3, f"MLflow sklearn flavor deps not found: {not_found_mlflow_sklearn}"

    def test_mlflow_pytorch_flavor_dependencies(self, tmp_path):
        """Test MLflow PyTorch flavor dependency detection."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        mlflow_pytorch_file = tmp_path / "mlflow_pytorch.py"
        mlflow_pytorch_file.write_text('''
"""MLflow PyTorch flavor usage."""

import mlflow
import mlflow.pytorch
import mlflow.tracking

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np


class SimpleNet(nn.Module):
    """Simple neural network."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def train_and_log_pytorch_model():
    """Train and log PyTorch model with MLflow."""

    # Generate sample data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))

    with mlflow.start_run():
        # Create model
        model = SimpleNet(10, 1)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y.float())
            loss.backward()
            optimizer.step()

        # Log model
        mlflow.pytorch.log_model(model, "pytorch_model")
        mlflow.log_metric("final_loss", loss.item())

        return model
''')

        result = analyzer.analyze_dependencies([str(mlflow_pytorch_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # MLflow PyTorch flavor dependencies
        expected_mlflow_pytorch = {"mlflow", "torch", "numpy"}
        found_mlflow_pytorch = expected_mlflow_pytorch.intersection(package_names)

        # Expect all three key dependencies
        assert expected_mlflow_pytorch.issubset(
            found_mlflow_pytorch
        ), f"Missing MLflow PyTorch deps: {expected_mlflow_pytorch - found_mlflow_pytorch}"

    def test_mlflow_custom_flavor_dependencies(self, tmp_path):
        """Test MLflow custom flavor dependency detection."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        custom_flavor_file = tmp_path / "mlflow_custom.py"
        custom_flavor_file.write_text('''
"""MLflow custom flavor implementation."""

import mlflow
from mlflow import pyfunc
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class CustomModelWrapper(pyfunc.PythonModel):
    """Custom model wrapper for MLflow."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        """Predict method for MLflow pyfunc."""
        return self.model.predict(model_input)


def create_custom_flavor():
    """Create and log custom MLflow flavor."""

    # Train a model
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier()
    model.fit(X, y)

    # Wrap in custom flavor
    custom_model = CustomModelWrapper(model)

    # Define signature
    signature = ModelSignature(
        inputs=Schema([ColSpec("double", f"feature_{i}") for i in range(5)]),
        outputs=Schema([ColSpec("long", "prediction")])
    )

    with mlflow.start_run():
        # Log custom model
        mlflow.pyfunc.log_model(
            "custom_model",
            python_model=custom_model,
            signature=signature,
            conda_env={
                "dependencies": [
                    "scikit-learn",
                    "pandas",
                    "numpy"
                ]
            }
        )

        return custom_model
''')

        result = analyzer.analyze_dependencies([str(custom_flavor_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Custom flavor dependencies
        expected_custom = {"mlflow", "joblib", "pandas", "numpy", "scikit-learn"}
        found_custom = expected_custom.intersection(package_names)

        assert len(found_custom) >= 4, f"MLflow custom flavor deps not found: {found_custom}"


class TestModernPythonProjectStructures:
    """Test modern Python project structure patterns."""

    def test_poetry_project_structure(self, tmp_path):
        """Test Poetry-based project structure analysis."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create Poetry project structure
        poetry_project = tmp_path / "poetry_project"
        poetry_project.mkdir()

        # pyproject.toml
        (poetry_project / "pyproject.toml").write_text("""
[tool.poetry]
name = "ml-poetry-project"
version = "0.1.0"
description = "ML project using Poetry"

[tool.poetry.dependencies]
python = "^3.8"
pandas = "^1.4.0"
numpy = "^1.21.0"
scikit-learn = "^1.1.0"
mlflow = "^1.28.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
black = "^22.0.0"
mypy = "^0.971"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
""")

        # Main module
        src_dir = poetry_project / "src" / "ml_poetry_project"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").touch()

        main_file = src_dir / "main.py"
        main_file.write_text('''
"""Main module for Poetry ML project."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


def train_model():
    """Train ML model."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)

    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X, y)

        mlflow.sklearn.log_model(model, "model")

    return model
''')

        result = analyzer.analyze_dependencies([str(main_file)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}
        expected_deps = {"pandas", "numpy", "scikit-learn", "mlflow"}
        found_deps = expected_deps.intersection(package_names)

        assert len(found_deps) >= 3, f"Poetry project deps not found: {found_deps}"

    def test_monorepo_structure(self, tmp_path):
        """Test monorepo structure with multiple packages."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create monorepo structure
        monorepo = tmp_path / "ml_monorepo"
        monorepo.mkdir()

        # Package 1: Data processing
        data_pkg = monorepo / "packages" / "data_processing"
        data_pkg.mkdir(parents=True)
        (data_pkg / "__init__.py").touch()

        data_main = data_pkg / "processor.py"
        data_main.write_text('''
"""Data processing package."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd


class DataProcessor:
    """Process data using multiple libraries."""

    def __init__(self):
        self.scaler = StandardScaler()

    def process_pandas(self, data):
        """Process with pandas."""
        df = pd.DataFrame(data)
        return self.scaler.fit_transform(df)

    def process_dask(self, data):
        """Process with dask."""
        df = dd.from_pandas(pd.DataFrame(data), npartitions=2)
        return df.mean().compute()
''')

        # Package 2: Model training
        model_pkg = monorepo / "packages" / "model_training"
        model_pkg.mkdir(parents=True)
        (model_pkg / "__init__.py").touch()

        model_main = model_pkg / "trainer.py"
        model_main.write_text('''
"""Model training package."""

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.pytorch
import mlflow.sklearn

# Import from other package in monorepo
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "data_processing"))
from processor import DataProcessor


class ModelTrainer:
    """Train models using different frameworks."""

    def __init__(self):
        self.data_processor = DataProcessor()

    def train_sklearn(self, X, y):
        """Train sklearn model."""
        processed_X = self.data_processor.process_pandas(X)

        with mlflow.start_run():
            model = RandomForestClassifier()
            model.fit(processed_X, y)
            mlflow.sklearn.log_model(model, "sklearn_model")

        return model

    def train_pytorch(self, X, y):
        """Train PyTorch model."""
        model = nn.Linear(X.shape[1], 1)

        with mlflow.start_run():
            # Training logic here
            mlflow.pytorch.log_model(model, "pytorch_model")

        return model
''')

        result = analyzer.analyze_dependencies([str(data_main), str(model_main)])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Should find dependencies from both packages
        expected_deps = {"pandas", "numpy", "scikit-learn", "dask", "torch", "mlflow"}
        found_deps = expected_deps.intersection(package_names)

        assert len(found_deps) >= 5, f"Monorepo deps not found: {found_deps}"

        # Should include cross-package references
        code_paths = result["code_paths"]
        assert any("data_processing" in path for path in code_paths)
        assert any("model_training" in path for path in code_paths)


class TestPerformanceAndScalability:
    """Test performance with large and complex codebases."""

    def test_large_codebase_performance(self, tmp_path):
        """Test analysis performance with large codebase."""
        import time

        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create large codebase (simplified for testing)
        large_project = tmp_path / "large_project"
        large_project.mkdir()

        # Create multiple files with complex dependencies
        files_to_analyze = []

        for i in range(20):  # Reduced from 100+ for test performance
            module_file = large_project / f"module_{i}.py"

            # Vary dependencies per module
            if i % 4 == 0:
                content = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn

def process_data():
    X = np.random.randn(100, 10)
    df = pd.DataFrame(X)
    model = RandomForestClassifier()
    return model, df
"""
            elif i % 4 == 1:
                content = """
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel
import mlflow.pytorch

def create_model():
    model = nn.Linear(10, 1)
    transformer = AutoModel.from_pretrained("bert-base-uncased")
    return model, transformer
"""
            elif i % 4 == 2:
                content = """
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats

def visualize_data(data):
    fig, ax = plt.subplots()
    sns.heatmap(data, ax=ax)
    return fig
"""
            else:
                content = f"""
import requests
import json
import yaml
from pathlib import Path
from module_{max(0, i - 1)} import process_data

def load_config():
    config = {{"module_id": {i}}}
    return config
"""

            module_file.write_text(content)
            files_to_analyze.append(str(module_file))

        # Measure analysis time
        start_time = time.time()
        result = analyzer.analyze_dependencies(files_to_analyze)
        analysis_time = time.time() - start_time

        # Should complete in reasonable time
        assert analysis_time < 30.0, f"Analysis too slow: {analysis_time:.2f}s"

        # Should find comprehensive dependencies
        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}
        expected_large_deps = {
            "pandas",
            "numpy",
            "scikit-learn",
            "mlflow",
            "torch",
            "transformers",
            "matplotlib",
            "seaborn",
            "plotly",
            "scipy",
            "requests",
            "yaml",
        }

        found_large_deps = expected_large_deps.intersection(package_names)
        assert len(found_large_deps) >= 8, f"Large codebase deps: {found_large_deps}"

    def test_deep_import_chains(self, tmp_path):
        """Test analysis of deep import chains."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create deep import chain
        chain_length = 10
        files_created = []

        for i in range(chain_length):
            module_file = tmp_path / f"chain_{i}.py"

            if i == 0:
                # Root module with external dependencies
                content = """
import pandas as pd
import numpy as np
import sklearn.ensemble
from chain_1 import next_function

def root_function():
    df = pd.DataFrame({"x": [1, 2, 3]})
    return next_function(df)
"""
            elif i == chain_length - 1:
                # Terminal module
                content = """
import matplotlib.pyplot as plt
import json

def terminal_function(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    return {"final": True}
"""
            else:
                # Intermediate modules
                content = f"""
import requests
import yaml
from chain_{i + 1} import next_function

def intermediate_function_{i}(data):
    config = yaml.safe_load("{{test: true}}")
    return next_function(data)

# Alias for import
next_function = intermediate_function_{i}
"""

            module_file.write_text(content)
            files_created.append(str(module_file))

        # Analyze the root module (should follow entire chain)
        result = analyzer.analyze_dependencies([files_created[0]])

        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Should find dependencies from entire chain
        expected_chain_deps = {"pandas", "numpy", "scikit-learn", "matplotlib", "requests", "yaml"}

        found_chain_deps = expected_chain_deps.intersection(package_names)
        assert len(found_chain_deps) >= 4, f"Deep chain deps: {found_chain_deps}"

        # Should include multiple chain files
        code_paths = result["code_paths"]
        chain_files_found = sum(1 for path in code_paths if "chain_" in path)
        assert chain_files_found >= 3, f"Chain files found: {chain_files_found}"


def test_comprehensive_integration_scenario(tmp_path):
    """Integration test combining multiple advanced scenarios."""

    # Create comprehensive project structure
    project = tmp_path / "comprehensive_ml_project"
    project.mkdir()

    # Create main application with lazy imports
    app_file = project / "app.py"
    app_file.write_text('''
"""Comprehensive ML application with lazy imports."""

import os
import json
from typing import Dict, Any, Optional

def create_ml_pipeline(framework: str, data_path: str):
    """Create ML pipeline with framework-specific imports."""

    if framework == "sklearn":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import pandas as pd
        import numpy as np
        import mlflow.sklearn

        # Load and process data
        data = pd.read_csv(data_path)
        X = data.drop("target", axis=1)
        y = data["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        with mlflow.start_run():
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "sklearn_model")

        return model

    elif framework == "pytorch":
        import torch
        import torch.nn as nn
        import pytorch_lightning as pl
        import mlflow.pytorch

        class LightningModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self.forward(x)
                loss = nn.functional.mse_loss(y_hat, y)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters())

        with mlflow.start_run():
            model = LightningModel()
            mlflow.pytorch.log_model(model, "pytorch_model")

        return model

    elif framework == "transformers":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import Trainer, TrainingArguments
        import mlflow.transformers

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )

        with mlflow.start_run():
            mlflow.transformers.log_model(
                transformers_model=model,
                artifact_path="transformers_model",
                tokenizer=tokenizer
            )

        return model

    else:
        raise ValueError(f"Unknown framework: {framework}")


def deploy_model(deployment_target: str, model_uri: str):
    """Deploy model with target-specific imports."""

    if deployment_target == "aws":
        import boto3
        import sagemaker
        from sagemaker.pytorch import PyTorchModel

        session = sagemaker.Session()
        role = sagemaker.get_execution_role()

        # Deploy to SageMaker
        model = PyTorchModel(
            model_data=model_uri,
            role=role,
            entry_point="inference.py",
            framework_version="1.12"
        )

        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.large"
        )

        return predictor

    elif deployment_target == "gcp":
        from google.cloud import aiplatform
        import google.cloud.storage as storage

        aiplatform.init(project="my-project")

        model = aiplatform.Model.upload(
            display_name="ml-model",
            artifact_uri=model_uri,
            serving_container_image_uri="gcr.io/my-project/predictor"
        )

        endpoint = model.deploy(
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3
        )

        return endpoint

    elif deployment_target == "local":
        import flask
        from flask import Flask, request, jsonify
        import mlflow.pyfunc

        app = Flask(__name__)
        model = mlflow.pyfunc.load_model(model_uri)

        @app.route("/predict", methods=["POST"])
        def predict():
            data = request.json
            predictions = model.predict(data)
            return jsonify(predictions.tolist())

        return app

    else:
        import mlflow.deployments

        client = mlflow.deployments.get_deploy_client(deployment_target)
        deployment = client.create_deployment(
            name="ml-model",
            model_uri=model_uri,
            config={"instance_type": "standard"}
        )

        return deployment
''')

    # Analyze comprehensive application
    analyzer = UnifiedDependencyAnalyzer(str(project))
    result = analyzer.analyze_dependencies([str(app_file)])

    requirements = set(result["requirements"])

    # Should detect comprehensive ML ecosystem despite lazy imports
    comprehensive_deps = {
        "scikit-learn",
        "pandas",
        "numpy",
        "torch",
        "pytorch-lightning",
        "transformers",
        "mlflow",
        "boto3",
        "sagemaker",
        "google-cloud-aiplatform",
        "flask",
    }

    found_comprehensive = comprehensive_deps.intersection(requirements)

    # This test demonstrates the current limitations
    # Many dependencies will be missed due to lazy import detection gap
    print(f"Comprehensive deps found: {found_comprehensive}")
    print(f"Total requirements found: {len(requirements)}")

    # Even with limitations, should find some dependencies
    assert len(found_comprehensive) >= 3, f"Comprehensive scenario deps: {found_comprehensive}"

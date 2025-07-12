"""
Tests for the UnifiedDependencyAnalyzer.
"""

import sys
from pathlib import Path

import pytest

from mlflow_dep_analyzer.unified_analyzer import (
    DependencyType,
    UnifiedDependencyAnalyzer,
    analyze_model_dependencies,
    get_model_code_paths,
    get_model_requirements,
)


@pytest.fixture(autouse=True)
def clean_imports():
    """Clean up import state between tests to prevent interference."""
    # Store original sys.path
    original_path = sys.path.copy()

    # Get list of modules that existed before test
    original_modules = set(sys.modules.keys())

    yield

    # Restore sys.path
    sys.path[:] = original_path

    # Remove any new modules that were imported during the test
    # (but keep built-ins and important ones)
    current_modules = set(sys.modules.keys())
    new_modules = current_modules - original_modules

    for module_name in new_modules:
        # Only remove test-related modules, not system ones
        if (
            module_name.startswith("tmp")
            or "test_" in module_name
            or module_name
            in [
                "utils",
                "helper",
                "shared",
                "mypackage",
                "projects",
                "module_a",
                "module_b",
                "sentiment",
                "preprocessing",
                "loader",
                "problematic_module",
                "local_mod",
            ]
        ):
            sys.modules.pop(module_name, None)
        # Also remove any modules that have file paths in temporary directories
        elif module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "__file__") and module.__file__:
                if "/tmp" in module.__file__ or "pytest-" in module.__file__:
                    sys.modules.pop(module_name, None)


class TestUnifiedDependencyAnalyzer:
    """Test cases for UnifiedDependencyAnalyzer."""

    def test_basic_dependency_analysis(self, tmp_path):
        """Test basic dependency analysis with mixed import types."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create a simple model file with different types of imports
        model_file = tmp_path / "model.py"
        model_file.write_text("""
import os  # stdlib
import pandas as pd  # external package
from utils import helper  # local file

def train_model():
    data = pd.DataFrame()
    return helper(data)
""")

        # Create local utility file
        utils_file = tmp_path / "utils.py"
        utils_file.write_text("""
import json  # stdlib

def helper(data):
    return json.dumps({"result": len(data)})
""")

        result = analyzer.analyze_dependencies([str(model_file)])

        # Check structure
        assert "requirements" in result
        assert "code_paths" in result
        assert "analysis" in result

        # Check that pandas is in requirements (external package)
        assert "pandas" in result["requirements"]

        # Check that os and json are NOT in requirements (stdlib)
        assert "os" not in result["requirements"]
        assert "json" not in result["requirements"]

        # Check that local files are in code paths
        code_paths = result["code_paths"]
        assert any("model.py" in path for path in code_paths)
        assert any("utils.py" in path for path in code_paths)

        # Check analysis metadata
        analysis = result["analysis"]
        assert analysis["total_modules"] > 0
        assert analysis["external_packages"] >= 1  # pandas
        assert analysis["local_files"] >= 2  # model.py, utils.py

    def test_package_structure_with_init_files(self, tmp_path):
        """Test analysis with proper Python package structure."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create package structure
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").touch()

        # Main model file
        model_file = pkg_dir / "model.py"
        model_file.write_text("""
from mypackage.utils import process_data
from mypackage.data.loader import load_data
import numpy as np

def train():
    data = load_data()
    return process_data(data)
""")

        # Utils module
        utils_file = pkg_dir / "utils.py"
        utils_file.write_text("""
import sklearn.linear_model as lm

def process_data(data):
    model = lm.LinearRegression()
    return model.fit(data, [1, 2, 3])
""")

        # Data subpackage
        data_dir = pkg_dir / "data"
        data_dir.mkdir()
        (data_dir / "__init__.py").touch()
        loader_file = data_dir / "loader.py"
        loader_file.write_text("""
import pathlib

def load_data():
    return [[1, 2], [3, 4]]
""")

        result = analyzer.analyze_dependencies([str(model_file)])

        # Check external packages
        requirements = result["requirements"]
        assert "numpy" in requirements
        assert "sklearn" in requirements

        # Check that stdlib is not in requirements
        assert "pathlib" not in requirements

        # Check code paths include all local files
        code_paths = result["code_paths"]
        assert any("mypackage/model.py" in path for path in code_paths)
        assert any("mypackage/utils.py" in path for path in code_paths)
        assert any("mypackage/data/loader.py" in path for path in code_paths)

    def test_src_directory_structure(self, tmp_path):
        """Test analysis with src/ directory structure."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create src/ structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        pkg_dir = src_dir / "myproject"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").touch()

        # Model in src/
        model_file = pkg_dir / "model.py"
        model_file.write_text("""
from myproject.core import Engine
import requests

class Model:
    def __init__(self):
        self.engine = Engine()
        self.session = requests.Session()
""")

        # Core module
        core_file = pkg_dir / "core.py"
        core_file.write_text("""
import datetime

class Engine:
    def __init__(self):
        self.created = datetime.datetime.now()
""")

        result = analyzer.analyze_dependencies([str(model_file)])

        # Check external package
        assert "requests" in result["requirements"]

        # Check stdlib not included
        assert "datetime" not in result["requirements"]

        # Check code paths (should be relative to repo root, not src/)
        code_paths = result["code_paths"]
        assert any("src/myproject/model.py" in path for path in code_paths)
        assert any("src/myproject/core.py" in path for path in code_paths)

    def test_relative_imports(self, tmp_path):
        """Test handling of relative imports."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create nested package structure
        pkg_dir = tmp_path / "package"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").touch()

        sub_dir = pkg_dir / "submodule"
        sub_dir.mkdir()
        (sub_dir / "__init__.py").touch()

        # Module with relative imports
        main_file = sub_dir / "main.py"
        main_file.write_text("""
from . import helper
from ..utils import shared_function
from .nested.deep import deep_func

def process():
    return helper.help() + shared_function() + deep_func()
""")

        # Sibling module
        helper_file = sub_dir / "helper.py"
        helper_file.write_text("""
def help():
    return "helped"
""")

        # Parent utils
        utils_file = pkg_dir / "utils.py"
        utils_file.write_text("""
def shared_function():
    return "shared"
""")

        # Nested module
        nested_dir = sub_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "__init__.py").touch()
        deep_file = nested_dir / "deep.py"
        deep_file.write_text("""
def deep_func():
    return "deep"
""")

        result = analyzer.analyze_dependencies([str(main_file)])

        # Should find all local files
        code_paths = result["code_paths"]
        assert any("package/submodule/main.py" in path for path in code_paths)
        assert any("package/submodule/helper.py" in path for path in code_paths)
        assert any("package/utils.py" in path for path in code_paths)
        assert any("package/submodule/nested/deep.py" in path for path in code_paths)

        # Should not have external requirements for this example
        assert len(result["requirements"]) == 0

    def test_circular_dependencies(self, tmp_path):
        """Test that circular dependencies don't cause infinite loops."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create circular dependency
        file_a = tmp_path / "module_a.py"
        file_a.write_text("""
from module_b import func_b

def func_a():
    return func_b() + "a"
""")

        file_b = tmp_path / "module_b.py"
        file_b.write_text("""
from module_a import func_a

def func_b():
    return "b"
""")

        # Should not hang or crash
        result = analyzer.analyze_dependencies([str(file_a)])

        # Should include both files
        code_paths = result["code_paths"]
        assert any("module_a.py" in path for path in code_paths)
        assert any("module_b.py" in path for path in code_paths)

    def test_missing_dependencies_handling(self, tmp_path):
        """Test handling of missing/uninstallable dependencies."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Model with missing external dependency
        model_file = tmp_path / "model.py"
        model_file.write_text("""
import nonexistent_package
from missing_local_module import something

def train():
    return nonexistent_package.do_something()
""")

        # Should not crash
        result = analyzer.analyze_dependencies([str(model_file)])

        # Missing packages should be treated as external
        assert "nonexistent_package" in result["requirements"]
        assert "missing_local_module" in result["requirements"]

    def test_module_classification(self, tmp_path):
        """Test accurate classification of different module types."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Test each classification method
        # Stdlib module
        os_info = analyzer._classify_and_resolve_module("os")
        assert os_info.dep_type == DependencyType.STDLIB_MODULE

        # Create a local module to test
        local_file = tmp_path / "local_mod.py"
        local_file.write_text("def func(): pass")

        local_info = analyzer._classify_and_resolve_module("local_mod")
        if local_info:  # May be None if import fails
            assert local_info.dep_type == DependencyType.LOCAL_FILE
            assert local_info.file_path is not None

    def test_multiple_entry_files(self, tmp_path):
        """Test analysis with multiple entry files."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create shared dependency
        shared_file = tmp_path / "shared.py"
        shared_file.write_text("""
import requests

def shared_func():
    return requests.get("http://example.com")
""")

        # First entry file
        entry1 = tmp_path / "entry1.py"
        entry1.write_text("""
from shared import shared_func
import pandas as pd

def process1():
    return pd.DataFrame()
""")

        # Second entry file
        entry2 = tmp_path / "entry2.py"
        entry2.write_text("""
from shared import shared_func
import numpy as np

def process2():
    return np.array([1, 2, 3])
""")

        result = analyzer.analyze_dependencies([str(entry1), str(entry2)])

        # Should include packages from both entry files
        requirements = result["requirements"]
        assert "requests" in requirements  # from shared
        assert "pandas" in requirements  # from entry1
        assert "numpy" in requirements  # from entry2

        # Should include all local files (no duplicates)
        code_paths = result["code_paths"]
        path_names = {Path(path).name for path in code_paths}
        assert "entry1.py" in path_names
        assert "entry2.py" in path_names
        assert "shared.py" in path_names

        # Verify no duplicate paths
        assert len(code_paths) == len(set(code_paths))


def test_convenience_functions(tmp_path):
    """Test convenience functions for backward compatibility."""
    # Create simple model
    model_file = tmp_path / "simple_model.py"
    model_file.write_text("""
import pandas as pd
import os

def train():
    return pd.DataFrame()
""")

    # Test analyze_model_dependencies
    result = analyze_model_dependencies(str(model_file), str(tmp_path))
    assert "requirements" in result
    assert "code_paths" in result
    assert "pandas" in result["requirements"]
    assert "os" not in result["requirements"]

    # Test get_model_requirements
    requirements = get_model_requirements(str(model_file), str(tmp_path))
    assert isinstance(requirements, list)
    assert "pandas" in requirements

    # Test get_model_code_paths
    code_paths = get_model_code_paths(str(model_file), str(tmp_path))
    assert isinstance(code_paths, list)
    assert any("simple_model.py" in path for path in code_paths)


def test_complex_real_world_scenario(tmp_path):
    """Test a complex scenario that mimics real-world usage."""
    analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

    # Create a realistic ML project structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    ml_pkg = src_dir / "ml_project"
    ml_pkg.mkdir()
    (ml_pkg / "__init__.py").touch()

    # Main model file
    model_file = ml_pkg / "model.py"
    model_file.write_text("""
import pandas as pd
import numpy as np
import sklearn.ensemble as ensemble
from ml_project.preprocessing import DataProcessor
from ml_project.utils.metrics import calculate_accuracy
import logging
import json

class MLModel:
    def __init__(self):
        self.processor = DataProcessor()
        self.model = ensemble.RandomForestClassifier()
        self.logger = logging.getLogger(__name__)

    def train(self, data_path):
        data = pd.read_csv(data_path)
        processed = self.processor.process(data)
        self.model.fit(processed.values, [1, 0, 1])
        accuracy = calculate_accuracy(self.model, processed)
        self.logger.info(f"Model accuracy: {accuracy}")
        return json.dumps({"accuracy": accuracy})
""")

    # Preprocessing module
    preprocessing_file = ml_pkg / "preprocessing.py"
    preprocessing_file.write_text("""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.created_at = datetime.datetime.now()

    def process(self, data):
        return pd.DataFrame(self.scaler.fit_transform(data))
""")

    # Utils package
    utils_dir = ml_pkg / "utils"
    utils_dir.mkdir()
    (utils_dir / "__init__.py").touch()

    metrics_file = utils_dir / "metrics.py"
    metrics_file.write_text("""
import numpy as np
from sklearn.metrics import accuracy_score
import math

def calculate_accuracy(model, data):
    predictions = model.predict(data)
    true_labels = [1] * len(predictions)
    return accuracy_score(true_labels, predictions)
""")

    result = analyzer.analyze_dependencies([str(model_file)])

    # Check external packages are correctly identified
    requirements = result["requirements"]
    expected_packages = {"pandas", "numpy", "sklearn"}
    assert expected_packages.issubset(set(requirements))

    # Check stdlib modules are NOT in requirements
    stdlib_modules = {"logging", "json", "datetime", "math"}
    assert stdlib_modules.isdisjoint(set(requirements))

    # Check all local files are included
    code_paths = result["code_paths"]
    expected_files = {"src/ml_project/model.py", "src/ml_project/preprocessing.py", "src/ml_project/utils/metrics.py"}

    for expected_file in expected_files:
        assert any(expected_file in path for path in code_paths), f"Missing {expected_file}"

    # Check analysis metadata
    analysis = result["analysis"]
    assert analysis["external_packages"] >= 3  # pandas, numpy, sklearn
    assert analysis["local_files"] >= 3  # model, preprocessing, metrics
    assert analysis["total_modules"] > analysis["external_packages"] + analysis["local_files"]  # includes stdlib


def test_edge_cases(tmp_path):
    """Test various edge cases and error conditions."""
    analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

    # Test with nonexistent entry file
    result = analyzer.analyze_dependencies(["/nonexistent/file.py"])
    assert result["requirements"] == []
    assert result["code_paths"] == []

    # Test with file that has syntax errors
    broken_file = tmp_path / "broken.py"
    broken_file.write_text("""
import pandas
def broken_function(
    # Missing closing parenthesis and colon
""")

    result = analyzer.analyze_dependencies([str(broken_file)])
    # Should handle gracefully and not crash
    assert isinstance(result["requirements"], list)
    assert isinstance(result["code_paths"], list)

    # Test with empty file
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")

    result = analyzer.analyze_dependencies([str(empty_file)])
    # Should only include the empty file itself
    assert len(result["requirements"]) == 0
    assert any("empty.py" in path for path in result["code_paths"])

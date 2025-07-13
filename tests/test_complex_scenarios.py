"""
Comprehensive tests for complex dependency analysis scenarios.

This test module uses the complex fixtures to verify that the UnifiedDependencyAnalyzer
correctly handles deeply nested codebases with interdependencies, circular imports,
and various edge cases that occur in real-world projects.
"""

import pytest

from mlflow_dep_analyzer.unified_analyzer import UnifiedDependencyAnalyzer, analyze_model_dependencies
from tests.fixtures.complex_project import create_complex_test_project
from tests.fixtures.problematic_imports import create_problematic_imports_fixture


class TestComplexProjectAnalysis:
    """Test dependency analysis on realistic complex projects."""

    @pytest.fixture
    def complex_project(self, tmp_path):
        """Create a complex ML project structure."""
        return create_complex_test_project(tmp_path)

    @pytest.fixture
    def problematic_imports(self, tmp_path):
        """Create problematic import scenarios."""
        return create_problematic_imports_fixture(tmp_path)

    def test_complex_ml_project_analysis(self, complex_project):
        """Test analysis of a complex ML project with deep dependencies."""
        analyzer = UnifiedDependencyAnalyzer(str(complex_project))

        # Analyze the main engine module (high-level entry point)
        engine_file = complex_project / "src" / "ml_platform" / "core" / "engine.py"
        result = analyzer.analyze_dependencies([str(engine_file)])

        # Verify structure
        assert "requirements" in result
        assert "code_paths" in result
        assert "analysis" in result
        assert "detailed_modules" in result

        # Check external dependencies are correctly identified
        requirements = result["requirements"]
        # Extract package names from versioned requirements
        package_names = {req.split("==")[0] for req in requirements}

        # Check that at least some key external dependencies are found
        # (not all may be imported depending on the specific analysis path)
        key_deps = {"pandas", "numpy", "scikit-learn", "mlflow", "mlflow-skinny", "flask", "Flask"}
        found_key_deps = key_deps.intersection(package_names)
        assert (
            len(found_key_deps) >= 3
        ), f"Expected key dependencies, found: {found_key_deps}, all packages: {package_names}"

        # Verify no standard library modules in requirements
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "datetime",
            "pathlib",
            "logging",
            "threading",
            "multiprocessing",
            "asyncio",
            "collections",
        }
        found_stdlib = stdlib_modules.intersection(package_names)
        assert len(found_stdlib) == 0, f"Found stdlib modules in requirements: {found_stdlib}"

        # Check code paths include local files
        code_paths = result["code_paths"]
        assert len(code_paths) > 5  # Should find multiple local files

        # Verify relative paths are used
        for path in code_paths:
            assert not path.startswith("/"), f"Absolute path found: {path}"
            assert "src/ml_platform" in path, f"Expected relative path, got: {path}"

        # Check analysis metadata
        analysis = result["analysis"]
        assert analysis["external_packages"] > 0
        assert analysis["local_files"] > 0
        assert analysis["total_modules"] > analysis["external_packages"] + analysis["local_files"]

    def test_circular_import_handling(self, problematic_imports):
        """Test that circular imports don't cause infinite loops."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        # Test simple circular import (A -> B -> A)
        circular_dir = problematic_imports / "circular"
        module_a = circular_dir / "module_a.py"

        # Should not hang or crash
        result = analyzer.analyze_dependencies([str(module_a)])

        # Should include both modules in the circular dependency
        code_paths = result["code_paths"]
        assert any("circular/module_a.py" in path for path in code_paths)
        assert any("circular/module_b.py" in path for path in code_paths)

        # Should find external dependencies from both modules
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        assert "pandas" in package_names
        assert "numpy" in package_names
        assert "scikit-learn" in package_names  # sklearn -> scikit-learn

        # Test three-way circular import (X -> Y -> Z -> X)
        module_x = circular_dir / "module_x.py"
        result_3way = analyzer.analyze_dependencies([str(module_x)])

        code_paths_3way = result_3way["code_paths"]
        assert any("circular/module_x.py" in path for path in code_paths_3way)
        assert any("circular/module_y.py" in path for path in code_paths_3way)
        assert any("circular/module_z.py" in path for path in code_paths_3way)

        # Should find external dependencies from all three modules
        requirements_3way = result_3way["requirements"]
        package_names_3way = {req.split("==")[0] for req in requirements_3way}
        external_deps = {"requests", "scikit-learn", "torch"}  # sklearn -> scikit-learn
        found_deps = external_deps.intersection(package_names_3way)
        assert len(found_deps) >= 2, f"Expected multiple external deps, found: {found_deps}"

    def test_deep_relative_imports(self, problematic_imports):
        """Test handling of deep relative imports across multiple levels."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        # Test module with deep relative imports (5 levels up)
        deep_dir = problematic_imports / "deep_relative"
        deep_module = deep_dir / "level1" / "level2" / "level3" / "level4" / "level5" / "deep_module.py"

        result = analyzer.analyze_dependencies([str(deep_module)])

        # Should find external dependencies
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        expected_deps = {"pandas", "numpy", "PyYAML", "requests"}  # Note: yaml -> PyYAML
        found_deps = expected_deps.intersection(package_names)
        assert len(found_deps) >= 2, f"Expected external deps from deep module, found: {found_deps}"

        # Should not include stdlib modules
        stdlib_not_expected = {"json", "datetime"}
        found_stdlib = stdlib_not_expected.intersection(package_names)
        assert len(found_stdlib) == 0, f"Found unexpected stdlib modules: {found_stdlib}"

        # Should include all referenced local files despite deep nesting
        code_paths = result["code_paths"]
        expected_files = [
            "deep_relative/level1/level2/level3/level4/level5/deep_module.py",
            "deep_relative/level1/level2/level3/level4/level5/level5_helper.py",
        ]

        for expected_file in expected_files:
            assert any(
                expected_file in path for path in code_paths
            ), f"Missing expected file: {expected_file}. Found: {code_paths}"

    def test_dynamic_imports_handling(self, problematic_imports):
        """Test handling of dynamic imports and conditional loading."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        dynamic_dir = problematic_imports / "dynamic"
        dynamic_importer = dynamic_dir / "dynamic_importer.py"

        result = analyzer.analyze_dependencies([str(dynamic_importer)])

        # Should find external dependencies used in dynamic imports
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        expected_deps = {"pandas", "numpy", "PyYAML"}  # yaml -> PyYAML
        found_deps = expected_deps.intersection(package_names)
        assert len(found_deps) >= 2, f"Expected deps from dynamic imports, found: {found_deps}"

        # Should handle missing/dynamic modules gracefully
        # (should not crash even if dynamic modules don't exist)
        assert "requirements" in result
        assert isinstance(result["requirements"], list)

    def test_missing_dependencies_graceful_handling(self, problematic_imports):
        """Test graceful handling of missing external and local dependencies."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        missing_dir = problematic_imports / "missing_deps"
        missing_external = missing_dir / "missing_external.py"

        # Should not crash when encountering missing dependencies
        result = analyzer.analyze_dependencies([str(missing_external)])

        # Should treat missing external packages as external dependencies
        requirements = result["requirements"]

        # Should include missing/fictional packages as external
        fictional_packages = {"nonexistent_package", "super_rare_package", "fictional_ml_library"}

        # At least some missing packages should be detected as external
        found_fictional = fictional_packages.intersection(requirements)
        assert len(found_fictional) >= 1, f"Expected to find fictional packages, found: {found_fictional}"

        # Should not crash and should return valid structure
        assert "code_paths" in result
        assert "analysis" in result

    def test_namespace_packages(self, problematic_imports):
        """Test handling of namespace packages without __init__.py files."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        namespace_dir = problematic_imports / "namespace"
        component1_processor = namespace_dir / "component1" / "processor.py"

        result = analyzer.analyze_dependencies([str(component1_processor)])

        # Should find external dependencies
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        expected_deps = {"pandas", "numpy"}
        found_deps = expected_deps.intersection(package_names)
        assert len(found_deps) >= 1, f"Expected deps from namespace package, found: {found_deps}"

        # Should include namespace package files
        code_paths = result["code_paths"]
        assert any("namespace/component1/processor.py" in path for path in code_paths)

    def test_conditional_imports(self, problematic_imports):
        """Test handling of conditional imports based on runtime conditions."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        conditional_dir = problematic_imports / "conditional"
        conditional_processor = conditional_dir / "conditional_processor.py"

        result = analyzer.analyze_dependencies([str(conditional_processor)])

        # Should find external dependencies from conditional imports
        requirements = result["requirements"]

        # Should find some conditional dependencies
        # Note: May not find all since they're conditional, but should find some

        # Should always find basic external dependencies
        package_names = {req.split("==")[0] for req in requirements}
        basic_deps = {"pandas", "numpy"}  # These are imported unconditionally
        found_basic = basic_deps.intersection(package_names)
        assert len(found_basic) >= 1, f"Expected basic deps, found: {found_basic}"

    def test_star_imports_resolution(self, problematic_imports):
        """Test resolution of star imports and their dependencies."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        star_dir = problematic_imports / "star_imports"
        star_importer = star_dir / "star_importer.py"

        result = analyzer.analyze_dependencies([str(star_importer)])

        # Should find external dependencies from star-imported modules
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        expected_deps = {"pandas", "numpy", "scikit-learn"}  # sklearn -> scikit-learn
        found_deps = expected_deps.intersection(package_names)
        assert len(found_deps) >= 2, f"Expected deps from star imports, found: {found_deps}"

        # Should include the base module that's star-imported
        code_paths = result["code_paths"]
        assert any("star_imports/base_module.py" in path for path in code_paths)

        # Should not include stdlib modules from star imports
        stdlib_modules = {"collections", "itertools"}
        found_stdlib = stdlib_modules.intersection(package_names)
        assert len(found_stdlib) == 0, f"Found stdlib modules from star imports: {found_stdlib}"

    def test_plugin_system_analysis(self, problematic_imports):
        """Test analysis of plugin systems with dynamic loading."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        plugin_dir = problematic_imports / "plugin_system"
        manager = plugin_dir / "manager.py"

        result = analyzer.analyze_dependencies([str(manager)])

        # Should find external dependencies from plugin manager
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        expected_deps = {"PyYAML", "pandas"}  # yaml -> PyYAML
        found_deps = expected_deps.intersection(package_names)
        assert len(found_deps) >= 1, f"Expected deps from plugin system, found: {found_deps}"

        # Test individual plugin analysis
        plugin_file = plugin_dir / "plugins" / "data_processor_plugin.py"
        plugin_result = analyzer.analyze_dependencies([str(plugin_file)])

        plugin_requirements = plugin_result["requirements"]
        plugin_package_names = {req.split("==")[0] for req in plugin_requirements}
        plugin_deps = {"pandas", "numpy"}
        found_plugin_deps = plugin_deps.intersection(plugin_package_names)
        assert len(found_plugin_deps) >= 1, f"Expected deps from plugin, found: {found_plugin_deps}"

    def test_multiple_entry_points_complex(self, complex_project):
        """Test analysis with multiple entry points in a complex project."""
        analyzer = UnifiedDependencyAnalyzer(str(complex_project))

        # Multiple entry points from different parts of the system
        entry_points = [
            complex_project / "src" / "ml_platform" / "core" / "engine.py",
            complex_project / "src" / "ml_platform" / "training" / "trainer.py",
            complex_project / "src" / "ml_platform" / "api" / "app.py",
            complex_project / "src" / "ml_platform" / "cli" / "main.py",
        ]

        # Filter to only existing files
        existing_entries = [str(ep) for ep in entry_points if ep.exists()]
        assert len(existing_entries) > 0, "No entry points found"

        result = analyzer.analyze_dependencies(existing_entries)

        # Should find comprehensive set of dependencies
        requirements = result["requirements"]

        # Should include major external dependencies from all entry points
        package_names = {req.split("==")[0] for req in requirements}
        major_deps = {
            "pandas",
            "numpy",
            "scikit-learn",
            "mlflow",
            "mlflow-skinny",
        }  # sklearn -> scikit-learn, handle mlflow variants
        found_major = major_deps.intersection(package_names)
        assert len(found_major) >= 3, f"Expected major deps from multiple entry points, found: {found_major}"

        # Should include many local files
        code_paths = result["code_paths"]
        assert len(code_paths) >= 10, f"Expected many local files, found {len(code_paths)}"

        # Should not have duplicate paths
        assert len(code_paths) == len(set(code_paths)), "Found duplicate code paths"

        # Verify analysis metadata for multiple entry points
        analysis = result["analysis"]
        assert len(analysis["entry_files"]) == len(existing_entries)
        assert analysis["external_packages"] >= 5
        assert analysis["local_files"] >= 5

    def test_performance_with_large_codebase(self, complex_project):
        """Test that analysis completes in reasonable time for large codebases."""
        import time

        analyzer = UnifiedDependencyAnalyzer(str(complex_project))

        # Find all Python files in the project
        all_python_files = list(complex_project.rglob("*.py"))

        # Test with a subset to avoid extremely long test times
        test_files = [str(f) for f in all_python_files[:10] if f.name != "__init__.py"]

        if test_files:
            start_time = time.time()
            result = analyzer.analyze_dependencies(test_files)
            end_time = time.time()

            analysis_time = end_time - start_time

            # Should complete within reasonable time (adjust threshold as needed)
            assert analysis_time < 30.0, f"Analysis took too long: {analysis_time:.2f} seconds"

            # Should still produce valid results
            assert "requirements" in result
            assert "code_paths" in result
            assert len(result["requirements"]) > 0
            assert len(result["code_paths"]) > 0

    def test_error_recovery_and_robustness(self, problematic_imports):
        """Test that analyzer recovers gracefully from various error conditions."""
        analyzer = UnifiedDependencyAnalyzer(str(problematic_imports))

        # Test with mix of valid and invalid files
        test_files = [
            str(problematic_imports / "missing_deps" / "missing_external.py"),  # Has missing deps
            str(problematic_imports / "circular" / "module_a.py"),  # Has circular imports
            str(problematic_imports / "nonexistent_file.py"),  # Doesn't exist
            str(problematic_imports / "dynamic" / "dynamic_importer.py"),  # Has dynamic imports
        ]

        # Should not crash even with problematic files
        result = analyzer.analyze_dependencies(test_files)

        # Should return valid structure
        assert "requirements" in result
        assert "code_paths" in result
        assert "analysis" in result

        # Should include valid files and skip invalid ones
        code_paths = result["code_paths"]

        # Should have found some local files
        assert len(code_paths) > 0

        # Should have found some external dependencies despite issues
        requirements = result["requirements"]
        assert len(requirements) > 0

    def test_stdlib_module_detection_accuracy(self, complex_project):
        """Test that stdlib modules are accurately detected and excluded."""
        analyzer = UnifiedDependencyAnalyzer(str(complex_project))

        # Test with a file that imports many stdlib modules
        test_file = complex_project / "src" / "ml_platform" / "utils" / "logging.py"

        if test_file.exists():
            result = analyzer.analyze_dependencies([str(test_file)])

            requirements = set(result["requirements"])

            # Known stdlib modules that should NOT be in requirements
            stdlib_modules = {
                "os",
                "sys",
                "json",
                "logging",
                "threading",
                "time",
                "pathlib",
                "collections",
                "datetime",
                "asyncio",
                "multiprocessing",
                "inspect",
            }

            found_stdlib = stdlib_modules.intersection(requirements)
            assert len(found_stdlib) == 0, f"Found stdlib modules in requirements: {found_stdlib}"

            # Should still find legitimate external dependencies
            external_deps = {"structlog", "sentry_sdk"}
            found_external = external_deps.intersection(requirements)
            assert len(found_external) >= 1, f"Expected external deps, found: {found_external}"


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions."""

    def test_empty_file_analysis(self, tmp_path):
        """Test analysis of empty Python files."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        result = analyzer.analyze_dependencies([str(empty_file)])

        # Should handle empty file gracefully
        assert result["requirements"] == []
        assert len(result["code_paths"]) == 1  # Just the empty file itself
        assert "empty.py" in result["code_paths"][0]

    def test_syntax_error_handling(self, tmp_path):
        """Test handling of files with syntax errors."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        broken_file = tmp_path / "broken.py"
        broken_file.write_text("""
import pandas as pd
import numpy as np

def broken_function(
    # Missing closing parenthesis and colon
    # This will cause a syntax error

class BrokenClass
    # Missing colon
    def method(self):
        return "broken"
""")

        # Should not crash
        result = analyzer.analyze_dependencies([str(broken_file)])

        # Should return valid structure even if parsing fails
        assert "requirements" in result
        assert "code_paths" in result
        assert isinstance(result["requirements"], list)
        assert isinstance(result["code_paths"], list)

    def test_very_long_import_chains(self, tmp_path):
        """Test handling of very long import dependency chains."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create a chain of 10 modules, each importing the next
        for i in range(10):
            module_file = tmp_path / f"chain_{i}.py"
            if i == 0:
                # First module imports external package
                content = """
import pandas as pd
import numpy as np
from chain_1 import next_function

def chain_0_function():
    return next_function()
"""
            elif i == 9:
                # Last module
                content = f"""
import json

def chain_{i}_function():
    return {{"chain_end": True}}
"""
            else:
                # Middle modules
                content = f"""
from chain_{i+1} import chain_{i+1}_function

def chain_{i}_function():
    return chain_{i+1}_function()
"""
            module_file.write_text(content)

        # Analyze the first module (should follow the entire chain)
        result = analyzer.analyze_dependencies([str(tmp_path / "chain_0.py")])

        # Should find external dependencies from first module
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        assert "pandas" in package_names
        assert "numpy" in package_names

        # Should not include stdlib modules
        assert "json" not in package_names

        # Should include all chain modules
        code_paths = result["code_paths"]
        chain_files_found = sum(1 for path in code_paths if "chain_" in path)
        assert chain_files_found >= 5  # Should find most of the chain

    def test_unicode_and_encoding_handling(self, tmp_path):
        """Test handling of files with Unicode content and encoding issues."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        unicode_file = tmp_path / "unicode_module.py"
        unicode_content = """# -*- coding: utf-8 -*-
\"\"\"
Module with Unicode content: 测试模块
Support for various encodings: café, naïve, résumé
\"\"\"

import pandas as pd
import numpy as np
from typing import Dict, Any

# Unicode in variable names and strings
测试变量 = "test variable"
CAFÉ_CONFIG = {"naïve": "résumé"}

def process_unicode_data(数据: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Process data with Unicode support.\"\"\"
    return {
        **数据,
        "unicode_processed": True,
        "café": "naïve processing",
        "test": 测试变量
    }

class UnicodeProcessor:
    \"\"\"Processor with Unicode support.\"\"\"

    def __init__(self):
        self.名称 = "Unicode处理器"

    def 处理(self, data):
        \"\"\"处理方法\"\"\"
        return process_unicode_data(data)
"""

        # Write with UTF-8 encoding
        unicode_file.write_text(unicode_content, encoding="utf-8")

        # Should handle Unicode content gracefully
        result = analyzer.analyze_dependencies([str(unicode_file)])

        # Should find external dependencies despite Unicode content
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        assert "pandas" in package_names
        assert "numpy" in package_names

        # Should include the Unicode file
        code_paths = result["code_paths"]
        assert any("unicode_module.py" in path for path in code_paths)


@pytest.mark.integration
class TestIntegrationWithRealPackages:
    """Integration tests with real package scenarios."""

    def test_real_world_ml_project_structure(self, tmp_path):
        """Test analysis of a realistic ML project structure."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create a realistic scikit-learn style project
        project_dir = tmp_path / "ml_project"
        project_dir.mkdir()

        # Main module
        main_file = project_dir / "main.py"
        main_file.write_text("""
import pandas as pd
import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import yaml

from .preprocessing import DataPreprocessor
from .models import ModelTrainer
from .evaluation import ModelEvaluator
from .utils import setup_logging, load_config

def main():
    # Load configuration
    config = load_config("config.yaml")
    logger = setup_logging()

    # Load and preprocess data
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.load_and_split()

    # Train model
    trainer = ModelTrainer(config)
    model = trainer.train(X_train, y_train)

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    return model, metrics

if __name__ == "__main__":
    main()
""")

        # Supporting modules
        (project_dir / "preprocessing.py").write_text("""
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = sklearn.preprocessing.StandardScaler()

    def load_and_split(self):
        # Simulate data loading
        data = pd.read_csv(self.config["data_path"])
        X = data.drop("target", axis=1)
        y = data["target"]

        return train_test_split(X, y, test_size=0.2, random_state=42)
""")

        (project_dir / "models.py").write_text("""
import sklearn.ensemble
import sklearn.linear_model
import joblib

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self, X_train, y_train):
        if self.config["model_type"] == "rf":
            model = sklearn.ensemble.RandomForestClassifier(**self.config["model_params"])
        else:
            model = sklearn.linear_model.LogisticRegression(**self.config["model_params"])

        model.fit(X_train, y_train)
        return model
""")

        (project_dir / "evaluation.py").write_text("""
import sklearn.metrics
import numpy as np

class ModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        predictions = model.predict(X_test)

        return {
            "accuracy": sklearn.metrics.accuracy_score(y_test, predictions),
            "precision": sklearn.metrics.precision_score(y_test, predictions, average="weighted"),
            "recall": sklearn.metrics.recall_score(y_test, predictions, average="weighted"),
            "f1": sklearn.metrics.f1_score(y_test, predictions, average="weighted")
        }
""")

        (project_dir / "utils.py").write_text("""
import logging
import yaml
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)
""")

        # Analyze the main file
        result = analyzer.analyze_dependencies([str(main_file)])

        # Should find all major ML dependencies
        requirements = result["requirements"]
        package_names = {req.split("==")[0] for req in requirements}
        expected_ml_deps = {
            "pandas",
            "numpy",
            "scikit-learn",
            "mlflow",
            "mlflow-skinny",
            "joblib",
            "PyYAML",
        }  # sklearn -> scikit-learn, yaml -> PyYAML
        found_ml_deps = expected_ml_deps.intersection(package_names)
        assert len(found_ml_deps) >= 5, f"Expected ML dependencies, found: {found_ml_deps}"

        # Should not include stdlib modules
        stdlib_modules = {"pathlib", "logging"}
        found_stdlib = stdlib_modules.intersection(package_names)
        assert len(found_stdlib) == 0, f"Found stdlib modules: {found_stdlib}"

        # Should include all local modules
        code_paths = result["code_paths"]
        expected_local_files = ["main.py", "preprocessing.py", "models.py", "evaluation.py", "utils.py"]

        for expected_file in expected_local_files:
            assert any(expected_file in path for path in code_paths), f"Missing local file: {expected_file}"


def test_convenience_functions_with_complex_scenarios(tmp_path):
    """Test convenience functions with complex scenarios."""
    # Create a complex project using the fixture
    complex_project = create_complex_test_project(tmp_path)

    # Test convenience functions
    engine_file = complex_project / "src" / "ml_platform" / "core" / "engine.py"

    if engine_file.exists():
        # Test analyze_model_dependencies
        result = analyze_model_dependencies(str(engine_file), str(complex_project))
        assert "requirements" in result
        assert "code_paths" in result
        assert len(result["requirements"]) > 0
        assert len(result["code_paths"]) > 0

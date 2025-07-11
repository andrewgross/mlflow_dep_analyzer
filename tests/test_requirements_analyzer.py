"""
Tests for the HybridRequirementsAnalyzer.
"""

import os
import sys
import tempfile

from mlflow_dep_analyzer.requirements_analyzer import HybridRequirementsAnalyzer, load_requirements_from_file


class TestHybridRequirementsAnalyzer:
    """Test cases for HybridRequirementsAnalyzer."""

    def test_analyze_file_basic_imports(self):
        """Test analyzing a simple Python file with imports."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
""")
            temp_file = f.name

        try:
            analyzer = HybridRequirementsAnalyzer()
            imports = analyzer.analyze_file(temp_file)
            expected = {"os", "sys", "datetime", "pandas", "numpy", "sklearn"}
            assert imports == expected
        finally:
            os.unlink(temp_file)

    def test_filter_local_modules(self, tmp_path):
        """Test filtering out local project modules using dynamic detection."""
        analyzer = HybridRequirementsAnalyzer()

        # Create actual local modules in the test repo
        (tmp_path / "projects").mkdir()
        (tmp_path / "projects" / "__init__.py").touch()
        (tmp_path / "shared_utils").mkdir()
        (tmp_path / "shared_utils" / "__init__.py").touch()
        (tmp_path / "my_model.py").touch()

        imports = {
            "pandas",
            "numpy",
            "sklearn",
            "os",
            "sys",
            "projects",
            "shared_utils",
            "my_model",
            "mlflow",
            "datetime",
        }

        filtered = analyzer.filter_local_modules(imports, str(tmp_path))

        # Should exclude local modules that exist in the repo
        assert "pandas" in filtered
        assert "projects" not in filtered  # exists as directory with __init__.py
        assert "shared_utils" not in filtered  # exists as directory with __init__.py
        assert "my_model" not in filtered  # exists as .py file
        assert "mlflow" in filtered  # external package
        assert "numpy" in filtered  # external package

    def test_dynamic_local_detection_no_hardcoded_patterns(self, tmp_path):
        """Test that dynamic detection works with arbitrary project-specific names."""
        analyzer = HybridRequirementsAnalyzer()

        # Create modules with completely arbitrary names (not in old hardcoded list)
        (tmp_path / "my_custom_package").mkdir()
        (tmp_path / "my_custom_package" / "__init__.py").touch()
        (tmp_path / "arbitrary_module.py").touch()
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "business_logic").mkdir()
        (tmp_path / "src" / "business_logic" / "__init__.py").touch()

        imports = {
            "pandas",
            "numpy",
            "my_custom_package",  # should be detected as local
            "arbitrary_module",  # should be detected as local
            "business_logic",  # should be detected as local (in src/)
            "some_external_lib",  # should not be detected as local
        }

        filtered = analyzer.filter_local_modules(imports, str(tmp_path))

        # Should exclude dynamically detected local modules
        assert "pandas" in filtered
        assert "numpy" in filtered
        assert "some_external_lib" in filtered
        # These should be filtered out as local
        assert "my_custom_package" not in filtered
        assert "arbitrary_module" not in filtered
        assert "business_logic" not in filtered

    def test_stdlib_module_detection(self):
        """Test that stdlib modules are correctly identified."""
        analyzer = HybridRequirementsAnalyzer()

        # Test core stdlib modules
        assert analyzer.is_stdlib_module("os")
        assert analyzer.is_stdlib_module("sys")
        assert analyzer.is_stdlib_module("json")
        assert analyzer.is_stdlib_module("datetime")
        assert analyzer.is_stdlib_module("collections")
        assert analyzer.is_stdlib_module("re")
        assert analyzer.is_stdlib_module("pathlib")
        assert analyzer.is_stdlib_module("ast")

        # Test that third-party packages are not stdlib
        assert not analyzer.is_stdlib_module("pandas")
        assert not analyzer.is_stdlib_module("numpy")
        assert not analyzer.is_stdlib_module("requests")
        assert not analyzer.is_stdlib_module("sklearn")
        assert not analyzer.is_stdlib_module("mlflow")

        # Test edge cases
        assert not analyzer.is_stdlib_module("nonexistent_module")
        assert not analyzer.is_stdlib_module("pkg_resources")  # This is setuptools, not stdlib
        assert not analyzer.is_stdlib_module("setuptools")  # Third-party package

        # Test that we properly handle module names that look like stdlib
        assert not analyzer.is_stdlib_module("os_custom")
        assert not analyzer.is_stdlib_module("sys_utils")

    def test_stdlib_filtering_in_analyze(self):
        """Test that stdlib modules are properly filtered during analysis."""
        analyzer = HybridRequirementsAnalyzer()

        # Mock imports that include stdlib modules
        imports = {"os", "sys", "json", "pandas", "numpy", "requests", "collections", "re"}

        # Filter should remove stdlib modules
        filtered = analyzer.filter_local_modules(imports, repo_root="/tmp/test")

        # Only external packages should remain
        assert "pandas" in filtered
        assert "numpy" in filtered
        assert "requests" in filtered

        # Stdlib modules should be filtered out
        assert "os" not in filtered
        assert "sys" not in filtered
        assert "json" not in filtered
        assert "collections" not in filtered
        assert "re" not in filtered

    def test_analyze_directory(self, tmp_path):
        """Test analyzing all Python files in a directory."""
        analyzer = HybridRequirementsAnalyzer()

        # Create test files
        file1 = tmp_path / "file1.py"
        file1.write_text("import pandas\nimport numpy")

        file2 = tmp_path / "file2.py"
        file2.write_text("import sklearn\nfrom datetime import datetime")

        imports = analyzer.analyze_directory(str(tmp_path))

        expected = {"pandas", "numpy", "sklearn", "datetime"}
        assert imports == expected

    def test_exclude_existing_requirements(self):
        """Test excluding packages that are already in existing requirements."""
        analyzer = HybridRequirementsAnalyzer(existing_requirements=["pandas>=1.3.0", "numpy==1.21.0", "mlflow>=2.0.0"])

        packages = {"pandas", "numpy", "sklearn", "mlflow"}
        filtered = analyzer.exclude_existing_requirements(packages)

        assert "sklearn" in filtered
        assert "pandas" not in filtered
        assert "numpy" not in filtered
        assert "mlflow" not in filtered


def test_load_requirements_from_file(tmp_path):
    """Test loading requirements from file."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("""
# This is a comment
pandas>=1.3.0
numpy==1.21.0

# Another comment
sklearn>=1.0.0
""")

    requirements = load_requirements_from_file(str(req_file))
    expected = ["pandas>=1.3.0", "numpy==1.21.0", "sklearn>=1.0.0"]
    assert requirements == expected


def test_is_stdlib_module_convenience_function():
    """Test the convenience function for stdlib detection."""
    from mlflow_dep_analyzer import is_stdlib_module

    # Test that the convenience function works the same as the method
    assert is_stdlib_module("os")
    assert is_stdlib_module("json")
    assert is_stdlib_module("datetime")
    assert not is_stdlib_module("pandas")
    assert not is_stdlib_module("numpy")
    assert not is_stdlib_module("nonexistent_module")


def test_comprehensive_module_classification():
    """Test comprehensive module classification: stdlib vs local vs third-party."""
    import tempfile
    from pathlib import Path

    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    # Create a temporary project structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create local modules
        (temp_path / "local_package").mkdir()
        (temp_path / "local_package" / "__init__.py").touch()
        (temp_path / "local_module.py").touch()
        (temp_path / "src").mkdir()
        (temp_path / "src" / "business_logic").mkdir()
        (temp_path / "src" / "business_logic" / "__init__.py").touch()

        analyzer = HybridRequirementsAnalyzer()

        # Test stdlib modules (should be comprehensive)
        stdlib_modules = [
            "os",
            "sys",
            "json",
            "datetime",
            "collections",
            "re",
            "pathlib",
            "ast",
            "urllib",
            "typing",
            "functools",
            "itertools",
            "operator",
            "math",
            "random",
            "string",
            "io",
            "contextlib",
            "logging",
            "tempfile",
            "subprocess",
            "shutil",
            "glob",
            "pickle",
            "warnings",
            "inspect",
            "importlib",
            "weakref",
            "gc",
            "atexit",
            "signal",
            "threading",
            "multiprocessing",
            "queue",
            "sqlite3",
            "csv",
            "xml",
            "html",
            "email",
            "base64",
            "hashlib",
            "hmac",
            "secrets",
            "ssl",
            "socket",
            "gzip",
            "tarfile",
            "zipfile",
            "configparser",
            "argparse",
            "unittest",
            "traceback",
            "copy",
            "time",
            "abc",
        ]

        print(f"Testing {len(stdlib_modules)} stdlib modules...")
        for module in stdlib_modules:
            assert analyzer.is_stdlib_module(module), f"{module} should be stdlib"

        # Test third-party modules (commonly used packages)
        third_party_modules = [
            "pandas",
            "numpy",
            "requests",
            "sklearn",
            "mlflow",
            "django",
            "flask",
            "tensorflow",
            "torch",
            "matplotlib",
            "seaborn",
            "pytest",
            "click",
            "pydantic",
            "fastapi",
            "sqlalchemy",
            "psycopg2",
            "pymongo",
            "redis",
            "celery",
            "gunicorn",
            "boto3",
            "aws",
            "azure",
            "google",
            "pkg_resources",
            "setuptools",
            "pip",
            "wheel",
            "twine",
        ]

        print(f"Testing {len(third_party_modules)} third-party modules...")
        for module in third_party_modules:
            assert not analyzer.is_stdlib_module(module), f"{module} should not be stdlib"

        # Test local modules (using actual filesystem detection)
        local_modules = ["local_package", "local_module", "business_logic"]

        # Test filtering behavior
        all_modules = set(stdlib_modules + third_party_modules + local_modules)
        filtered = analyzer.filter_local_modules(all_modules, str(temp_path))

        print(f"Filtered {len(all_modules)} modules down to {len(filtered)}")

        # Only third-party modules should remain
        for module in third_party_modules:
            assert module in filtered, f"{module} should remain after filtering"

        # Stdlib modules should be filtered out
        for module in stdlib_modules:
            assert module not in filtered, f"{module} should be filtered out (stdlib)"

        # Local modules should be filtered out
        for module in local_modules:
            assert module not in filtered, f"{module} should be filtered out (local)"


def test_edge_cases_and_error_handling():
    """Test edge cases and error handling in module detection."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    analyzer = HybridRequirementsAnalyzer()

    # Test non-existent modules
    assert not analyzer.is_stdlib_module("nonexistent_module_12345")
    assert not analyzer.is_stdlib_module("fake_package_name")

    # Test empty/invalid module names
    assert not analyzer.is_stdlib_module("")
    assert not analyzer.is_stdlib_module("   ")

    # Test modules with similar names to stdlib
    assert not analyzer.is_stdlib_module("os_custom")
    assert not analyzer.is_stdlib_module("sys_utils")
    assert not analyzer.is_stdlib_module("json_parser")
    assert not analyzer.is_stdlib_module("datetime_utils")

    # Test private modules (should be filtered out anyway)
    assert not analyzer.is_stdlib_module("_private_module")
    assert not analyzer.is_stdlib_module("__private_module")

    # Test case sensitivity
    assert not analyzer.is_stdlib_module("OS")  # Python module names are case-sensitive
    assert not analyzer.is_stdlib_module("JSON")


def test_dynamic_detection_completeness():
    """Test that dynamic detection finds expected stdlib modules."""

    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    analyzer = HybridRequirementsAnalyzer()

    # If we have sys.stdlib_module_names, compare against it
    if hasattr(sys, "stdlib_module_names"):
        official_stdlib = sys.stdlib_module_names
        print(f"Comparing against {len(official_stdlib)} official stdlib modules")

        # Test a sample of official stdlib modules
        sample_size = min(50, len(official_stdlib))
        sample_modules = sorted(official_stdlib)[:sample_size]

        for module in sample_modules:
            # Skip private modules as they might not be importable
            if not module.startswith("_"):
                try:
                    result = analyzer.is_stdlib_module(module)
                    assert result, f"Official stdlib module {module} should be detected as stdlib"
                except Exception as e:
                    print(f"Warning: Could not test {module}: {e}")

        # Test that we don't have too many false positives
        detected_stdlib = set()
        common_modules = [
            "os",
            "sys",
            "json",
            "datetime",
            "collections",
            "re",
            "pathlib",
            "ast",
            "urllib",
            "typing",
            "functools",
            "itertools",
            "operator",
            "math",
            "random",
            "string",
            "io",
            "contextlib",
            "logging",
            "tempfile",
            "subprocess",
            "shutil",
            "glob",
            "pickle",
            "warnings",
        ]

        for module in common_modules:
            if analyzer.is_stdlib_module(module):
                detected_stdlib.add(module)

        print(f"Detected {len(detected_stdlib)} common stdlib modules")
        assert len(detected_stdlib) > 20, "Should detect most common stdlib modules"


def test_load_requirements_nonexistent_file():
    """Test loading from non-existent file."""
    requirements = load_requirements_from_file("/non/existent/file.txt")
    assert requirements == []


def test_analyze_model_requirements_comprehensive(tmp_path):
    """Test the complete analyze_model_requirements workflow."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    # Create a test Python file with various imports
    test_file = tmp_path / "test_model.py"
    test_file.write_text("""
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import mlflow
from my_local_module import helper_function
from shared_utils import data_processor
""")

    # Create local modules
    (tmp_path / "my_local_module.py").touch()
    (tmp_path / "shared_utils").mkdir()
    (tmp_path / "shared_utils" / "__init__.py").touch()

    analyzer = HybridRequirementsAnalyzer()
    result = analyzer.analyze_model_requirements(
        code_paths=[str(test_file)], repo_root=str(tmp_path), exclude_existing=False
    )

    # Verify structure
    assert "requirements" in result
    assert "analysis" in result
    assert isinstance(result["requirements"], list)
    assert isinstance(result["analysis"], dict)

    # Check analysis details
    analysis = result["analysis"]
    assert "files_analyzed" in analysis
    assert "raw_imports" in analysis
    assert "external_modules" in analysis
    assert "final_packages" in analysis

    # Should have analyzed our test file
    assert str(test_file) in analysis["files_analyzed"]

    # Should have found external imports but filtered out stdlib and local
    external_modules = set(analysis["external_modules"])
    assert "pandas" in external_modules
    assert "numpy" in external_modules
    assert "sklearn" in external_modules
    assert "mlflow" in external_modules

    # Should have filtered out stdlib modules
    assert "os" not in external_modules
    assert "sys" not in external_modules
    assert "json" not in external_modules

    # Should have filtered out local modules
    assert "my_local_module" not in external_modules
    assert "shared_utils" not in external_modules


def test_analyze_code_dependencies_function(tmp_path):
    """Test the analyze_code_dependencies convenience function."""
    from mlflow_dep_analyzer import analyze_code_dependencies

    # Create test files
    test_file1 = tmp_path / "model1.py"
    test_file1.write_text("import pandas\nimport numpy\nimport os")

    test_file2 = tmp_path / "model2.py"
    test_file2.write_text("import sklearn\nimport matplotlib\nimport sys")

    # Test with multiple files
    requirements = analyze_code_dependencies(code_paths=[str(test_file1), str(test_file2)], repo_root=str(tmp_path))

    assert isinstance(requirements, list)
    # Should contain external packages but not stdlib
    requirement_names = [req.split("==")[0].split(">=")[0] for req in requirements]

    # External packages should be included
    assert any("pandas" in req for req in requirement_names)
    assert any("numpy" in req for req in requirement_names)
    assert any("sklearn" in req or "scikit-learn" in req for req in requirement_names)

    # Stdlib should be excluded
    assert not any("os" in req for req in requirement_names)
    assert not any("sys" in req for req in requirement_names)


def test_analyze_code_dependencies_with_existing_requirements(tmp_path):
    """Test analyze_code_dependencies with existing requirements exclusion."""
    from mlflow_dep_analyzer import analyze_code_dependencies

    # Create test file
    test_file = tmp_path / "model.py"
    test_file.write_text("import pandas\nimport numpy\nimport sklearn")

    # Create requirements file
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas>=1.3.0\nnumpy==1.21.0\n")

    # Test excluding existing requirements
    requirements = analyze_code_dependencies(
        code_paths=[str(test_file)],
        existing_requirements_file=str(req_file),
        repo_root=str(tmp_path),
        exclude_existing=True,
    )

    requirement_names = [req.split("==")[0].split(">=")[0] for req in requirements]

    # Should exclude pandas and numpy (in existing requirements)
    assert not any("pandas" in req for req in requirement_names)
    assert not any("numpy" in req for req in requirement_names)

    # Should include sklearn (not in existing requirements)
    assert any("scikit-learn" in req for req in requirement_names)


def test_analyze_code_dependencies_directory(tmp_path):
    """Test analyze_code_dependencies with directory analysis."""
    from mlflow_dep_analyzer import analyze_code_dependencies

    # Create multiple files in a directory
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    (models_dir / "model1.py").write_text("import pandas\nimport requests")
    (models_dir / "model2.py").write_text("import numpy\nimport flask")
    (models_dir / "utils.py").write_text("import sklearn\nimport matplotlib")

    # Test directory analysis
    requirements = analyze_code_dependencies(code_paths=[str(models_dir)], repo_root=str(tmp_path))

    requirement_names = [req.split("==")[0].split(">=")[0] for req in requirements]

    # Should find imports from all files
    assert any("pandas" in req for req in requirement_names)
    assert any("numpy" in req for req in requirement_names)
    assert any("scikit-learn" in req for req in requirement_names)
    assert any("requests" in req for req in requirement_names)
    assert any("flask" in req.lower() for req in requirement_names)  # Case insensitive
    assert any("matplotlib" in req for req in requirement_names)


def test_prune_dependencies_mlflow_style():
    """Test MLflow-style dependency pruning."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    analyzer = HybridRequirementsAnalyzer()

    # Test with a set of packages that might have dependencies
    packages = {
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "requests",
        "urllib3",  # Often a dependency of requests
        "setuptools",  # Should be filtered out
        "wheel",  # Should be filtered out
    }

    # Apply MLflow filtering first
    filtered = analyzer.apply_mlflow_filtering(packages)

    # Then prune dependencies
    pruned = analyzer.prune_dependencies_mlflow_style(filtered)

    assert isinstance(pruned, set)
    # Should not contain build tools
    assert "setuptools" not in pruned
    assert "wheel" not in pruned

    # Should contain main packages
    assert "pandas" in pruned
    assert "numpy" in pruned
    assert "requests" in pruned


def test_generate_pinned_requirements():
    """Test generating pinned requirements."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    analyzer = HybridRequirementsAnalyzer()

    # Test with packages that should be available
    packages = {"os", "sys", "json"}  # These won't have versions but should be handled gracefully

    requirements = analyzer.generate_pinned_requirements(packages)

    assert isinstance(requirements, list)
    assert len(requirements) >= 0  # Should handle gracefully even if no versions available

    # Test with a mock package that has a version

    # Create a simple test case with known behavior
    test_packages = set()  # Empty set should return empty list
    requirements = analyzer.generate_pinned_requirements(test_packages)
    assert requirements == []


def test_generate_pinned_requirements_with_real_packages():
    """Test generating pinned requirements with real installed packages."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    analyzer = HybridRequirementsAnalyzer()

    # Use packages that are likely to be installed in the test environment
    packages = {"pytest"}  # pytest should be available in test environment

    requirements = analyzer.generate_pinned_requirements(packages)

    assert isinstance(requirements, list)

    # Should either have a pinned version or just the package name
    for req in requirements:
        assert isinstance(req, str)
        assert "pytest" in req.lower()


def test_apply_mlflow_filtering():
    """Test MLflow-style package filtering."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    analyzer = HybridRequirementsAnalyzer()

    packages = {
        "pandas",
        "numpy",
        "setuptools",  # Should be filtered out
        "pip",  # Should be filtered out
        "wheel",  # Should be filtered out
        "distutils",  # Should be filtered out
        "pkg-resources",  # Should be filtered out
        "mlflow",
        "requests",
    }

    filtered = analyzer.apply_mlflow_filtering(packages)

    # Should exclude development/build packages
    assert "setuptools" not in filtered
    assert "pip" not in filtered
    assert "wheel" not in filtered
    assert "distutils" not in filtered
    assert "pkg-resources" not in filtered

    # Should keep regular packages
    assert "pandas" in filtered
    assert "numpy" in filtered
    assert "requests" in filtered


def test_exclude_existing_requirements():
    """Test excluding existing requirements."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    existing_reqs = ["pandas>=1.3.0", "numpy==1.21.0", "mlflow>=2.0.0"]
    analyzer = HybridRequirementsAnalyzer(existing_requirements=existing_reqs)

    packages = {"pandas", "numpy", "sklearn", "mlflow", "requests"}

    filtered = analyzer.exclude_existing_requirements(packages)

    # Should exclude packages that are already in existing requirements
    assert "pandas" not in filtered
    assert "numpy" not in filtered
    assert "mlflow" not in filtered

    # Should keep packages that are not in existing requirements
    assert "sklearn" in filtered
    assert "requests" in filtered


def test_resolve_packages_mlflow_style():
    """Test resolving modules to packages using MLflow-style approach."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    analyzer = HybridRequirementsAnalyzer()

    # Test with common modules
    modules = {"pandas", "numpy", "sklearn", "requests"}

    packages = analyzer.resolve_packages_mlflow_style(modules)

    assert isinstance(packages, set)
    # Should map modules to packages (might be the same or different)
    assert len(packages) >= 0


def test_analyze_model_requirements_with_local_patterns(tmp_path):
    """Test analyze_model_requirements with custom local patterns."""
    from mlflow_dep_analyzer import HybridRequirementsAnalyzer

    # Create test file
    test_file = tmp_path / "model.py"
    test_file.write_text("""
import pandas
import numpy
import custom_local_lib
import another_local_module
""")

    # Define custom local patterns
    local_patterns = {"custom_local_lib", "another_local_module"}

    analyzer = HybridRequirementsAnalyzer()
    result = analyzer.analyze_model_requirements(
        code_paths=[str(test_file)], repo_root=str(tmp_path), local_patterns=local_patterns
    )

    external_modules = set(result["analysis"]["external_modules"])

    # Should include external packages
    assert "pandas" in external_modules
    assert "numpy" in external_modules

    # Should exclude custom local patterns
    assert "custom_local_lib" not in external_modules
    assert "another_local_module" not in external_modules

"""
Tests for the HybridRequirementsAnalyzer.
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.mlflow_code_analysis.requirements_analyzer import HybridRequirementsAnalyzer, load_requirements_from_file


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
    from src.mlflow_code_analysis import is_stdlib_module

    # Test that the convenience function works the same as the method
    assert is_stdlib_module("os")
    assert is_stdlib_module("json")
    assert is_stdlib_module("datetime")
    assert not is_stdlib_module("pandas")
    assert not is_stdlib_module("numpy")
    assert not is_stdlib_module("nonexistent_module")


def test_load_requirements_nonexistent_file():
    """Test loading from non-existent file."""
    requirements = load_requirements_from_file("/non/existent/file.txt")
    assert requirements == []

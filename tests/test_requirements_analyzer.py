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

    def test_filter_local_modules(self):
        """Test filtering out local project modules."""
        analyzer = HybridRequirementsAnalyzer()
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

        filtered = analyzer.filter_local_modules(imports, "/test/repo")

        # Should exclude local patterns
        assert "pandas" in filtered
        assert "projects" not in filtered
        assert "shared_utils" not in filtered

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


def test_load_requirements_nonexistent_file():
    """Test loading from non-existent file."""
    requirements = load_requirements_from_file("/non/existent/file.txt")
    assert requirements == []

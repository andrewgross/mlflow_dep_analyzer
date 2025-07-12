"""
Tests for the CodePathAnalyzer.
"""

from mlflow_dep_analyzer.code_path_analyzer import CodePathAnalyzer, analyze_code_paths, find_model_code_paths


class TestCodePathAnalyzer:
    """Test cases for CodePathAnalyzer."""

    def test_analyze_file_local_imports(self, tmp_path):
        """Test analyzing a file for local imports."""
        analyzer = CodePathAnalyzer(str(tmp_path))

        # Create a local module so it gets detected as local
        (tmp_path / "projects").mkdir()
        (tmp_path / "projects" / "__init__.py").touch()

        test_file = tmp_path / "test_model.py"
        test_file.write_text("""
import os
import pandas as pd
from projects.shared import common
import external_package
""")

        imports = analyzer.analyze_file(str(test_file))

        # Should detect local imports only (full module names)
        assert "projects.shared" in imports
        # External packages should not be in the results (this method returns only local imports)
        assert "pandas" not in imports
        assert "os" not in imports
        assert "external_package" not in imports

        # Test that we detect the number of local imports we expect
        assert len(imports) == 1  # Only projects.shared should be local

    def test_is_local_import(self, tmp_path):
        """Test local import detection logic using dynamic detection."""
        analyzer = CodePathAnalyzer(str(tmp_path))
        repo_name = tmp_path.name

        # Create actual local modules in the test repo
        (tmp_path / "projects").mkdir()
        (tmp_path / "projects" / "__init__.py").touch()
        (tmp_path / "shared_utils").mkdir()
        (tmp_path / "shared_utils" / "__init__.py").touch()
        (tmp_path / "local_module.py").touch()

        # Create a repo-named module
        (tmp_path / repo_name).mkdir()
        (tmp_path / repo_name / "__init__.py").touch()

        # Test various import patterns - should detect as local
        assert analyzer._is_local_import("projects.my_model", repo_name)
        assert analyzer._is_local_import("shared_utils.base", repo_name)
        assert analyzer._is_local_import(f"{repo_name}.module", repo_name)
        assert analyzer._is_local_import("local_module", repo_name)

        # External packages should not be local
        assert not analyzer._is_local_import("pandas", repo_name)
        assert not analyzer._is_local_import("numpy.array", repo_name)
        assert not analyzer._is_local_import("sklearn.linear_model", repo_name)
        assert not analyzer._is_local_import("nonexistent_module", repo_name)

    def test_analyze_code_paths_complete(self, tmp_path):
        """Test complete code path analysis."""
        analyzer = CodePathAnalyzer(str(tmp_path))

        # Create test project structure
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        (projects_dir / "__init__.py").touch()

        model_file = projects_dir / "model.py"
        model_file.write_text("""
import pandas as pd
from projects.utils import helper
""")

        utils_file = projects_dir / "utils.py"
        utils_file.write_text("def helper(): pass")

        result = analyzer.analyze_code_paths([str(model_file)])

        assert "entry_files" in result
        assert "required_files" in result
        assert "relative_paths" in result
        assert "dependencies" in result
        assert "analysis" in result

        # Should include entry file
        assert str(model_file) in result["entry_files"]
        assert str(model_file) in result["required_files"]
        assert str(utils_file) in result["required_files"]

        # Verify we have exactly the expected number of files
        assert (
            len(result["required_files"]) == 2
        ), f"Expected 2 files, got {len(result['required_files'])}: {result['required_files']}"

        # Verify relative paths are correct
        relative_paths = result["relative_paths"]
        assert any("projects/model.py" in path for path in relative_paths)
        assert any("projects/utils.py" in path for path in relative_paths)

        # Verify analysis metrics
        assert result["analysis"]["total_files"] == 2
        assert result["analysis"]["total_dependencies"] >= 1


def test_analyze_code_paths_convenience(tmp_path):
    """Test convenience function for code path analysis."""
    # Create test file
    model_file = tmp_path / "model.py"
    model_file.write_text("import pandas")

    paths = analyze_code_paths(entry_files=[str(model_file)], repo_root=str(tmp_path))

    assert isinstance(paths, list)
    assert any("model.py" in path for path in paths)


def test_find_model_code_paths(tmp_path):
    """Test finding code paths for a single model."""
    # Test case 1: Model with no local dependencies
    model_file = tmp_path / "sentiment_model.py"
    model_file.write_text("""
import pandas
import numpy
""")

    paths = find_model_code_paths(str(model_file), str(tmp_path))
    assert isinstance(paths, list)
    # Entry file is always included, even with no local imports
    assert len(paths) == 1, f"Expected 1 path (entry file), got {len(paths)}: {paths}"
    assert any("sentiment_model.py" in path for path in paths), f"Entry file not found in paths: {paths}"

    # Test case 2: Model with local dependency
    utils_file = tmp_path / "utils.py"
    utils_file.write_text("def helper(): pass")

    model_with_local = tmp_path / "model_with_local.py"
    model_with_local.write_text("""
import pandas
from utils import helper
""")

    paths_with_local = find_model_code_paths(str(model_with_local), str(tmp_path))
    assert isinstance(paths_with_local, list)

    # Should find both the model file and its dependency
    assert len(paths_with_local) == 2, f"Expected 2 paths, got {len(paths_with_local)}: {paths_with_local}"

    # Check that the model file is included
    assert any(
        "model_with_local.py" in path for path in paths_with_local
    ), f"Model file not found in paths: {paths_with_local}"

    # Should also include the utils dependency
    assert any("utils.py" in path for path in paths_with_local), f"Utils file not found in paths: {paths_with_local}"


def test_deep_recursive_dependency_collection(tmp_path):
    """Test that all files in a deep dependency chain are collected.

    This test would have caught the original bug where only immediate
    dependencies were being collected, not all files in the dependency tree.
    """
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create a complex dependency chain: model.py -> utils.py -> helpers.py -> base.py
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    # Level 1: Entry file
    model_file = projects_dir / "model.py"
    model_file.write_text("""
import pandas as pd
from projects.utils import process_data
""")

    # Level 2: First dependency
    utils_file = projects_dir / "utils.py"
    utils_file.write_text("""
import numpy as np
from projects.helpers import transform_data

def process_data(data):
    return transform_data(data)
""")

    # Level 3: Second dependency
    helpers_file = projects_dir / "helpers.py"
    helpers_file.write_text("""
from projects.base import BaseTransformer

def transform_data(data):
    transformer = BaseTransformer()
    return transformer.transform(data)
""")

    # Level 4: Deepest dependency
    base_file = projects_dir / "base.py"
    base_file.write_text("""
class BaseTransformer:
    def transform(self, data):
        return data
""")

    # Also create __init__.py files to make it a proper package
    (projects_dir / "__init__.py").touch()

    # Analyze the dependency chain
    result = analyzer.analyze_code_paths([str(model_file)])

    # Verify structure
    assert "entry_files" in result
    assert "required_files" in result
    assert "relative_paths" in result
    assert "dependencies" in result

    # Critical test: ALL files in the dependency chain should be in required_files
    required_files = result["required_files"]

    # Check that all 4 files are present
    assert str(model_file) in required_files, "Entry file should be in required_files"
    assert str(utils_file) in required_files, "First dependency should be in required_files"
    assert str(helpers_file) in required_files, "Second dependency should be in required_files"
    assert str(base_file) in required_files, "Deepest dependency should be in required_files"

    # Should have exactly 4 files (no duplicates)
    assert len(required_files) == 4, f"Expected 4 files, got {len(required_files)}: {required_files}"

    # Check relative paths contain all files
    relative_paths = result["relative_paths"]
    assert any("projects/model.py" in path for path in relative_paths)
    assert any("projects/utils.py" in path for path in relative_paths)
    assert any("projects/helpers.py" in path for path in relative_paths)
    assert any("projects/base.py" in path for path in relative_paths)

    # Verify the dependency chain is properly tracked
    dependencies = result["dependencies"]
    assert str(model_file) in dependencies

    # Check that the recursive collection worked
    model_deps = dependencies[str(model_file)]
    assert len(model_deps) == 4  # All 4 files should be in the dependency collection


def test_multiple_entry_files_with_shared_dependencies(tmp_path):
    """Test that shared dependencies are not duplicated when analyzing multiple entry files."""
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create shared dependency
    shared_file = tmp_path / "shared.py"
    shared_file.write_text("def shared_function(): pass")

    # Create first entry file that uses shared
    entry1_file = tmp_path / "entry1.py"
    entry1_file.write_text("from shared import shared_function")

    # Create second entry file that also uses shared
    entry2_file = tmp_path / "entry2.py"
    entry2_file.write_text("from shared import shared_function")

    # Analyze both entry files
    result = analyzer.analyze_code_paths([str(entry1_file), str(entry2_file)])

    # Should have all 3 files but no duplicates
    required_files = result["required_files"]
    assert len(required_files) == 3, f"Expected 3 unique files, got {len(required_files)}: {required_files}"

    # All files should be present
    assert str(entry1_file) in required_files
    assert str(entry2_file) in required_files
    assert str(shared_file) in required_files


def test_circular_dependency_handling(tmp_path):
    """Test that circular dependencies don't cause infinite loops."""
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create circular dependency: a.py imports b.py, b.py imports a.py
    file_a = tmp_path / "a.py"
    file_a.write_text("from b import func_b")

    file_b = tmp_path / "b.py"
    file_b.write_text("from a import func_a")

    # This should not hang or crash
    result = analyzer.analyze_code_paths([str(file_a)])

    # Should collect both files
    required_files = result["required_files"]
    assert str(file_a) in required_files
    assert str(file_b) in required_files
    assert len(required_files) == 2

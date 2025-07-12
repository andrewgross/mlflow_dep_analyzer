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

        # Verify we have exactly the expected number of files (2 .py files + 1 __init__.py)
        assert (
            len(result["required_files"]) == 3
        ), f"Expected 3 files, got {len(result['required_files'])}: {result['required_files']}"

        # Verify relative paths are correct
        relative_paths = result["relative_paths"]
        assert any("projects/model.py" in path for path in relative_paths)
        assert any("projects/utils.py" in path for path in relative_paths)

        # Verify analysis metrics
        assert result["analysis"]["total_files"] == 3
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

    # Should have exactly 5 files (4 .py files + 1 __init__.py file)
    assert len(required_files) == 5, f"Expected 5 files, got {len(required_files)}: {required_files}"

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
    assert len(model_deps) == 5  # All 5 files should be in the dependency collection


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


def test_inspect_module_resolution_src_structure(tmp_path):
    """Test inspect-based module resolution with src/ structure."""
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create src/ directory structure like real Python projects
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pkg_dir = src_dir / "mypackage"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()

    # Create modules in the package
    model_file = pkg_dir / "model.py"
    model_file.write_text("""
from mypackage.utils import helper_function
from mypackage.data.loader import load_data

def main_function():
    data = load_data()
    return helper_function(data)
""")

    utils_file = pkg_dir / "utils.py"
    utils_file.write_text("""
def helper_function(data):
    return data * 2
""")

    # Create nested package
    data_dir = pkg_dir / "data"
    data_dir.mkdir()
    (data_dir / "__init__.py").touch()
    loader_file = data_dir / "loader.py"
    loader_file.write_text("""
import os
def load_data():
    return [1, 2, 3]
""")

    # Analyze the model file
    result = analyzer.analyze_code_paths([str(model_file)])
    required_files = result["required_files"]

    # Should find all files using proper module resolution
    assert str(model_file) in required_files, "Model file should be found"
    assert str(utils_file) in required_files, "Utils file should be found via inspect"
    assert str(loader_file) in required_files, "Loader file should be found via inspect"

    # Should include package __init__.py files
    assert str(pkg_dir / "__init__.py") in required_files, "Package __init__.py should be found"
    assert str(data_dir / "__init__.py") in required_files, "Data package __init__.py should be found"


def test_inspect_cross_package_imports(tmp_path):
    """Test inspect resolution with imports across different packages."""
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create multiple packages
    for pkg_name in ["package_a", "package_b", "shared"]:
        pkg_dir = tmp_path / pkg_name
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").touch()

    # Package A imports from Package B and shared
    pkg_a_module = tmp_path / "package_a" / "module.py"
    pkg_a_module.write_text("""
from package_b.service import process_request
from shared.common import utility_function

def handle_request(data):
    processed = process_request(data)
    return utility_function(processed)
""")

    # Package B imports from shared
    pkg_b_service = tmp_path / "package_b" / "service.py"
    pkg_b_service.write_text("""
from shared.common import base_function

def process_request(data):
    return base_function(data)
""")

    # Shared package
    shared_common = tmp_path / "shared" / "common.py"
    shared_common.write_text("""
def utility_function(data):
    return data

def base_function(data):
    return data.upper() if isinstance(data, str) else str(data)
""")

    # Analyze package A module
    result = analyzer.analyze_code_paths([str(pkg_a_module)])
    required_files = result["required_files"]

    # Should find all files across packages
    assert str(pkg_a_module) in required_files
    assert str(pkg_b_service) in required_files
    assert str(shared_common) in required_files

    # Should include all package __init__.py files
    for pkg_name in ["package_a", "package_b", "shared"]:
        init_file = str(tmp_path / pkg_name / "__init__.py")
        assert init_file in required_files, f"Missing {pkg_name}/__init__.py"


def test_inspect_relative_imports(tmp_path):
    """Test inspect resolution with relative imports."""
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create package structure for relative imports
    pkg_dir = tmp_path / "mypackage"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()

    # Create subpackage
    sub_dir = pkg_dir / "subpackage"
    sub_dir.mkdir()
    (sub_dir / "__init__.py").touch()

    # Module with relative imports
    main_module = sub_dir / "main.py"
    main_module.write_text("""
from . import sibling
from ..utils import helper
from .nested.deep import deep_function

def main():
    return sibling.func() + helper.help() + deep_function()
""")

    # Sibling module
    sibling_module = sub_dir / "sibling.py"
    sibling_module.write_text("""
def func():
    return "sibling"
""")

    # Parent utils
    utils_module = pkg_dir / "utils.py"
    utils_module.write_text("""
def help():
    return "helper"
""")

    # Nested deep module
    nested_dir = sub_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "__init__.py").touch()
    deep_module = nested_dir / "deep.py"
    deep_module.write_text("""
def deep_function():
    return "deep"
""")

    # Analyze main module
    result = analyzer.analyze_code_paths([str(main_module)])
    required_files = result["required_files"]

    # Should resolve relative imports correctly
    assert str(main_module) in required_files
    assert str(sibling_module) in required_files
    assert str(utils_module) in required_files
    assert str(deep_module) in required_files


def test_inspect_module_fallback_to_manual_resolution(tmp_path):
    """Test that manual resolution works when inspect fails."""
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create a scenario where inspect might fail but manual resolution works
    # This could happen with malformed modules or import errors

    # Create a module that imports something that might not import cleanly
    main_file = tmp_path / "main.py"
    main_file.write_text("""
# This import might fail at runtime but we still want to find the file
from problematic_module import some_function

def main():
    return "main"
""")

    # Create the problematic module (syntactically invalid to cause import errors)
    problematic_file = tmp_path / "problematic_module.py"
    problematic_file.write_text("""
# This module has syntax errors that prevent import
def some_function():
    return "result"

# Add syntax error to trigger ImportError
class BrokenClass
    pass  # Missing colon will cause SyntaxError
""")

    # Despite import errors, the analyzer should still find both files
    result = analyzer.analyze_code_paths([str(main_file)])
    required_files = result["required_files"]

    # Should find both files using fallback resolution
    assert str(main_file) in required_files
    assert str(problematic_file) in required_files


def test_inspect_long_dependency_chain_performance(tmp_path):
    """Test performance with very long dependency chains."""
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create a chain of 10 modules, each importing the next
    chain_length = 10
    files = []

    for i in range(chain_length):
        filename = f"module_{i}.py"
        filepath = tmp_path / filename
        files.append(filepath)

        if i < chain_length - 1:
            # Import the next module in chain
            next_module = f"module_{i + 1}"
            content = f"""
from {next_module} import function_{i + 1}

def function_{i}():
    return function_{i + 1}() + {i}
"""
        else:
            # Last module in chain
            content = f"""
def function_{i}():
    return {i}
"""

        filepath.write_text(content)

    # Analyze the first module (should trace the entire chain)
    result = analyzer.analyze_code_paths([str(files[0])])
    required_files = result["required_files"]

    # Should find all files in the chain
    assert len(required_files) == chain_length, f"Expected {chain_length} files, got {len(required_files)}"

    for i, file_path in enumerate(files):
        assert str(file_path) in required_files, f"Missing file {i}: {file_path}"


def test_inspect_mixed_import_styles(tmp_path):
    """Test inspect resolution with mixed import styles in one file."""
    analyzer = CodePathAnalyzer(str(tmp_path))

    # Create modules for different import styles
    utils_file = tmp_path / "utils.py"
    utils_file.write_text("def utility(): pass")

    helpers_file = tmp_path / "helpers.py"
    helpers_file.write_text("def helper(): pass")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "__init__.py").touch()
    processor_file = data_dir / "processor.py"
    processor_file.write_text("def process(): pass")

    # Main file with mixed import styles
    main_file = tmp_path / "main.py"
    main_file.write_text("""
# Direct import
import utils

# From import
from helpers import helper

# Package import
from data.processor import process

# Import with alias
import helpers as h

def main():
    utils.utility()
    helper()
    process()
    h.helper()
""")

    result = analyzer.analyze_code_paths([str(main_file)])
    required_files = result["required_files"]

    # Should find all imported files
    assert str(main_file) in required_files
    assert str(utils_file) in required_files
    assert str(helpers_file) in required_files
    assert str(processor_file) in required_files
    assert str(data_dir / "__init__.py") in required_files

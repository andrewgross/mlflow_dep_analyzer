"""
Tests for the internal methods of UnifiedDependencyAnalyzer.

These tests focus on the smaller, refactored methods to ensure they work correctly
in isolation and provide good test coverage for the internal logic.
"""

import ast
import sys
from unittest.mock import patch

from mlflow_dep_analyzer.unified_analyzer import DependencyType, ModuleInfo, UnifiedDependencyAnalyzer


class TestUnifiedDependencyAnalyzerInternals:
    """Test cases for internal methods of UnifiedDependencyAnalyzer."""

    def test_detect_stdlib_from_filesystem(self, tmp_path):
        """Test stdlib module detection from filesystem."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # This should not crash and should return a set
        stdlib_modules = analyzer._detect_stdlib_from_filesystem()
        assert isinstance(stdlib_modules, set)

        # Some modules might be detected, others might not depending on Python installation
        # Just verify it doesn't crash and returns reasonable results

    def test_get_stdlib_module_names(self, tmp_path):
        """Test that stdlib module names are correctly retrieved."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        stdlib_modules = analyzer._get_stdlib_module_names()

        # Check some known stdlib modules are included
        expected_modules = {"os", "sys", "json", "datetime", "pathlib", "collections"}
        assert expected_modules.issubset(stdlib_modules)

        # Should be a reasonable number of modules
        assert len(stdlib_modules) > 40

        # If Python 3.10+, should use sys.stdlib_module_names
        if hasattr(sys, "stdlib_module_names"):
            # Should have many more modules from the official list
            assert len(stdlib_modules) > 200
        else:
            # Fallback should have essential modules only
            assert len(stdlib_modules) < 100

    def test_is_stdlib_module(self, tmp_path):
        """Test stdlib module identification."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Known stdlib modules
        assert analyzer._is_stdlib_module("os")
        assert analyzer._is_stdlib_module("sys")
        assert analyzer._is_stdlib_module("json")
        assert analyzer._is_stdlib_module("datetime.datetime")  # submodule

        # Known external packages
        assert not analyzer._is_stdlib_module("pandas")
        assert not analyzer._is_stdlib_module("numpy")
        assert not analyzer._is_stdlib_module("nonexistent_package")

    def test_extract_package_name(self, tmp_path):
        """Test package name extraction."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Simple package names
        assert analyzer._extract_package_name("pandas") == "pandas"
        assert analyzer._extract_package_name("numpy") == "numpy"

        # Submodules should return top-level package
        assert analyzer._extract_package_name("pandas.core.frame") == "pandas"
        assert analyzer._extract_package_name("sklearn.linear_model") == "sklearn"

        # Problematic names should be filtered out
        assert analyzer._extract_package_name("") is None
        assert analyzer._extract_package_name("_") is None
        assert analyzer._extract_package_name("__") is None
        assert analyzer._extract_package_name("test") is None
        assert analyzer._extract_package_name("tests") is None

    def test_convert_to_relative_paths(self, tmp_path):
        """Test conversion of absolute paths to relative paths."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create some test files
        file1 = tmp_path / "model.py"
        file1.touch()
        file2 = tmp_path / "subdir" / "utils.py"
        file2.parent.mkdir()
        file2.touch()

        # External file (outside repo)
        external_file = tmp_path.parent / "external.py"
        external_file.touch()

        local_files = {str(file1), str(file2), str(external_file)}

        relative_paths = analyzer._convert_to_relative_paths(local_files)

        # Should include files within repo
        assert "model.py" in relative_paths
        assert "subdir/utils.py" in relative_paths

        # Should exclude files outside repo (external_file should be excluded)
        assert len(relative_paths) == 2

    def test_setup_import_paths(self, tmp_path):
        """Test that import paths are correctly set up."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create some directories (using common Python project patterns)
        (tmp_path / "src").mkdir()
        (tmp_path / "lib").mkdir()

        original_path = sys.path.copy()
        try:
            analyzer._setup_import_paths()

            # Should have added repo paths to sys.path
            assert str(tmp_path) in sys.path
            assert str(tmp_path / "src") in sys.path
            assert str(tmp_path / "lib") in sys.path

        finally:
            sys.path[:] = original_path

    def test_is_problematic_module_name(self, tmp_path):
        """Test identification of problematic module names."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Normal module names should not be problematic
        assert not analyzer._is_problematic_module_name("pandas")
        assert not analyzer._is_problematic_module_name("my_package.module")

        # Problematic patterns should be detected
        assert analyzer._is_problematic_module_name("some.module.in.venv.test")
        assert analyzer._is_problematic_module_name("package.site-packages.module")
        assert analyzer._is_problematic_module_name("parent...child")
        assert analyzer._is_problematic_module_name("module.__pycache__.compiled")

        # Single-part names should not be problematic regardless of content
        assert not analyzer._is_problematic_module_name("venv")  # no dot
        assert not analyzer._is_problematic_module_name("__pycache__")  # no dot

    def test_get_module_file_path(self, tmp_path):
        """Test getting file path from imported module."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Test with a real module that has a file
        import tempfile

        file_path = analyzer._get_module_file_path(tempfile)
        assert file_path is not None
        assert file_path.endswith(".py")

        # Test with a built-in module (no file)
        import sys

        file_path = analyzer._get_module_file_path(sys)
        # sys might or might not have a file depending on Python implementation
        # Just verify it doesn't crash

    def test_parse_python_file(self, tmp_path):
        """Test Python file parsing."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create a valid Python file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
import os
import pandas as pd
from utils import helper

def function():
    return os.path.join('a', 'b')
""")

        tree = analyzer._parse_python_file(str(test_file))
        assert isinstance(tree, ast.AST)

        # Should be able to walk the AST
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])

        assert "os" in imports
        assert "pandas" in imports

    def test_process_import_node(self, tmp_path):
        """Test processing of import nodes."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create an import node
        import_node = ast.Import(
            names=[
                ast.alias(name="os", asname=None),
                ast.alias(name="pandas", asname="pd"),
            ]
        )

        imports = set()
        analyzer._process_import_node(import_node, imports)

        assert "os" in imports
        assert "pandas" in imports
        assert len(imports) == 2

    def test_calculate_target_directory(self, tmp_path):
        """Test calculation of target directory for relative imports."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create nested directory structure
        deep_dir = tmp_path / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)
        test_file = deep_dir / "test.py"
        test_file.touch()

        # Level 1 should be current directory (level3)
        target = analyzer._calculate_target_directory(str(test_file), 1)
        assert target == deep_dir

        # Level 2 should go up one directory (level2)
        target = analyzer._calculate_target_directory(str(test_file), 2)
        assert target == deep_dir.parent

        # Level 3 should go up two directories (level1)
        target = analyzer._calculate_target_directory(str(test_file), 3)
        assert target == deep_dir.parent.parent

        # Going beyond repo root should return None
        target = analyzer._calculate_target_directory(str(test_file), 10)
        assert target is None

    def test_build_module_name_from_path(self, tmp_path):
        """Test building module names from directory paths."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Test with nested directory
        nested_dir = tmp_path / "package" / "subpackage"
        nested_dir.mkdir(parents=True)

        # Without module name
        module_name = analyzer._build_module_name_from_path(nested_dir, None)
        assert module_name == "package.subpackage"

        # With module name
        module_name = analyzer._build_module_name_from_path(nested_dir, "module")
        assert module_name == "package.subpackage.module"

        # Test with src/ directory (should be stripped)
        src_dir = tmp_path / "src" / "mypackage"
        src_dir.mkdir(parents=True)

        module_name = analyzer._build_module_name_from_path(src_dir, None)
        assert module_name == "mypackage"

        # Test with repo root
        module_name = analyzer._build_module_name_from_path(tmp_path, "module")
        assert module_name == "module"

        # Test with repo root and no module
        module_name = analyzer._build_module_name_from_path(tmp_path, None)
        assert module_name == ""

    def test_categorize_modules(self, tmp_path):
        """Test module categorization."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create test modules
        local_file = tmp_path / "local_module.py"
        local_file.touch()

        modules = {
            "pandas": ModuleInfo("pandas", DependencyType.EXTERNAL_PACKAGE),
            "os": ModuleInfo("os", DependencyType.STDLIB_MODULE),
            "local_module": ModuleInfo("local_module", DependencyType.LOCAL_FILE, str(local_file)),
            "sklearn.linear_model": ModuleInfo("sklearn.linear_model", DependencyType.EXTERNAL_PACKAGE),
            "test": ModuleInfo("test", DependencyType.EXTERNAL_PACKAGE),  # Should be filtered out
        }

        external_packages, local_files = analyzer._categorize_modules(modules)

        assert "pandas" in external_packages
        assert "sklearn" in external_packages  # Top-level package
        assert "test" not in external_packages  # Should be filtered out
        assert "os" not in external_packages  # Stdlib module

        assert str(local_file) in local_files

    def test_build_analysis_result(self, tmp_path):
        """Test building the final analysis result."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create test data
        entry_files = ["model.py"]
        modules = {
            "pandas": ModuleInfo("pandas", DependencyType.EXTERNAL_PACKAGE),
            "os": ModuleInfo("os", DependencyType.STDLIB_MODULE),
            "local": ModuleInfo("local", DependencyType.LOCAL_FILE, "local.py"),
        }
        external_packages = {"pandas"}
        relative_code_paths = ["model.py", "local.py"]

        result = analyzer._build_analysis_result(entry_files, modules, external_packages, relative_code_paths)

        # Check structure
        assert "requirements" in result
        assert "code_paths" in result
        assert "analysis" in result
        assert "detailed_modules" in result

        # Check contents
        assert result["requirements"] == ["pandas"]
        assert result["code_paths"] == ["local.py", "model.py"]  # Should be sorted

        analysis = result["analysis"]
        assert analysis["total_modules"] == 3
        assert analysis["external_packages"] == 1
        assert analysis["local_files"] == 1
        assert analysis["stdlib_modules"] == 1
        assert analysis["entry_files"] == entry_files

    @patch("importlib.import_module")
    def test_handle_import_error(self, mock_import, tmp_path):
        """Test handling of import errors."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create a local file
        local_file = tmp_path / "local_module.py"
        local_file.touch()

        # Test when local module exists
        result = analyzer._handle_import_error("local_module")
        assert result is not None
        assert result.dep_type == DependencyType.LOCAL_FILE
        assert result.file_path == str(local_file)

        # Test when module doesn't exist locally
        result = analyzer._handle_import_error("nonexistent_module")
        assert result is not None
        assert result.dep_type == DependencyType.EXTERNAL_PACKAGE
        assert result.file_path is None

    def test_process_entry_files(self, tmp_path):
        """Test processing of entry files."""
        analyzer = UnifiedDependencyAnalyzer(str(tmp_path))

        # Create test files
        file1 = tmp_path / "file1.py"
        file1.write_text("import os")

        file2 = tmp_path / "file2.py"
        file2.write_text("import json")

        nonexistent = tmp_path / "nonexistent.py"

        all_modules = {}
        processed_files = set()

        # This should not crash even with nonexistent file
        analyzer._process_entry_files([str(file1), str(file2), str(nonexistent)], all_modules, processed_files)

        # Should have processed the existing files
        assert str(file1) in processed_files
        assert str(file2) in processed_files
        assert str(nonexistent) not in processed_files

        # Should have discovered some modules
        assert len(all_modules) > 0

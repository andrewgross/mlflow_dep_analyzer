"""
Code Path Analysis for MLflow Models

This module provides utilities to analyze Python code and determine
the minimal set of files needed for MLflow model deployment with code_paths.

The goal is to identify which files are actually used by a model so that
only necessary code is bundled with the model artifact.
"""

import ast
import importlib
import inspect
import os
import sys
from pathlib import Path


class CodePathAnalyzer:
    """
    Analyzer for determining minimal code paths for MLflow models.

    This analyzer:
    1. Uses AST to find local imports without code execution
    2. Recursively discovers all dependencies
    3. Filters out unnecessary files
    4. Provides detailed analysis for debugging
    """

    def __init__(self, repo_root: str):
        """
        Initialize the code path analyzer.

        Args:
            repo_root: Root directory of the repository
        """
        self.repo_root = Path(repo_root).resolve()

    def analyze_file(self, file_path: str) -> set[str]:
        """
        Find all local imports in a Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Set of local import module names
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return set()

        local_imports = set()
        repo_name = self.repo_root.name

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    # Check if it's a local import
                    if self._is_local_import(module_name, repo_name):
                        local_imports.add(module_name)
            elif isinstance(node, ast.ImportFrom):
                # Handle relative imports (. or ..)
                if node.level > 0:
                    # This is a relative import, we need to resolve it based on the file's location
                    if node.module:
                        # from .module import something
                        relative_module = self._resolve_relative_import(file_path, node.level, node.module)
                        if relative_module and self._is_local_import(relative_module, repo_name):
                            local_imports.add(relative_module)
                    else:
                        # from . import something - need to handle each imported name
                        base_module = self._resolve_relative_import(file_path, node.level, None)
                        if base_module is not None:
                            for alias in node.names:
                                if base_module:
                                    full_module = f"{base_module}.{alias.name}"
                                else:
                                    full_module = alias.name
                                if self._is_local_import(full_module, repo_name):
                                    local_imports.add(full_module)
                elif node.module:
                    # Absolute imports that start with repo name or known local patterns
                    if self._is_local_import(node.module, repo_name):
                        local_imports.add(node.module)

        return local_imports

    def _resolve_relative_import(self, file_path: str, level: int, module: str | None) -> str | None:
        """Resolve a relative import to an absolute module name."""
        try:
            file_obj = Path(file_path).resolve()  # Resolve symlinks consistently

            # Start from the file's directory
            current_dir = file_obj.parent

            # For level 1 (single dot), we stay in the current directory
            # For level 2 (double dot), we go up one directory, etc.
            for _ in range(level - 1):
                if current_dir == self.repo_root or current_dir == current_dir.parent:
                    return None  # Can't go up further
                current_dir = current_dir.parent

            # Convert directory path back to module path
            try:
                relative_to_root = current_dir.relative_to(self.repo_root)
                if relative_to_root == Path("."):
                    # We're at the repo root
                    package_parts = []
                else:
                    package_parts = list(relative_to_root.parts)

                # Handle src/ directory specially
                if package_parts and package_parts[0] == "src":
                    package_parts = package_parts[1:]

                # Add the module name if provided
                if module:
                    package_parts.append(module)

                if package_parts:
                    return ".".join(package_parts)
                else:
                    return module or ""

            except ValueError:
                # File is outside repo root
                return None

        except Exception:
            return None

    def _is_local_import(self, module_name: str, repo_name: str) -> bool:
        """Check if a module name represents a local import using dynamic detection."""
        if not module_name:
            return False

        # Check if starts with repo name
        if repo_name and module_name.startswith(repo_name):
            return True

        # Get the top-level module name
        top_level_module = module_name.split(".")[0]

        # Check if this module exists as a directory or file in the repo
        return self._module_exists_in_repo(top_level_module)

    def _module_exists_in_repo(self, module_name: str) -> bool:
        """Check if a module exists as a local file or directory in the repository."""
        # Check common locations where Python modules might exist
        search_paths = [
            self.repo_root,  # Root level
            self.repo_root / "src",  # src/ directory
            self.repo_root / "lib",  # lib/ directory
            self.repo_root / "packages",  # packages/ directory
        ]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            # Check for package directory (with __init__.py)
            package_dir = search_path / module_name
            if package_dir.is_dir():
                # It's a local package if it has __init__.py or contains .py files
                if (package_dir / "__init__.py").exists() or any(package_dir.glob("*.py")):
                    return True

            # Check for module file
            module_file = search_path / f"{module_name}.py"
            if module_file.is_file():
                return True

        return False

    def collect_dependencies(self, entry_file: str) -> dict[str, set[str]]:
        """
        Recursively collect all local dependencies from an entry file.

        Args:
            entry_file: Path to the main file to start analysis from

        Returns:
            Dictionary mapping file paths to their local imports
        """
        dependencies = {}
        # Convert entry_file to absolute path for consistency
        entry_file_abs = str(Path(entry_file).resolve())
        to_process = {entry_file_abs}
        processed = set()

        while to_process:
            current = to_process.pop()
            if current in processed:
                continue
            processed.add(current)

            # Get imports for current file
            imports = self.analyze_file(current)
            dependencies[current] = imports

            # Also add the current file's own package __init__.py files
            current_package_inits = self._get_package_init_files(current)
            for init_file in current_package_inits:
                if init_file not in processed:
                    to_process.add(init_file)

            # Find corresponding files for imports
            for imp in imports:
                # Convert module path to file paths
                potential_files = self._module_to_file_paths(imp)
                for file_path in potential_files:
                    if file_path and file_path not in processed:
                        to_process.add(file_path)

        return dependencies

    def _module_to_file_paths(self, module_name: str) -> list[str]:
        """
        Convert a module name to file paths using proper module resolution.

        Args:
            module_name: Python module name (e.g., "projects.my_model.sentiment")

        Returns:
            List of file paths that contain this module
        """
        if not module_name:
            return []

        paths = []

        # First try manual resolution to avoid import errors
        manual_paths = self._manual_module_resolution(module_name)
        if manual_paths:
            paths.extend(manual_paths)

        # Then try inspect-based resolution if manual didn't find anything or to complement it
        if not paths or True:  # Always try inspect to get additional info
            # Add the repo root to sys.path temporarily to enable proper module resolution
            repo_root_str = str(self.repo_root)
            src_path = str(self.repo_root / "src")

            original_path = sys.path.copy()
            paths_to_add = [repo_root_str]
            if (self.repo_root / "src").exists():
                paths_to_add.append(src_path)

            try:
                # Add our paths to the beginning of sys.path
                for path in reversed(paths_to_add):
                    if path not in sys.path:
                        sys.path.insert(0, path)

                # Try to import the module and get its file path
                try:
                    module = importlib.import_module(module_name)

                    # Use inspect to get the source file
                    try:
                        source_file = inspect.getsourcefile(module)
                        if source_file and self._is_file_in_repo(source_file):
                            if source_file not in paths:
                                paths.append(source_file)
                    except (TypeError, OSError):
                        pass

                    # Also try __file__ attribute
                    try:
                        if hasattr(module, "__file__") and module.__file__:
                            file_path = str(Path(module.__file__).resolve())
                            if self._is_file_in_repo(file_path):
                                if file_path not in paths:
                                    paths.append(file_path)
                    except (AttributeError, OSError):
                        pass

                except (ImportError, SyntaxError, ValueError):
                    # Import failed, rely on manual resolution
                    pass

            finally:
                # Restore original sys.path
                sys.path[:] = original_path

        return list(set(paths))  # Remove duplicates

    def _is_file_in_repo(self, file_path: str) -> bool:
        """Check if a file path is within the repository."""
        try:
            resolved_path = Path(file_path).resolve()
            return str(resolved_path).startswith(str(self.repo_root))
        except (OSError, ValueError):
            return False

    def _manual_module_resolution(self, module_name: str) -> list[str]:
        """Fallback manual resolution when importlib fails."""
        paths = []

        # Convert dots to path separators
        module_path = module_name.replace(".", "/")

        # Search in common Python package locations
        search_locations = [
            self.repo_root,
            self.repo_root / "src",
            self.repo_root / "lib",
            self.repo_root / "packages",
        ]

        for base_dir in search_locations:
            if not base_dir.exists():
                continue

            # Try as a package (directory with __init__.py)
            package_dir = base_dir / module_path
            package_init = package_dir / "__init__.py"
            if package_init.exists():
                paths.append(str(package_init))

                # Also add intermediate __init__.py files in the package hierarchy
                parts = module_name.split(".")
                current_path = base_dir
                for part in parts:
                    current_path = current_path / part
                    init_file = current_path / "__init__.py"
                    if init_file.exists() and str(init_file) not in paths:
                        paths.append(str(init_file))

            # Try as a direct .py file
            module_file = base_dir / (module_path + ".py")
            if module_file.exists():
                paths.append(str(module_file))

                # Also add intermediate __init__.py files for the module's package
                if "." in module_name:
                    parts = module_name.split(".")[:-1]  # Exclude the module name itself
                    current_path = base_dir
                    for part in parts:
                        current_path = current_path / part
                        init_file = current_path / "__init__.py"
                        if init_file.exists() and str(init_file) not in paths:
                            paths.append(str(init_file))

        return paths

    def _get_package_init_files(self, file_path: str) -> list[str]:
        """Get all __init__.py files for the package hierarchy of a given file."""
        init_files = []
        file_obj = Path(file_path)

        # Start from the file's directory and walk up to repo root
        current_dir = file_obj.parent

        while current_dir != self.repo_root and current_dir != current_dir.parent:
            init_file = current_dir / "__init__.py"
            if init_file.exists():
                init_files.append(str(init_file))
            current_dir = current_dir.parent

        return init_files

    def analyze_code_paths(
        self,
        entry_files: list[str],
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict:
        """
        Analyze code paths for a set of entry files.

        Args:
            entry_files: List of main files to analyze
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude

        Returns:
            Dictionary with analysis results
        """
        if include_patterns is None:
            include_patterns = ["**/*.py"]
        if exclude_patterns is None:
            exclude_patterns = ["**/__pycache__/**", "**/.git/**"]

        all_dependencies = {}
        all_required_files: set[str] = set()

        # Analyze each entry file
        for entry_file in entry_files:
            if not os.path.exists(entry_file):
                print(f"Warning: Entry file does not exist: {entry_file}")
                continue

            file_deps = self.collect_dependencies(entry_file)
            all_dependencies[entry_file] = file_deps

            # Add all discovered files - both from the dependency keys and the resolved file paths
            all_required_files.update(file_deps.keys())  # All files that were analyzed
            # Also add any additional files discovered through module resolution
            for deps in file_deps.values():
                for dep_files in [self._module_to_file_paths(dep) for dep in deps]:
                    all_required_files.update(dep_files)

        # Filter files based on patterns
        filtered_files = self._filter_files(all_required_files, include_patterns, exclude_patterns)

        # Generate relative paths for MLflow
        relative_paths = []
        for file_path in filtered_files:
            try:
                rel_path = os.path.relpath(file_path, self.repo_root)
                if not rel_path.startswith(".."):  # Only include files within repo
                    relative_paths.append(rel_path)
            except ValueError:
                # Path is on different drive (Windows)
                pass

        return {
            "entry_files": entry_files,
            "required_files": sorted(filtered_files),
            "relative_paths": sorted(relative_paths),
            "dependencies": all_dependencies,
            "analysis": {
                "total_files": len(filtered_files),
                "total_dependencies": sum(len(deps) for deps in all_dependencies.values()),
                "include_patterns": include_patterns,
                "exclude_patterns": exclude_patterns,
            },
        }

    def _filter_files(self, files: set[str], include_patterns: list[str], exclude_patterns: list[str]) -> set[str]:
        """Filter files based on include/exclude patterns."""
        from fnmatch import fnmatch

        filtered = set()

        for file_path in files:
            file_path_obj = Path(file_path)

            # Check if file matches include patterns
            included = False
            for pattern in include_patterns:
                if fnmatch(str(file_path_obj), pattern) or fnmatch(file_path_obj.name, pattern):
                    included = True
                    break

            if not included:
                continue

            # Check if file matches exclude patterns
            excluded = False
            for pattern in exclude_patterns:
                if fnmatch(str(file_path_obj), pattern) or fnmatch(file_path_obj.name, pattern):
                    excluded = True
                    break

            if not excluded:
                filtered.add(file_path)

        return filtered


def analyze_code_paths(
    entry_files: list[str],
    repo_root: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """
    Analyze code paths and return relative paths for MLflow code_paths.

    This is a convenience function that returns just the relative paths
    needed for MLflow's code_paths parameter.

    Args:
        entry_files: List of main Python files to analyze
        repo_root: Root directory of the repository
        include_patterns: Glob patterns for files to include
        exclude_patterns: Glob patterns for files to exclude

    Returns:
        List of relative file paths for MLflow code_paths
    """
    analyzer = CodePathAnalyzer(repo_root)
    result = analyzer.analyze_code_paths(
        entry_files=entry_files, include_patterns=include_patterns, exclude_patterns=exclude_patterns
    )
    return result["relative_paths"]


def find_model_code_paths(model_file: str, repo_root: str | None = None) -> list[str]:
    """
    Find all code paths needed for a specific model file.

    This is a high-level convenience function for the common use case
    of finding code paths for a single model file.

    Args:
        model_file: Path to the main model Python file
        repo_root: Root directory of the repository (auto-detected if None)

    Returns:
        List of relative file paths for MLflow code_paths
    """
    if repo_root is None:
        # Auto-detect repo root by looking for common markers
        current_dir = Path(model_file).parent
        while current_dir != current_dir.parent:
            if any(
                (current_dir / marker).exists() for marker in [".git", "pyproject.toml", "setup.py", "requirements.txt"]
            ):
                repo_root = str(current_dir)
                break
            current_dir = current_dir.parent

        if repo_root is None:
            repo_root = str(Path(model_file).parent)

    return analyze_code_paths(
        entry_files=[model_file],
        repo_root=repo_root,
        exclude_patterns=["**/__pycache__/**", "**/.git/**", "**/tests/**", "test_*.py"],
    )

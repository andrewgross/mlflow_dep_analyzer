"""
Smart requirements analysis for MLflow models.

This module provides utilities to analyze Python code and generate minimal requirements.txt
files that include only the dependencies actually needed to run the model.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pkg_resources


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract import statements from Python code."""

    def __init__(self):
        self.imports = set()
        self.from_imports = set()

    def visit_Import(self, node):
        """Visit import statements like 'import numpy'."""
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])  # Get top-level module

    def visit_ImportFrom(self, node):
        """Visit from-import statements like 'from sklearn import model_selection'."""
        if node.module:
            self.from_imports.add(node.module.split(".")[0])  # Get top-level module

    def get_all_imports(self) -> set[str]:
        """Get all imports found in the code."""
        return self.imports.union(self.from_imports)


class RequirementsAnalyzer:
    """Analyzes Python files to determine minimal requirements."""

    def __init__(self):
        self.stdlib_modules = self._get_stdlib_modules()
        self.package_mapping = self._get_package_mapping()

    def _get_stdlib_modules(self) -> set[str]:
        """Get set of standard library modules."""
        # Common stdlib modules (not exhaustive but covers most cases)
        stdlib = {
            "os",
            "sys",
            "datetime",
            "json",
            "pickle",
            "logging",
            "pathlib",
            "tempfile",
            "subprocess",
            "shutil",
            "glob",
            "collections",
            "re",
            "urllib",
            "http",
            "functools",
            "itertools",
            "operator",
            "math",
            "random",
            "string",
            "io",
            "contextlib",
            "typing",
            "dataclasses",
            "abc",
            "copy",
            "time",
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
            "ftplib",
            "gzip",
            "tarfile",
            "zipfile",
            "configparser",
            "argparse",
            "getopt",
            "unittest",
            "doctest",
            "pdb",
            "cProfile",
            "profile",
            "trace",
            "ast",
            "pkg_resources",
            "setuptools",
            "distutils",
        }
        return stdlib

    def _get_package_mapping(self) -> dict[str, str]:
        """Get mapping from import names to package names."""
        # Some imports don't match their package names
        mapping = {
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "sklearn": "scikit-learn",
            "skimage": "scikit-image",
            "bs4": "beautifulsoup4",
            "serial": "pyserial",
            "MySQLdb": "MySQL-python",
            "psycopg2": "psycopg2-binary",
        }
        return mapping

    def analyze_file(self, file_path: str) -> set[str]:
        """Analyze a single Python file for imports."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            analyzer = ImportAnalyzer()
            analyzer.visit(tree)

            return analyzer.get_all_imports()

        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
            return set()

    def analyze_directory(self, directory: str, patterns: list[str] = None) -> set[str]:
        """Analyze all Python files in a directory."""
        if patterns is None:
            patterns = ["**/*.py"]

        all_imports = set()
        directory_path = Path(directory)

        for pattern in patterns:
            for file_path in directory_path.glob(pattern):
                if file_path.is_file() and file_path.suffix == ".py":
                    imports = self.analyze_file(str(file_path))
                    all_imports.update(imports)

        return all_imports

    def filter_external_packages(self, imports: set[str]) -> set[str]:
        """Filter out stdlib modules and local modules, keeping only external packages."""
        external = set()

        # Common local module patterns to exclude
        local_patterns = {
            "projects",
            "shared_utils",
            "text_utils",
            "validation",
            "constants",
            "databricks",
            "my_model",
            "inference",
        }

        for module in imports:
            # Skip stdlib modules
            if module in self.stdlib_modules:
                continue

            # Skip relative imports and special cases
            if module.startswith(".") or not module:
                continue

            # Skip known local modules
            if module in local_patterns:
                continue

            # Skip modules that start with local patterns
            if any(module.startswith(pattern + ".") for pattern in local_patterns):
                continue

            external.add(module)

        return external

    def resolve_package_names(self, imports: set[str]) -> set[str]:
        """Resolve import names to actual package names."""
        packages = set()

        for import_name in imports:
            # Check if we have a known mapping
            package_name = self.package_mapping.get(import_name, import_name)
            packages.add(package_name)

        return packages

    def get_installed_version(self, package_name: str) -> str:
        """Get the currently installed version of a package."""
        try:
            # Try the mapped name first, then the original name
            possible_names = [package_name]
            if package_name in self.package_mapping.values():
                # Find the import name for this package
                for import_name, pkg_name in self.package_mapping.items():
                    if pkg_name == package_name:
                        possible_names.append(import_name)

            for name in possible_names:
                try:
                    distribution = pkg_resources.get_distribution(name)
                    return distribution.version
                except pkg_resources.DistributionNotFound:
                    continue

            # Try importing the module to see if it's available
            try:
                spec = importlib.util.find_spec(package_name)
                if spec is not None:
                    return "unknown"  # Package exists but version unknown
            except ImportError:
                pass

            return None

        except Exception:
            return None

    def generate_requirements(
        self, file_paths: list[str] = None, directory_paths: list[str] = None, include_versions: bool = True
    ) -> list[str]:
        """Generate requirements list from files or directories."""
        all_imports = set()

        # Analyze individual files
        if file_paths:
            for file_path in file_paths:
                imports = self.analyze_file(file_path)
                all_imports.update(imports)

        # Analyze directories
        if directory_paths:
            for directory in directory_paths:
                imports = self.analyze_directory(directory)
                all_imports.update(imports)

        # Filter to external packages only
        external_packages = self.filter_external_packages(all_imports)

        # Resolve to package names
        package_names = self.resolve_package_names(external_packages)

        # Generate requirements list
        requirements = []
        for package in sorted(package_names):
            if include_versions:
                version = self.get_installed_version(package)
                if version and version != "unknown":
                    requirements.append(f"{package}=={version}")
                else:
                    requirements.append(package)
            else:
                requirements.append(package)

        return requirements

    def save_requirements(self, requirements: list[str], output_path: str = "requirements.txt"):
        """Save requirements to a file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated requirements file\n")
            f.write("# Contains only packages needed for model execution\n")
            f.write("\n")
            for req in requirements:
                f.write(f"{req}\n")

        print(f"Requirements saved to {output_path}")
        print(f"Found {len(requirements)} dependencies:")
        for req in requirements:
            print(f"  - {req}")


def analyze_code_paths(code_paths: list[str], output_path: str = None) -> list[str]:
    """
    Analyze code paths and generate requirements for MLflow model.

    This is the main entry point for analyzing MLflow code_paths and generating
    a minimal requirements.txt file.

    Args:
        code_paths: List of file/directory paths to analyze
        output_path: Path to save requirements.txt (optional)

    Returns:
        List of requirement strings
    """
    analyzer = RequirementsAnalyzer()

    files = []
    directories = []

    for path in code_paths:
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            directories.append(path)
        else:
            print(f"Warning: Path does not exist: {path}")

    requirements = analyzer.generate_requirements(file_paths=files, directory_paths=directories, include_versions=True)

    if output_path:
        analyzer.save_requirements(requirements, output_path)

    return requirements


# Example usage functions
def example_analyze_current_project():
    """Example: Analyze the current project."""
    print("🔍 Analyzing current project...")

    # Analyze the projects directory
    code_paths = ["projects/my_model", "projects/shared_utils"]

    requirements = analyze_code_paths(code_paths, "model_requirements.txt")

    print(f"\n✅ Generated requirements with {len(requirements)} packages")
    return requirements


if __name__ == "__main__":
    example_analyze_current_project()

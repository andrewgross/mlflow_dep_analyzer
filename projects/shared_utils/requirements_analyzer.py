"""
Smart requirements analysis for MLflow models.

This module provides utilities to analyze Python code and generate minimal requirements.txt
files that include only the dependencies actually needed to run the model.

Uses a combination of:
1. AST-based static analysis (safe, no code execution)
2. MLflow-style package detection using importlib_metadata
3. Support for filtering against existing requirements
"""

import ast
import importlib.metadata
import importlib.util
import os
from pathlib import Path


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

    def __init__(self, existing_requirements: list[str] = None):
        """
        Initialize the analyzer.

        Args:
            existing_requirements: List of already-installed packages to exclude
        """
        self.stdlib_modules = self._get_stdlib_modules()
        self.custom_package_mapping = self._get_custom_package_mapping()
        self.existing_requirements = set(existing_requirements or [])
        self._packages_to_modules = self._get_packages_to_modules_mapping()
        self._modules_to_packages = self._get_modules_to_packages_mapping()

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
            "traceback",  # Missing from original list
        }
        return stdlib

    def _get_custom_package_mapping(self) -> dict[str, str]:
        """Get mapping from import names to package names for special cases."""
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
            # Add MLflow-style mappings
            "databricks": "databricks-sdk",
        }
        return mapping

    def _get_packages_to_modules_mapping(self) -> dict[str, list[str]]:
        """Get mapping from package names to their provided modules using importlib.metadata."""
        try:
            from importlib.metadata import packages_distributions

            return packages_distributions()
        except ImportError:
            # Fallback for older Python versions
            try:
                import pkg_resources

                mapping = {}
                for dist in pkg_resources.working_set:
                    if dist.has_metadata("top_level.txt"):
                        modules = dist.get_metadata("top_level.txt").split()
                        for module in modules:
                            if module not in mapping:
                                mapping[module] = []
                            mapping[module].append(dist.project_name)
                return mapping
            except Exception:
                return {}

    def _get_modules_to_packages_mapping(self) -> dict[str, list[str]]:
        """Reverse mapping: modules to packages."""
        modules_to_packages = {}
        for module, packages in self._packages_to_modules.items():
            if module not in modules_to_packages:
                modules_to_packages[module] = []
            modules_to_packages[module].extend(packages)
        return modules_to_packages

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
        """Filter out stdlib modules and local modules using MLflow-style approach."""
        external = set()

        # MLflow-style filtering rules
        excluded_patterns = {
            # Skip private/internal modules (MLflow approach)
            lambda m: m.startswith("_"),
            # Skip empty or relative imports
            lambda m: not m or m.startswith("."),
            # Skip known local patterns
            lambda m: m.startswith("projects"),
            lambda m: m.startswith("shared_utils"),
            lambda m: m.startswith("text_utils"),
            lambda m: m.startswith("validation"),
            lambda m: m.startswith("constants"),
            lambda m: m.startswith("databricks") and not self._is_external_databricks(m),
            lambda m: m.startswith("my_model"),
            lambda m: m.startswith("inference"),
        }

        for module in imports:
            # Apply MLflow-style filtering
            if any(pattern(module) for pattern in excluded_patterns):
                continue

            # Apply stdlib filtering (fallback for AST analysis)
            if module in self.stdlib_modules:
                continue

            external.add(module)

        return external

    def _is_external_databricks(self, module: str) -> bool:
        """Check if a databricks module is external (like databricks-sdk)."""
        # External databricks packages that should be included
        external_databricks = {"databricks.sdk", "databricks.cli", "databricks.connect"}
        return any(module.startswith(ext) for ext in external_databricks)

    def resolve_package_names(self, imports: set[str]) -> set[str]:
        """Resolve import names to actual package names using MLflow-style approach."""
        packages = set()

        for import_name in imports:
            resolved_packages = self._resolve_single_import(import_name)
            packages.update(resolved_packages)

        # Apply MLflow-style post-filtering
        packages = self._apply_mlflow_style_filtering(packages)
        return packages

    def _apply_mlflow_style_filtering(self, packages: set[str]) -> set[str]:
        """Apply MLflow-style package filtering after resolution."""
        # MLflow excludes these packages
        mlflow_excluded = {
            "setuptools",
            "pip",
            "wheel",
            "distutils",
            "pkg-resources",  # Often a false positive
        }

        # Also exclude MLflow itself and variants
        mlflow_packages = self._modules_to_packages.get("mlflow", [])
        mlflow_excluded.update(mlflow_packages)
        mlflow_excluded.update(["mlflow-skinny", "mlflow"])

        # Filter out excluded packages
        filtered = {pkg for pkg in packages if pkg.lower() not in {p.lower() for p in mlflow_excluded}}

        return filtered

    def _resolve_single_import(self, import_name: str) -> list[str]:
        """Resolve a single import name to package names."""
        # 1. Check custom mappings first
        if import_name in self.custom_package_mapping:
            return [self.custom_package_mapping[import_name]]

        # 2. Check importlib.metadata mapping
        if import_name in self._modules_to_packages:
            return self._modules_to_packages[import_name]

        # 3. Try to find the package through importlib
        try:
            spec = importlib.util.find_spec(import_name)
            if spec and spec.origin:
                # Try to find which package this module belongs to
                for dist in importlib.metadata.distributions():
                    if dist.files:
                        for file in dist.files:
                            if str(file).startswith(import_name.replace(".", "/")):
                                return [dist.metadata["name"]]
        except Exception:
            pass

        # 4. Last resort: assume import name is package name
        return [import_name]

    def get_installed_version(self, package_name: str) -> str:
        """Get the currently installed version of a package using importlib.metadata."""
        try:
            # Normalize package name (PEP 503)
            normalized_name = self._normalize_package_name(package_name)

            # Try importlib.metadata first (preferred for Python 3.8+)
            try:
                return importlib.metadata.version(normalized_name)
            except importlib.metadata.PackageNotFoundError:
                pass

            # Try the original name
            try:
                return importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                pass

            # Fallback: try to find by checking all distributions
            for dist in importlib.metadata.distributions():
                if self._normalize_package_name(dist.metadata["name"]) == normalized_name:
                    return dist.version

            return None

        except Exception:
            return None

    def _normalize_package_name(self, name: str) -> str:
        """Normalize package name according to PEP 503."""
        import re

        return re.sub(r"[-_.]+", "-", name).lower()

    def generate_requirements(
        self,
        file_paths: list[str] = None,
        directory_paths: list[str] = None,
        include_versions: bool = True,
        exclude_existing: bool = True,
    ) -> list[str]:
        """Generate requirements list from files or directories.

        Args:
            file_paths: List of Python files to analyze
            directory_paths: List of directories to analyze recursively
            include_versions: Whether to pin to specific versions
            exclude_existing: Whether to exclude packages in existing_requirements
        """
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

        # Remove existing packages if requested
        if exclude_existing and self.existing_requirements:
            normalized_existing = {
                self._normalize_package_name(pkg.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0])
                for pkg in self.existing_requirements
            }

            # Special handling for MLflow packages - if mlflow is in requirements, exclude mlflow-skinny too
            if "mlflow" in normalized_existing:
                normalized_existing.add("mlflow-skinny")

            package_names = {
                pkg for pkg in package_names if self._normalize_package_name(pkg) not in normalized_existing
            }

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


def find_local_imports(file_path: str, repo_root: str) -> set[str]:
    """
    Find all local imports in a Python file.

    This is an improved version that detects local project imports more accurately.

    Args:
        file_path: Path to the Python file to analyze
        repo_root: Root directory of the repository

    Returns:
        Set of local import names
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return set()

    local_imports = set()
    repo_name = Path(repo_root).name

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                # Check if it's a local import
                if (
                    module_name.startswith(repo_name)
                    or module_name.startswith("projects")
                    or "." not in module_name.split(".")[0]
                ):  # Top-level single word modules might be local
                    local_imports.add(module_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Relative imports (. or ..)
                if node.level > 0:
                    local_imports.add(node.module or "")
                # Absolute imports that start with repo name or known local patterns
                elif node.module.startswith(repo_name) or node.module.startswith("projects"):
                    local_imports.add(node.module)

    return local_imports


def collect_dependencies(model_file: str, repo_root: str) -> set[str]:
    """
    Recursively collect all dependencies from a model file.

    This follows import chains to find all local dependencies.

    Args:
        model_file: Path to the main model file
        repo_root: Root directory of the repository

    Returns:
        Set of all local dependency module names
    """
    dependencies = set()
    to_process = {model_file}
    processed = set()

    while to_process:
        current = to_process.pop()
        if current in processed:
            continue
        processed.add(current)

        imports = find_local_imports(current, repo_root)
        dependencies.update(imports)

        # Find corresponding files for imports
        for imp in imports:
            # Convert module path to file path
            imp_path = Path(repo_root) / imp.replace(".", "/") / "__init__.py"
            if imp_path.exists():
                to_process.add(str(imp_path))
            else:
                # Try as a direct .py file
                imp_file = Path(repo_root) / (imp.replace(".", "/") + ".py")
                if imp_file.exists():
                    to_process.add(str(imp_file))

    return dependencies


def load_requirements_from_file(requirements_file: str) -> list[str]:
    """
    Load requirements from a requirements.txt file.

    Args:
        requirements_file: Path to requirements.txt file

    Returns:
        List of requirement strings (comments and empty lines removed)
    """
    if not os.path.exists(requirements_file):
        return []

    requirements = []
    with open(requirements_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                requirements.append(line)

    return requirements


def analyze_code_paths_mlflow_style(imports: set[str]) -> list[str]:
    """
    Analyze imports using MLflow's exact approach.

    This demonstrates how MLflow would handle the same input.
    MLflow's key insight: Only modules that appear in importlib_metadata
    are real packages that need to be installed.
    """
    try:
        from importlib.metadata import packages_distributions

        modules_to_packages = packages_distributions()
    except ImportError:
        # Fallback for older Python
        modules_to_packages = {}

    # MLflow's core logic: Convert modules to packages
    packages = []
    for module in imports:
        # Skip private modules (MLflow approach)
        if module.startswith("_"):
            continue

        # Get packages for this module (empty list if not found)
        module_packages = modules_to_packages.get(module, [])
        packages.extend(module_packages)

    # MLflow's exclusion list
    excluded = {"setuptools", "mlflow", "mlflow-skinny", "pip", "wheel"}

    # Filter and deduplicate
    result = sorted(set(packages) - excluded)

    return result


def analyze_code_paths(
    code_paths: list[str],
    output_path: str = None,
    existing_requirements: list[str] = None,
    existing_requirements_file: str = None,
) -> list[str]:
    """
    Analyze code paths and generate requirements for MLflow model.

    This is the main entry point for analyzing MLflow code_paths and generating
    a minimal requirements.txt file.

    Args:
        code_paths: List of file/directory paths to analyze
        output_path: Path to save requirements.txt (optional)
        existing_requirements: List of already-installed packages to exclude
        existing_requirements_file: Path to requirements.txt with existing packages

    Returns:
        List of requirement strings
    """
    # Load existing requirements from file if provided
    if existing_requirements_file and not existing_requirements:
        existing_requirements = load_requirements_from_file(existing_requirements_file)

    analyzer = RequirementsAnalyzer(existing_requirements=existing_requirements)

    files = []
    directories = []

    for path in code_paths:
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            directories.append(path)
        else:
            print(f"Warning: Path does not exist: {path}")

    requirements = analyzer.generate_requirements(
        file_paths=files,
        directory_paths=directories,
        include_versions=True,
        exclude_existing=bool(existing_requirements),
    )

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

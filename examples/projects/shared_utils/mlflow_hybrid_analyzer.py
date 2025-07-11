"""
Hybrid Requirements Analyzer combining MLflow's production utilities with AST analysis.

This module provides the best of both worlds:
1. MLflow's battle-tested package detection and filtering
2. Our safe AST-based import analysis for precise dependency pruning

Key advantages:
- Uses MLflow's robust module-to-package mapping
- Leverages MLflow's dependency pruning logic
- Incorporates MLflow's PyPI validation
- Maintains AST analysis safety (no code execution)
- Provides precise model-specific requirements
"""

import ast
import os
from pathlib import Path

try:
    # Import MLflow's production utilities
    from mlflow.utils.requirements_utils import (
        _MODULES_TO_PACKAGES,
        _get_pinned_requirement,
        _init_modules_to_packages_map,
        _load_pypi_package_index,
        _normalize_package_name,
        _prune_packages,
    )

    MLFLOW_AVAILABLE = True
except ImportError:
    # Fallback if MLflow not available or version incompatible
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow utilities not available, using fallback implementation")


class MLflowHybridAnalyzer:
    """
    Hybrid analyzer combining MLflow's package detection with AST-based import analysis.

    This provides maximum accuracy and safety by:
    1. Using AST to discover all imports (safe, no execution)
    2. Using MLflow's proven package resolution logic
    3. Applying MLflow's dependency pruning and validation
    """

    def __init__(self, existing_requirements: list[str] = None):
        """Initialize the hybrid analyzer."""
        self.existing_requirements = set(existing_requirements or [])
        self._pypi_index = None

        if MLFLOW_AVAILABLE:
            try:
                # Initialize MLflow's module-to-package mapping
                _init_modules_to_packages_map()
                # Verify it was initialized properly
                if _MODULES_TO_PACKAGES is None:
                    print("Warning: MLflow mapping not initialized, using fallback")
                    self._init_fallback_mapping()
                    self._use_mlflow_mapping = False
                else:
                    self._use_mlflow_mapping = True
            except Exception as e:
                print(f"Warning: MLflow initialization failed: {e}, using fallback")
                self._init_fallback_mapping()
                self._use_mlflow_mapping = False
        else:
            # Fallback to importlib.metadata
            self._init_fallback_mapping()
            self._use_mlflow_mapping = False

    def _init_fallback_mapping(self):
        """Fallback package mapping if MLflow not available."""
        try:
            from importlib.metadata import packages_distributions

            self._modules_to_packages = packages_distributions()
        except ImportError:
            self._modules_to_packages = {}

        # Initialize stdlib detection for fallback filtering
        self._stdlib_modules = self._get_stdlib_modules()

    def _get_stdlib_modules(self) -> set:
        """Get stdlib modules for fallback filtering."""
        try:
            import sys

            if hasattr(sys, "stdlib_module_names"):
                return set(sys.stdlib_module_names)
        except Exception:
            pass

        # Minimal stdlib list for fallback
        return {
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
            "traceback",
        }

    def analyze_file(self, file_path: str) -> set[str]:
        """Extract all imports from a Python file using AST."""
        try:
            with open(file_path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return set()

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])  # Get top-level module
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # Only absolute imports
                    imports.add(node.module.split(".")[0])

        return imports

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

    def filter_local_modules(self, imports: set[str], repo_root: str = None) -> set[str]:
        """Filter out local project modules using MLflow-style rules."""
        filtered = set()
        repo_name = Path(repo_root).name if repo_root else ""

        for module in imports:
            # Apply MLflow-style filtering
            if self._should_exclude_module(module, repo_name):
                continue

            # Apply stdlib filtering in fallback mode
            if not self._use_mlflow_mapping and hasattr(self, "_stdlib_modules"):
                if module in self._stdlib_modules:
                    continue

            filtered.add(module)

        return filtered

    def _should_exclude_module(self, module: str, repo_name: str = "") -> bool:
        """Determine if a module should be excluded using MLflow-style rules."""
        # MLflow excludes private modules
        if module.startswith("_"):
            return True

        # Skip empty or relative imports
        if not module or module.startswith("."):
            return True

        # Skip known local patterns
        local_patterns = {
            "projects",
            "shared_utils",
            "text_utils",
            "validation",
            "constants",
            "my_model",
            "inference",
            repo_name,
        }

        if module in local_patterns or any(module.startswith(p + ".") for p in local_patterns):
            return True

        # Special handling for databricks (allow external databricks packages)
        if module.startswith("databricks"):
            external_databricks = {"databricks.sdk", "databricks.cli", "databricks.connect"}
            return not any(module.startswith(ext) for ext in external_databricks)

        return False

    def resolve_packages_mlflow_style(self, modules: set[str]) -> set[str]:
        """Convert modules to packages using MLflow's approach."""
        packages = set()

        if self._use_mlflow_mapping and _MODULES_TO_PACKAGES is not None:
            # Use MLflow's mapping
            for module in modules:
                module_packages = _MODULES_TO_PACKAGES.get(module, [])
                packages.update(module_packages)
        else:
            # Fallback mapping
            for module in modules:
                module_packages = self._modules_to_packages.get(module, [module])
                if isinstance(module_packages, list):
                    packages.update(module_packages)
                else:
                    packages.add(module_packages)

        return packages

    def apply_mlflow_filtering(self, packages: set[str]) -> set[str]:
        """Apply MLflow's production filtering rules."""
        excluded_packages = ["setuptools"]  # Base exclusions

        if self._use_mlflow_mapping and _MODULES_TO_PACKAGES is not None:
            # Add MLflow variants from mapping
            mlflow_packages = _MODULES_TO_PACKAGES.get("mlflow", [])
            excluded_packages.extend(mlflow_packages)

        # Additional development packages to exclude
        excluded_packages.extend(["pip", "wheel", "distutils", "pkg-resources"])

        # Filter out excluded packages (case-insensitive)
        excluded_lower = {pkg.lower() for pkg in excluded_packages}
        filtered = {pkg for pkg in packages if pkg.lower() not in excluded_lower}

        return filtered

    def prune_dependencies_mlflow_style(self, packages: set[str]) -> set[str]:
        """Apply MLflow's dependency pruning logic."""
        if not self._use_mlflow_mapping or not MLFLOW_AVAILABLE:
            return packages

        try:
            # Use MLflow's battle-tested pruning logic
            return set(_prune_packages(packages))
        except Exception as e:
            print(f"Warning: MLflow pruning failed: {e}")
            return packages

    def validate_against_pypi(self, packages: set[str]) -> tuple[set[str], set[str]]:
        """Validate packages against PyPI index using MLflow's approach."""
        if not self._use_mlflow_mapping or not MLFLOW_AVAILABLE:
            return packages, set()

        try:
            if self._pypi_index is None:
                self._pypi_index = _load_pypi_package_index()

            # Split into recognized and unrecognized packages
            recognized = packages & self._pypi_index.package_names
            unrecognized = packages - recognized

            # MLflow allows certain special packages
            special_packages = {"mlflow[gateway]"}
            unrecognized = unrecognized - special_packages
            recognized = recognized | (packages & special_packages)

            return recognized, unrecognized
        except Exception as e:
            print(f"Warning: PyPI validation failed: {e}")
            return packages, set()

    def generate_pinned_requirements(self, packages: set[str]) -> list[str]:
        """Generate pinned requirements using MLflow's utilities."""
        requirements = []

        for package in sorted(packages):
            try:
                if self._use_mlflow_mapping and MLFLOW_AVAILABLE:
                    # Use MLflow's pinning logic
                    req = _get_pinned_requirement(package)
                    requirements.append(req)
                else:
                    # Fallback: try to get version manually
                    try:
                        import importlib.metadata

                        version = importlib.metadata.version(package)
                        requirements.append(f"{package}=={version}")
                    except Exception:
                        requirements.append(package)
            except Exception as e:
                print(f"Warning: Could not pin {package}: {e}")
                requirements.append(package)

        return requirements

    def exclude_existing_requirements(self, packages: set[str]) -> set[str]:
        """Exclude packages that are already in existing requirements."""
        if not self.existing_requirements:
            return packages

        # Normalize existing requirements (remove version constraints)
        normalized_existing = set()
        for req in self.existing_requirements:
            # Remove version constraints
            pkg_name = req.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
            if self._use_mlflow_mapping and MLFLOW_AVAILABLE:
                pkg_name = _normalize_package_name(pkg_name)
            else:
                pkg_name = pkg_name.lower().replace("_", "-")
            normalized_existing.add(pkg_name)

        # Special handling for MLflow packages - if mlflow is in requirements, exclude mlflow-skinny too
        if "mlflow" in normalized_existing:
            normalized_existing.add("mlflow-skinny")

        # Filter out existing packages
        filtered = set()
        for pkg in packages:
            normalized_pkg = (
                _normalize_package_name(pkg)
                if self._use_mlflow_mapping and MLFLOW_AVAILABLE
                else pkg.lower().replace("_", "-")
            )
            if normalized_pkg not in normalized_existing:
                filtered.add(pkg)

        return filtered

    def analyze_model_requirements(
        self, code_paths: list[str], repo_root: str = None, exclude_existing: bool = True
    ) -> dict:
        """
        Complete analysis using MLflow-style approach with AST discovery.

        Returns detailed analysis results including intermediate steps.
        """
        # Step 1: Discover all imports using AST (safe)
        all_imports = set()
        files_analyzed = []
        directories_analyzed = []

        for path in code_paths:
            if os.path.isfile(path):
                imports = self.analyze_file(path)
                all_imports.update(imports)
                files_analyzed.append(path)
            elif os.path.isdir(path):
                imports = self.analyze_directory(path)
                all_imports.update(imports)
                directories_analyzed.append(path)

        # Step 2: Filter out local modules
        external_modules = self.filter_local_modules(all_imports, repo_root)

        # Step 3: Convert modules to packages (MLflow-style)
        packages = self.resolve_packages_mlflow_style(external_modules)

        # Step 4: Apply MLflow's filtering rules
        filtered_packages = self.apply_mlflow_filtering(packages)

        # Step 5: Prune dependencies (MLflow-style)
        pruned_packages = self.prune_dependencies_mlflow_style(filtered_packages)

        # Step 6: Validate against PyPI (if available)
        validated_packages, unrecognized = self.validate_against_pypi(pruned_packages)

        # Step 7: Exclude existing requirements if requested
        if exclude_existing:
            final_packages = self.exclude_existing_requirements(validated_packages)
        else:
            final_packages = validated_packages

        # Step 8: Generate pinned requirements
        requirements = self.generate_pinned_requirements(final_packages)

        # Return comprehensive analysis
        return {
            "requirements": requirements,
            "analysis": {
                "files_analyzed": files_analyzed,
                "directories_analyzed": directories_analyzed,
                "raw_imports": sorted(all_imports),
                "external_modules": sorted(external_modules),
                "resolved_packages": sorted(packages),
                "filtered_packages": sorted(filtered_packages),
                "pruned_packages": sorted(pruned_packages),
                "validated_packages": sorted(validated_packages),
                "unrecognized_packages": sorted(unrecognized),
                "final_packages": sorted(final_packages),
                "excluded_existing": len(packages) - len(final_packages) if exclude_existing else 0,
                "mlflow_available": MLFLOW_AVAILABLE,
            },
        }


def analyze_model_requirements_hybrid(
    code_paths: list[str], existing_requirements: list[str] = None, repo_root: str = None, exclude_existing: bool = True
) -> list[str]:
    """
    Convenience function for hybrid MLflow + AST requirements analysis.

    Args:
        code_paths: List of Python files/directories to analyze
        existing_requirements: List of already-installed packages to exclude
        repo_root: Root directory of the repository (for local module detection)
        exclude_existing: Whether to exclude existing requirements

    Returns:
        List of pinned requirements needed for the model
    """
    analyzer = MLflowHybridAnalyzer(existing_requirements=existing_requirements)
    result = analyzer.analyze_model_requirements(
        code_paths=code_paths, repo_root=repo_root, exclude_existing=exclude_existing
    )
    return result["requirements"]


# Example usage
if __name__ == "__main__":
    # Test the hybrid analyzer
    analyzer = MLflowHybridAnalyzer()

    # Analyze the auto-logging model
    result = analyzer.analyze_model_requirements(
        code_paths=["../my_model/auto_logging_sentiment_model.py", "../shared_utils"],
        repo_root="../../..",
        exclude_existing=True,
    )

    print("🔬 Hybrid MLflow + AST Analysis Results")
    print("=" * 40)
    print(f"📦 Final requirements: {len(result['requirements'])}")
    for req in result["requirements"]:
        print(f"  • {req}")

    print("\n📊 Analysis Details:")
    analysis = result["analysis"]
    print(f"  • Raw imports found: {len(analysis['raw_imports'])}")
    print(f"  • External modules: {len(analysis['external_modules'])}")
    print(f"  • Final packages: {len(analysis['final_packages'])}")
    print(f"  • MLflow utilities: {'✅' if analysis['mlflow_available'] else '❌'}")

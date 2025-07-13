"""
Unified Dependency Analyzer for MLflow Models

This module provides a unified approach to analyzing Python code dependencies,
combining both requirements (external packages) and code paths (local files)
analysis into a single cohesive flow.

The analyzer:
1. Uses AST to find imports without code execution
2. Uses inspect module to get actual file paths from imported modules
3. Classifies dependencies into: external packages, stdlib modules, local files
4. Recursively discovers all dependencies
5. Returns both requirements and code paths in a unified result
"""

import ast
import importlib
import inspect
import os
import sys
from pathlib import Path

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import PackageNotFoundError, version  # type: ignore


class DependencyType:
    """Enumeration of dependency types."""

    EXTERNAL_PACKAGE = "external_package"
    STDLIB_MODULE = "stdlib_module"
    LOCAL_FILE = "local_file"


class ModuleInfo:
    """Information about a discovered module."""

    def __init__(self, name: str, dep_type: str, file_path: str | None = None):
        self.name = name
        self.dep_type = dep_type
        self.file_path = file_path

    def __repr__(self):
        return f"ModuleInfo(name='{self.name}', type='{self.dep_type}', path='{self.file_path}')"


class UnifiedDependencyAnalyzer:
    """
    Unified analyzer for determining both requirements and code paths.

    This analyzer uses Python's introspection capabilities to accurately
    determine what files and packages are actually needed by a model.
    """

    def __init__(self, repo_root: str):
        """
        Initialize the unified dependency analyzer.

        Args:
            repo_root: Root directory of the repository
        """
        self.repo_root = Path(repo_root).resolve()
        self._stdlib_modules = self._get_stdlib_modules()

    def analyze_dependencies(self, entry_files: list[str]) -> dict:
        """
        Analyze all dependencies for the given entry files.

        Args:
            entry_files: List of Python files to analyze

        Returns:
            Dictionary containing:
            - requirements: List of external package requirements
            - code_paths: List of relative paths to local files
            - analysis: Analysis metadata
        """
        all_modules: dict[str, ModuleInfo] = {}
        processed_files: set[str] = set()

        # Process all entry files
        self._process_entry_files(entry_files, all_modules, processed_files)

        # Separate modules by type
        external_packages, local_files = self._categorize_modules(all_modules)

        # Convert local files to relative paths
        relative_code_paths = self._convert_to_relative_paths(local_files)

        # Build final result
        return self._build_analysis_result(entry_files, all_modules, external_packages, relative_code_paths)

    def _process_entry_files(
        self, entry_files: list[str], all_modules: dict[str, ModuleInfo], processed_files: set[str]
    ) -> None:
        """Process all entry files and discover their dependencies."""
        for entry_file in entry_files:
            if not os.path.exists(entry_file):
                print(f"Warning: Entry file does not exist: {entry_file}")
                continue
            self._discover_dependencies_recursive(entry_file, all_modules, processed_files)

    def _categorize_modules(self, all_modules: dict[str, ModuleInfo]) -> tuple[set[str], set[str]]:
        """Categorize modules into external packages and local files."""
        external_packages = set()
        local_files = set()

        for module_info in all_modules.values():
            if module_info.dep_type == DependencyType.EXTERNAL_PACKAGE:
                package_name = self._extract_package_name(module_info.name)
                if package_name:
                    external_packages.add(package_name)
            elif module_info.dep_type == DependencyType.LOCAL_FILE and module_info.file_path:
                if self._is_file_in_repo(module_info.file_path):
                    local_files.add(module_info.file_path)

        return external_packages, local_files

    def _extract_package_name(self, module_name: str) -> str | None:
        """Extract top-level package name from module name and map to actual package name."""
        import_name = module_name.split(".")[0]
        # Filter out empty or problematic package names
        if not import_name or import_name in {"", "_", "__", "test", "tests"}:
            return None

        # Map import name to actual package name
        actual_package_name = self._map_import_to_package_name(import_name)
        return actual_package_name

    def _map_import_to_package_name(self, import_name: str) -> str | None:
        """
        Map an import name to the actual package name using importlib.metadata.

        This handles cases where the import name differs from the package name,
        such as 'sklearn' -> 'scikit-learn', 'cv2' -> 'opencv-python', etc.
        """
        try:
            # Try to find the distribution that provides this import name
            package_name = self._find_distribution_for_import(import_name)
            if package_name:
                return package_name

        except Exception:
            # Any error, continue with fallback
            pass

        # Fallback to import name if we can't determine the package name
        return import_name

    def _find_distribution_for_import(self, import_name: str) -> str | None:
        """Find the distribution name that provides the given import name."""
        try:
            # Cache for performance - only build once per analyzer instance
            if not hasattr(self, "_import_to_package_cache"):
                self._import_to_package_cache = self._build_import_to_package_mapping()

            return self._import_to_package_cache.get(import_name)

        except Exception:
            return None

    def _build_import_to_package_mapping(self) -> dict[str, str]:
        """Build a mapping from import names to package names by scanning all distributions."""
        mapping = {}
        conflicts = {}  # Track multiple packages providing the same import

        try:
            from importlib.metadata import distributions

            # First pass: collect all mappings and track conflicts
            for dist in distributions():
                dist_name = dist.metadata["Name"]

                # Get top-level imports for this distribution
                top_level_imports = self._get_top_level_imports_for_dist(dist)

                # Map each import to this distribution
                for import_name in top_level_imports:
                    if import_name not in conflicts:
                        conflicts[import_name] = []
                    conflicts[import_name].append(dist_name)

                # Also map the package name to itself (for cases like 'requests' -> 'requests')
                normalized_dist_name = dist_name.lower().replace("-", "_").replace(".", "_")
                if normalized_dist_name not in conflicts:
                    conflicts[normalized_dist_name] = []
                conflicts[normalized_dist_name].append(dist_name)

            # Second pass: resolve conflicts using priority rules
            for import_name, candidates in conflicts.items():
                if len(candidates) == 1:
                    mapping[import_name] = candidates[0]
                else:
                    # Multiple candidates - choose the best one
                    mapping[import_name] = self._resolve_package_conflict(import_name, candidates)

        except Exception:
            pass

        return mapping

    def _resolve_package_conflict(self, import_name: str, candidates: list[str]) -> str:
        """Resolve conflicts by determining which package would actually be imported."""
        try:
            # Try using packages_distributions() first - this is most reliable
            try:
                import importlib.metadata

                pkg_to_dist = importlib.metadata.packages_distributions()
                if import_name in pkg_to_dist:
                    # Get all distributions that provide this import
                    providing_dists = pkg_to_dist[import_name]
                    # Return the first one that's in our candidates list
                    for dist_name in providing_dists:
                        if dist_name in candidates:
                            return dist_name
            except Exception:
                pass

            # Fallback: check which package Python would actually import from
            import importlib.util

            spec = importlib.util.find_spec(import_name)
            if spec and (spec.submodule_search_locations or spec.origin):
                # Check which candidate distribution has the most files for this import
                best_match = None
                best_count = 0

                for candidate in candidates:
                    try:
                        dist = importlib.metadata.distribution(candidate)
                        if dist.files:
                            count = sum(1 for f in dist.files if f.parts and f.parts[0] == import_name)
                            if count > best_count:
                                best_count = count
                                best_match = candidate
                    except Exception:
                        continue

                if best_match:
                    return best_match

            # Final fallback: return first candidate
            return candidates[0]

        except Exception:
            return candidates[0]

    def _get_top_level_imports_for_dist(self, dist) -> set[str]:
        """Get the top-level import names provided by a distribution."""
        top_level_imports = set()

        try:
            # Method 1: Try to read top_level.txt metadata
            try:
                top_level_txt = dist.read_text("top_level.txt")
                if top_level_txt:
                    for line in top_level_txt.strip().split("\n"):
                        if line.strip():
                            top_level_imports.add(line.strip())
            except Exception:
                pass

            # Method 2: If no top_level.txt, infer from files
            if not top_level_imports and hasattr(dist, "files") and dist.files:
                for file in dist.files:
                    if file.suffix in (".py", ".so", ".pyd") or file.suffix == "":
                        parts = str(file).replace("\\", "/").split("/")
                        if parts[0] and not parts[0].endswith(".dist-info"):
                            # Extract top-level directory/module name
                            name = parts[0]
                            if "." in name:
                                name = name.split(".")[0]
                            if name and not name.startswith("_") and name.isidentifier():
                                top_level_imports.add(name)

        except Exception:
            pass

        return top_level_imports

    def _get_package_version(self, package_name: str) -> str | None:
        """Get the version of an installed package."""
        try:
            return version(package_name)
        except PackageNotFoundError:
            return None
        except Exception:
            # Handle any other unexpected errors
            return None

    def _format_requirement_with_version(self, package_name: str) -> str:
        """Format a package requirement with version if available."""
        package_version = self._get_package_version(package_name)
        if package_version:
            return f"{package_name}=={package_version}"
        else:
            return package_name

    def _convert_to_relative_paths(self, local_files: set[str]) -> list[str]:
        """Convert absolute file paths to relative paths for MLflow."""
        relative_code_paths = []
        for file_path in local_files:
            try:
                rel_path = os.path.relpath(file_path, self.repo_root)
                if not rel_path.startswith(".."):  # Only include files within repo
                    relative_code_paths.append(rel_path)
            except ValueError:
                # Path is on different drive (Windows)
                pass
        return relative_code_paths

    def _build_analysis_result(
        self,
        entry_files: list[str],
        all_modules: dict[str, ModuleInfo],
        external_packages: set[str],
        relative_code_paths: list[str],
    ) -> dict:
        """Build the final analysis result dictionary."""
        # Format requirements with versions
        versioned_requirements = [self._format_requirement_with_version(pkg) for pkg in external_packages]

        return {
            "requirements": sorted(versioned_requirements),
            "code_paths": sorted(relative_code_paths),
            "analysis": {
                "total_modules": len(all_modules),
                "external_packages": len(external_packages),
                "local_files": len([m for m in all_modules.values() if m.dep_type == DependencyType.LOCAL_FILE]),
                "stdlib_modules": len([m for m in all_modules.values() if m.dep_type == DependencyType.STDLIB_MODULE]),
                "entry_files": entry_files,
            },
            "detailed_modules": all_modules,  # For debugging/advanced use
        }

    def _discover_dependencies_recursive(
        self, file_path: str, all_modules: dict[str, ModuleInfo], processed_files: set[str]
    ) -> None:
        """
        Recursively discover all dependencies starting from a file.

        Args:
            file_path: Python file to analyze
            all_modules: Dictionary to store discovered modules
            processed_files: Set of already processed file paths
        """
        file_path = str(Path(file_path).resolve())

        if file_path in processed_files:
            return
        processed_files.add(file_path)

        # Add the file itself as a local dependency
        if self._is_file_in_repo(file_path):
            rel_name = self._file_path_to_module_name(file_path)
            if rel_name:
                all_modules[rel_name] = ModuleInfo(rel_name, DependencyType.LOCAL_FILE, file_path)

        # Find imports in this file
        imports = self._extract_imports_from_file(file_path)

        # Process each import
        for import_name in imports:
            if import_name in all_modules:
                continue  # Already processed

            module_info = self._classify_and_resolve_module(import_name)
            if module_info:
                all_modules[import_name] = module_info

                # If it's a local file, recurse into it
                if module_info.dep_type == DependencyType.LOCAL_FILE and module_info.file_path:
                    self._discover_dependencies_recursive(module_info.file_path, all_modules, processed_files)

    def _extract_imports_from_file(self, file_path: str) -> set[str]:
        """
        Extract all import statements from a Python file using AST.

        Args:
            file_path: Path to the Python file

        Returns:
            Set of imported module names
        """
        try:
            tree = self._parse_python_file(file_path)
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return set()

        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._process_import_node(node, imports)
            elif isinstance(node, ast.ImportFrom):
                self._process_import_from_node(node, file_path, imports)

        return imports

    def _parse_python_file(self, file_path: str) -> ast.AST:
        """Parse a Python file and return its AST."""
        with open(file_path, encoding="utf-8") as f:
            return ast.parse(f.read())

    def _process_import_node(self, node: ast.Import, imports: set[str]) -> None:
        """Process a regular import node (import module)."""
        for alias in node.names:
            imports.add(alias.name)

    def _process_import_from_node(self, node: ast.ImportFrom, file_path: str, imports: set[str]) -> None:
        """Process an import-from node (from module import name)."""
        if node.module:
            if node.level > 0:
                # Relative import - resolve to absolute
                abs_module = self._resolve_relative_import(file_path, node.level, node.module)
                if abs_module:
                    imports.add(abs_module)
            else:
                # Absolute import
                imports.add(node.module)
        elif node.level > 0:
            # Relative import without module (from . import name)
            self._process_relative_import_without_module(node, file_path, imports)

    def _process_relative_import_without_module(self, node: ast.ImportFrom, file_path: str, imports: set[str]) -> None:
        """Process relative imports without explicit module (from . import name)."""
        base_module = self._resolve_relative_import(file_path, node.level, None)
        if base_module is not None:
            for alias in node.names:
                if base_module:
                    full_module = f"{base_module}.{alias.name}"
                else:
                    full_module = alias.name
                imports.add(full_module)

    def _classify_and_resolve_module(self, module_name: str) -> ModuleInfo | None:
        """
        Classify a module and resolve its file path using inspect.

        Args:
            module_name: Name of the module to classify

        Returns:
            ModuleInfo object or None if module cannot be resolved
        """
        # Check if it's a stdlib module first (fast check)
        if self._is_stdlib_module(module_name):
            return ModuleInfo(module_name, DependencyType.STDLIB_MODULE)

        # Try to import and classify the module
        return self._import_and_classify_module(module_name)

    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of the standard library."""
        top_level = module_name.split(".")[0]
        return top_level in self._stdlib_modules

    def _import_and_classify_module(self, module_name: str) -> ModuleInfo | None:
        """Import a module and classify it based on its file path."""
        original_path = sys.path.copy()
        try:
            self._setup_import_paths()

            try:
                if self._is_problematic_module_name(module_name):
                    raise ImportError(f"Skipping problematic module path: {module_name}")

                module = importlib.import_module(module_name)
                return self._classify_imported_module(module_name, module)

            except ImportError:
                return self._handle_import_error(module_name)

        finally:
            sys.path[:] = original_path

    def _setup_import_paths(self) -> None:
        """Add repository paths to sys.path for local module resolution."""
        repo_paths = [str(self.repo_root)]
        # Common Python project directory patterns
        for subdir in ["src", "lib", "packages"]:
            if (self.repo_root / subdir).exists():
                repo_paths.append(str(self.repo_root / subdir))

        for path in reversed(repo_paths):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _is_problematic_module_name(self, module_name: str) -> bool:
        """Check if module name contains problematic patterns."""
        if "." not in module_name:
            return False

        problematic_patterns = [".venv", "site-packages", "..", "__pycache__"]
        return any(pattern in module_name for pattern in problematic_patterns)

    def _classify_imported_module(self, module_name: str, module) -> ModuleInfo:
        """Classify a successfully imported module."""
        file_path = self._get_module_file_path(module)

        if file_path:
            if self._is_file_in_repo(file_path):
                return ModuleInfo(module_name, DependencyType.LOCAL_FILE, file_path)
            else:
                # Check if there's a local version in the current repo
                local_path = self._find_local_module_path(module_name)
                if local_path:
                    return ModuleInfo(module_name, DependencyType.LOCAL_FILE, local_path)
                else:
                    return ModuleInfo(module_name, DependencyType.EXTERNAL_PACKAGE, file_path)
        else:
            # No file path available (built-in module, etc.)
            if self._is_likely_stdlib(module):
                return ModuleInfo(module_name, DependencyType.STDLIB_MODULE)
            else:
                return ModuleInfo(module_name, DependencyType.EXTERNAL_PACKAGE)

    def _get_module_file_path(self, module) -> str | None:
        """Get the file path of an imported module."""
        try:
            return inspect.getsourcefile(module)
        except (TypeError, OSError):
            # Fallback to __file__ attribute
            if hasattr(module, "__file__") and module.__file__:
                return str(Path(module.__file__).resolve())
        return None

    def _handle_import_error(self, module_name: str) -> ModuleInfo | None:
        """Handle cases where module cannot be imported."""
        # Check if it exists locally before assuming it's an external package
        local_path = self._find_local_module_path(module_name)
        if local_path:
            return ModuleInfo(module_name, DependencyType.LOCAL_FILE, local_path)

        # Assume it's an external package
        return ModuleInfo(module_name, DependencyType.EXTERNAL_PACKAGE)

    def _resolve_relative_import(self, file_path: str, level: int, module: str | None) -> str | None:
        """
        Resolve a relative import to an absolute module name.

        Args:
            file_path: Path of the file containing the import
            level: Number of parent directories to go up
            module: Module name (if any)

        Returns:
            Absolute module name or None if cannot be resolved
        """
        try:
            target_dir = self._calculate_target_directory(file_path, level)
            if target_dir is None:
                return None

            return self._build_module_name_from_path(target_dir, module)

        except Exception:
            return None

    def _calculate_target_directory(self, file_path: str, level: int) -> Path | None:
        """Calculate the target directory by going up the specified number of levels."""
        file_obj = Path(file_path).resolve()
        current_dir = file_obj.parent

        # Go up 'level-1' directories (level 1 = current dir)
        for _ in range(level - 1):
            if current_dir == self.repo_root or current_dir == current_dir.parent:
                return None
            current_dir = current_dir.parent

        return current_dir

    def _build_module_name_from_path(self, target_dir: Path, module: str | None) -> str | None:
        """Build a module name from a directory path and optional module name."""
        try:
            relative_to_root = target_dir.relative_to(self.repo_root)
            if relative_to_root == Path("."):
                package_parts = []
            else:
                package_parts = list(relative_to_root.parts)

            # Handle src/ directory
            if package_parts and package_parts[0] == "src":
                package_parts = package_parts[1:]

            # Add module name if provided
            if module:
                package_parts.append(module)

            if package_parts:
                return ".".join(package_parts)
            else:
                return module or ""

        except ValueError:
            return None

    def _is_file_in_repo(self, file_path: str) -> bool:
        """Check if a file path is within the repository (excluding virtual envs and external packages)."""
        try:
            resolved_path = Path(file_path).resolve()
            resolved_str = str(resolved_path)
            repo_root_str = str(self.repo_root)

            # Must be under repo root
            if not resolved_str.startswith(repo_root_str):
                return False

            # Exclude virtual environments and package installations
            excluded_patterns = [
                "/.venv/",
                "/venv/",
                "/env/",
                "/site-packages/",
                "/dist-packages/",
                "/__pycache__/",
                "/.git/",
                "/node_modules/",
            ]

            for pattern in excluded_patterns:
                if pattern in resolved_str:
                    return False

            return True
        except (OSError, ValueError):
            return False

    def _file_path_to_module_name(self, file_path: str) -> str | None:
        """Convert a file path to a Python module name."""
        try:
            file_obj = Path(file_path)

            # Get relative path from repo root
            rel_path = file_obj.relative_to(self.repo_root)

            # Handle src/ directory
            parts = list(rel_path.parts)
            if parts and parts[0] == "src":
                parts = parts[1:]

            # Remove .py extension and convert to module name
            if parts:
                if parts[-1].endswith(".py"):
                    parts[-1] = parts[-1][:-3]
                elif parts[-1] == "__init__.py":
                    parts = parts[:-1]  # Package directory

                if parts:
                    return ".".join(parts)

            return None

        except (ValueError, Exception):
            return None

    def _get_stdlib_modules(self) -> set[str]:
        """Get set of Python standard library module names."""
        stdlib_modules: set[str] = set()

        # Built-in modules
        stdlib_modules.update(sys.builtin_module_names)

        # Try to detect stdlib modules from filesystem
        stdlib_modules.update(self._detect_stdlib_from_filesystem())

        # Add stdlib modules using sys.stdlib_module_names or fallback
        stdlib_modules.update(self._get_stdlib_module_names())

        return stdlib_modules

    def _detect_stdlib_from_filesystem(self) -> set[str]:
        """Detect stdlib modules by scanning the Python installation directory."""
        stdlib_modules: set[str] = set()

        try:
            import sysconfig

            stdlib_path = sysconfig.get_path("stdlib")
            if stdlib_path:
                stdlib_dir = Path(stdlib_path)
                if stdlib_dir.exists():
                    for py_file in stdlib_dir.glob("*.py"):
                        stdlib_modules.add(py_file.stem)
                    for pkg_dir in stdlib_dir.iterdir():
                        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                            stdlib_modules.add(pkg_dir.name)
        except Exception:
            pass

        return stdlib_modules

    def _get_stdlib_module_names(self) -> set[str]:
        """Get Python standard library module names (Python 3.11+ has sys.stdlib_module_names)."""
        # Python 3.11+ always has sys.stdlib_module_names
        return set(sys.stdlib_module_names)

    def _is_likely_stdlib(self, module) -> bool:
        """Check if a module is likely from the standard library."""
        try:
            if hasattr(module, "__file__") and module.__file__:
                file_path = Path(module.__file__)
                # Check if it's in Python's installation directory
                import sysconfig

                stdlib_path = sysconfig.get_path("stdlib")
                if stdlib_path and str(file_path).startswith(stdlib_path):
                    return True

            # Check if it's a built-in module
            if hasattr(module, "__name__"):
                return module.__name__ in sys.builtin_module_names

        except Exception:
            pass

        return False

    def _check_if_local_module_exists(self, module_name: str) -> bool:
        """Check if a module exists as a local file in the repository."""
        return self._find_local_module_path(module_name) is not None

    def _find_local_module_path(self, module_name: str) -> str | None:
        """Find the file path for a local module by searching the repo."""
        # Convert module name to potential file paths
        module_path = module_name.replace(".", "/")

        # Search in common Python project directory patterns
        search_locations = [
            self.repo_root,
            self.repo_root / "src",
            self.repo_root / "lib",
            self.repo_root / "packages",
        ]

        for base_dir in search_locations:
            if not base_dir.exists():
                continue

            # Try as a direct .py file
            module_file = base_dir / (module_path + ".py")
            if module_file.exists():
                return str(module_file)

            # Try as a package (directory with __init__.py)
            package_dir = base_dir / module_path
            package_init = package_dir / "__init__.py"
            if package_init.exists():
                return str(package_init)

        return None


# Convenience functions for backward compatibility
def analyze_model_dependencies(model_file: str, repo_root: str | None = None) -> dict:
    """
    Analyze dependencies for a single model file.

    Args:
        model_file: Path to the main model Python file
        repo_root: Root directory of the repository (auto-detected if None)

    Returns:
        Dictionary with requirements and code_paths
    """
    if repo_root is None:
        # Auto-detect repo root
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

    analyzer = UnifiedDependencyAnalyzer(repo_root)
    return analyzer.analyze_dependencies([model_file])


def get_model_requirements(model_file: str, repo_root: str | None = None) -> list[str]:
    """Get just the requirements for a model file."""
    result = analyze_model_dependencies(model_file, repo_root)
    return result["requirements"]


def get_model_code_paths(model_file: str, repo_root: str | None = None) -> list[str]:
    """Get just the code paths for a model file."""
    result = analyze_model_dependencies(model_file, repo_root)
    return result["code_paths"]


def get_mlflow_dependencies(model_file: str, repo_root: str | None = None) -> dict:
    """
    Get dependencies in a format ready for MLflow logging.

    Args:
        model_file: Path to the main model Python file
        repo_root: Root directory of the repository (auto-detected if None)

    Returns:
        Dictionary with 'requirements' (list of versioned requirements) and
        'code_paths' (list of relative paths) that can be directly used with
        mlflow.log_model() or mlflow.save_model()

    Example:
        deps = get_mlflow_dependencies("model.py")
        mlflow.sklearn.log_model(
            model,
            "my_model",
            pip_requirements=deps["requirements"],
            code_paths=deps["code_paths"]
        )
    """
    result = analyze_model_dependencies(model_file, repo_root)
    return {"requirements": result["requirements"], "code_paths": result["code_paths"]}

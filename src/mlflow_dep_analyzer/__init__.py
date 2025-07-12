"""
MLflow Dependency Analyzer

Smart dependency analysis and minimal requirements generation for MLflow models.

Uses Python's importlib and inspect modules for accurate, universal dependency discovery
without hardcoded patterns. Works with any Python project structure.

Example:
    >>> from mlflow_dep_analyzer import analyze_model_dependencies
    >>> result = analyze_model_dependencies("model.py")
    >>> print(result["requirements"])  # External packages
    >>> print(result["code_paths"])    # Local files
"""

from .unified_analyzer import (
    UnifiedDependencyAnalyzer,
    analyze_model_dependencies,
    get_model_code_paths,
    get_model_requirements,
)

__version__ = "0.2.0"

__all__ = [
    "UnifiedDependencyAnalyzer",
    "analyze_model_dependencies",
    "get_model_requirements",
    "get_model_code_paths",
]

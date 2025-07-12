"""
MLflow Dependency Analyzer

Smart dependency analysis and minimal requirements generation for MLflow models.

This library provides three specialized analyzers:
- UnifiedDependencyAnalyzer: Complete analysis combining requirements and code paths
- CodePathAnalyzer: Local file dependency discovery
- HybridRequirementsAnalyzer: External package dependency analysis

Example:
    >>> from mlflow_dep_analyzer import analyze_model_dependencies
    >>> result = analyze_model_dependencies("model.py")
    >>> print(result["requirements"])  # External packages
    >>> print(result["code_paths"])    # Local files
"""

from .code_path_analyzer import CodePathAnalyzer, analyze_code_paths, find_model_code_paths
from .requirements_analyzer import HybridRequirementsAnalyzer, analyze_code_dependencies, is_stdlib_module
from .unified_analyzer import (
    UnifiedDependencyAnalyzer,
    analyze_model_dependencies,
    get_model_code_paths,
    get_model_requirements,
)

__version__ = "0.2.0"

__all__ = [
    # Main unified analyzer (recommended)
    "UnifiedDependencyAnalyzer",
    "analyze_model_dependencies",
    "get_model_requirements",
    "get_model_code_paths",
    # Specialized analyzers
    "HybridRequirementsAnalyzer",
    "analyze_code_dependencies",
    "CodePathAnalyzer",
    "analyze_code_paths",
    "find_model_code_paths",
    # Utilities
    "is_stdlib_module",
]

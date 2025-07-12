# MLflow Dependency Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/andrewgross/mlflow-dep-analyzer/workflows/Tests/badge.svg)](https://github.com/andrewgross/mlflow-dep-analyzer/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-green.svg)](https://github.com/andrewgross/mlflow-dep-analyzer)

**Smart dependency analysis and minimal requirements generation for MLflow models.**

Automatically detect and generate minimal `code_paths` and `requirements` lists for your MLflow models using safe AST-based analysis. Ensure portable and reproducible model deployments without dependency bloat.

## 🚀 Features

- **🔍 Unified Analysis**: Complete dependency analysis combining requirements and code paths
- **🧠 Smart Detection**: Uses Python's `importlib` and `inspect` for accurate module resolution
- **🔒 Safe Analysis**: AST-based import discovery - no code execution required
- **📦 MLflow Integration**: Built-in support for MLflow's production utilities
- **🎯 Minimal Dependencies**: Intelligent pruning eliminates unnecessary packages
- **🔄 Recursive Discovery**: Follows deep dependency chains automatically
- **🛡️ Robust Error Handling**: Graceful handling of circular dependencies and import errors
- **⚡ Production Ready**: Comprehensive test coverage with real-world scenarios

## 📦 Installation

```bash
pip install mlflow-dep-analyzer
```

## 🎯 Quick Start

### Simple Model Analysis

```python
from mlflow_dep_analyzer import analyze_model_dependencies

# Analyze a single model file
result = analyze_model_dependencies("model.py")

print("📦 External packages needed:")
print(result["requirements"])

print("📂 Local files needed:")
print(result["code_paths"])
```

### MLflow Integration

```python
import mlflow
import mlflow.sklearn
from mlflow_dep_analyzer import analyze_model_dependencies
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
# ... training code ...

# Analyze dependencies
deps = analyze_model_dependencies("model.py")

# Log with minimal dependencies
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "classifier",
        code_paths=deps["code_paths"],
        pip_requirements=deps["requirements"]
    )
```

## 📚 API Reference

### Unified Analyzer (Recommended)

The `UnifiedDependencyAnalyzer` provides complete analysis in a single operation:

```python
from mlflow_dep_analyzer import UnifiedDependencyAnalyzer, analyze_model_dependencies

# Class-based usage
analyzer = UnifiedDependencyAnalyzer(repo_root=".")
result = analyzer.analyze_dependencies(["model.py", "utils.py"])

# Function-based usage (recommended for simple cases)
result = analyze_model_dependencies("model.py")

# Result structure
{
    "requirements": ["pandas", "scikit-learn"],  # External packages
    "code_paths": ["model.py", "utils.py"],      # Local files
    "analysis": {
        "total_modules": 15,
        "external_packages": 2,
        "local_files": 2,
        "stdlib_modules": 11
    }
}
```

### Specialized Analyzers

For advanced use cases, use the specialized analyzers:

```python
from mlflow_dep_analyzer import CodePathAnalyzer, HybridRequirementsAnalyzer

# Find only local file dependencies
code_analyzer = CodePathAnalyzer(repo_root=".")
code_paths = code_analyzer.analyze_code_paths(["model.py"])

# Find only external package dependencies
req_analyzer = HybridRequirementsAnalyzer(repo_root=".")
requirements = req_analyzer.analyze(["model.py"])
```

### Convenience Functions

```python
from mlflow_dep_analyzer import (
    get_model_requirements,
    get_model_code_paths,
    analyze_code_dependencies,
    find_model_code_paths
)

# Get just the requirements
packages = get_model_requirements("model.py")

# Get just the code paths
files = get_model_code_paths("model.py")

# Legacy functions (still supported)
requirements = analyze_code_dependencies(["model.py"], repo_root=".")
code_paths = find_model_code_paths("model.py")
```

## 🏗️ Architecture

The library provides three specialized analyzers:

```
┌─────────────────────────┐
│  UnifiedDependencyAnalyzer │  ← Recommended for most use cases
│  (Complete Analysis)       │
└─────────────────────────┘
           │
           ├─── Uses inspect + importlib for module resolution
           ├─── Classifies: external packages | stdlib | local files
           └─── Returns unified result

┌─────────────────────────┐  ┌─────────────────────────┐
│   CodePathAnalyzer      │  │ HybridRequirementsAnalyzer │
│   (Local Files)         │  │ (External Packages)       │
└─────────────────────────┘  └─────────────────────────┘
           │                            │
           ├─── AST-based discovery     ├─── MLflow integration
           ├─── Recursive traversal     ├─── PyPI validation
           └─── Package structure       └─── Smart filtering
```

## 🔍 How It Works

1. **AST Parsing**: Safely extracts import statements without executing code
2. **Module Resolution**: Uses `importlib.import_module()` + `inspect.getsourcefile()`
3. **Smart Classification**: Automatically categorizes modules:
   - 📦 **External packages** → Added to requirements
   - 🐍 **Standard library** → Ignored (built into Python)
   - 📁 **Local files** → Added to code_paths and analyzed recursively
4. **Dependency Discovery**: Recursively follows imports to build complete dependency graph
5. **Path Optimization**: Generates minimal file lists and package requirements

## 🌟 Advanced Usage

### Complex Project Structure

```python
from mlflow_dep_analyzer import UnifiedDependencyAnalyzer

# Analyze a complex project with src/ structure
analyzer = UnifiedDependencyAnalyzer(repo_root="/path/to/project")
result = analyzer.analyze_dependencies([
    "src/models/classifier.py",
    "src/models/preprocessor.py",
    "src/utils/data_loader.py"
])

print(f"Found {result['analysis']['total_modules']} total modules")
print(f"External packages: {result['analysis']['external_packages']}")
print(f"Local files: {result['analysis']['local_files']}")
```

### Custom Package Detection

```python
from mlflow_dep_analyzer import HybridRequirementsAnalyzer

# Use specialized analyzer with custom configuration
analyzer = HybridRequirementsAnalyzer(
    repo_root=".",
    use_mlflow_mapping=True,  # Use MLflow's package mapping
    exclude_dev_deps=True     # Exclude development dependencies
)

requirements = analyzer.analyze(
    code_paths=["model.py"],
    existing_requirements=["numpy>=1.20.0"]  # Exclude already-specified packages
)
```

### Error Handling

```python
from mlflow_dep_analyzer import analyze_model_dependencies

try:
    result = analyze_model_dependencies("model.py")
except FileNotFoundError:
    print("Model file not found")
except ImportError as e:
    print(f"Import resolution failed: {e}")
```

## 🧪 Examples

See the [examples/](examples/) directory for complete working examples:

- **[Basic Usage](examples/demo_smart_requirements.py)**: Simple model analysis
- **[MLflow Integration](examples/projects/)**: Real-world MLflow projects
- **[Complex Projects](examples/projects/my_model/)**: Multi-file analysis

## 🛠️ Development

### Setup

This project uses `uv` for dependency management:

```bash
git clone https://github.com/andrewgross/mlflow-dep-analyzer
cd mlflow-dep-analyzer
uv sync
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/mlflow_dep_analyzer --cov-report=html

# Run specific test categories
uv run pytest tests/test_unified_analyzer.py -v
```

### Code Quality

```bash
# Linting and formatting
uv run ruff check
uv run ruff format

# Type checking
uv run mypy src/

# Pre-commit hooks
uv run pre-commit run --all-files
```

### Requirements

- **Python**: 3.8+ (developed with 3.11.11 for Databricks Runtime 15.4 LTS compatibility)
- **Core dependencies**: MLflow 2.0+
- **Development**: pytest, ruff, mypy, pre-commit

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Run the test suite: `uv run pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on MLflow's production-tested dependency resolution utilities
- Inspired by the need for reliable, minimal MLflow model deployments
- Thanks to the Python AST and importlib developers for robust introspection tools

## 📈 Roadmap

- [ ] Configuration file support
- [ ] Plugin system for custom analyzers
- [ ] Integration with other ML frameworks
- [ ] Dependency vulnerability scanning
- [ ] Performance optimizations with caching

---

<div align="center">

**[Documentation](https://github.com/andrewgross/mlflow-dep-analyzer)** •
**[Issues](https://github.com/andrewgross/mlflow-dep-analyzer/issues)** •
**[Contributing](CONTRIBUTING.md)**

Made with ❤️ for the MLflow community

</div>

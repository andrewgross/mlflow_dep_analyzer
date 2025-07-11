# MLflow Dependency Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Smart dependency analysis and minimal requirements generation for MLflow models.**

Automatically detect and generate minimal `requirements.txt` files for your MLflow models using AST-based analysis, ensuring portable and reproducible model deployments without dependency bloat.

## 🚀 Features

### **🔍 Smart Dependency Detection**
- **AST-based analysis**: Safely analyze Python imports without code execution
- **Dynamic stdlib detection**: Automatically excludes Python standard library modules
- **Local module filtering**: Intelligently detects and excludes project-specific code
- **MLflow integration**: Leverages MLflow's proven package resolution and pruning logic

### **📦 Minimal Requirements Generation**
- **Hybrid analysis**: Combines AST discovery with MLflow's production-tested utilities
- **Dependency pruning**: Removes redundant dependencies using MLflow's algorithms
- **Version pinning**: Generates pinned requirements for reproducible environments
- **Existing requirements handling**: Excludes packages already specified in your requirements

### **🛠️ Code Path Analysis**
- **Dependency mapping**: Finds all local code dependencies for MLflow `code_paths`
- **Minimal bundling**: Only includes necessary files for model portability
- **Framework agnostic**: Works with any Python ML framework (scikit-learn, PyTorch, etc.)

## 📦 Installation

```bash
pip install mlflow-dep-analyzer
```

Or install from source:
```bash
git clone https://github.com/your-username/mlflow-dep-analyzer
cd mlflow-dep-analyzer
pip install -e .
```

## 🏃 Quick Start

### Generate Minimal Requirements

```python
from mlflow_dep_analyzer import analyze_code_dependencies

# Analyze a single file
requirements = analyze_code_dependencies(
    code_paths=["src/my_model.py"],
    repo_root="/path/to/your/project"
)

print("Minimal requirements:")
for req in requirements:
    print(f"  {req}")
```

### Find Code Paths for MLflow

```python
from mlflow_dep_analyzer import analyze_code_paths

# Find all dependencies for MLflow code_paths
code_paths = analyze_code_paths(
    entry_files=["src/my_model.py"],
    repo_root="/path/to/your/project"
)

# Use with MLflow
import mlflow.sklearn
mlflow.sklearn.log_model(
    model,
    "my_model",
    code_paths=code_paths  # ← Minimal, portable model
)
```

### Complete Workflow Example

```python
import mlflow
from mlflow_dep_analyzer import HybridRequirementsAnalyzer, analyze_code_paths
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 1. Train your model
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 2. Analyze dependencies
analyzer = HybridRequirementsAnalyzer()
result = analyzer.analyze_model_requirements(
    code_paths=["my_model_training.py"],
    repo_root="."
)

# 3. Generate requirements.txt
with open("model_requirements.txt", "w") as f:
    for req in result["requirements"]:
        f.write(f"{req}\\n")

# 4. Find code dependencies
code_paths = analyze_code_paths(
    entry_files=["my_model_training.py"],
    repo_root="."
)

# 5. Log model with minimal dependencies
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "random_forest",
        code_paths=code_paths,
        pip_requirements="model_requirements.txt"
    )

print("✅ Model logged with minimal, portable dependencies!")
```

## 📚 Detailed Examples

### 1. Basic Requirements Analysis

```python
from mlflow_dep_analyzer import analyze_code_dependencies

# Analyze a directory of model files
requirements = analyze_code_dependencies(
    code_paths=["src/models/", "src/preprocessing/"],
    repo_root="/path/to/project",
    exclude_existing=True,  # Skip packages in existing requirements.txt
    existing_requirements_file="requirements.txt"
)

# Save minimal requirements
with open("model_requirements.txt", "w") as f:
    for req in requirements:
        f.write(f"{req}\\n")
```

### 2. Advanced Analysis with Custom Patterns

```python
from mlflow_dep_analyzer import HybridRequirementsAnalyzer

# Create analyzer with custom configuration
analyzer = HybridRequirementsAnalyzer(
    existing_requirements=["pandas>=1.0.0", "numpy>=1.20.0"]
)

# Detailed analysis with custom local patterns
result = analyzer.analyze_model_requirements(
    code_paths=["src/"],
    repo_root="/path/to/project",
    local_patterns={"my_company_utils", "internal_libs"},  # Custom local modules
    exclude_existing=True
)

# Inspect the analysis
analysis = result["analysis"]
print(f"Files analyzed: {len(analysis['files_analyzed'])}")
print(f"Raw imports found: {len(analysis['raw_imports'])}")
print(f"External modules: {len(analysis['external_modules'])}")
print(f"Final requirements: {len(result['requirements'])}")

# Detailed breakdown
print("\\nAnalysis breakdown:")
print(f"  Raw imports: {analysis['raw_imports']}")
print(f"  External modules: {analysis['external_modules']}")
print(f"  Final packages: {analysis['final_packages']}")
```

### 3. Code Path Analysis for MLflow

```python
from mlflow_dep_analyzer import CodePathAnalyzer

# Detailed code path analysis
analyzer = CodePathAnalyzer(repo_root="/path/to/project")

result = analyzer.analyze_code_paths(
    entry_files=["src/train_model.py", "src/predict.py"],
    include_patterns=["**/*.py"],
    exclude_patterns=["**/tests/**", "**/__pycache__/**"]
)

print(f"Required files: {len(result['required_files'])}")
print(f"Dependencies found: {result['analysis']['total_dependencies']}")

# Use the relative paths with MLflow
import mlflow.sklearn
mlflow.sklearn.log_model(
    model,
    "my_model",
    code_paths=result["relative_paths"]
)
```

### 4. Integration with MLflow Training

```python
import mlflow
import mlflow.sklearn
from mlflow_dep_analyzer import analyze_code_dependencies, analyze_code_paths

def train_and_log_model():
    # Your training code here
    model = train_your_model()

    # Automatically generate minimal requirements
    requirements = analyze_code_dependencies(
        code_paths=["src/"],
        repo_root=".",
        exclude_existing=True
    )

    # Find necessary code paths
    code_paths = analyze_code_paths(
        entry_files=["src/model.py", "src/preprocessing.py"],
        repo_root="."
    )

    # Log with minimal dependencies
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model,
            "model",
            code_paths=code_paths,
            pip_requirements=requirements
        )

    return model
```

## 🔧 API Reference

### Core Functions

#### `analyze_code_dependencies(code_paths, **kwargs)`
Generate minimal requirements for given code paths.

**Parameters:**
- `code_paths` (List[str]): Files/directories to analyze
- `repo_root` (str, optional): Repository root for local module detection
- `existing_requirements` (List[str], optional): Already installed packages to exclude
- `existing_requirements_file` (str, optional): Path to existing requirements.txt
- `exclude_existing` (bool): Whether to exclude existing requirements (default: True)
- `local_patterns` (Set[str], optional): Custom local module patterns

**Returns:** List[str] - Minimal requirements

#### `analyze_code_paths(entry_files, repo_root, **kwargs)`
Find code dependencies for MLflow `code_paths` parameter.

**Parameters:**
- `entry_files` (List[str]): Main Python files to analyze
- `repo_root` (str): Repository root directory
- `include_patterns` (List[str], optional): File patterns to include
- `exclude_patterns` (List[str], optional): File patterns to exclude

**Returns:** List[str] - Relative file paths for MLflow

### Classes

#### `HybridRequirementsAnalyzer`
Advanced requirements analyzer with detailed control.

**Methods:**
- `analyze_model_requirements()`: Complete analysis with detailed results
- `analyze_file()`: Analyze single Python file
- `analyze_directory()`: Analyze directory of Python files
- `filter_local_modules()`: Remove local/project modules
- `is_stdlib_module()`: Check if module is in Python stdlib

#### `CodePathAnalyzer`
Analyzer for finding minimal code dependencies.

**Methods:**
- `analyze_code_paths()`: Find dependencies for entry files
- `collect_dependencies()`: Recursively collect all dependencies
- `analyze_file()`: Find local imports in a file

## 🛡️ Why Use This?

### **Problem: Bloated Model Dependencies**
```python
# Traditional approach - includes everything
pip freeze > requirements.txt  # 📦 200+ packages

mlflow.sklearn.log_model(model, "model", pip_requirements="requirements.txt")
```

### **Solution: Minimal Dependencies**
```python
# Smart approach - only what you need
from mlflow_dep_analyzer import analyze_code_dependencies

requirements = analyze_code_dependencies(["src/model.py"])  # 📦 5-10 packages
mlflow.sklearn.log_model(model, "model", pip_requirements=requirements)
```

### **Benefits:**
- ⚡ **Faster deployments**: Smaller container images, faster pip installs
- 🔒 **Better security**: Fewer dependencies = smaller attack surface
- 🎯 **Clearer dependencies**: Know exactly what your model needs
- 📱 **Portable models**: Models work anywhere with minimal setup
- 🔄 **Reproducible builds**: Pinned versions ensure consistency

## 🧪 Development

### Setup Development Environment

```bash
git clone https://github.com/your-username/mlflow-dep-analyzer
cd mlflow-dep-analyzer
uv sync  # Install dependencies
```

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/mlflow_dep_analyzer

# Run specific test categories
uv run pytest tests/test_requirements_analyzer.py -v
uv run pytest tests/test_code_path_analyzer.py -v
```

### Code Quality

```bash
# Linting and formatting
uv run ruff check
uv run ruff format

# Type checking
uv run mypy src/
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `uv run pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MLflow Team**: For the excellent MLflow framework and production-tested utilities
- **Python AST Module**: For safe code analysis capabilities
- **Community**: For feedback and contributions

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/andrewgross/mlflow-dep-analyzer/issues)

---

**Made with ❤️ for the MLflow community**

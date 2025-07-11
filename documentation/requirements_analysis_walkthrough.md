# Requirements Finding and Pruning: A Comprehensive Walkthrough

## Overview for Junior Engineers

This document explains our hybrid approach to automatically determining the minimal set of Python packages needed to run MLflow models in production. As a junior engineer, you'll learn about dependency analysis, package resolution, and how we combine safety with accuracy to solve a critical production problem.

## Updated Project Structure

This project has been restructured for better organization:

```
├── examples/                          # Example implementations and demos
│   ├── projects/                      # Example model projects
│   │   ├── my_model/                 # Sentiment analysis model example
│   │   │   ├── auto_logging_sentiment_model.py
│   │   │   └── sentiment_model.py
│   │   ├── shared_utils/             # Example utilities and helpers
│   │   │   ├── mlflow_hybrid_analyzer.py  # Legacy analyzer (examples)
│   │   │   └── base_model.py
│   │   └── inference/                # Inference pipeline examples
│   ├── tests/                        # Tests for example implementations
│   ├── notebooks/                    # Jupyter notebooks for demos
│   └── demo_smart_requirements.py    # Standalone demo script
├── src/                              # Reusable library components
│   └── mlflow_code_analysis/         # Main library package
│       ├── requirements_analyzer.py   # Core requirements analysis
│       └── code_path_analyzer.py     # Code path discovery
├── tests/                            # Tests for src library
├── documentation/                    # Technical documentation
└── Makefile                          # Build and test commands
```

## The Problem: Dependency Hell in ML Models

When you save an ML model with MLflow, you need to know which Python packages are required to load and use that model later. Too many packages create bloated environments and security risks. Too few packages cause runtime failures. The challenge is finding the **exact minimal set** needed.

### Real-World Example

Consider this simple sentiment model:

```python
# examples/projects/my_model/auto_logging_sentiment_model.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import mlflow

class SentimentModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression())
        ])
```

**Question**: What packages do we need to install to run this model?

- Obviously: `scikit-learn`, `pandas`, `mlflow`
- Not obvious: Does `sklearn` import require `scipy`? What about `numpy`?
- Definitely not: `requests`, `matplotlib`, or 200+ other packages in your dev environment

**Our goal**: Automatically discover that we need exactly `scikit-learn==1.3.0`, `pandas==2.0.3`, `mlflow==2.8.1`, `numpy==1.24.0`, `scipy==1.11.0` and nothing else.

## The Hybrid Solution: Best of Both Worlds

We developed a **hybrid approach** that combines:

1. **AST-based import discovery** (safe, comprehensive)
2. **MLflow's production-tested package resolution** (accurate, battle-tested)

### Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Python Files  │───▶│   AST Analysis   │───▶│   Raw Imports   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Final Requirements│◀───│ MLflow's Package │◀───│ Filtered Imports│
└─────────────────┘    │   Resolution     │    └─────────────────┘
                       └──────────────────┘
```

## Step 1: AST-Based Import Discovery

### The Safe Way to Find Imports

Instead of executing code (dangerous!), we parse the Python syntax tree to find all import statements:

```python
# src/mlflow_code_analysis/requirements_analyzer.py
import ast

def analyze_file(self, file_path: str) -> set[str]:
    """Extract all imports from a Python file using AST."""
    with open(file_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())  # Parse without executing!

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])  # Get top-level module
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # Only absolute imports
                imports.add(node.module.split(".")[0])

    return imports
```

**Example**: For our sentiment model, this discovers:
```python
{
    'sklearn', 'pandas', 'mlflow', 'datetime', 'os', 'sys',
    'examples', 'tempfile', 'importlib', 'traceback', ...
}
```

### Why AST is Safe and Effective

✅ **Safe**: No code execution, no security risks
✅ **Comprehensive**: Finds all imports, even in unused code paths
✅ **Fast**: Parsing is much faster than importing
✅ **Reliable**: Works even if dependencies aren't installed

❌ **Challenge**: Raw imports include stdlib, local modules, and false positives

## Step 2: Smart Filtering - Local vs External

Not all imports need to be installed! We need to filter out:

### Standard Library Modules
```python
# These are built into Python - no installation needed
stdlib_modules = {
    'os', 'sys', 'datetime', 'json', 'pickle', 'logging',
    'pathlib', 'tempfile', 'subprocess', 'collections',
    're', 'urllib', 'functools', 'itertools', ...
}
```

### Local Project Modules
```python
# src/mlflow_code_analysis/requirements_analyzer.py
def _should_exclude_module(self, module: str, local_patterns: set[str]) -> bool:
    """Determine if a module should be excluded using MLflow-style rules."""

    # MLflow excludes private modules
    if module.startswith("_"):
        return True

    # Skip known local patterns
    local_patterns = {
        "examples", "projects", "shared_utils", "text_utils",
        "validation", "constants", "my_model", "inference", "src"
    }

    if module in local_patterns:
        return True

    return False
```

**After filtering our example**:
```python
# Before: {'sklearn', 'pandas', 'mlflow', 'datetime', 'os', 'examples', ...}
# After:  {'sklearn', 'pandas', 'mlflow'}  # Only external packages!
```

## Step 3: MLflow's Battle-Tested Package Resolution

Now comes the tricky part: converting module names to package names.

### The Import Name vs Package Name Problem

| Import Statement | Module Name | Package Name |
|------------------|-------------|--------------|
| `import sklearn` | `sklearn` | `scikit-learn` |
| `import cv2` | `cv2` | `opencv-python` |
| `import PIL` | `PIL` | `Pillow` |
| `import yaml` | `yaml` | `PyYAML` |

**This is where most homegrown solutions fail!**

### MLflow's Production Solution

MLflow has solved this in production through `importlib.metadata`:

```python
# src/mlflow_code_analysis/requirements_analyzer.py
from mlflow.utils.requirements_utils import _MODULES_TO_PACKAGES

def resolve_packages_mlflow_style(self, modules: set[str]) -> set[str]:
    """Convert modules to packages using MLflow's approach."""
    packages = set()

    for module in modules:
        # MLflow maintains a comprehensive mapping
        module_packages = _MODULES_TO_PACKAGES.get(module, [])
        packages.update(module_packages)

    return packages
```

**MLflow's mapping is comprehensive**:
```python
_MODULES_TO_PACKAGES = {
    'sklearn': ['scikit-learn'],
    'cv2': ['opencv-python'],
    'PIL': ['Pillow'],
    'yaml': ['PyYAML'],
    'bs4': ['beautifulsoup4'],
    # ... hundreds more
}
```

## Step 4: Dependency Pruning - The MLflow Magic

MLflow has another production insight: **dependency pruning**.

### The Transitive Dependency Problem

When you install `scikit-learn`, pip automatically installs:
- `numpy` (scikit-learn depends on it)
- `scipy` (scikit-learn depends on it)
- `joblib` (scikit-learn depends on it)

**Question**: Should our requirements.txt include all of these?

**MLflow's Answer**: No! Only specify top-level packages. Let pip handle the rest.

```python
# src/mlflow_code_analysis/requirements_analyzer.py
def prune_dependencies_mlflow_style(self, packages: set[str]) -> set[str]:
    """Apply MLflow's dependency pruning logic."""
    try:
        # Use MLflow's battle-tested pruning logic
        return set(_prune_packages(packages))
    except Exception as e:
        return packages
```

### Why Pruning Matters

**Without pruning**:
```
scikit-learn==1.3.0
numpy==1.24.0
scipy==1.11.0
joblib==1.3.2
threadpoolctl==3.2.0
```

**With pruning**:
```
scikit-learn==1.3.0
```

✅ **Cleaner**: Fewer version conflicts
✅ **More flexible**: Pip chooses compatible versions
✅ **More maintainable**: One source of truth

## Step 5: Version Pinning and Validation

### Getting Exact Versions

For reproducibility, we pin to specific versions:

```python
# src/mlflow_code_analysis/requirements_analyzer.py
def generate_pinned_requirements(self, packages: set[str]) -> list[str]:
    """Generate pinned requirements using MLflow's utilities."""
    requirements = []

    for package in sorted(packages):
        # Use MLflow's pinning logic
        req = _get_pinned_requirement(package)
        requirements.append(req)

    return requirements
```

### PyPI Validation

MLflow validates against PyPI to catch typos:

```python
def validate_against_pypi(self, packages: set[str]) -> tuple[set[str], set[str]]:
    """Validate packages against PyPI index."""
    pypi_index = _load_pypi_package_index()

    recognized = packages & pypi_index.package_names
    unrecognized = packages - recognized

    return recognized, unrecognized
```

## Using the Reusable Library

### Core Library Components

The `src/mlflow_code_analysis/` package provides two main analyzers:

#### 1. HybridRequirementsAnalyzer

```python
# Import the library
from src.mlflow_code_analysis import HybridRequirementsAnalyzer

# Initialize analyzer
analyzer = HybridRequirementsAnalyzer(
    existing_requirements=["mlflow>=2.0.0", "pandas>=1.3.0"]
)

# Analyze requirements
result = analyzer.analyze_model_requirements(
    code_paths=["examples/projects/my_model/"],
    repo_root="/path/to/repo",
    exclude_existing=True
)

print(result["requirements"])  # ['scikit-learn==1.3.0']
```

#### 2. CodePathAnalyzer

```python
# Import the library
from src.mlflow_code_analysis import CodePathAnalyzer

# Initialize analyzer
analyzer = CodePathAnalyzer(repo_root="/path/to/repo")

# Find code paths needed for model
result = analyzer.analyze_code_paths(
    entry_files=["examples/projects/my_model/auto_logging_sentiment_model.py"]
)

print(result["relative_paths"])  # ['examples/projects/my_model/auto_logging_sentiment_model.py', ...]
```

### Convenience Functions

For simple use cases, use the convenience functions:

```python
# Analyze dependencies
from src.mlflow_code_analysis import analyze_code_dependencies

requirements = analyze_code_dependencies(
    code_paths=["examples/projects/my_model/"],
    existing_requirements_file="requirements.txt"
)

# Analyze code paths
from src.mlflow_code_analysis import analyze_code_paths

code_paths = analyze_code_paths(
    entry_files=["examples/projects/my_model/auto_logging_sentiment_model.py"],
    repo_root="/path/to/repo"
)
```

## Complete Workflow Example

Let's trace through our sentiment model using the new structure:

### Input: Model Code
```python
# examples/projects/my_model/auto_logging_sentiment_model.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import mlflow
from ..shared_utils.base_model import BaseModelV3
```

### Step-by-Step Analysis

```python
# Step 1: AST discovers all imports
raw_imports = {
    'sklearn', 'pandas', 'mlflow', 'examples', 'datetime', 'os',
    'sys', 'tempfile', 'importlib', 'traceback', 'joblib'
}

# Step 2: Filter out stdlib and local modules
external_modules = {
    'sklearn', 'pandas', 'mlflow', 'joblib'
}

# Step 3: Resolve to package names (MLflow mapping)
resolved_packages = {
    'scikit-learn', 'pandas', 'mlflow', 'joblib'
}

# Step 4: Prune dependencies (MLflow logic)
pruned_packages = {
    'scikit-learn', 'pandas', 'mlflow'  # joblib removed (scikit-learn dependency)
}

# Step 5: Exclude existing packages (already in requirements.txt)
base_requirements = ['mlflow>=2.0.0', 'pandas>=1.3.0', ...]
final_packages = {
    'scikit-learn'  # Only this is missing from base requirements!
}

# Step 6: Pin to specific versions
final_requirements = [
    'scikit-learn==1.3.0'
]
```

### Result: Perfect Optimization!

Our model that imports 10+ modules only needs **1 additional package** beyond the base environment. This is the power of intelligent dependency analysis.

## Code Integration Example

Here's how this integrates into our auto-logging model using the new library:

```python
# examples/projects/my_model/auto_logging_sentiment_model.py
def _generate_and_log_requirements(self):
    """Generate smart requirements.txt for model dependencies."""
    # Import from the reusable library
    from src.mlflow_code_analysis import HybridRequirementsAnalyzer

    # Get base requirements from project
    base_requirements = self._load_base_requirements()

    # Initialize hybrid analyzer
    analyzer = HybridRequirementsAnalyzer(existing_requirements=base_requirements)

    # Analyze model dependencies
    result = analyzer.analyze_model_requirements(
        code_paths=[__file__, "../shared_utils/"],
        repo_root=self.repo_root,
        exclude_existing=True
    )

    requirements = result["requirements"]

    # Log to MLflow for reproducibility
    mlflow.log_param("requirements_summary",
                     "None - perfect optimization!" if not requirements
                     else ", ".join(requirements))

    return requirements
```

## Running Tests

The project includes comprehensive tests for both the library and examples:

```bash
# Test the core library
make test-src
# OR: uv run pytest tests/ -v

# Test the examples
make test-examples
# OR: uv run pytest examples/tests/ -v

# Test everything
make test
```

## Development Workflow

### Adding New Functionality

1. **Core logic** goes in `src/mlflow_code_analysis/`
2. **Examples** go in `examples/projects/`
3. **Tests for core** go in `tests/`
4. **Tests for examples** go in `examples/tests/`

### Using the Library in Your Projects

```python
# Option 1: Direct import (if src/ is in your path)
from src.mlflow_code_analysis import HybridRequirementsAnalyzer

# Option 2: Install as package (future)
# pip install mlflow-code-analysis
# from mlflow_code_analysis import HybridRequirementsAnalyzer
```

## Key Insights for Junior Engineers

### 1. Safety First
- **Never execute untrusted code** for dependency analysis
- AST parsing is safe and comprehensive
- Always validate inputs and handle exceptions

### 2. Leverage Production Knowledge
- Don't reinvent the wheel - MLflow has solved hard problems
- Use battle-tested mappings and logic where possible
- Production systems encode years of edge case handling

### 3. Layered Architecture
- **`src/`** contains reusable, well-tested components
- **`examples/`** shows how to use the library
- Clear separation of concerns enables maintainability

### 4. Real-World Complexity
- Import names ≠ package names (always!)
- Transitive dependencies create complexity
- Version conflicts are common in practice
- Local modules must be filtered correctly

### 5. Testing Strategy
- Library components have focused unit tests
- Examples have integration tests
- Both test suites validate the complete workflow

## Performance Characteristics

| Metric | AST-Only | MLflow-Only | Hybrid Approach |
|--------|----------|-------------|-----------------|
| **Safety** | ✅ Safe | ❌ Execution | ✅ Safe |
| **Accuracy** | ❌ Poor mapping | ✅ Excellent | ✅ Excellent |
| **Completeness** | ✅ Comprehensive | ❌ Runtime only | ✅ Comprehensive |
| **Speed** | ✅ Fast | ❌ Slow | ✅ Fast |
| **Production Ready** | ❌ No | ✅ Yes | ✅ Yes |
| **Reusability** | ❌ Project-specific | ❌ MLflow-specific | ✅ Library + Examples |

## Future Improvements

As you grow as an engineer, consider these enhancements:

1. **Package Distribution**: Publish `src/` as a proper Python package
2. **Caching**: Cache PyPI index and package mappings
3. **Parallelization**: Analyze multiple files concurrently
4. **ML-based mapping**: Use ML to improve import→package mapping
5. **Integration**: Hook into CI/CD for automatic requirements updates
6. **Metrics**: Track optimization rates and accuracy over time

## Conclusion

Our hybrid requirements analyzer solves a critical production problem by combining:
- **AST safety** with **MLflow accuracy**
- **Comprehensive discovery** with **intelligent filtering**
- **Automatic optimization** with **manual control**
- **Reusable library** with **practical examples**

The restructured project provides:
- **Clean separation** between library and examples
- **Comprehensive testing** for both components
- **Easy integration** into existing workflows
- **Clear documentation** for future development

The result is a system that consistently generates minimal, accurate requirements.txt files for MLflow models, enabling reliable production deployments while minimizing dependency bloat.

**Key takeaway**: Good engineering often means combining the best aspects of different approaches rather than choosing just one. Our hybrid solution and clean architecture demonstrate this principle in action.

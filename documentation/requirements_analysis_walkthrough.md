# Requirements Finding and Pruning: A Comprehensive Walkthrough

## Overview for Junior Engineers

This document explains our hybrid approach to automatically determining the minimal set of Python packages needed to run MLflow models in production. As a junior engineer, you'll learn about dependency analysis, package resolution, and how we combine safety with accuracy to solve a critical production problem.

## The Problem: Dependency Hell in ML Models

When you save an ML model with MLflow, you need to know which Python packages are required to load and use that model later. Too many packages create bloated environments and security risks. Too few packages cause runtime failures. The challenge is finding the **exact minimal set** needed.

### Real-World Example

Consider this simple sentiment model:

```python
# projects/my_model/auto_logging_sentiment_model.py
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
# projects/shared_utils/mlflow_hybrid_analyzer.py
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
    'projects', 'tempfile', 'importlib', 'traceback', ...
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
# projects/shared_utils/mlflow_hybrid_analyzer.py
def _should_exclude_module(self, module: str, repo_name: str = "") -> bool:
    """Determine if a module should be excluded using MLflow-style rules."""

    # MLflow excludes private modules
    if module.startswith("_"):
        return True

    # Skip known local patterns
    local_patterns = {
        "projects", "shared_utils", "text_utils",
        "validation", "constants", "my_model", "inference"
    }

    if module in local_patterns:
        return True

    return False
```

**After filtering our example**:
```python
# Before: {'sklearn', 'pandas', 'mlflow', 'datetime', 'os', 'projects', ...}
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
# projects/shared_utils/mlflow_hybrid_analyzer.py
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
# projects/shared_utils/mlflow_hybrid_analyzer.py
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
# projects/shared_utils/mlflow_hybrid_analyzer.py
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

## Complete Workflow Example

Let's trace through our sentiment model:

### Input: Model Code
```python
# projects/my_model/auto_logging_sentiment_model.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import mlflow
from projects.shared_utils.base_model import BaseModelV3
```

### Step-by-Step Analysis

```python
# Step 1: AST discovers all imports
raw_imports = {
    'sklearn', 'pandas', 'mlflow', 'projects', 'datetime', 'os',
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

Here's how this integrates into our auto-logging model:

```python
# projects/my_model/auto_logging_sentiment_model.py
def _generate_and_log_requirements(self):
    """Generate smart requirements.txt for model dependencies."""
    from ..shared_utils.mlflow_hybrid_analyzer import MLflowHybridAnalyzer

    # Get base requirements from project
    base_requirements = load_requirements_from_file("requirements.txt")

    # Initialize hybrid analyzer
    analyzer = MLflowHybridAnalyzer(existing_requirements=base_requirements)

    # Analyze model dependencies
    result = analyzer.analyze_model_requirements(
        code_paths=[__file__, "shared_utils/"],
        repo_root=repo_root,
        exclude_existing=True
    )

    requirements = result["requirements"]

    # Log to MLflow for reproducibility
    mlflow.log_param("requirements_summary",
                     "None - perfect optimization!" if not requirements
                     else ", ".join(requirements))

    return requirements
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

### 3. Layered Approach
- Break complex problems into clear steps
- Each step should be testable and understandable
- Provide detailed analysis for debugging

### 4. Real-World Complexity
- Import names ≠ package names (always!)
- Transitive dependencies create complexity
- Version conflicts are common in practice
- Local modules must be filtered correctly

### 5. Optimization Mindset
- Minimal requirements = fewer conflicts
- Let package managers handle transitive dependencies
- Pin versions for reproducibility
- Measure and validate results

## Testing and Validation

Our hybrid analyzer includes comprehensive testing:

```python
# Test the analyzer works correctly
analyzer = MLflowHybridAnalyzer()
result = analyzer.analyze_model_requirements(
    code_paths=["projects/my_model/"],
    exclude_existing=True
)

# Verify results
assert len(result["requirements"]) <= 5  # Should be minimal
assert "mlflow" not in result["final_packages"]  # Should be excluded
assert all("==" in req for req in result["requirements"])  # Should be pinned
```

## Performance Characteristics

| Metric | AST-Only | MLflow-Only | Hybrid Approach |
|--------|----------|-------------|-----------------|
| **Safety** | ✅ Safe | ❌ Execution | ✅ Safe |
| **Accuracy** | ❌ Poor mapping | ✅ Excellent | ✅ Excellent |
| **Completeness** | ✅ Comprehensive | ❌ Runtime only | ✅ Comprehensive |
| **Speed** | ✅ Fast | ❌ Slow | ✅ Fast |
| **Production Ready** | ❌ No | ✅ Yes | ✅ Yes |

## Future Improvements

As you grow as an engineer, consider these enhancements:

1. **Caching**: Cache PyPI index and package mappings
2. **Parallelization**: Analyze multiple files concurrently
3. **ML-based mapping**: Use ML to improve import→package mapping
4. **Integration**: Hook into CI/CD for automatic requirements updates
5. **Metrics**: Track optimization rates and accuracy over time

## Conclusion

Our hybrid requirements analyzer solves a critical production problem by combining:
- **AST safety** with **MLflow accuracy**
- **Comprehensive discovery** with **intelligent filtering**
- **Automatic optimization** with **manual control**

The result is a system that consistently generates minimal, accurate requirements.txt files for MLflow models, enabling reliable production deployments while minimizing dependency bloat.

**Key takeaway**: Good engineering often means combining the best aspects of different approaches rather than choosing just one. Our hybrid solution demonstrates this principle in action.

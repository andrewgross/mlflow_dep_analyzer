# Contributing to MLflow Dependency Analyzer

Thank you for your interest in contributing to MLflow Dependency Analyzer! We welcome contributions from the community and are grateful for your support.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/mlflow-dep-analyzer.git`
3. Set up development environment: `uv sync`
4. Create a feature branch: `git checkout -b feature-name`
5. Make your changes with tests
6. Run the test suite: `uv run pytest`
7. Submit a pull request

## 📋 Development Setup

### Prerequisites

- Python 3.8+ (developed with 3.11.11 for Databricks Runtime 15.4 LTS compatibility)
- [uv](https://github.com/astral-sh/uv) for dependency management

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/andrewgross/mlflow-dep-analyzer.git
cd mlflow-dep-analyzer

# Install dependencies and set up environment
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/mlflow_dep_analyzer --cov-report=html

# Run specific test files
uv run pytest tests/test_unified_analyzer.py -v

# Run tests with different markers
uv run pytest -m "not slow"  # Skip slow tests
```

### Code Quality

We maintain high code quality standards:

```bash
# Linting
uv run ruff check

# Formatting
uv run ruff format

# Type checking
uv run mypy src/

# All quality checks
uv run pre-commit run --all-files
```

## 🛠️ Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all functions and methods
- Write comprehensive docstrings in Google/NumPy style
- Keep functions focused and small (single responsibility principle)
- Use meaningful variable and function names

### Testing

- Write tests for all new functionality
- Maintain or improve test coverage (currently at 100%)
- Use `pytest` for all tests
- Include both unit tests and integration tests
- Test edge cases and error conditions

### Documentation

- Update docstrings for any API changes
- Add examples for new features
- Update README.md if needed
- Include type hints in all function signatures

## 🔍 Project Architecture

The project is organized into three main analyzers:

```
src/mlflow_dep_analyzer/
├── __init__.py              # Public API
├── unified_analyzer.py      # Main analyzer (recommended)
├── code_path_analyzer.py    # Local file discovery
├── requirements_analyzer.py # External package discovery
└── config.py               # Configuration system
```

### Key Components

1. **UnifiedDependencyAnalyzer**: Complete analysis combining requirements and code paths
2. **CodePathAnalyzer**: Specialized for finding local file dependencies
3. **HybridRequirementsAnalyzer**: Focused on external package dependencies
4. **AnalyzerConfig**: Configuration system for customizing behavior

## 📝 Pull Request Process

### Before Submitting

1. **Run the full test suite**: `uv run pytest`
2. **Check code quality**: `uv run pre-commit run --all-files`
3. **Update documentation**: If you've changed the API
4. **Add tests**: For any new functionality
5. **Update CHANGELOG**: If applicable

### PR Guidelines

- **Clear title**: Briefly describe what the PR does
- **Detailed description**: Explain the motivation and approach
- **Link issues**: Reference any related issues
- **Small, focused changes**: Easier to review and merge
- **Test coverage**: Maintain or improve coverage

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for functionality
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🐛 Bug Reports

### Before Reporting

1. Check existing [issues](https://github.com/andrewgross/mlflow-dep-analyzer/issues)
2. Try the latest version
3. Create a minimal reproduction example

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Code example
2. Expected vs actual behavior

**Environment**
- Python version:
- MLflow version:
- mlflow-dep-analyzer version:
- OS:

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

We welcome feature requests! Please:

1. Check if the feature already exists
2. Describe the use case and motivation
3. Provide examples of how it would work
4. Consider if it fits the project scope

## 🏷️ Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📦 Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Create GitHub release with tag
4. Automated CI publishes to PyPI

## 🤝 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions
- Collaborate openly

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Publishing private information
- Any conduct deemed inappropriate

## 📚 Resources

### Learning Materials

- [Python AST Documentation](https://docs.python.org/3/library/ast.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Pytest Documentation](https://docs.pytest.org/)

### Project Links

- [Documentation](https://github.com/andrewgross/mlflow-dep-analyzer)
- [Issue Tracker](https://github.com/andrewgross/mlflow-dep-analyzer/issues)
- [Discussions](https://github.com/andrewgross/mlflow-dep-analyzer/discussions)

## ❓ Questions?

- 💬 [GitHub Discussions](https://github.com/andrewgross/mlflow-dep-analyzer/discussions) for general questions
- 🐛 [GitHub Issues](https://github.com/andrewgross/mlflow-dep-analyzer/issues) for bugs and feature requests
- 📧 Contact maintainers for security issues

---

Thank you for contributing to MLflow Dependency Analyzer! 🙏

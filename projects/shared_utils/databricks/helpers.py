import os
from typing import Any

import mlflow


def get_projects_source_path() -> str:
    """Get the path to the projects directory for code_paths."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate back to find the projects directory
    projects_path = os.path.join(current_dir, "..", "..")
    return os.path.abspath(projects_path)


def log_model(
    model: Any,
    artifact_path: str,
    artifacts: dict[str, str] | None = None,
    conda_env: str | None = None,
    code_paths: list | None = None,
    **kwargs,
) -> None:
    """
    Enhanced log_model function that can include code_paths for portability.

    Args:
        model: The model instance to log
        artifact_path: MLflow artifact path
        artifacts: Dictionary of artifact names to paths
        conda_env: Conda environment specification
        code_paths: List of code paths to include (for portability)
        **kwargs: Additional arguments for mlflow.pyfunc.log_model
    """
    # Prepare artifacts if the model has prepare_artifacts method
    if hasattr(model, "prepare_artifacts") and artifacts is None:
        artifacts = model.prepare_artifacts()

    # Log model info
    if hasattr(model, "log_model_info"):
        model.log_model_info(f"Logging model to {artifact_path}")
        if code_paths:
            model.log_model_info(f"Including code paths: {code_paths}")

    # Log the model with MLflow
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=model,
        artifacts=artifacts,
        conda_env=conda_env,
        code_paths=code_paths,
        **kwargs,
    )


def log_model_with_code_paths(
    model: Any, artifact_path: str, artifacts: dict[str, str] | None = None, conda_env: str | None = None, **kwargs
) -> None:
    """
    Convenience function that automatically includes the projects source code.
    This is the recommended approach for portable model deployment.
    """
    projects_path = get_projects_source_path()

    log_model(
        model=model,
        artifact_path=artifact_path,
        artifacts=artifacts,
        conda_env=conda_env,
        code_paths=[projects_path],
        **kwargs,
    )

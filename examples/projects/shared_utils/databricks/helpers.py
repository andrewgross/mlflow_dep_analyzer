import logging
import os
from typing import Any

import mlflow
import pandas as pd

from ..constants import (
    MAX_PROCESSING_TIME_SECONDS,
    NEGATIVE_LABEL,
    NEGATIVE_THRESHOLD,
    NEUTRAL_LABEL,
    POSITIVE_LABEL,
    POSITIVE_THRESHOLD,
)
from ..text_utils import clean_text, extract_features, validate_text_length
from ..validation import sanitize_input, validate_model_config, validate_model_input

logger = logging.getLogger(__name__)


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


def preprocess_text_data(data: Any, clean_data: bool = True, validate_lengths: bool = True) -> pd.DataFrame:
    """
    Comprehensive text preprocessing function that uses shared utilities.

    Args:
        data: Input data (DataFrame, list of strings, or single string)
        clean_data: Whether to clean text using shared text utilities
        validate_lengths: Whether to validate text lengths

    Returns:
        Processed DataFrame with 'text' column
    """
    logger.info("Starting text preprocessing with databricks helpers")

    # Validate input data
    is_valid, errors = validate_model_input(data)
    if not is_valid:
        logger.error(f"Input validation failed: {errors}")
        raise ValueError(f"Input validation errors: {errors}")

    # Sanitize input
    data = sanitize_input(data)

    # Convert to DataFrame if needed
    if isinstance(data, list):
        df = pd.DataFrame({"text": data})
    elif isinstance(data, str):
        df = pd.DataFrame({"text": [data]})
    else:
        df = data.copy()

    # Clean text data if requested
    if clean_data:
        logger.info("Cleaning text data using shared utilities")
        df["text"] = df["text"].apply(lambda x: clean_text(x, remove_urls=True, remove_special_chars=True))

    # Validate text lengths if requested
    if validate_lengths:
        logger.info("Validating text lengths")
        valid_mask = df["text"].apply(validate_text_length)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} texts with invalid lengths, filtering them out")
            df = df[valid_mask].reset_index(drop=True)

    # Extract features for logging
    sample_features = extract_features(df["text"].iloc[0] if len(df) > 0 else "")
    logger.info(f"Sample text features: {sample_features}")

    logger.info(f"Preprocessing complete. Final dataset shape: {df.shape}")
    return df


def postprocess_predictions(predictions: list[Any], probabilities: list[list[float]] | None = None) -> pd.DataFrame:
    """
    Postprocess model predictions using shared constants and logic.

    Args:
        predictions: Raw model predictions
        probabilities: Optional prediction probabilities

    Returns:
        DataFrame with processed predictions and confidence scores
    """
    logger.info("Postprocessing predictions using databricks helpers")

    results = []

    for i, pred in enumerate(predictions):
        result = {"prediction": pred}

        if probabilities is not None and i < len(probabilities):
            prob = probabilities[i]

            # Apply thresholding logic using shared constants
            if len(prob) >= 2:  # Binary classification
                positive_prob = prob[1] if len(prob) > 1 else prob[0]
                negative_prob = prob[0] if len(prob) > 1 else 1 - prob[0]

                result["positive_probability"] = positive_prob
                result["negative_probability"] = negative_prob

                # Apply confidence thresholding
                if positive_prob >= POSITIVE_THRESHOLD:
                    result["confidence_label"] = POSITIVE_LABEL
                elif negative_prob >= NEGATIVE_THRESHOLD:
                    result["confidence_label"] = NEGATIVE_LABEL
                else:
                    result["confidence_label"] = NEUTRAL_LABEL

                result["confidence_score"] = max(positive_prob, negative_prob)
            else:
                result["confidence_score"] = prob[0] if prob else 0.0
        else:
            result["confidence_score"] = 0.5  # Default neutral confidence
            result["confidence_label"] = NEUTRAL_LABEL

        results.append(result)

    df_results = pd.DataFrame(results)
    logger.info(f"Postprocessing complete. Results shape: {df_results.shape}")
    return df_results


def create_model_metadata(model_type: str, version: str, **kwargs) -> dict[str, Any]:
    """
    Create standardized model metadata using shared constants.

    Args:
        model_type: Type of the model
        version: Model version
        **kwargs: Additional metadata fields

    Returns:
        Dictionary with standardized metadata
    """
    metadata = {
        "model_type": model_type,
        "version": version,
        "created_with": "databricks_helpers",
        "processing_config": {
            "max_processing_time": MAX_PROCESSING_TIME_SECONDS,
            "positive_threshold": POSITIVE_THRESHOLD,
            "negative_threshold": NEGATIVE_THRESHOLD,
            "labels": {"positive": POSITIVE_LABEL, "negative": NEGATIVE_LABEL, "neutral": NEUTRAL_LABEL},
        },
    }

    # Add any additional metadata
    metadata.update(kwargs)

    # Validate metadata
    is_valid, errors = validate_model_config(metadata)
    if not is_valid:
        logger.warning(f"Model metadata validation warnings: {errors}")

    return metadata


def save_model_with_metadata(
    model: Any, artifact_path: str, model_type: str, version: str, include_code_paths: bool = True, **kwargs
) -> str:
    """
    Save model with comprehensive metadata and optional code paths.

    Args:
        model: The model instance to save
        artifact_path: MLflow artifact path
        model_type: Type of the model
        version: Model version
        include_code_paths: Whether to include code paths for portability
        **kwargs: Additional arguments

    Returns:
        MLflow run ID
    """
    logger.info(f"Saving model with metadata: {model_type} v{version}")

    # Create metadata
    metadata = create_model_metadata(model_type, version, **kwargs)

    # Update model metadata if it has the attribute
    if hasattr(model, "metadata"):
        model.metadata.update(metadata)

    # Choose logging function based on code paths preference
    if include_code_paths:
        log_model_with_code_paths(model, artifact_path)
    else:
        log_model(model, artifact_path)

    # Log metadata as MLflow parameters
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, str(value))

    run_id = mlflow.active_run().info.run_id
    logger.info(f"Model saved successfully with run ID: {run_id}")
    return run_id

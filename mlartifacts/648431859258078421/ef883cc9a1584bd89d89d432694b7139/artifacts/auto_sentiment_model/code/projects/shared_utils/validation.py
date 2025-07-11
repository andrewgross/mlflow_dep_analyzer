"""
Data validation utilities for ML pipelines.
"""

import logging
from typing import Any

import pandas as pd

from .constants import MAX_PROCESSING_TIME_SECONDS, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: list[str] = None) -> tuple[bool, list[str]]:
    """
    Validate DataFrame structure and content.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    errors = []

    # Check if DataFrame is empty
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors

    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Check for null values in required columns
    for col in required_columns:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} null values")

    # Check data types
    if "text" in df.columns:
        non_string_mask = ~df["text"].apply(lambda x: isinstance(x, str))
        if non_string_mask.any():
            non_string_count = non_string_mask.sum()
            errors.append(f"Column 'text' has {non_string_count} non-string values")

    return len(errors) == 0, errors


def validate_model_input(data: Any) -> tuple[bool, list[str]]:
    """
    Validate input data for model prediction.

    Args:
        data: Input data to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if isinstance(data, pd.DataFrame):
        is_valid, validation_errors = validate_dataframe(data)
        errors.extend(validation_errors)
    elif isinstance(data, list):
        if not data:
            errors.append("Input list is empty")
        else:
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    errors.append(f"Item at index {i} is not a string")
    elif isinstance(data, str):
        if not data.strip():
            errors.append("Input string is empty")
    else:
        errors.append(f"Unsupported input type: {type(data)}")

    return len(errors) == 0, errors


def validate_model_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate model configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required config keys
    required_keys = ["model_type", "version"]
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        errors.append(f"Missing required config keys: {missing_keys}")

    # Validate specific values
    if "model_type" in config and not isinstance(config["model_type"], str):
        errors.append("model_type must be a string")

    if "version" in config and not isinstance(config["version"], str):
        errors.append("version must be a string")

    if "max_processing_time" in config:
        max_time = config["max_processing_time"]
        if not isinstance(max_time, int | float) or max_time <= 0:
            errors.append("max_processing_time must be a positive number")
        elif max_time > MAX_PROCESSING_TIME_SECONDS:
            errors.append(f"max_processing_time exceeds limit of {MAX_PROCESSING_TIME_SECONDS} seconds")

    return len(errors) == 0, errors


def sanitize_input(data: Any) -> Any:
    """
    Sanitize input data for processing.

    Args:
        data: Input data to sanitize

    Returns:
        Sanitized data
    """
    if isinstance(data, pd.DataFrame):
        # Remove any completely empty rows
        data = data.dropna(how="all")

        # Fill empty text values with empty string
        if "text" in data.columns:
            data["text"] = data["text"].fillna("")

        return data
    elif isinstance(data, list):
        # Remove None values and convert to strings
        return [str(item) if item is not None else "" for item in data]
    elif isinstance(data, str):
        return data.strip()
    else:
        return str(data) if data is not None else ""

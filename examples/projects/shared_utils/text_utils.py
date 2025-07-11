"""
Text processing utilities shared across the project.
"""

import logging
import re
from typing import Any

from .constants import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH, STOPWORDS_CUSTOM

logger = logging.getLogger(__name__)


def clean_text(text: str, remove_urls: bool = True, remove_special_chars: bool = True) -> str:
    """
    Clean and normalize text for processing.

    Args:
        text: Input text to clean
        remove_urls: Whether to remove URLs
        remove_special_chars: Whether to remove special characters

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower().strip()

    # Remove URLs if requested
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove special characters if requested
    if remove_special_chars:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_text_length(text: str, min_length: int = MIN_TEXT_LENGTH, max_length: int = MAX_TEXT_LENGTH) -> bool:
    """
    Validate text length against configured limits.

    Args:
        text: Text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        True if text length is valid
    """
    if not isinstance(text, str):
        return False

    text_length = len(text.strip())
    return min_length <= text_length <= max_length


def remove_stopwords(text: str, language: str = "english", custom_stopwords: list[str] | None = None) -> str:
    """
    Remove stopwords from text.

    Args:
        text: Input text
        language: Language for stopwords
        custom_stopwords: Additional stopwords to remove

    Returns:
        Text with stopwords removed
    """
    if not isinstance(text, str):
        return ""

    words = text.split()
    stopwords = set(STOPWORDS_CUSTOM.get(language, []))

    if custom_stopwords:
        stopwords.update(custom_stopwords)

    filtered_words = [word for word in words if word.lower() not in stopwords]
    return " ".join(filtered_words)


def extract_features(text: str) -> dict[str, Any]:
    """
    Extract basic features from text.

    Args:
        text: Input text

    Returns:
        Dictionary of extracted features
    """
    if not isinstance(text, str):
        return {}

    words = text.split()

    features = {
        "char_count": len(text),
        "word_count": len(words),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "sentence_count": len(text.split(".")),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
    }

    return features


def batch_process_texts(texts: list[str], batch_size: int = 32) -> list[str]:
    """
    Process texts in batches for efficiency.

    Args:
        texts: List of texts to process
        batch_size: Size of each batch

    Returns:
        List of processed texts
    """
    processed_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        processed_batch = [clean_text(text) for text in batch]
        processed_texts.extend(processed_batch)

        logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    return processed_texts

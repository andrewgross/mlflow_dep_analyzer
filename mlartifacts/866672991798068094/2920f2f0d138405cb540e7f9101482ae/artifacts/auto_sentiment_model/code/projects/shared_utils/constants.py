"""
Constants used across the project for text processing and sentiment analysis.
"""

# Text preprocessing constants
MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = 5000
DEFAULT_BATCH_SIZE = 32

# Sentiment analysis constants
POSITIVE_THRESHOLD = 0.6
NEGATIVE_THRESHOLD = 0.4
NEUTRAL_LABEL = "neutral"
POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"

# Text cleaning patterns
STOPWORDS_CUSTOM = {
    "english": ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"],
    "extended": ["http", "https", "www", "com", "org", "net", "edu", "gov"],
}

# Model configuration
MODEL_ARTIFACTS_DIR = "model_artifacts"
PREPROCESSING_CACHE_DIR = "preprocessing_cache"
MODEL_METADATA_KEYS = [
    "model_type",
    "version",
    "features",
    "preprocessing_steps",
    "training_date",
    "performance_metrics",
]

# Data validation constants
REQUIRED_COLUMNS = ["text"]
OPTIONAL_COLUMNS = ["label", "confidence", "metadata"]
SUPPORTED_FORMATS = ["csv", "json", "parquet"]

# Performance thresholds
MAX_PROCESSING_TIME_SECONDS = 300
MEMORY_LIMIT_MB = 1024
MAX_CONCURRENT_REQUESTS = 10

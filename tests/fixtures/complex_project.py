"""
Complex project fixture for testing dependency analysis on realistic nested codebases.

This module creates a comprehensive test fixture that mimics a real-world Python project
with deeply nested packages, complex interdependencies, and various import patterns.
"""

import shutil
from pathlib import Path


class ComplexProjectFixture:
    """Creates a complex project structure for testing dependency analysis."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.project_root = self.base_path / "complex_ml_project"

    def create_project(self) -> Path:
        """Create the complete complex project structure."""
        if self.project_root.exists():
            shutil.rmtree(self.project_root)

        self.project_root.mkdir(parents=True)

        # Create all components
        self._create_src_structure()
        self._create_tests_structure()
        self._create_config_and_scripts()
        self._create_data_pipelines()
        self._create_model_components()
        self._create_utils_and_helpers()
        self._create_web_api()
        self._create_cli_tools()

        return self.project_root

    def _create_src_structure(self):
        """Create the main src/ structure with core packages."""
        src = self.project_root / "src"
        src.mkdir()

        # Main package
        main_pkg = src / "ml_platform"
        main_pkg.mkdir()
        (main_pkg / "__init__.py").write_text('''
"""ML Platform - A comprehensive machine learning platform."""

__version__ = "1.0.0"

# Re-export main components for easier access
from .core.engine import MLEngine
from .models.registry import ModelRegistry
from .data.loader import DataLoader

__all__ = ["MLEngine", "ModelRegistry", "DataLoader"]
''')

        # Create core package with engine
        self._create_core_package(main_pkg)
        self._create_models_package(main_pkg)
        self._create_data_package(main_pkg)
        self._create_training_package(main_pkg)
        self._create_inference_package(main_pkg)

    def _create_core_package(self, main_pkg: Path):
        """Create core package with foundational classes."""
        core = main_pkg / "core"
        core.mkdir()
        (core / "__init__.py").write_text("""
from .engine import MLEngine
from .config import Config
from .exceptions import MLPlatformError

__all__ = ["MLEngine", "Config", "MLPlatformError"]
""")

        # Main engine with complex dependencies
        (core / "engine.py").write_text('''
import logging
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import threading
import multiprocessing

# External dependencies
import pandas as pd
import numpy as np
import sklearn.base
import mlflow
import mlflow.tracking
from flask import Flask

# Internal imports
from ..models.registry import ModelRegistry
from ..data.loader import DataLoader
from ..training.trainer import ModelTrainer
from ..inference.predictor import Predictor
from .config import Config
from .exceptions import MLPlatformError
from ..utils.logging import setup_logging
from ..utils.metrics import MetricsCollector
from ..utils.cache import CacheManager


@dataclass
class MLEngine:
    """Main ML platform engine coordinating all components."""

    config: Config = field(default_factory=Config)
    model_registry: Optional[ModelRegistry] = None
    data_loader: Optional[DataLoader] = None
    trainer: Optional[ModelTrainer] = None
    predictor: Optional[Predictor] = None
    metrics_collector: Optional[MetricsCollector] = None
    cache_manager: Optional[CacheManager] = None
    _app: Optional[Flask] = None
    _executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        """Initialize components after creation."""
        self.logger = setup_logging(self.__class__.__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all platform components."""
        try:
            with self._lock:
                self.model_registry = ModelRegistry(self.config)
                self.data_loader = DataLoader(self.config)
                self.trainer = ModelTrainer(self.config, self.model_registry)
                self.predictor = Predictor(self.config, self.model_registry)
                self.metrics_collector = MetricsCollector(self.config)
                self.cache_manager = CacheManager(self.config)
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.max_workers
                )
                self.logger.info("ML Engine initialized successfully")
        except Exception as e:
            raise MLPlatformError(f"Failed to initialize ML Engine: {e}")

    async def train_model_async(self, model_config: Dict[str, Any]) -> str:
        """Train a model asynchronously."""
        loop = asyncio.get_event_loop()

        def _train():
            return self.trainer.train(model_config)

        return await loop.run_in_executor(self._executor, _train)

    def parallel_inference(self, data: List[Dict], model_id: str) -> List[Any]:
        """Run inference on multiple data points in parallel."""
        with multiprocessing.Pool(processes=self.config.inference_processes) as pool:
            results = pool.starmap(
                self.predictor.predict_single,
                [(model_id, item) for item in data]
            )
        return results
''')

        # Config management
        (core / "config.py").write_text('''
import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

# External dependencies
import pydantic
from pydantic import BaseSettings, validator

# Internal imports
from .exceptions import ConfigurationError


class Settings(BaseSettings):
    """Pydantic-based settings management."""

    database_url: str = "sqlite:///ml_platform.db"
    redis_url: str = "redis://localhost:6379"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_storage_path: str = "./models"
    data_storage_path: str = "./data"
    log_level: str = "INFO"
    max_workers: int = 4
    inference_processes: int = 2
    cache_ttl: int = 3600

    class Config:
        env_file = ".env"
        env_prefix = "ML_PLATFORM_"


@dataclass
class Config:
    """Configuration management for ML Platform."""

    settings: Settings = field(default_factory=Settings)
    custom_config: Dict[str, Any] = field(default_factory=dict)
    _config_path: Optional[Path] = None

    def __post_init__(self):
        """Load configuration from file if available."""
        self._load_config_file()

    def _load_config_file(self):
        """Load configuration from YAML or JSON file."""
        for config_file in ["config.yaml", "config.yml", "config.json"]:
            config_path = Path(config_file)
            if config_path.exists():
                self._config_path = config_path
                with open(config_path) as f:
                    if config_path.suffix in [".yaml", ".yml"]:
                        self.custom_config = yaml.safe_load(f) or {}
                    else:
                        self.custom_config = json.load(f) or {}
                break

    @property
    def database_url(self) -> str:
        return self.custom_config.get("database_url", self.settings.database_url)

    @property
    def max_workers(self) -> int:
        return self.custom_config.get("max_workers", self.settings.max_workers)

    @property
    def inference_processes(self) -> int:
        return self.custom_config.get("inference_processes", self.settings.inference_processes)
''')

        # Custom exceptions
        (core / "exceptions.py").write_text('''
"""Custom exceptions for ML Platform."""


class MLPlatformError(Exception):
    """Base exception for ML Platform errors."""
    pass


class ConfigurationError(MLPlatformError):
    """Raised when there are configuration issues."""
    pass


class ModelError(MLPlatformError):
    """Raised when there are model-related issues."""
    pass


class DataError(MLPlatformError):
    """Raised when there are data-related issues."""
    pass


class TrainingError(MLPlatformError):
    """Raised when there are training-related issues."""
    pass


class InferenceError(MLPlatformError):
    """Raised when there are inference-related issues."""
    pass


class ValidationError(MLPlatformError):
    """Raised when data validation fails."""
    pass
''')

    def _create_models_package(self, main_pkg: Path):
        """Create models package with registry and model definitions."""
        models = main_pkg / "models"
        models.mkdir()
        (models / "__init__.py").write_text("""
from .registry import ModelRegistry
from .base import BaseModel
from .sklearn_models import SKLearnWrapper
from .pytorch_models import PyTorchWrapper

__all__ = ["ModelRegistry", "BaseModel", "SKLearnWrapper", "PyTorchWrapper"]
""")

        # Model registry with complex dependencies
        (models / "registry.py").write_text('''
import pickle
import json
import hashlib
import datetime
from typing import Dict, List, Optional, Any, Type, Union
from pathlib import Path
from dataclasses import dataclass, field
import threading
import sqlite3
from contextlib import contextmanager

# External dependencies
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import redis
from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, Text
from sqlalchemy.orm import sessionmaker

# Internal imports
from .base import BaseModel
from .sklearn_models import SKLearnWrapper
from .pytorch_models import PyTorchWrapper
from ..core.config import Config
from ..core.exceptions import ModelError
from ..utils.serialization import ModelSerializer
from ..utils.validation import ModelValidator


@dataclass
class ModelMetadata:
    """Metadata for registered models."""

    model_id: str
    name: str
    version: str
    model_type: str
    created_at: datetime.datetime
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    file_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "performance_metrics": json.dumps(self.performance_metrics),
            "hyperparameters": json.dumps(self.hyperparameters),
            "training_data_hash": self.training_data_hash,
            "file_path": self.file_path,
            "tags": json.dumps(self.tags)
        }


class ModelRegistry:
    """Central registry for managing ML models."""

    def __init__(self, config: Config):
        self.config = config
        self.storage_path = Path(config.settings.model_storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Database connection
        self.engine = create_engine(config.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Redis cache
        self.redis_client = redis.from_url(config.settings.redis_url)

        # MLflow tracking
        mlflow.set_tracking_uri(config.settings.mlflow_tracking_uri)

        # Model serializer and validator
        self.serializer = ModelSerializer()
        self.validator = ModelValidator()

        # Thread safety
        self._lock = threading.RLock()

        # Initialize database
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database tables."""
        # Create tables if they don't exist
        # (simplified for example)
        pass

    def register_model(self, model: BaseModel, metadata: ModelMetadata) -> str:
        """Register a new model."""
        with self._lock:
            try:
                # Validate model
                self.validator.validate(model)

                # Generate unique model ID
                model_id = self._generate_model_id(metadata)
                metadata.model_id = model_id

                # Serialize and save model
                model_path = self.storage_path / f"{model_id}.pkl"
                self.serializer.save(model, model_path)
                metadata.file_path = str(model_path)

                # Save to database
                self._save_metadata(metadata)

                # Cache in Redis
                self._cache_model_metadata(metadata)

                # Log to MLflow
                self._log_to_mlflow(model, metadata)

                return model_id

            except Exception as e:
                raise ModelError(f"Failed to register model: {e}")

    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """Retrieve a model by ID."""
        try:
            # Try cache first
            cached_path = self.redis_client.get(f"model_path:{model_id}")
            if cached_path:
                return self.serializer.load(cached_path.decode())

            # Fall back to database
            metadata = self._get_metadata(model_id)
            if metadata and metadata.file_path:
                model = self.serializer.load(metadata.file_path)
                # Cache for next time
                self.redis_client.setex(
                    f"model_path:{model_id}",
                    self.config.settings.cache_ttl,
                    metadata.file_path
                )
                return model

        except Exception as e:
            raise ModelError(f"Failed to retrieve model {model_id}: {e}")

        return None
''')

        # Base model interface
        (models / "base.py").write_text('''
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> 'BaseModel':
        """Load model from file."""
        pass
''')

    def _create_data_package(self, main_pkg: Path):
        """Create data package with loaders and processors."""
        data = main_pkg / "data"
        data.mkdir()
        (data / "__init__.py").write_text("""
from .loader import DataLoader
from .processor import DataProcessor
from .validator import DataValidator

__all__ = ["DataLoader", "DataProcessor", "DataValidator"]
""")

        # Complex data loader with multiple dependencies
        (data / "loader.py").write_text('''
import json
import csv
import sqlite3
import asyncio
from typing import Dict, List, Optional, Any, Union, Iterator, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

# External dependencies
import pandas as pd
import numpy as np
import boto3
import psycopg2
import pymongo
import redis
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine
import dask.dataframe as dd
from kafka import KafkaConsumer, KafkaProducer

# Internal imports
from .processor import DataProcessor
from .validator import DataValidator
from ..core.config import Config
from ..core.exceptions import DataError
from ..utils.cache import CacheManager
from ..utils.monitoring import DataMonitor


@dataclass
class DataSource:
    """Configuration for a data source."""

    source_type: str  # 'file', 'database', 's3', 'kafka', etc.
    connection_params: Dict[str, Any]
    query_params: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None


class DataLoader:
    """Flexible data loader supporting multiple sources and formats."""

    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor(config)
        self.validator = DataValidator(config)
        self.cache_manager = CacheManager(config)
        self.monitor = DataMonitor(config)

        # Database connections
        self._db_engines = {}
        self._mongo_clients = {}
        self._redis_client = redis.from_url(config.settings.redis_url)

        # AWS clients
        self._s3_client = boto3.client('s3')

        # Kafka clients
        self._kafka_producer = None
        self._kafka_consumer = None

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=config.settings.max_workers)
        self._lock = threading.RLock()

        self.logger = logging.getLogger(__name__)

    def load_data(self, source: DataSource) -> pd.DataFrame:
        """Load data from specified source."""
        try:
            # Check cache first
            if source.cache_key:
                cached_data = self.cache_manager.get(source.cache_key)
                if cached_data is not None:
                    self.logger.info(f"Data loaded from cache: {source.cache_key}")
                    return cached_data

            # Load based on source type
            if source.source_type == 'file':
                data = self._load_file(source)
            elif source.source_type == 'database':
                data = self._load_database(source)
            elif source.source_type == 's3':
                data = self._load_s3(source)
            elif source.source_type == 'kafka':
                data = self._load_kafka(source)
            elif source.source_type == 'mongodb':
                data = self._load_mongodb(source)
            else:
                raise DataError(f"Unsupported source type: {source.source_type}")

            # Validate data
            self.validator.validate(data)

            # Cache if requested
            if source.cache_key:
                self.cache_manager.set(source.cache_key, data)

            # Monitor data quality
            self.monitor.track_data_load(source, data)

            return data

        except Exception as e:
            self.logger.error(f"Failed to load data from {source.source_type}: {e}")
            raise DataError(f"Data loading failed: {e}")

    def _load_file(self, source: DataSource) -> pd.DataFrame:
        """Load data from file."""
        file_path = Path(source.connection_params['path'])

        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path, **source.query_params or {})
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == '.xlsx':
            return pd.read_excel(file_path)
        else:
            raise DataError(f"Unsupported file format: {file_path.suffix}")

    async def load_data_async(self, sources: List[DataSource]) -> List[pd.DataFrame]:
        """Load data from multiple sources asynchronously."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self._executor, self.load_data, source)
            for source in sources
        ]
        return await asyncio.gather(*tasks)

    def stream_data(self, source: DataSource, batch_size: int = 1000) -> Iterator[pd.DataFrame]:
        """Stream data in batches."""
        if source.source_type == 'kafka':
            yield from self._stream_kafka(source, batch_size)
        elif source.source_type == 'database':
            yield from self._stream_database(source, batch_size)
        else:
            # For other sources, load all and yield in chunks
            data = self.load_data(source)
            for i in range(0, len(data), batch_size):
                yield data.iloc[i:i + batch_size]
''')

    def _create_training_package(self, main_pkg: Path):
        """Create training package with trainers and experiments."""
        training = main_pkg / "training"
        training.mkdir()
        (training / "__init__.py").write_text("""
from .trainer import ModelTrainer
from .experiment import ExperimentManager
from .pipeline import TrainingPipeline

__all__ = ["ModelTrainer", "ExperimentManager", "TrainingPipeline"]
""")

        # Comprehensive trainer with hyperparameter optimization
        (training / "trainer.py").write_text('''
import json
import time
import random
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import threading

# External dependencies
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import optuna
import mlflow
import mlflow.sklearn
import wandb
from hyperopt import hp, fmin, tpe, Trials

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
import pytorch_lightning as pl

# Internal imports
from ..models.registry import ModelRegistry, ModelMetadata
from ..models.base import BaseModel
from ..models.sklearn_models import SKLearnWrapper
from ..models.pytorch_models import PyTorchWrapper
from ..data.loader import DataLoader
from ..data.processor import DataProcessor
from ..core.config import Config
from ..core.exceptions import TrainingError
from ..utils.metrics import MetricsCollector
from ..utils.visualization import TrainingVisualizer


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: str
    hyperparameters: Dict[str, Any]
    optimization_metric: str = "accuracy"
    optimization_direction: str = "maximize"
    n_trials: int = 100
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    early_stopping_patience: int = 10
    max_epochs: int = 100
    use_gpu: bool = False
    distributed_training: bool = False
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ModelTrainer:
    """Comprehensive model trainer with hyperparameter optimization."""

    def __init__(self, config: Config, model_registry: ModelRegistry):
        self.config = config
        self.model_registry = model_registry
        self.data_loader = DataLoader(config)
        self.data_processor = DataProcessor(config)
        self.metrics_collector = MetricsCollector(config)
        self.visualizer = TrainingVisualizer(config)

        # MLflow and Weights & Biases setup
        mlflow.set_tracking_uri(config.settings.mlflow_tracking_uri)

        # Optuna study for hyperparameter optimization
        self._study = None
        self._best_params = None

        self.logger = logging.getLogger(__name__)

    def train(self, training_config: TrainingConfig,
              train_data: pd.DataFrame,
              target_column: str) -> str:
        """Train a model with the given configuration."""

        try:
            # Start MLflow run
            with mlflow.start_run(run_name=training_config.experiment_name):
                # Log training configuration
                mlflow.log_params(training_config.hyperparameters)
                mlflow.set_tags({tag: True for tag in training_config.tags})

                # Prepare data
                X, y = self._prepare_training_data(train_data, target_column)
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                    X, y,
                    test_size=training_config.test_size,
                    random_state=training_config.random_state
                )

                # Optimize hyperparameters if requested
                if training_config.n_trials > 1:
                    best_params = self._optimize_hyperparameters(
                        training_config, X_train, y_train
                    )
                    training_config.hyperparameters.update(best_params)

                # Train final model
                model = self._train_model(training_config, X_train, y_train)

                # Evaluate model
                metrics = self._evaluate_model(model, X_test, y_test, training_config)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Create model metadata
                metadata = ModelMetadata(
                    model_id="",  # Will be generated
                    name=training_config.experiment_name or f"model_{int(time.time())}",
                    version="1.0.0",
                    model_type=training_config.model_type,
                    created_at=time.time(),
                    performance_metrics=metrics,
                    hyperparameters=training_config.hyperparameters,
                    training_data_hash=self._hash_data(train_data),
                    tags=training_config.tags
                )

                # Register model
                model_id = self.model_registry.register_model(model, metadata)

                # Log model to MLflow
                if isinstance(model, SKLearnWrapper):
                    mlflow.sklearn.log_model(model.model, "model")
                elif isinstance(model, PyTorchWrapper):
                    mlflow.pytorch.log_model(model.model, "model")

                self.logger.info(f"Model training completed. Model ID: {model_id}")
                return model_id

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise TrainingError(f"Model training failed: {e}")

    def _optimize_hyperparameters(self, config: TrainingConfig, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""

        def objective(trial):
            # Generate hyperparameters based on model type
            params = self._suggest_hyperparameters(trial, config.model_type)

            # Train and evaluate with cross-validation
            model = self._create_model(config.model_type, params)
            scores = sklearn.model_selection.cross_val_score(
                model, X, y,
                cv=config.cv_folds,
                scoring=config.optimization_metric,
                n_jobs=-1
            )

            return scores.mean()

        # Create or reuse study
        study_name = f"optimize_{config.model_type}_{int(time.time())}"
        self._study = optuna.create_study(
            study_name=study_name,
            direction=config.optimization_direction
        )

        # Optimize
        self._study.optimize(objective, n_trials=config.n_trials)

        self.logger.info(f"Best hyperparameters: {self._study.best_params}")
        return self._study.best_params

    def train_distributed(self, config: TrainingConfig, train_data: pd.DataFrame, target_column: str) -> str:
        """Train model using distributed processing."""

        if config.model_type in ["pytorch", "neural_network"]:
            return self._train_distributed_pytorch(config, train_data, target_column)
        else:
            return self._train_distributed_sklearn(config, train_data, target_column)

    def _train_distributed_pytorch(self, config: TrainingConfig, train_data: pd.DataFrame, target_column: str) -> str:
        """Train PyTorch model with distributed training."""

        # Setup distributed training
        if torch.cuda.device_count() > 1:
            # Multi-GPU training
            pass
        else:
            # Multi-process training
            pass

        # Implement distributed PyTorch training
        # (simplified for example)
        return self.train(config, train_data, target_column)
''')

    def _create_inference_package(self, main_pkg: Path):
        """Create inference package with predictors and inference pipeline."""
        inference = main_pkg / "inference"
        inference.mkdir()
        (inference / "__init__.py").write_text("""
from .predictor import Predictor
from .pipeline import InferencePipeline
from .batch_processor import BatchProcessor

__all__ = ["Predictor", "InferencePipeline", "BatchProcessor"]
""")

        # Main predictor with inference capabilities
        (inference / "predictor.py").write_text('''
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
import concurrent.futures
import threading

# External dependencies
import pandas as pd
import numpy as np
import torch
import onnx
import onnxruntime
import tritonclient.http as httpclient

# Internal imports
from ..models.registry import ModelRegistry
from ..models.base import BaseModel
from ..core.config import Config
from ..core.exceptions import InferenceError
from ..utils.metrics import MetricsCollector
from ..utils.cache import CacheManager


@dataclass
class PredictionRequest:
    """Request for model prediction."""

    model_id: str
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]]
    return_probabilities: bool = False
    explain_predictions: bool = False
    batch_size: Optional[int] = None


@dataclass
class PredictionResponse:
    """Response from model prediction."""

    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    model_id: str = ""
    inference_time_ms: float = 0.0
    metadata: Dict[str, Any] = None


class Predictor:
    """High-performance model predictor with caching and batching."""

    def __init__(self, config: Config, model_registry: ModelRegistry):
        self.config = config
        self.model_registry = model_registry
        self.metrics_collector = MetricsCollector(config)
        self.cache_manager = CacheManager(config)

        # Model cache
        self._model_cache = {}
        self._cache_lock = threading.RLock()

        # Async executor
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.settings.max_workers
        )

        self.logger = logging.getLogger(__name__)

    def predict(self, model_id: str, input_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Any]:
        """Make predictions using the specified model."""

        start_time = time.time()

        try:
            # Get model
            model = self._get_cached_model(model_id)
            if not model:
                raise InferenceError(f"Model not found: {model_id}")

            # Prepare input
            if isinstance(input_data, dict):
                input_data = [input_data]

            # Make predictions
            predictions = []
            for item in input_data:
                pred = model.predict(self._prepare_input(item))
                predictions.append(pred)

            # Track metrics
            inference_time = (time.time() - start_time) * 1000
            self.metrics_collector.record_histogram("inference_latency_ms", inference_time)
            self.metrics_collector.increment_counter("predictions_total", len(predictions))

            return predictions

        except Exception as e:
            self.metrics_collector.increment_counter("prediction_errors_total")
            self.logger.error(f"Prediction failed for model {model_id}: {e}")
            raise InferenceError(f"Prediction failed: {e}")

    async def predict_async(self, model_id: str, input_data: Dict[str, Any]) -> Any:
        """Make async prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.predict,
            model_id,
            input_data
        )

    def predict_batch(self, model_id: str, input_batch: List[Dict[str, Any]], batch_size: int = 32) -> List[Any]:
        """Make batch predictions with optimized batching."""

        all_predictions = []

        # Process in batches
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i + batch_size]
            batch_predictions = self.predict(model_id, batch)
            all_predictions.extend(batch_predictions)

        return all_predictions

    async def predict_batch_async(self, model_id: str, input_batch: List[Dict[str, Any]]) -> List[Any]:
        """Make async batch predictions."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.predict_batch,
            model_id,
            input_batch
        )

    def _get_cached_model(self, model_id: str) -> Optional[BaseModel]:
        """Get model from cache or load it."""
        with self._cache_lock:
            if model_id in self._model_cache:
                return self._model_cache[model_id]

            # Load model
            model = self.model_registry.get_model(model_id)
            if model:
                self._model_cache[model_id] = model

            return model

    def _prepare_input(self, input_data: Dict[str, Any]) -> Any:
        """Prepare input data for model."""
        # Convert dict to appropriate format for model
        return input_data
''')

        # Inference pipeline
        (inference / "pipeline.py").write_text('''
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging

# External dependencies
import pandas as pd
import numpy as np

# Internal imports
from .predictor import Predictor, PredictionRequest, PredictionResponse
from ..data.processor import DataProcessor
from ..utils.validation import DataValidator
from ..core.config import Config
from ..core.exceptions import InferenceError


@dataclass
class PipelineStage:
    """A stage in the inference pipeline."""

    name: str
    processor: Callable[[Any], Any]
    is_async: bool = False
    required: bool = True


class InferencePipeline:
    """Multi-stage inference pipeline with preprocessing and postprocessing."""

    def __init__(self, config: Config, predictor: Predictor):
        self.config = config
        self.predictor = predictor
        self.data_processor = DataProcessor(config)
        self.data_validator = DataValidator(config)

        self.stages = []
        self.logger = logging.getLogger(__name__)

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a processing stage to the pipeline."""
        self.stages.append(stage)

    def process(self, request: PredictionRequest) -> PredictionResponse:
        """Process prediction request through the pipeline."""

        try:
            # Validate input
            if not self.data_validator.validate(request.input_data):
                raise InferenceError("Input validation failed")

            # Process through stages
            processed_data = request.input_data
            for stage in self.stages:
                if stage.is_async:
                    processed_data = asyncio.run(stage.processor(processed_data))
                else:
                    processed_data = stage.processor(processed_data)

            # Make prediction
            predictions = self.predictor.predict(request.model_id, processed_data)

            # Create response
            response = PredictionResponse(
                predictions=predictions,
                model_id=request.model_id
            )

            return response

        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            raise InferenceError(f"Pipeline failed: {e}")
''')

        # Batch processor
        (inference / "batch_processor.py").write_text('''
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass
import logging
import time

# External dependencies
import pandas as pd
import numpy as np

# Internal imports
from .predictor import Predictor
from ..core.config import Config
from ..core.exceptions import InferenceError


@dataclass
class BatchJob:
    """Batch inference job."""

    job_id: str
    model_id: str
    input_data: List[Dict[str, Any]]
    batch_size: int = 32
    status: str = "pending"
    results: Optional[List[Any]] = None
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None


class BatchProcessor:
    """Batch processor for large-scale inference."""

    def __init__(self, config: Config, predictor: Predictor):
        self.config = config
        self.predictor = predictor

        self.jobs = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.settings.max_workers
        )

        self.logger = logging.getLogger(__name__)

    def submit_batch_job(self, job: BatchJob) -> str:
        """Submit a batch inference job."""

        job.created_at = time.time()
        job.status = "submitted"
        self.jobs[job.job_id] = job

        # Submit to executor
        future = self.executor.submit(self._process_batch_job, job)

        return job.job_id

    def _process_batch_job(self, job: BatchJob) -> None:
        """Process a batch job."""

        try:
            job.status = "processing"

            # Process in batches
            all_results = []
            for i in range(0, len(job.input_data), job.batch_size):
                batch = job.input_data[i:i + job.batch_size]
                batch_results = self.predictor.predict_batch(
                    job.model_id,
                    batch,
                    job.batch_size
                )
                all_results.extend(batch_results)

            job.results = all_results
            job.status = "completed"
            job.completed_at = time.time()

            self.logger.info(f"Batch job {job.job_id} completed successfully")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = time.time()

            self.logger.error(f"Batch job {job.job_id} failed: {e}")

    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get the status of a batch job."""
        return self.jobs.get(job_id)

    def get_job_results(self, job_id: str) -> Optional[List[Any]]:
        """Get results from a completed batch job."""
        job = self.jobs.get(job_id)
        if job and job.status == "completed":
            return job.results
        return None
''')

    def _create_tests_structure(self):
        """Create test structure for the ML platform."""
        tests = self.project_root / "tests"
        tests.mkdir()
        (tests / "__init__.py").touch()

        # Unit tests
        (tests / "test_core.py").write_text("""
import pytest
import pandas as pd
import numpy as np
from ml_platform.core.engine import MLEngine
from ml_platform.core.config import Config

class TestMLEngine:
    def test_engine_initialization(self):
        config = Config()
        engine = MLEngine(config)
        assert engine.config is not None
""")

        # Integration tests
        (tests / "test_integration.py").write_text("""
import pytest
import tempfile
from pathlib import Path
from ml_platform.core.engine import MLEngine
from ml_platform.core.config import Config

class TestIntegration:
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config()
            engine = MLEngine(config)
            # Basic integration test
            assert engine is not None
""")

    def _create_config_and_scripts(self):
        """Create configuration files and scripts."""
        # Configuration files
        (self.project_root / "config.yaml").write_text("""
database_url: "sqlite:///ml_platform.db"
redis_url: "redis://localhost:6379"
mlflow_tracking_uri: "http://localhost:5000"
model_storage_path: "./models"
data_storage_path: "./data"
log_level: "INFO"
max_workers: 4
""")

        (self.project_root / "requirements.txt").write_text("""
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
mlflow>=1.20.0
fastapi>=0.70.0
uvicorn>=0.15.0
redis>=4.0.0
""")

        # Setup script
        (self.project_root / "setup.py").write_text("""
from setuptools import setup, find_packages

setup(
    name="ml-platform",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "mlflow>=1.20.0",
    ]
)
""")

    def _create_data_pipelines(self):
        """Create data pipeline modules."""
        pipelines = self.project_root / "pipelines"
        pipelines.mkdir()
        (pipelines / "__init__.py").touch()

        (pipelines / "etl_pipeline.py").write_text("""
import pandas as pd
import numpy as np
from typing import Dict, Any
from ml_platform.data.loader import DataLoader
from ml_platform.data.processor import DataProcessor

class ETLPipeline:
    def __init__(self, config):
        self.data_loader = DataLoader(config)
        self.data_processor = DataProcessor(config)

    def run(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        # Extract
        raw_data = self.data_loader.load_data(source_config)

        # Transform
        processed_data = self.data_processor.process(raw_data)

        # Load (return for now)
        return processed_data
""")

    def _create_model_components(self):
        """Create additional model components."""
        # Add missing model files
        models = self.project_root / "src" / "ml_platform" / "models"

        (models / "sklearn_models.py").write_text("""
import pickle
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import sklearn.base
from .base import BaseModel

class SKLearnWrapper(BaseModel):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self
""")

        (models / "pytorch_models.py").write_text("""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from .base import BaseModel

class PyTorchWrapper(BaseModel):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        # Simplified training
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        return self.model(X).detach().numpy()

    def get_feature_importance(self):
        return None

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
        return self
""")

    def _create_utils_and_helpers(self):
        """Create comprehensive utils package."""
        utils = self.project_root / "src" / "ml_platform" / "utils"
        utils.mkdir()
        (utils / "__init__.py").write_text("""
from .logging import setup_logging
from .metrics import MetricsCollector
from .cache import CacheManager
from .serialization import ModelSerializer
from .validation import ModelValidator, DataValidator

__all__ = [
    "setup_logging", "MetricsCollector", "CacheManager",
    "ModelSerializer", "ModelValidator", "DataValidator"
]
""")

        # Complex logging setup
        (utils / "logging.py").write_text('''
import logging
import logging.handlers
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import threading

# External dependencies
import structlog
from pythonjsonlogger import jsonlogger
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration


class JSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional metadata."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = time.time()
        log_record['level'] = record.levelname
        log_record['thread_id'] = threading.get_ident()


def setup_logging(name: str,
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 structured: bool = True,
                 sentry_dsn: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging configuration."""

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    if structured:
        # Use structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Console handler with JSON format
        console_handler = logging.StreamHandler(sys.stdout)
        json_formatter = JSONFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        console_handler.setFormatter(json_formatter)
        logger.addHandler(console_handler)

    else:
        # Standard logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        if structured:
            file_handler.setFormatter(json_formatter)
        else:
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Sentry integration
    if sentry_dsn:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,        # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send errors as events
        )
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[sentry_logging]
        )

    return logger
''')

        # Metrics collection
        (utils / "metrics.py").write_text('''
import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import statistics

# External dependencies
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import statsd

# Internal imports
from ..core.config import Config


@dataclass
class MetricData:
    """Container for metric data."""

    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsCollector:
    """Comprehensive metrics collection and reporting."""

    def __init__(self, config: Config):
        self.config = config

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._counters = {}
        self._histograms = {}
        self._gauges = {}

        # StatsD client
        self.statsd_client = statsd.StatsClient('localhost', 8125)

        # In-memory metrics storage
        self._metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()

        # Performance tracking
        self._operation_times = defaultdict(list)

    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            # Prometheus
            if name not in self._counters:
                self._counters[name] = Counter(
                    name, f'Counter for {name}',
                    list(tags.keys()) if tags else [],
                    registry=self.registry
                )

            if tags:
                self._counters[name].labels(**tags).inc(value)
            else:
                self._counters[name].inc(value)

            # StatsD
            self.statsd_client.incr(name, value)

            # Internal storage
            metric = MetricData(name, value, time.time(), tags or {}, "counter")
            self._metrics_buffer[name].append(metric)

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self._lock:
            # Prometheus
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name, f'Histogram for {name}',
                    list(tags.keys()) if tags else [],
                    registry=self.registry
                )

            if tags:
                self._histograms[name].labels(**tags).observe(value)
            else:
                self._histograms[name].observe(value)

            # StatsD
            self.statsd_client.timing(name, value)

            # Internal storage
            metric = MetricData(name, value, time.time(), tags or {}, "histogram")
            self._metrics_buffer[name].append(metric)

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            # Prometheus
            if name not in self._gauges:
                self._gauges[name] = Gauge(
                    name, f'Gauge for {name}',
                    list(tags.keys()) if tags else [],
                    registry=self.registry
                )

            if tags:
                self._gauges[name].labels(**tags).set(value)
            else:
                self._gauges[name].set(value)

            # StatsD
            self.statsd_client.gauge(name, value)

            # Internal storage
            metric = MetricData(name, value, time.time(), tags or {}, "gauge")
            self._metrics_buffer[name].append(metric)

    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)

    def get_metrics_summary(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self._lock:
            if metric_name:
                metrics = list(self._metrics_buffer.get(metric_name, []))
                if not metrics:
                    return {}

                values = [m.value for m in metrics]
                return {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else None
                }
            else:
                return {
                    name: self.get_metrics_summary(name)
                    for name in self._metrics_buffer.keys()
                }


class OperationTimer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, operation_name: str):
        self.collector = collector
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_histogram(
                f"{self.operation_name}_duration_seconds",
                duration
            )

            if exc_type:
                self.collector.increment_counter(
                    f"{self.operation_name}_errors_total"
                )
            else:
                self.collector.increment_counter(
                    f"{self.operation_name}_success_total"
                )
''')

    def _create_web_api(self):
        """Create web API package with Flask/FastAPI."""
        api = self.project_root / "src" / "ml_platform" / "api"
        api.mkdir()
        (api / "__init__.py").write_text("")

        # FastAPI application
        (api / "app.py").write_text('''
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

# External dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, Field
import redis
import celery

# Internal imports
from ..core.engine import MLEngine
from ..core.config import Config
from ..models.registry import ModelRegistry
from ..inference.predictor import Predictor
from ..utils.metrics import MetricsCollector
from .middleware import RequestLoggingMiddleware, MetricsMiddleware
from .auth import authenticate_user
from .models import PredictionRequest, PredictionResponse, TrainingRequest


# Pydantic models for API
class PredictionRequest(BaseModel):
    model_id: str
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    return_probabilities: bool = False
    explain_prediction: bool = False


class PredictionResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    model_id: str
    inference_time_ms: float


class TrainingRequest(BaseModel):
    model_type: str
    hyperparameters: Dict[str, Any]
    data_source: Dict[str, Any]
    target_column: str
    experiment_name: Optional[str] = None
    optimization_trials: int = Field(default=10, ge=1, le=1000)


# Global app instance
app = FastAPI(
    title="ML Platform API",
    description="Comprehensive ML platform with training and inference capabilities",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Security
security = HTTPBearer()

# Global dependencies
config = Config()
ml_engine = MLEngine(config)
metrics_collector = MetricsCollector(config)
redis_client = redis.from_url(config.settings.redis_url)

# Celery for background tasks
celery_app = celery.Celery(
    'ml_platform',
    broker=config.settings.redis_url,
    backend=config.settings.redis_url
)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user."""
    return authenticate_user(credentials.credentials)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    user = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Make predictions using a trained model."""

    start_time = time.time()

    try:
        # Get predictor
        predictor = ml_engine.predictor

        # Make prediction
        if isinstance(request.data, list):
            predictions = await predictor.predict_batch_async(
                request.model_id,
                request.data
            )
        else:
            predictions = [await predictor.predict_async(
                request.model_id,
                request.data
            )]

        # Get probabilities if requested
        probabilities = None
        if request.return_probabilities:
            probabilities = await predictor.predict_proba_async(
                request.model_id,
                request.data
            )

        # Get explanations if requested
        explanations = None
        if request.explain_prediction:
            explanations = await predictor.explain_predictions_async(
                request.model_id,
                request.data
            )

        inference_time = (time.time() - start_time) * 1000

        # Log metrics in background
        background_tasks.add_task(
            log_prediction_metrics,
            request.model_id,
            len(predictions),
            inference_time
        )

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            explanations=explanations,
            model_id=request.model_id,
            inference_time_ms=inference_time
        )

    except Exception as e:
        metrics_collector.increment_counter("prediction_errors_total")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(
    request: TrainingRequest,
    user = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Start model training (asynchronous)."""

    try:
        # Start training in background
        task = train_model_background.delay(request.dict())

        return {
            "task_id": task.id,
            "status": "started",
            "message": "Training started successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@celery_app.task(bind=True)
def train_model_background(self, training_request: Dict[str, Any]):
    """Background task for model training."""

    try:
        # Update task status
        self.update_state(state='PROGRESS', meta={'step': 'loading_data'})

        # Load training data
        data_loader = ml_engine.data_loader
        # ... implementation

        self.update_state(state='PROGRESS', meta={'step': 'training'})

        # Train model
        trainer = ml_engine.trainer
        # ... implementation

        return {
            'status': 'completed',
            'model_id': 'trained_model_id',
            'metrics': {}
        }

    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise


def log_prediction_metrics(model_id: str, batch_size: int, inference_time: float):
    """Log prediction metrics."""
    metrics_collector.record_histogram("prediction_latency_ms", inference_time)
    metrics_collector.increment_counter("predictions_total", batch_size)
    metrics_collector.set_gauge("active_model_requests", 1, {"model_id": model_id})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )
''')

    def _create_cli_tools(self):
        """Create CLI tools package."""
        cli = self.project_root / "src" / "ml_platform" / "cli"
        cli.mkdir()
        (cli / "__init__.py").write_text("")

        # Main CLI application
        (cli / "main.py").write_text('''
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# External dependencies
import click
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd

# Internal imports
from ..core.engine import MLEngine
from ..core.config import Config
from ..data.loader import DataSource
from ..training.trainer import TrainingConfig
from ..models.registry import ModelMetadata


console = Console()
config = Config()
ml_engine = MLEngine(config)


@click.group()
@click.option('--config-file', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
def cli(config_file, log_level):
    """ML Platform CLI - Comprehensive machine learning platform."""

    if config_file:
        # Load custom config
        pass

    # Setup logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.group()
def data():
    """Data management commands."""
    pass


@data.command('load')
@click.argument('source_config', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output file path')
@click.option('--format', type=click.Choice(['csv', 'parquet', 'json']), default='csv')
def load_data(source_config, output, format):
    """Load data from configured source."""

    try:
        with open(source_config) as f:
            source_dict = json.load(f)

        source = DataSource(**source_dict)
        data = ml_engine.data_loader.load_data(source)

        if output:
            if format == 'csv':
                data.to_csv(output, index=False)
            elif format == 'parquet':
                data.to_parquet(output)
            elif format == 'json':
                data.to_json(output)

        console.print(f"[green]Data loaded successfully: {len(data)} rows[/green]")

        # Show sample
        table = Table(title="Data Sample")
        for col in data.columns[:5]:  # Show first 5 columns
            table.add_column(col)

        for _, row in data.head().iterrows():
            table.add_row(*[str(row[col]) for col in data.columns[:5]])

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        sys.exit(1)


@cli.group()
def models():
    """Model management commands."""
    pass


@models.command('list')
@click.option('--limit', default=10, help='Number of models to show')
def list_models(limit):
    """List registered models."""

    try:
        # Get models from registry
        # (simplified implementation)

        table = Table(title="Registered Models")
        table.add_column("Model ID")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Created")
        table.add_column("Accuracy")

        # Add sample data
        table.add_row("model_001", "RandomForest_v1", "sklearn", "2023-01-01", "0.95")
        table.add_row("model_002", "XGBoost_v1", "xgboost", "2023-01-02", "0.97")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")


@models.command('train')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--data-file', type=click.Path(exists=True), required=True)
@click.option('--target-column', required=True, help='Target column name')
@click.option('--experiment-name', help='Experiment name')
def train_model(config_file, data_file, target_column, experiment_name):
    """Train a new model."""

    try:
        # Load training configuration
        with open(config_file) as f:
            config_dict = json.load(f)

        training_config = TrainingConfig(**config_dict)
        if experiment_name:
            training_config.experiment_name = experiment_name

        # Load training data
        data = pd.read_csv(data_file)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task("Training model...", total=None)

            # Train model
            model_id = ml_engine.trainer.train(training_config, data, target_column)

            progress.update(task, description="Training completed!")

        console.print(f"[green]Model trained successfully: {model_id}[/green]")

    except Exception as e:
        console.print(f"[red]Error training model: {e}[/red]")
        sys.exit(1)


@cli.group()
def inference():
    """Inference commands."""
    pass


@inference.command('predict')
@click.argument('model_id')
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output file for predictions')
def predict(model_id, data_file, output):
    """Make predictions using a trained model."""

    try:
        # Load data
        data = pd.read_csv(data_file)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task("Making predictions...", total=None)

            # Make predictions
            predictions = ml_engine.predictor.predict_batch(model_id, data.to_dict('records'))

            progress.update(task, description="Predictions completed!")

        # Create results DataFrame
        results = data.copy()
        results['prediction'] = predictions

        if output:
            results.to_csv(output, index=False)
            console.print(f"[green]Predictions saved to {output}[/green]")
        else:
            # Show sample predictions
            table = Table(title="Predictions Sample")
            for col in results.columns:
                table.add_column(col)

            for _, row in results.head().iterrows():
                table.add_row(*[str(row[col]) for col in results.columns])

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error making predictions: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    cli()
''')


def create_complex_test_project(base_path: Path) -> Path:
    """Create a complex test project for comprehensive testing."""
    fixture = ComplexProjectFixture(base_path)
    return fixture.create_project()

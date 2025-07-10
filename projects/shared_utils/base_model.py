import os
import pickle
import logging
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.pyfunc
import pandas as pd
from pyspark.ml.feature import StringIndexerModel


logger = logging.getLogger(__name__)


class BaseModelV3(mlflow.pyfunc.PythonModel):
    """Base model class for MLflow models with artifact management."""
    
    def __init__(self):
        self.model = None
        self.artifacts = {}
        self.metadata = {}
        
    def prepare_artifacts(self) -> Dict[str, str]:
        """Prepare artifacts for serialization. Override in subclasses."""
        artifact_paths = {}
        
        # Save any StringIndexerModels or other Spark artifacts
        for name, artifact in self.artifacts.items():
            if isinstance(artifact, StringIndexerModel):
                artifact_path = f"{name}_indexer"
                artifact.write().overwrite().save(artifact_path)
                artifact_paths[name] = artifact_path
            else:
                # For other artifacts, pickle them
                artifact_path = f"{name}.pkl"
                with open(artifact_path, 'wb') as f:
                    pickle.dump(artifact, f)
                artifact_paths[name] = artifact_path
                
        return artifact_paths
        
    def load_context(self, context):
        """Load artifacts from MLflow context."""
        for name, path in context.artifacts.items():
            if name.endswith('_indexer'):
                # Load StringIndexerModel
                from pyspark.ml.feature import StringIndexerModel
                self.artifacts[name.replace('_indexer', '')] = StringIndexerModel.load(path)
            else:
                # Load pickled artifacts
                with open(path, 'rb') as f:
                    self.artifacts[name] = pickle.load(f)
                    
    def predict(self, context, model_input):
        """Base predict method. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")
        
    def log_model_info(self, message: str):
        """Helper method for logging model information."""
        logger.info(f"[{self.__class__.__name__}] {message}")
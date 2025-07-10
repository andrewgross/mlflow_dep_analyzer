import logging

import mlflow

logger = logging.getLogger(__name__)


class BaseModelV3(mlflow.pyfunc.PythonModel):
    """Base model class for MLflow models with artifact management."""

    def __init__(self):
        self.model = None
        self.artifacts = {}
        self.metadata = {}

    def prepare_artifacts(self) -> dict[str, str]:
        """
        Prepare artifacts for serialization. Override in subclasses.

        Returns:
            Dict mapping artifact names to local file paths
        """
        artifact_paths = {}

        # Save any StringIndexerModels or other Spark artifacts
        for name, artifact in self.artifacts.items():
            # Check if it's a Spark StringIndexerModel
            if hasattr(artifact, "write") and hasattr(artifact, "save"):
                # Spark model - save to directory
                artifact_path = f"{name}_indexer"
                artifact.write().overwrite().save(artifact_path)
                artifact_paths[f"{name}_indexer"] = artifact_path
            else:
                # For other artifacts, let MLflow handle serialization
                # Just mark them for inclusion - MLflow will serialize appropriately
                pass

        return artifact_paths

    def load_context(self, context):
        """Load artifacts from MLflow context."""
        # MLflow automatically makes artifacts available via context.artifacts
        # Each artifact name maps to the local path where MLflow extracted it
        for name, path in context.artifacts.items():
            if name.endswith("_indexer"):
                # Load StringIndexerModel from directory
                try:
                    from pyspark.ml.feature import StringIndexerModel

                    original_name = name.replace("_indexer", "")
                    self.artifacts[original_name] = StringIndexerModel.load(path)
                except Exception as e:
                    logger.warning(f"Failed to load StringIndexer {name}: {e}")
            # Other artifacts are handled by subclasses if needed

    def predict(self, context, model_input):
        """Base predict method. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")

    def log_model_info(self, message: str):
        """Helper method for logging model information."""
        logger.info(f"[{self.__class__.__name__}] {message}")

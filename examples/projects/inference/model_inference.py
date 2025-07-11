"""
Production inference pipeline for loading and running MLflow models.

This module provides a clean interface for loading saved models and running inference,
demonstrating that models are completely independent from the original repo code.
"""

import logging
from typing import Any

import mlflow
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInferencePipeline:
    """
    Production inference pipeline for MLflow models.

    This class demonstrates that saved models with code_paths are completely
    independent and can be loaded without any dependency on the original repo code.
    """

    def __init__(self, mlflow_tracking_uri: str = "http://127.0.0.1:5000"):
        """
        Initialize the inference pipeline.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.loaded_models = {}  # Cache for loaded models
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Initialized inference pipeline with MLflow URI: {mlflow_tracking_uri}")

    def load_model(self, model_uri: str, alias: str | None = None) -> None:
        """
        Load a model from MLflow.

        Args:
            model_uri: MLflow model URI (e.g., "runs:/run_id/artifact_path")
            alias: Optional alias to store the model under for quick access
        """
        try:
            logger.info(f"Loading model from URI: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)

            # Store model with alias or URI as key
            key = alias if alias else model_uri
            self.loaded_models[key] = {"model": model, "uri": model_uri}

            logger.info(f"Successfully loaded model with key: {key}")

        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {e}")
            raise

    def predict(self, model_key: str, input_data: Any) -> pd.DataFrame:
        """
        Run inference on loaded model.

        Args:
            model_key: Key of the loaded model (alias or URI)
            input_data: Input data for prediction

        Returns:
            Prediction results as DataFrame
        """
        if model_key not in self.loaded_models:
            raise ValueError(f"Model '{model_key}' not loaded. Available models: {list(self.loaded_models.keys())}")

        model_info = self.loaded_models[model_key]
        model = model_info["model"]

        logger.info(f"Running inference with model: {model_key}")
        logger.info(
            f"Input data type: {type(input_data)}, length: {len(input_data) if hasattr(input_data, '__len__') else 'N/A'}"
        )

        try:
            predictions = model.predict(input_data)
            logger.info(
                f"Inference completed. Output shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}"
            )
            return predictions

        except Exception as e:
            logger.error(f"Inference failed for model {model_key}: {e}")
            raise

    def batch_predict(self, model_key: str, input_batches: list[Any]) -> list[pd.DataFrame]:
        """
        Run batch inference on multiple input batches.

        Args:
            model_key: Key of the loaded model
            input_batches: List of input batches

        Returns:
            List of prediction DataFrames
        """
        logger.info(f"Running batch inference with {len(input_batches)} batches")
        results = []

        for i, batch in enumerate(input_batches):
            logger.info(f"Processing batch {i+1}/{len(input_batches)}")
            batch_results = self.predict(model_key, batch)
            results.append(batch_results)

        logger.info("Batch inference completed")
        return results

    def get_model_info(self, model_key: str) -> dict[str, Any]:
        """
        Get information about a loaded model.

        Args:
            model_key: Key of the loaded model

        Returns:
            Dictionary with model information
        """
        if model_key not in self.loaded_models:
            raise ValueError(f"Model '{model_key}' not loaded")

        model_info = self.loaded_models[model_key]

        # Try to get MLflow model info
        try:
            mlflow_info = mlflow.models.get_model_info(model_info["uri"])
            return {
                "uri": model_info["uri"],
                "flavors": list(mlflow_info.flavors.keys()) if mlflow_info.flavors else [],
                "model_uuid": mlflow_info.model_uuid,
                "run_id": mlflow_info.run_id,
                "artifact_path": mlflow_info.artifact_path,
                "signature": str(mlflow_info.signature) if mlflow_info.signature else None,
            }
        except Exception as e:
            logger.warning(f"Could not get MLflow info: {e}")
            return {"uri": model_info["uri"], "error": str(e)}

    def list_loaded_models(self) -> list[str]:
        """
        List all currently loaded models.

        Returns:
            List of model keys
        """
        return list(self.loaded_models.keys())

    def unload_model(self, model_key: str) -> None:
        """
        Unload a model from memory.

        Args:
            model_key: Key of the model to unload
        """
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            logger.info(f"Unloaded model: {model_key}")
        else:
            logger.warning(f"Model '{model_key}' not found for unloading")


class SentimentInferencePipeline(ModelInferencePipeline):
    """
    Specialized inference pipeline for sentiment analysis models.

    This class shows how to create domain-specific inference pipelines
    while maintaining complete independence from training code.
    """

    def analyze_sentiment(self, model_key: str, texts: list[str]) -> dict[str, Any]:
        """
        Analyze sentiment of texts with detailed results.

        Args:
            model_key: Key of the loaded sentiment model
            texts: List of texts to analyze

        Returns:
            Detailed sentiment analysis results
        """
        predictions = self.predict(model_key, texts)

        # Process results for sentiment analysis
        results = {
            "total_texts": len(texts),
            "predictions": [],
            "summary": {"positive": 0, "negative": 0, "neutral": 0},
        }

        for i, text in enumerate(texts):
            pred = predictions.iloc[i]

            prediction_result = {
                "text": text,
                "prediction": int(pred["prediction"]),
                "sentiment": "positive" if pred["prediction"] == 1 else "negative",
            }

            # Add confidence info if available
            if "confidence_label" in pred:
                prediction_result["confidence_label"] = pred["confidence_label"]
            if "confidence_score" in pred:
                prediction_result["confidence_score"] = float(pred["confidence_score"])
            if "positive_probability" in pred:
                prediction_result["positive_probability"] = float(pred["positive_probability"])
            if "negative_probability" in pred:
                prediction_result["negative_probability"] = float(pred["negative_probability"])

            results["predictions"].append(prediction_result)

            # Update summary
            if "confidence_label" in pred:
                label = pred["confidence_label"]
                if label in results["summary"]:
                    results["summary"][label] += 1
            else:
                sentiment = prediction_result["sentiment"]
                results["summary"][sentiment] += 1

        return results

    def batch_sentiment_analysis(self, model_key: str, text_batches: list[list[str]]) -> list[dict[str, Any]]:
        """
        Run sentiment analysis on multiple batches of texts.

        Args:
            model_key: Key of the loaded sentiment model
            text_batches: List of text batches

        Returns:
            List of sentiment analysis results for each batch
        """
        logger.info(f"Running batch sentiment analysis on {len(text_batches)} batches")
        results = []

        for i, batch in enumerate(text_batches):
            logger.info(f"Analyzing sentiment for batch {i+1}/{len(text_batches)} ({len(batch)} texts)")
            batch_results = self.analyze_sentiment(model_key, batch)
            results.append(batch_results)

        return results


def create_inference_pipeline(pipeline_type: str = "general", **kwargs) -> ModelInferencePipeline:
    """
    Factory function to create inference pipelines.

    Args:
        pipeline_type: Type of pipeline ("general" or "sentiment")
        **kwargs: Additional arguments for pipeline initialization

    Returns:
        Configured inference pipeline
    """
    if pipeline_type == "sentiment":
        return SentimentInferencePipeline(**kwargs)
    else:
        return ModelInferencePipeline(**kwargs)


# Example usage functions for testing
def example_basic_inference():
    """Example of basic model inference."""
    print("🚀 Basic Inference Example")

    # This example assumes you have a trained model
    # In practice, you would get the model URI from your training runs
    print("To use: create_inference_pipeline()")

    # Example model URI (replace with actual URI)
    # model_uri = "runs:/some-run-id/sentiment_model"
    # pipeline.load_model(model_uri, alias="sentiment_v1")

    # Example predictions
    # results = pipeline.predict("sentiment_v1", ["Great product!", "Poor quality"])
    # print(f"Predictions: {results}")

    print("✅ Basic inference pipeline ready (add model URI to test)")


def example_sentiment_inference():
    """Example of specialized sentiment inference."""
    print("🚀 Sentiment Inference Example")

    print("To use: create_inference_pipeline('sentiment')")

    # Example sentiment analysis
    # model_uri = "runs:/some-run-id/auto_sentiment_model"
    # pipeline.load_model(model_uri, alias="auto_sentiment")

    # texts = ["Amazing product!", "Terrible service", "Average quality"]
    # results = pipeline.analyze_sentiment("auto_sentiment", texts)
    # print(f"Sentiment Analysis: {results}")

    print("✅ Sentiment inference pipeline ready (add model URI to test)")


if __name__ == "__main__":
    example_basic_inference()
    example_sentiment_inference()

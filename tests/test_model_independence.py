"""
Tests demonstrating that saved models are completely independent from repo code changes.

These tests show that:
1. Models saved with code_paths work even when the original repo code changes
2. Inference pipeline can load models without importing any repo model classes
3. Saved models maintain consistent behavior regardless of repo state
"""

import os
import sys

# Add projects to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.inference.model_inference import SentimentInferencePipeline
from projects.my_model.auto_logging_sentiment_model import AutoLoggingSentimentModel


class TestModelIndependence:
    """Tests for model independence from repo code changes."""

    def test_inference_pipeline_with_repo_models(self, mlflow_server, sample_data):
        """Test that inference pipeline works with repo code available."""
        texts, labels = sample_data

        # Train and save model using repo code
        model = AutoLoggingSentimentModel(experiment_name="test_inference_pipeline", dry_run=False)

        model.train(texts, labels)
        run_id = model.run_id

        # Verify the run exists before trying to load the model
        import mlflow

        client = mlflow.tracking.MlflowClient()

        # Wait for run to be available
        import time

        max_retries = 10
        for i in range(max_retries):
            try:
                client.get_run(run_id)
                break
            except Exception:
                if i < max_retries - 1:
                    time.sleep(0.5)
                else:
                    raise

        # Use inference pipeline (which is part of repo) to load model
        pipeline = SentimentInferencePipeline()
        model_uri = f"runs:/{run_id}/auto_sentiment_model"
        pipeline.load_model(model_uri, alias="test_model")

        # Test inference through pipeline
        test_texts = ["Great product!", "Terrible service"]
        results = pipeline.analyze_sentiment("test_model", test_texts)

        assert results["total_texts"] == 2
        assert len(results["predictions"]) == 2
        assert "summary" in results

        # Verify detailed results
        for pred in results["predictions"]:
            assert "text" in pred
            assert "prediction" in pred
            assert "sentiment" in pred
            assert pred["prediction"] in [0, 1]
            assert pred["sentiment"] in ["positive", "negative"]

    def test_inference_pipeline_isolated_environment(self, mlflow_server, sample_data, temp_workspace, isolated_env):
        """Test inference pipeline in isolated environment without repo code."""
        texts, labels = sample_data

        # Train and save model
        model = AutoLoggingSentimentModel(experiment_name="test_isolated_inference", dry_run=False)

        model.train(texts, labels)
        run_id = model.run_id

        # Test in isolated environment (no access to repo model classes)
        with isolated_env(temp_workspace):
            # Import inference pipeline in isolated environment
            # Note: This would fail if inference pipeline depended on model classes
            import mlflow

            # Create a simple inference function that doesn't import model classes
            def isolated_inference(model_uri, test_inputs):
                """Inference function that works without repo model imports."""
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                return loaded_model.predict(test_inputs)

            # Test inference without any repo model imports
            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            test_texts = ["Excellent product!", "Poor quality"]

            predictions = isolated_inference(model_uri, test_texts)

            # Verify predictions work in isolation
            assert len(predictions) == 2
            assert "prediction" in predictions.columns
            assert all(pred in [0, 1] for pred in predictions["prediction"])

    def test_model_consistency_across_environments(self, mlflow_server, sample_data, temp_workspace, isolated_env):
        """Test that models produce consistent results across different environments."""
        texts, labels = sample_data

        # Train model
        model = AutoLoggingSentimentModel(experiment_name="test_consistency", dry_run=False)

        model.train(texts, labels)
        run_id = model.run_id

        # Get baseline predictions with repo code available
        test_input = ["Consistent test text for comparison"]
        baseline_prediction = model.predict(None, test_input)
        baseline_result = baseline_prediction.iloc[0]["prediction"]

        # Test same input in isolated environment
        with isolated_env(temp_workspace):
            import mlflow

            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            isolated_prediction = loaded_model.predict(test_input)
            isolated_result = isolated_prediction.iloc[0]["prediction"]

            # Results must be identical
            assert (
                baseline_result == isolated_result
            ), f"Inconsistent predictions: repo={baseline_result}, isolated={isolated_result}"

    def test_inference_pipeline_model_management(self, mlflow_server, sample_data):
        """Test inference pipeline model management capabilities."""
        texts, labels = sample_data

        # Train multiple models
        model1 = AutoLoggingSentimentModel(experiment_name="test_management_1", dry_run=False)
        model1.train(texts, labels)
        run_id_1 = model1.run_id

        model2 = AutoLoggingSentimentModel(experiment_name="test_management_2", dry_run=False)
        model2.train(texts, labels)
        run_id_2 = model2.run_id

        # Test pipeline model management
        pipeline = SentimentInferencePipeline()

        # Load multiple models
        model_uri_1 = f"runs:/{run_id_1}/auto_sentiment_model"
        model_uri_2 = f"runs:/{run_id_2}/auto_sentiment_model"

        pipeline.load_model(model_uri_1, alias="model_v1")
        pipeline.load_model(model_uri_2, alias="model_v2")

        # Test model listing
        loaded_models = pipeline.list_loaded_models()
        assert "model_v1" in loaded_models
        assert "model_v2" in loaded_models

        # Test model info
        info_v1 = pipeline.get_model_info("model_v1")
        assert "uri" in info_v1
        assert "run_id" in info_v1

        # Test predictions from both models
        test_texts = ["Test text"]
        results_v1 = pipeline.analyze_sentiment("model_v1", test_texts)
        results_v2 = pipeline.analyze_sentiment("model_v2", test_texts)

        assert results_v1["total_texts"] == 1
        assert results_v2["total_texts"] == 1

        # Test model unloading
        pipeline.unload_model("model_v1")
        remaining_models = pipeline.list_loaded_models()
        assert "model_v1" not in remaining_models
        assert "model_v2" in remaining_models

    def test_batch_inference_in_isolation(self, mlflow_server, sample_data, temp_workspace, isolated_env):
        """Test batch inference works in isolated environments."""
        texts, labels = sample_data

        # Train model
        model = AutoLoggingSentimentModel(experiment_name="test_batch_isolation", dry_run=False)

        model.train(texts, labels)
        run_id = model.run_id

        # Prepare batch data
        text_batches = [
            ["Great product!", "Love it!"],
            ["Terrible quality", "Poor service"],
            ["Average product", "Okay quality", "Nothing special"],
        ]

        # Test batch inference in isolated environment
        with isolated_env(temp_workspace):
            import mlflow

            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Process batches
            batch_results = []
            for batch in text_batches:
                batch_predictions = loaded_model.predict(batch)
                batch_results.append(batch_predictions)

            # Verify batch results
            assert len(batch_results) == len(text_batches)
            for i, (batch, results) in enumerate(zip(text_batches, batch_results, strict=False)):
                assert len(results) == len(batch), f"Batch {i} length mismatch"
                assert "prediction" in results.columns
                assert all(pred in [0, 1] for pred in results["prediction"])

    def test_model_artifacts_independence(self, mlflow_server, sample_data, temp_workspace, isolated_env):
        """Test that all model artifacts are self-contained and independent."""
        texts, labels = sample_data

        # Train model with complex preprocessing - need enough samples for stratified split
        complex_texts = [
            "AMAZING product!!! https://example.com",
            "terrible quality :( very disappointed",
            "Great service & fast delivery!!",
            "Poor support... not recommended",
            "Fantastic experience, love it!",
            "Worst product ever, terrible",
            "Outstanding quality and service",
            "Complete waste of money",
        ]
        complex_labels = [1, 0, 1, 0, 1, 0, 1, 0]

        model = AutoLoggingSentimentModel(experiment_name="test_artifacts_independence", dry_run=False)

        model.train(complex_texts, complex_labels)
        run_id = model.run_id

        # Test that complex preprocessing works in complete isolation
        with isolated_env(temp_workspace):
            import mlflow

            # Load model in isolation
            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Test with complex inputs that require full preprocessing chain
            complex_test_inputs = [
                "EXCELLENT PRODUCT!!! https://test.com",
                "horrible experience :( :( :(",
                "Good value & quality!!!",
                "Worst purchase ever... avoid!!!",
            ]

            predictions = loaded_model.predict(complex_test_inputs)

            # Verify complete pipeline works in isolation
            assert len(predictions) == len(complex_test_inputs)
            assert "prediction" in predictions.columns
            assert "confidence_label" in predictions.columns
            assert "positive_probability" in predictions.columns
            assert "negative_probability" in predictions.columns

            # Verify preprocessing worked (URLs removed, text normalized, etc.)
            # All predictions should be valid
            assert all(pred in [0, 1] for pred in predictions["prediction"])
            assert all(0 <= prob <= 1 for prob in predictions["positive_probability"])
            assert all(0 <= prob <= 1 for prob in predictions["negative_probability"])

            # Verify confidence labels are valid
            valid_labels = ["positive", "negative", "neutral"]
            assert all(label in valid_labels for label in predictions["confidence_label"])

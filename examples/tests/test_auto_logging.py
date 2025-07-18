"""
Tests for the auto-logging sentiment model that automatically logs to MLflow during training.
"""

import os
import sys

import mlflow

# Add projects to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.my_model.auto_logging_sentiment_model import AutoLoggingSentimentModel


class TestAutoLoggingSentimentModel:
    """Tests for auto-logging sentiment model functionality."""

    def test_auto_logging_training_with_mlflow_server(self, mlflow_server, sample_data):
        """Test that the model automatically logs itself during training."""
        texts, labels = sample_data

        # Create model with auto-logging enabled
        model = AutoLoggingSentimentModel(
            experiment_name="test_auto_logging",
            dry_run=False,  # Actually save the model
        )

        print(f"Training auto-logging model with {len(texts)} samples")

        # Train the model - this should automatically log everything
        trained_pipeline = model.train(texts, labels)

        # Verify the model was trained
        assert trained_pipeline is not None
        assert model.trained is True
        assert model.run_id is not None
        assert model.training_date is not None

        # Verify the model can make predictions
        test_data = ["Great product!", "Terrible service"]
        predictions = model.predict(None, test_data)

        assert len(predictions) == 2
        assert "prediction" in predictions.columns
        assert "confidence_label" in predictions.columns

        # Verify the model was logged to MLflow
        model_uri = f"runs:/{model.run_id}/auto_sentiment_model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Test the loaded model
        loaded_predictions = loaded_model.predict(test_data)
        assert len(loaded_predictions) == 2

    def test_auto_logging_dry_run_mode(self, mlflow_server, sample_data):
        """Test that dry run mode trains but doesn't save the model."""
        texts, labels = sample_data

        # Create model with dry run enabled
        model = AutoLoggingSentimentModel(
            experiment_name="test_dry_run",
            dry_run=True,  # Don't save the model
        )

        print("Training model in dry run mode")

        # Train the model
        trained_pipeline = model.train(texts, labels)

        # Verify the model was trained
        assert trained_pipeline is not None
        assert model.trained is True
        assert model.run_id is not None

        # In dry run mode, the model artifacts shouldn't be saved
        # But the run should still exist with parameters and metrics
        run = mlflow.get_run(model.run_id)

        # Check that parameters were logged
        assert "dataset_size" in run.data.params
        assert "train_size" in run.data.params
        assert "test_size" in run.data.params

        # Check that metrics were logged
        assert "train_accuracy" in run.data.metrics
        assert "test_accuracy" in run.data.metrics

    def test_auto_logging_experiment_creation(self, mlflow_server, sample_data):
        """Test that the model creates the specified experiment."""
        texts, labels = sample_data
        experiment_name = "test_experiment_creation"

        # Verify experiment doesn't exist yet
        try:
            existing_exp = mlflow.get_experiment_by_name(experiment_name)
            if existing_exp:
                mlflow.delete_experiment(existing_exp.experiment_id)
        except Exception:
            pass  # Experiment doesn't exist, which is what we want

        # Create and train model
        model = AutoLoggingSentimentModel(experiment_name=experiment_name, dry_run=True)

        model.train(texts, labels)

        # Verify experiment was created
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None
        assert experiment.name == experiment_name

    def test_auto_logging_run_naming(self, mlflow_server, sample_data):
        """Test that runs are named with timestamp and model type."""
        texts, labels = sample_data

        model = AutoLoggingSentimentModel(experiment_name="test_run_naming", dry_run=True)

        model.train(texts, labels)

        # Get the run and check its name
        run = mlflow.get_run(model.run_id)
        run_name = run.data.tags.get("mlflow.runName", "")

        # Run name should contain timestamp and model type
        assert "auto_logging_sentiment" in run_name
        assert len(run_name) > 20  # Should have timestamp prefix

    def test_auto_logging_comprehensive_metrics(self, mlflow_server, sample_data):
        """Test that comprehensive metrics are logged."""
        texts, labels = sample_data

        model = AutoLoggingSentimentModel(experiment_name="test_comprehensive_metrics", dry_run=True)

        model.train(texts, labels)

        run = mlflow.get_run(model.run_id)
        metrics = run.data.metrics

        # Check that all expected metrics are logged
        expected_metrics = [
            "train_accuracy",
            "test_accuracy",
            "precision_class_0",
            "recall_class_0",
            "f1_class_0",
            "precision_class_1",
            "recall_class_1",
            "f1_class_1",
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 0 <= metrics[metric] <= 1, f"Invalid metric value for {metric}: {metrics[metric]}"

    def test_auto_logging_parameters(self, mlflow_server, sample_data):
        """Test that training parameters are logged."""
        texts, labels = sample_data

        model = AutoLoggingSentimentModel(experiment_name="test_parameters", dry_run=True)

        model.train(texts, labels)

        run = mlflow.get_run(model.run_id)
        params = run.data.params

        # Check expected parameters
        expected_params = [
            "dataset_size",
            "train_size",
            "test_size",
            "positive_samples",
            "negative_samples",
            "training_date",
        ]

        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

        # Verify parameter values make sense
        assert int(params["dataset_size"]) == len(texts)
        assert int(params["positive_samples"]) + int(params["negative_samples"]) == len(labels)

    def test_auto_logging_model_loading_and_prediction(self, mlflow_server, sample_data):
        """Test loading the auto-logged model and making predictions."""
        texts, labels = sample_data

        model = AutoLoggingSentimentModel(
            experiment_name="test_model_loading",
            dry_run=False,  # Actually save the model
        )

        # Train and auto-log the model
        model.train(texts, labels)

        # Load the model from MLflow
        model_uri = f"runs:/{model.run_id}/auto_sentiment_model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Test predictions with various inputs
        test_cases = [
            ["Amazing product, love it!"],
            ["Terrible quality, very disappointed"],
            ["Average, nothing special", "Great value for money"],
        ]

        for test_input in test_cases:
            predictions = loaded_model.predict(test_input)
            assert len(predictions) == len(test_input)
            assert "prediction" in predictions.columns
            assert "confidence_label" in predictions.columns
            assert "positive_probability" in predictions.columns
            assert "negative_probability" in predictions.columns

    def test_auto_logging_model_isolated_environment(self, mlflow_server, sample_data, temp_workspace, isolated_env):
        """Test that auto-logged models work in completely isolated environments."""
        texts, labels = sample_data

        # Train and auto-log model
        model = AutoLoggingSentimentModel(
            experiment_name="test_isolated_environment",
            dry_run=False,  # Actually save the model
        )

        model.train(texts, labels)
        run_id = model.run_id

        # Test loading in isolated environment (no access to repo code)
        with isolated_env(temp_workspace):
            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Test predictions in isolated environment
            test_data = ["Great product!", "Terrible service"]
            predictions = loaded_model.predict(test_data)

            assert len(predictions) == 2
            assert "prediction" in predictions.columns
            assert "confidence_label" in predictions.columns

    def test_auto_logging_model_independence_from_repo_changes(
        self, mlflow_server, sample_data, temp_workspace, isolated_env
    ):
        """Test that auto-logged models are independent of repo code changes."""
        texts, labels = sample_data

        # Train and save model
        model = AutoLoggingSentimentModel(experiment_name="test_independence", dry_run=False)

        model.train(texts, labels)
        run_id = model.run_id

        # Get baseline prediction from current model
        baseline_prediction = model.predict(None, ["Test text for consistency"])
        baseline_result = baseline_prediction.iloc[0]["prediction"]

        # Load model in isolated environment (simulating different deployment)
        with isolated_env(temp_workspace):
            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Test same input in isolated environment
            isolated_prediction = loaded_model.predict(["Test text for consistency"])
            isolated_result = isolated_prediction.iloc[0]["prediction"]

            # Results should be identical - model is independent
            assert (
                baseline_result == isolated_result
            ), f"Model behavior changed: baseline={baseline_result}, isolated={isolated_result}"

    def test_auto_logging_with_complex_preprocessing_in_isolation(
        self, mlflow_server, sample_data, temp_workspace, isolated_env
    ):
        """Test that complex preprocessing pipeline works in isolation."""
        texts, labels = sample_data

        # Use texts with special characters and URLs to test preprocessing
        # Need enough samples for stratified split
        complex_texts = [
            "I LOVE this product!!! https://example.com/product",
            "Terrible quality :( Would not recommend!",
            "Amazing service & fast delivery!!",
            "Poor customer support... Very disappointed",
            "EXCELLENT experience!!! https://test.com",
            "horrible service :( very disappointed",
            "GREAT quality & fast shipping!",
            "worst purchase ever... avoid this!",
        ]
        complex_labels = [1, 0, 1, 0, 1, 0, 1, 0]

        # Train model with complex preprocessing
        model = AutoLoggingSentimentModel(experiment_name="test_complex_preprocessing", dry_run=False)

        model.train(complex_texts, complex_labels)
        run_id = model.run_id

        # Test complex preprocessing in isolated environment
        with isolated_env(temp_workspace):
            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Test with similar complex input
            test_input = ["EXCELLENT product!!! https://test.com", "horrible service :("]
            predictions = loaded_model.predict(test_input)

            assert len(predictions) == 2
            assert "prediction" in predictions.columns
            # Verify preprocessing worked (should handle caps, URLs, special chars)
            assert all(pred in [0, 1] for pred in predictions["prediction"])

    def test_auto_logging_model_artifacts_isolation(self, mlflow_server, sample_data, temp_workspace, isolated_env):
        """Test that all model artifacts are properly isolated and self-contained."""
        texts, labels = sample_data

        model = AutoLoggingSentimentModel(experiment_name="test_artifacts_isolation", dry_run=False)

        model.train(texts, labels)
        run_id = model.run_id

        # Load in isolated environment and verify all components work
        with isolated_env(temp_workspace):
            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Test that all the complex preprocessing chain works
            test_cases = [
                "Simple positive text",
                "Simple negative text",
                "Complex text with URLs http://example.com and symbols!@#",
                "UPPERCASE TEXT WITH EMOTIONS!!!",
            ]

            predictions = loaded_model.predict(test_cases)

            # Verify complete pipeline works
            assert len(predictions) == len(test_cases)
            required_columns = ["prediction", "confidence_label", "positive_probability", "negative_probability"]
            for col in required_columns:
                assert col in predictions.columns

            # Verify predictions are reasonable
            assert all(0 <= prob <= 1 for prob in predictions["positive_probability"])
            assert all(0 <= prob <= 1 for prob in predictions["negative_probability"])
            assert all(pred in [0, 1] for pred in predictions["prediction"])

#!/usr/bin/env python3
"""
Standalone script to test loading models outside of the repo code.
This simulates loading a model in a completely different environment.
"""

import os
import sys
import tempfile


def test_model_loading_with_code_paths():
    """Test that we can load a model with code paths in an isolated environment."""
    print("🧪 Testing model loading with code paths...")

    # First, we need to train and save a model (this would normally be done in a notebook)
    import mlflow

    # Add projects to path temporarily for training
    project_root = os.path.dirname(__file__)
    sys.path.insert(0, project_root)

    try:
        from projects.my_model.sentiment_model import SentimentModel

        # Create and train model
        print("🤖 Training model...")
        model = SentimentModel()

        sample_texts = [
            "I love this product! Amazing quality.",
            "Terrible service, very disappointed.",
            "Great value for money!",
            "Poor quality, waste of money.",
        ]
        sample_labels = [1, 0, 1, 0]

        model.train(sample_texts, sample_labels)

        # Save model with code paths
        print("💾 Saving model with code paths...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        with mlflow.start_run(run_name="standalone_test"):
            run_id = model.save_model(artifact_path="test_model", include_code_paths=True)
            print(f"📝 Model saved with run ID: {run_id}")

    finally:
        # Remove projects from path
        if project_root in sys.path:
            sys.path.remove(project_root)

    # Now simulate loading in a completely isolated environment
    print("\n🔒 Testing in isolated environment (no access to project code)...")

    # Create temporary directory to simulate different workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Remove all project modules from sys.modules if they exist
            modules_to_remove = [key for key in sys.modules.keys() if key.startswith("projects")]
            for module in modules_to_remove:
                del sys.modules[module]

            # Try to load model (should work because of code_paths)
            model_uri = f"runs:/{run_id}/test_model"
            print(f"🔄 Loading model from: {model_uri}")

            loaded_model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Model loaded successfully!")

            # Test prediction
            test_data = ["This is fantastic!", "Horrible experience"]
            predictions = loaded_model.predict(test_data)

            print("🎯 Testing predictions:")
            for i, text in enumerate(test_data):
                pred = predictions.iloc[i]
                print(f"  Text: '{text}'")
                print(f"  Prediction: {pred['prediction']}")
                print(f"  Confidence: {pred['confidence_label']}")
                print()

            print("✅ All tests passed! Model works outside repo code.")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

        finally:
            os.chdir(original_cwd)

    return True


def test_model_loading_without_code_paths():
    """Test that model loading fails without code paths in isolated environment."""
    print("\n🧪 Testing model loading WITHOUT code paths...")

    # Train and save model without code paths
    import mlflow

    project_root = os.path.dirname(__file__)
    sys.path.insert(0, project_root)

    try:
        from projects.my_model.sentiment_model import SentimentModel
        from projects.shared_utils.databricks.helpers import log_model

        model = SentimentModel()
        sample_texts = ["Good product", "Bad service"]
        sample_labels = [1, 0]
        model.train(sample_texts, sample_labels)

        # Save without code paths
        print("💾 Saving model WITHOUT code paths...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        with mlflow.start_run(run_name="standalone_test_no_code"):
            log_model(model=model, artifact_path="test_model_no_code")
            run_id = mlflow.active_run().info.run_id
            print(f"📝 Model saved with run ID: {run_id}")

    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)

    # Try loading in isolated environment (should fail)
    print("🔒 Testing in isolated environment...")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Remove project modules
            modules_to_remove = [key for key in sys.modules.keys() if key.startswith("projects")]
            for module in modules_to_remove:
                del sys.modules[module]

            model_uri = f"runs:/{run_id}/test_model_no_code"
            print(f"🔄 Attempting to load model: {model_uri}")

            try:
                mlflow.pyfunc.load_model(model_uri)
                print("❌ Unexpected success - model should have failed to load!")
                return False
            except Exception as e:
                print(f"✅ Expected failure: {type(e).__name__}")
                print("✅ Model correctly failed to load without code paths")
                return True

        finally:
            os.chdir(original_cwd)


def test_auto_logging_model_loading_with_code_paths():
    """Test that auto-logging models work in isolated environments."""
    print("\n🧪 Testing auto-logging model with code paths...")

    # Train and save auto-logging model
    import mlflow

    project_root = os.path.dirname(__file__)
    sys.path.insert(0, project_root)

    try:
        from projects.my_model.auto_logging_sentiment_model import AutoLoggingSentimentModel

        # Create and train auto-logging model
        print("🤖 Training auto-logging model...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        model = AutoLoggingSentimentModel(
            experiment_name="standalone_auto_logging_test",
            dry_run=False,  # Actually save the model
        )

        sample_texts = [
            "Excellent product, highly recommend!",
            "Poor quality, very disappointed.",
            "Great customer service experience.",
            "Worst purchase I've ever made.",
        ]
        sample_labels = [1, 0, 1, 0]

        # Train with auto-logging (this automatically logs everything)
        model.train(sample_texts, sample_labels)
        run_id = model.run_id

        print(f"📝 Auto-logged model with run ID: {run_id}")

    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)

    # Test in isolated environment
    print("🔒 Testing in isolated environment...")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Remove project modules
            modules_to_remove = [key for key in sys.modules.keys() if key.startswith("projects")]
            for module in modules_to_remove:
                del sys.modules[module]

            # Load auto-logged model
            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            print(f"🔄 Loading auto-logged model: {model_uri}")

            loaded_model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Auto-logged model loaded successfully!")

            # Test predictions
            test_data = ["Amazing experience!", "Terrible service!"]
            predictions = loaded_model.predict(test_data)

            print("🎯 Testing predictions on auto-logged model:")
            for i, text in enumerate(test_data):
                pred = predictions.iloc[i]
                print(f"  Text: '{text}'")
                print(f"  Prediction: {pred['prediction']}")
                print(f"  Confidence: {pred['confidence_label']}")
                print()

            print("✅ Auto-logging model works perfectly in isolated environment!")
            return True

        except Exception as e:
            print(f"❌ Auto-logging test failed: {e}")
            return False

        finally:
            os.chdir(original_cwd)


def test_auto_logging_model_independence():
    """Test that auto-logged models are independent of repo code changes."""
    print("\n🧪 Testing auto-logging model independence from repo changes...")

    import mlflow

    project_root = os.path.dirname(__file__)
    sys.path.insert(0, project_root)

    try:
        from projects.my_model.auto_logging_sentiment_model import AutoLoggingSentimentModel

        # Train and save model
        print("🤖 Training and auto-logging model...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        model = AutoLoggingSentimentModel(experiment_name="independence_test", dry_run=False)

        sample_texts = ["Great product!", "Bad service!"]
        sample_labels = [1, 0]

        model.train(sample_texts, sample_labels)
        run_id = model.run_id

        print(f"📝 Model saved with run ID: {run_id}")

        # Test original model
        original_prediction = model.predict(None, ["Test text"])
        original_result = original_prediction.iloc[0]["prediction"]
        print(f"🎯 Original model prediction: {original_result}")

    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)

    # Now simulate "changing" the repo code by loading in isolated environment
    print("🔄 Testing model after 'repo changes' (isolated loading)...")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Remove all project modules to simulate fresh environment
            modules_to_remove = [key for key in sys.modules.keys() if key.startswith("projects")]
            for module in modules_to_remove:
                del sys.modules[module]

            # Load the saved model (which includes its own code via code_paths)
            model_uri = f"runs:/{run_id}/auto_sentiment_model"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Test that it still works the same way
            loaded_prediction = loaded_model.predict(["Test text"])
            loaded_result = loaded_prediction.iloc[0]["prediction"]

            print(f"🎯 Loaded model prediction: {loaded_result}")
            print(f"🔍 Predictions match: {original_result == loaded_result}")

            if original_result == loaded_result:
                print("✅ Model is independent of repo changes!")
                return True
            else:
                print("❌ Model behavior changed - not independent!")
                return False

        except Exception as e:
            print(f"❌ Independence test failed: {e}")
            return False

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    print("🚀 Starting standalone model loading tests\n")

    success1 = test_model_loading_with_code_paths()
    success2 = test_model_loading_without_code_paths()
    success3 = test_auto_logging_model_loading_with_code_paths()
    success4 = test_auto_logging_model_independence()

    if all([success1, success2, success3, success4]):
        print("\n🎉 All standalone tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

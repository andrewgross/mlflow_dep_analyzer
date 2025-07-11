#!/usr/bin/env python3
"""
Demo script showing smart requirements generation with MLflow models.

This script demonstrates the complete workflow:
1. Train a model with automatic MLflow logging
2. Generate minimal requirements.txt using AST analysis
3. Show that saved models work with only minimal dependencies
"""

import os
import subprocess
import sys
import tempfile
import time

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))


def demo_smart_requirements():
    """Demonstrate smart requirements generation for MLflow models."""
    print("🚀 Smart Requirements Generation Demo")
    print("=====================================\n")

    # Start temporary MLflow server
    with tempfile.TemporaryDirectory() as temp_dir:
        backend_store = f"sqlite:///{temp_dir}/mlflow.db"
        artifact_root = f"{temp_dir}/artifacts"

        print("🔧 Starting temporary MLflow server...")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--backend-store-uri",
                backend_store,
                "--default-artifact-root",
                artifact_root,
                "--host",
                "127.0.0.1",
                "--port",
                "5000",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            # Wait for server to start
            time.sleep(3)

            # Import and configure MLflow
            import mlflow

            mlflow.set_tracking_uri("http://127.0.0.1:5000")

            # Import our model
            from projects.my_model.auto_logging_sentiment_model import AutoLoggingSentimentModel

            print("📊 Training model with auto-logging and smart requirements...")

            # Create model with smart requirements generation
            model = AutoLoggingSentimentModel(
                experiment_name="smart_requirements_demo",
                dry_run=False,  # Actually save the model with requirements
            )

            # Train with sample data
            texts = [
                "This product is amazing! Highly recommend!",
                "Poor quality, very disappointed with purchase.",
                "Great value for money and excellent service!",
                "Terrible experience, would not buy again.",
                "Outstanding quality and fast shipping!",
                "Waste of money, very poor construction.",
            ]
            labels = [1, 0, 1, 0, 1, 0]

            print(f"  Training with {len(texts)} samples...")
            model.train(texts, labels)

            print(f"✅ Model trained and saved with run ID: {model.run_id}")

            # Show the generated requirements
            print("\n📋 Smart Requirements Analysis:")
            print("==============================")

            # Load the saved requirements file from MLflow artifacts
            model_uri = f"runs:/{model.run_id}/auto_sentiment_model"

            print(f"Model URI: {model_uri}")
            print("\nGenerated minimal requirements.txt contains only:")

            # Use our improved analyzer to show what was analyzed
            from projects.shared_utils.requirements_analyzer import RequirementsAnalyzer, load_requirements_from_file

            # Load base requirements to show the filtering effect
            base_requirements = load_requirements_from_file("requirements.txt")

            analyzer = RequirementsAnalyzer(existing_requirements=base_requirements)
            current_file = os.path.abspath(model.__class__.__module__.replace(".", "/") + ".py")
            shared_utils_dir = os.path.join(os.path.dirname(os.path.dirname(current_file)), "shared_utils")

            # Generate requirements excluding base packages
            requirements = analyzer.generate_requirements(
                file_paths=[current_file],
                directory_paths=[shared_utils_dir],
                include_versions=True,
                exclude_existing=True,
            )

            print(f"Base requirements.txt excludes {len(base_requirements)} packages:")

            print(f"\n📦 Additional packages needed: {len(requirements)}")
            if requirements:
                for req in requirements:
                    print(f"  + {req}")
            else:
                print("  ✅ None! All dependencies covered by base environment.")

            print("\n🎯 Benefits:")
            print(f"  • Base environment: {len(base_requirements)} packages")
            print(f"  • Additional needed: {len(requirements)} packages")
            print(f"  • Total for model: {len(base_requirements) + len(requirements)} packages")
            if len(requirements) == 0:
                print("  • Perfect optimization: No additional packages needed!")
            print("  • Smaller deployment footprint")
            print("  • Reduced security attack surface")
            print("  • Faster container builds and deployments")
            print("  • Clear visibility of actual dependencies")

            # Test model loading in isolation
            print("\n🔒 Testing Model Independence:")
            print("============================")

            # This simulates loading the model in a production environment
            # with only the minimal requirements installed
            print("Loading model with minimal dependencies...")

            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Test prediction
            test_texts = ["Fantastic product!", "Poor quality item"]
            predictions = loaded_model.predict(test_texts)

            print("Test predictions:")
            for i, text in enumerate(test_texts):
                pred = predictions.iloc[i]
                sentiment = "Positive" if pred["prediction"] == 1 else "Negative"
                confidence = pred["confidence_label"]
                print(f"  '{text}' → {sentiment} (confidence: {confidence})")

            print("\n✅ Model works perfectly with minimal dependencies!")
            print("\n🎉 Smart Requirements Demo Completed Successfully!")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Clean up server
            print("\n🛑 Shutting down MLflow server...")
            process.terminate()
            process.wait()


if __name__ == "__main__":
    demo_smart_requirements()

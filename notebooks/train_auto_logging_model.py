"""
Training notebook for auto-logging sentiment analysis model.

This notebook demonstrates the auto-logging approach where the train() method
automatically logs the model to MLflow after training completes.
"""

import os
import sys

import mlflow

# Add projects to path so we can import our model
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "projects"))

from my_model.auto_logging_sentiment_model import AutoLoggingSentimentModel

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def create_sample_data():
    """Create sample training data for sentiment analysis."""
    sample_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst thing I've ever bought. Terrible quality.",
        "Great service and fast delivery. Very satisfied!",
        "Poor customer support. I'm very disappointed.",
        "Excellent quality and value for money. Highly recommend!",
        "The product broke after one day. Complete waste of money.",
        "Outstanding performance! Exceeded my expectations.",
        "Horrible experience. Would not recommend to anyone.",
        "Perfect! Exactly what I was looking for.",
        "Defective product. Asked for refund immediately.",
        "Amazing features and easy to use interface.",
        "Terrible design and confusing instructions.",
        "Best purchase I've made this year!",
        "Completely useless and overpriced product.",
        "Good quality materials and solid construction.",
        "Cheap quality and poor workmanship.",
        "Fantastic customer service team!",
        "Worst company I've ever dealt with.",
        "Impressive technology and innovative design.",
        "Outdated and unreliable system.",
    ]

    sample_labels = [
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,  # First 10
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,  # Next 10
    ]

    return sample_texts, sample_labels


def main():
    """Main training function demonstrating auto-logging."""
    print("🚀 Starting auto-logging sentiment model training")

    # Create sample data
    print("📊 Creating sample training data")
    texts, labels = create_sample_data()

    print(f"Dataset size: {len(texts)} samples")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")

    # Initialize model with auto-logging enabled
    print("🤖 Initializing auto-logging sentiment model")
    model = AutoLoggingSentimentModel(
        experiment_name="auto_logging_sentiment_demo",
        dry_run=False,  # Actually save the model
    )

    print("🔧 Training model with automatic logging...")
    print("   (This will automatically create MLflow run, log metrics, and save model)")

    # The train method automatically handles all MLflow logging!
    model.train(texts, labels)

    print("✅ Training completed with automatic logging!")
    print(f"📝 Run ID: {model.run_id}")
    print(f"🕐 Training Date: {model.training_date}")

    # Get experiment info
    experiment = mlflow.get_experiment_by_name(model.experiment_name)
    print(f"🔗 MLflow UI: http://127.0.0.1:5000/#/experiments/{experiment.experiment_id}/runs/{model.run_id}")

    # Test prediction on new data
    print("\n🧪 Testing model with sample predictions")
    test_texts = [
        "This product is fantastic!",
        "I hate this terrible service.",
        "Average quality, nothing special.",
    ]

    predictions = model.predict(None, test_texts)
    print("\nSample predictions:")
    for i, text in enumerate(test_texts):
        pred = predictions.iloc[i]
        print(f"Text: '{text}'")
        print(f"  Prediction: {pred['prediction']}")
        print(f"  Confidence: {pred['confidence_label']} ({pred['confidence_score']:.3f})")
        print()

    # Demonstrate loading the auto-logged model
    print("🔄 Testing model loading from MLflow...")
    model_uri = f"runs:/{model.run_id}/auto_sentiment_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    loaded_predictions = loaded_model.predict(["Great product!"])
    print("✅ Successfully loaded and tested model from MLflow!")
    print(f"Loaded model prediction: {loaded_predictions.iloc[0]['prediction']}")


if __name__ == "__main__":
    main()

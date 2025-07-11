"""
Training notebook for sentiment analysis model.

This notebook demonstrates:
1. Importing the model class
2. Loading sample data
3. Preprocessing the data
4. Training the model
5. Saving the model with MLflow
"""

import os
import sys

import mlflow

# Add projects to path so we can import our model
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "projects"))

from my_model.sentiment_model import SentimentModel

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "sentiment_analysis_training"

try:
    experiment_id = mlflow.create_experiment(experiment_name)
except Exception:
    # Experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)


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
    """Main training function."""
    print("🚀 Starting sentiment model training")

    # Create sample data
    print("📊 Creating sample training data")
    texts, labels = create_sample_data()

    print(f"Dataset size: {len(texts)} samples")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")

    # Start MLflow run
    with mlflow.start_run(run_name="sentiment_model_training") as run:
        print(f"🔬 Started MLflow run: {run.info.run_id}")

        # Log training parameters
        mlflow.log_param("dataset_size", len(texts))
        mlflow.log_param("positive_samples", sum(labels))
        mlflow.log_param("negative_samples", len(labels) - sum(labels))

        # Initialize and train model
        print("🤖 Initializing sentiment model")
        model = SentimentModel()

        print("🔧 Training model with preprocessing...")
        model.train(texts, labels)

        # Log training metrics (simple accuracy on training set)
        print("📈 Evaluating model performance")
        train_predictions = model.predict(None, texts)
        accuracy = (train_predictions["prediction"] == labels).mean()
        mlflow.log_metric("training_accuracy", accuracy)

        print(f"Training accuracy: {accuracy:.3f}")

        # Save model using the new helper function
        print("💾 Saving model with metadata...")
        run_id = model.save_model(artifact_path="sentiment_model", include_code_paths=True)

        print("✅ Model saved successfully!")
        print(f"📝 Run ID: {run_id}")
        print(f"🔗 MLflow UI: http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id}")

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


if __name__ == "__main__":
    main()

"""
Training notebook simulation - demonstrates the problem and solution.
This simulates what would happen in a Databricks notebook.
"""

import sys
import os
import mlflow
import pandas as pd

# Add projects to path (simulates being in the workspace)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# These imports work when running from the repo
from projects.shared_utils.databricks.helpers import log_model, log_model_with_code_paths
from projects.my_model.sentiment_model import SentimentModel


def create_sample_data():
    """Create sample training data."""
    texts = [
        "I love this product, it's amazing!",
        "This is terrible, worst purchase ever.",
        "Great quality and fast shipping.",
        "Not worth the money, poor quality.",
        "Excellent service, highly recommend!",
        "Waste of time and money.",
        "Perfect for my needs, very satisfied.",
        "Disappointed with the results."
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    
    return texts, labels


def train_and_log_model_without_code_paths():
    """Train and log model WITHOUT code_paths (problematic approach)."""
    print("=== Training Model WITHOUT code_paths ===")
    
    # Create and train model
    model = SentimentModel()
    texts, labels = create_sample_data()
    model.train(texts, labels)
    
    # Log model without code_paths
    with mlflow.start_run(run_name="sentiment_model_no_code_paths"):
        log_model(
            model=model,
            artifact_path="sentiment_model",
            conda_env={
                'channels': ['defaults'],
                'dependencies': [
                    'python=3.8',
                    'pip',
                    {'pip': ['scikit-learn', 'pandas', 'numpy', 'mlflow']}
                ]
            }
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"Model logged without code_paths. Run ID: {run_id}")
        return run_id


def train_and_log_model_with_code_paths():
    """Train and log model WITH code_paths (solution approach)."""
    print("\n=== Training Model WITH code_paths ===")
    
    # Create and train model
    model = SentimentModel()
    texts, labels = create_sample_data()
    model.train(texts, labels)
    
    # Log model with code_paths
    with mlflow.start_run(run_name="sentiment_model_with_code_paths"):
        log_model_with_code_paths(
            model=model,
            artifact_path="sentiment_model",
            conda_env={
                'channels': ['defaults'],
                'dependencies': [
                    'python=3.8',
                    'pip',
                    {'pip': ['scikit-learn', 'pandas', 'numpy', 'mlflow']}
                ]
            }
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"Model logged with code_paths. Run ID: {run_id}")
        return run_id


if __name__ == "__main__":
    # Set up MLflow
    mlflow.set_experiment("sentiment_analysis_experiment")
    
    # Train and log both versions
    run_id_no_code = train_and_log_model_without_code_paths()
    run_id_with_code = train_and_log_model_with_code_paths()
    
    print(f"\nModels logged:")
    print(f"Without code_paths: {run_id_no_code}")
    print(f"With code_paths: {run_id_with_code}")
    
    # Create test data for inference
    test_texts = [
        "This is fantastic!",
        "I hate this product."
    ]
    
    test_df = pd.DataFrame({'text': test_texts})
    
    print(f"\nTesting inference with sample data:")
    print(test_df)
"""
Test script to demonstrate the import problem and solution.
This simulates loading models in a different environment where the original repo is not available.
"""

import sys
import os
import tempfile
import shutil
import mlflow
import mlflow.pyfunc
import pandas as pd
from unittest.mock import patch


def simulate_clean_environment():
    """Simulate a clean environment without access to the original repo."""
    # Remove projects from sys.path if it exists
    paths_to_remove = [p for p in sys.path if 'projects' in p or p.endswith('test_mlflow')]
    for path in paths_to_remove:
        if path in sys.path:
            sys.path.remove(path)
    
    # Remove any imported modules from our project
    modules_to_remove = [m for m in sys.modules.keys() if m.startswith('projects')]
    for module in modules_to_remove:
        del sys.modules[module]


def test_model_loading_without_code_paths(run_id):
    """Test loading a model that was logged without code_paths."""
    print(f"\n=== Testing Model Loading WITHOUT code_paths ===")
    print(f"Run ID: {run_id}")
    
    try:
        # Simulate clean environment
        simulate_clean_environment()
        
        # Try to load the model
        model_uri = f"runs:/{run_id}/sentiment_model"
        print(f"Attempting to load model from: {model_uri}")
        
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Model loaded successfully!")
        
        # Test prediction
        test_data = pd.DataFrame({'text': ["This is great!", "This is terrible!"]})
        predictions = loaded_model.predict(test_data)
        print("✅ Predictions successful!")
        print(predictions)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        return False
    
    return True


def test_model_loading_with_code_paths(run_id):
    """Test loading a model that was logged with code_paths."""
    print(f"\n=== Testing Model Loading WITH code_paths ===")
    print(f"Run ID: {run_id}")
    
    try:
        # Simulate clean environment
        simulate_clean_environment()
        
        # Try to load the model
        model_uri = f"runs:/{run_id}/sentiment_model"
        print(f"Attempting to load model from: {model_uri}")
        
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Model loaded successfully!")
        
        # Test prediction
        test_data = pd.DataFrame({'text': ["This is great!", "This is terrible!"]})
        predictions = loaded_model.predict(test_data)
        print("✅ Predictions successful!")
        print(predictions)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        return False
    
    return True


def test_model_in_isolated_directory(run_id):
    """Test loading model from a completely different directory."""
    print(f"\n=== Testing Model Loading in Isolated Directory ===")
    print(f"Run ID: {run_id}")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary directory: {temp_dir}")
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Simulate clean environment
            simulate_clean_environment()
            
            # Try to load the model
            model_uri = f"runs:/{run_id}/sentiment_model"
            print(f"Attempting to load model from: {model_uri}")
            
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Model loaded successfully!")
            
            # Test prediction
            test_data = pd.DataFrame({'text': ["This is great!", "This is terrible!"]})
            predictions = loaded_model.predict(test_data)
            print("✅ Predictions successful!")
            print(predictions)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"Error type: {type(e).__name__}")
            return False
        finally:
            # Change back to original directory
            os.chdir(original_cwd)


if __name__ == "__main__":
    # You need to provide run IDs from the training script
    print("This script demonstrates testing model loading in different environments.")
    print("Run the train_model.py script first to get run IDs.")
    
    # Example usage (replace with actual run IDs):
    # test_model_loading_without_code_paths("your_run_id_here")
    # test_model_loading_with_code_paths("your_run_id_here")
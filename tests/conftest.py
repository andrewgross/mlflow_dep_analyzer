import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

import mlflow
import mlflow.pyfunc
import pytest
import requests
from pyspark.sql import SparkSession


def find_free_port():
    """Find a free port for MLflow server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session for testing."""
    spark = (
        SparkSession.builder.appName("MLFlowPackagingTest")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def mlflow_server():
    """Start MLflow server for testing with proper lifecycle management."""
    # Create temporary directory for MLflow artifacts
    temp_dir = tempfile.mkdtemp(prefix="mlflow_test_")
    mlflow_dir = os.path.join(temp_dir, "mlflow_runs")
    os.makedirs(mlflow_dir, exist_ok=True)

    port = find_free_port()

    # Start MLflow server
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "mlflow",
            "server",
            "--backend-store-uri",
            f"sqlite:///{mlflow_dir}/mlflow.db",
            "--default-artifact-root",
            f"{mlflow_dir}/artifacts",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    server_url = f"http://127.0.0.1:{port}"
    max_attempts = 60
    for _attempt in range(max_attempts):
        try:
            response = requests.get(f"{server_url}/health", timeout=1)
            if response.status_code == 200:
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        time.sleep(0.5)
    else:
        # Get any error output
        stdout, stderr = process.communicate(timeout=5)
        process.terminate()
        raise RuntimeError(
            f"MLflow server failed to start after {max_attempts} attempts.\n"
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    # Set MLflow tracking URI
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(server_url)

    yield server_url

    # Cleanup
    mlflow.set_tracking_uri(original_uri)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    temp_dir = tempfile.mkdtemp(prefix="workspace_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    texts = [
        "I love this product, it's amazing!",
        "This is terrible, worst purchase ever.",
        "Great quality and fast shipping.",
        "Not worth the money, poor quality.",
        "Excellent service, highly recommend!",
        "Waste of time and money.",
        "Perfect for my needs, very satisfied.",
        "Disappointed with the results.",
        "Outstanding customer support!",
        "Complete garbage, avoid at all costs.",
    ]

    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

    return texts, labels


@contextmanager
def isolated_environment(temp_workspace):
    """Context manager to simulate isolated environment without access to project code."""
    # Save original state
    original_cwd = os.getcwd()
    original_sys_path = sys.path.copy()
    original_modules = set(sys.modules.keys())

    try:
        # Change to temp workspace
        os.chdir(temp_workspace)

        # Remove project paths from sys.path
        project_paths = [p for p in sys.path if "test_mlflow" in p]
        for path in project_paths:
            if path in sys.path:
                sys.path.remove(path)

        # Remove project modules from sys.modules
        project_modules = [m for m in sys.modules.keys() if m.startswith("projects")]
        for module in project_modules:
            if module in sys.modules:
                del sys.modules[module]

        yield

    finally:
        # Restore original state
        os.chdir(original_cwd)
        sys.path.clear()
        sys.path.extend(original_sys_path)

        # Clean up any new modules that were imported
        new_modules = set(sys.modules.keys()) - original_modules
        for module in new_modules:
            if module in sys.modules:
                del sys.modules[module]


@pytest.fixture
def isolated_env():
    """Fixture that provides the isolated environment context manager."""
    return isolated_environment


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    # Add project path to sys.path if not already there
    project_root = os.path.join(os.path.dirname(__file__), "..")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from projects.my_model.sentiment_model import SentimentModel

    texts, labels = sample_data
    model = SentimentModel()
    model.train(texts, labels)
    return model


@pytest.fixture
def test_predictions_data():
    """Create test data for predictions."""
    import pandas as pd

    return pd.DataFrame(
        {"text": ["This is fantastic!", "I hate this product.", "Great value for money!", "Terrible experience."]}
    )

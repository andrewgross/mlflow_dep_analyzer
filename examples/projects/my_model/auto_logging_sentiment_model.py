"""
Auto-logging sentiment analysis model that automatically logs training runs to MLflow.
This model follows the pattern of training and then automatically logging itself.
"""

import datetime

import mlflow
import pandas as pd
from projects.shared_utils.base_model import BaseModelV3
from projects.shared_utils.databricks.helpers import (
    postprocess_predictions,
    preprocess_text_data,
    save_model_with_metadata,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class AutoLoggingSentimentModel(BaseModelV3):
    """
    Sentiment analysis model with automatic MLflow logging.

    This model automatically logs itself to MLflow after training completes,
    similar to production ML training workflows.
    """

    def __init__(self, experiment_name: str = "auto_logging_sentiment", dry_run: bool = False):
        super().__init__()
        self.pipeline = None
        self.metadata = {"model_type": "auto_logging_sentiment", "version": "1.0.0", "features": ["text"]}
        self.experiment_name = experiment_name
        self.dry_run = dry_run
        self.training_date = None
        self.run_id = None
        self.trained = False

    def preprocess_input_data(self, data):
        """Preprocess input data using databricks helpers."""
        return preprocess_text_data(data, clean_data=True, validate_lengths=True)

    def train(self, texts: list, labels: list) -> Pipeline:
        """
        Train the sentiment model with automatic MLflow logging.

        This method:
        1. Sets up MLflow experiment and run
        2. Preprocesses data
        3. Trains the model with train/test split
        4. Logs parameters and metrics
        5. Automatically saves the model to MLflow
        6. Returns the trained pipeline
        """
        self.training_date = datetime.datetime.now(datetime.UTC)
        run_name = self.training_date.strftime("%Y_%m_%d_%H_%M_%S_%f") + f"__{self.metadata['model_type']}"

        self.log_model_info("Starting auto-logging training...")

        # Preprocess the training data
        self.log_model_info("Preprocessing training data...")
        processed_df = self.preprocess_input_data(texts)
        processed_texts = processed_df["text"].tolist()

        # Convert to DataFrame for easier handling
        df = pd.DataFrame({"text": processed_texts, "label": labels})

        # Set up MLflow experiment
        mlflow.set_experiment(self.experiment_name)

        # Start MLflow run and train with logging
        with mlflow.start_run(run_name=run_name) as mlflow_run:
            self.run_id = mlflow_run.info.run_id
            return self._train_with_logging(df)

    def _train_with_logging(self, df: pd.DataFrame) -> Pipeline:
        """Train model with comprehensive MLflow logging."""
        X = df["text"]
        y = df["label"]

        # Split the data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Log training parameters
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("positive_samples", sum(y))
        mlflow.log_param("negative_samples", len(y) - sum(y))
        mlflow.log_param("training_date", self.training_date.isoformat())

        # Create and train pipeline
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )

        self.log_model_info("Training pipeline...")
        # Train the model
        self.pipeline.fit(X_train, y_train)

        # Evaluate the model
        train_predictions = self.pipeline.predict(X_train)
        test_predictions = self.pipeline.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Log detailed classification report
        report = classification_report(y_test, test_predictions, output_dict=True)
        mlflow.log_metric("precision_class_0", report["0"]["precision"])
        mlflow.log_metric("recall_class_0", report["0"]["recall"])
        mlflow.log_metric("f1_class_0", report["0"]["f1-score"])
        mlflow.log_metric("precision_class_1", report["1"]["precision"])
        mlflow.log_metric("recall_class_1", report["1"]["recall"])
        mlflow.log_metric("f1_class_1", report["1"]["f1-score"])

        self.trained = True

        # Auto-save model if not dry run
        if not self.dry_run:
            self.log_model_info("Auto-logging model to MLflow...")

            # Generate smart requirements for model dependencies
            self._generate_and_log_requirements()

            # Use the save_model_with_metadata helper for comprehensive logging
            save_model_with_metadata(
                model=self,
                artifact_path="auto_sentiment_model",
                model_type=self.metadata["model_type"],
                version=self.metadata["version"],
                include_code_paths=True,
                features=self.metadata.get("features", []),
                training_framework="scikit-learn",
                training_date=self.training_date.isoformat(),
                dataset_size=len(df),
                test_accuracy=test_accuracy,
            )

        self._print_model_info(X_test, y_test, test_predictions)
        self.log_model_info("Auto-logging training completed!")

        return self.pipeline

    def _print_model_info(self, X_test, y_test, predictions):
        """Print model evaluation information."""
        accuracy = accuracy_score(y_test, predictions)

        self.log_model_info("Model Performance:")
        self.log_model_info(f"  Test Accuracy: {accuracy:.3f}")
        self.log_model_info(f"  Test Set Size: {len(X_test)}")

        # Print confusion matrix
        cm = confusion_matrix(y_test, predictions)
        self.log_model_info("  Confusion Matrix:")
        self.log_model_info(f"    [[{cm[0,0]}, {cm[0,1]}],")
        self.log_model_info(f"     [{cm[1,0]}, {cm[1,1]}]]")

    def _generate_and_log_requirements(self):
        """Generate smart requirements.txt for model dependencies using hybrid MLflow + AST analyzer."""
        try:
            import os

            # Get the model file and its dependencies first
            import sys
            import tempfile

            current_file = os.path.abspath(__file__)
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

            # Import from our new package
            from mlflow_dep_analyzer import HybridRequirementsAnalyzer, analyze_code_dependencies

            self.log_model_info("Generating smart requirements using hybrid MLflow + AST analyzer...")

            # Get all files that should be analyzed
            model_dir = os.path.dirname(current_file)
            shared_utils_dir = os.path.join(os.path.dirname(model_dir), "shared_utils")

            # Check if there's an existing requirements.txt to use as base
            existing_requirements_file = os.path.join(repo_root, "requirements.txt")
            base_requirements = []
            if os.path.exists(existing_requirements_file):
                with open(existing_requirements_file) as f:
                    base_requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                self.log_model_info(f"Found base requirements.txt with {len(base_requirements)} packages")

            # Analyze using our new hybrid approach
            code_paths = [current_file, shared_utils_dir]
            requirements = analyze_code_dependencies(
                code_paths=code_paths,
                repo_root=repo_root,
                existing_requirements=base_requirements,
                exclude_existing=True,
            )

            # Get detailed analysis for logging
            analyzer = HybridRequirementsAnalyzer(existing_requirements=base_requirements)
            result = analyzer.analyze_model_requirements(
                code_paths=code_paths, repo_root=repo_root, exclude_existing=True
            )
            analysis = result["analysis"]

            # Create temporary requirements file with detailed info
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("# Auto-generated minimal requirements for model execution\n")
                f.write("# Generated by hybrid MLflow + AST analyzer\n")
                f.write("# Combines MLflow's battle-tested package resolution with AST safety\n")
                if base_requirements:
                    f.write(f"# Excludes {len(base_requirements)} packages from base requirements.txt\n")
                f.write(
                    f"# Analysis: {analysis['raw_imports']} raw imports → {analysis['external_modules']} external → {len(requirements)} final\n"
                )
                f.write(f"# MLflow utilities: {'available' if analysis['mlflow_available'] else 'fallback mode'}\n")
                f.write("\n")
                for req in requirements:
                    f.write(f"{req}\n")
                temp_path = f.name

            # Log requirements as artifact
            import mlflow

            mlflow.log_artifact(temp_path, "requirements.txt")

            # Log comprehensive metrics
            mlflow.log_metric("requirements_count", len(requirements))
            mlflow.log_metric("base_requirements_count", len(base_requirements))
            mlflow.log_metric("raw_imports_count", len(analysis["raw_imports"]))
            mlflow.log_metric("external_modules_count", len(analysis["external_modules"]))
            mlflow.log_metric("resolved_packages_count", len(analysis["resolved_packages"]))
            mlflow.log_metric("final_packages_count", len(analysis["final_packages"]))

            # Log analyzer info
            mlflow.log_param("analyzer_type", "hybrid_mlflow_ast")
            mlflow.log_param("mlflow_utilities_available", analysis["mlflow_available"])
            mlflow.log_param("files_analyzed", len(analysis["files_analyzed"]))
            mlflow.log_param("directories_analyzed", len(analysis["directories_analyzed"]))

            # Log requirements summary
            if requirements:
                req_summary = ", ".join(requirements[:10])
                if len(requirements) > 10:
                    req_summary += f" ... and {len(requirements) - 10} more"
                mlflow.log_param("requirements_summary", req_summary)
            else:
                mlflow.log_param("requirements_summary", "None - perfect optimization!")

            # Log analysis details
            if analysis["unrecognized_packages"]:
                mlflow.log_param("unrecognized_packages", ", ".join(analysis["unrecognized_packages"]))

            # Enhanced logging output
            self.log_model_info("🔬 Hybrid Analysis Complete:")
            self.log_model_info(f"  • Raw imports discovered: {len(analysis['raw_imports'])}")
            self.log_model_info(f"  • External modules: {len(analysis['external_modules'])}")
            self.log_model_info(f"  • MLflow utilities: {'✅' if analysis['mlflow_available'] else '❌ (fallback)'}")

            if len(requirements) == 0:
                self.log_model_info("🎉 Perfect optimization! All dependencies covered by base environment.")
                self.log_model_info(f"  • Base packages: {len(base_requirements)}")
                self.log_model_info("  • Additional needed: 0")
            else:
                self.log_model_info(f"📦 Generated {len(requirements)} additional requirements:")
                self.log_model_info(f"  • Base packages (excluded): {len(base_requirements)}")
                for req in requirements:
                    self.log_model_info(f"  + {req}")

            if analysis["unrecognized_packages"]:
                self.log_model_info(f"⚠️  Unrecognized packages: {analysis['unrecognized_packages']}")

            # Clean up temp file
            os.unlink(temp_path)

        except Exception as e:
            self.log_model_info(f"Warning: Could not generate requirements: {e}")
            import traceback

            traceback.print_exc()

    def predict(self, context, model_input):
        """Predict sentiment for input text with preprocessing."""
        # Preprocess input data
        processed_df = self.preprocess_input_data(model_input)
        texts = processed_df["text"].tolist()

        # Make predictions
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)

        # Use databricks helper for postprocessing
        results = postprocess_predictions(predictions.tolist(), probabilities.tolist())

        return results

    def prepare_artifacts(self) -> dict[str, str]:
        """Prepare artifacts for MLflow logging."""
        artifact_paths = super().prepare_artifacts()

        # Save the trained pipeline
        import joblib

        pipeline_path = "auto_sentiment_pipeline.pkl"
        joblib.dump(self.pipeline, pipeline_path)
        artifact_paths["pipeline"] = pipeline_path

        return artifact_paths

    def load_context(self, context):
        """Load artifacts from MLflow context."""
        super().load_context(context)

        # Load the pipeline
        import joblib

        if "pipeline" in context.artifacts:
            self.pipeline = joblib.load(context.artifacts["pipeline"])

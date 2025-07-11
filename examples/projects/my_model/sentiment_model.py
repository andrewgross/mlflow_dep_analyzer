from projects.shared_utils.base_model import BaseModelV3
from projects.shared_utils.databricks.helpers import (
    postprocess_predictions,
    preprocess_text_data,
    save_model_with_metadata,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class SentimentModel(BaseModelV3):
    """Example sentiment analysis model that inherits from BaseModelV3."""

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.metadata = {"model_type": "sentiment_analysis", "version": "1.0.0", "features": ["text"]}

    def preprocess_input_data(self, data):
        """Preprocess input data using databricks helpers."""
        return preprocess_text_data(data, clean_data=True, validate_lengths=True)

    def train(self, texts: list, labels: list):
        """Train the sentiment model with preprocessing."""
        self.log_model_info("Starting training...")

        # Preprocess the training data
        self.log_model_info("Preprocessing training data...")
        processed_df = self.preprocess_input_data(texts)
        processed_texts = processed_df["text"].tolist()

        # Create pipeline with TF-IDF and Logistic Regression
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )

        # Train the model on preprocessed data
        self.pipeline.fit(processed_texts, labels)

        self.log_model_info("Training completed!")

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

    def save_model(self, artifact_path: str = "sentiment_model", include_code_paths: bool = True):
        """Save model using databricks helper with metadata."""
        return save_model_with_metadata(
            model=self,
            artifact_path=artifact_path,
            model_type=self.metadata["model_type"],
            version=self.metadata["version"],
            include_code_paths=include_code_paths,
            features=self.metadata.get("features", []),
            training_framework="scikit-learn",
        )

    def prepare_artifacts(self) -> dict[str, str]:
        """Prepare artifacts for MLflow logging."""
        artifact_paths = super().prepare_artifacts()

        # Save the trained pipeline
        import joblib

        pipeline_path = "sentiment_pipeline.pkl"
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

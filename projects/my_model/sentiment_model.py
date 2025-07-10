import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from projects.shared_utils.base_model import BaseModelV3


class SentimentModel(BaseModelV3):
    """Example sentiment analysis model that inherits from BaseModelV3."""

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.metadata = {"model_type": "sentiment_analysis", "version": "1.0.0", "features": ["text"]}

    def train(self, texts: list, labels: list):
        """Train the sentiment model."""
        self.log_model_info("Starting training...")

        # Create pipeline with TF-IDF and Logistic Regression
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )

        # Train the model
        self.pipeline.fit(texts, labels)

        self.log_model_info("Training completed!")

    def predict(self, context, model_input):
        """Predict sentiment for input text."""
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        else:
            texts = model_input

        # Make predictions
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)

        # Return results as DataFrame
        results = pd.DataFrame(
            {
                "prediction": predictions,
                "positive_probability": probabilities[:, 1],
                "negative_probability": probabilities[:, 0],
            }
        )

        return results

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_utils import load_sentiment_data, save_normalized_csv


MODEL_PATH = Path("models/sentiment_model.pkl")


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_and_evaluate(
    data_path: str | Path = "data.csv",
    model_path: str | Path = MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    data = load_sentiment_data(data_path)

    train_text, test_text, train_label, test_label = train_test_split(
        data["review"],
        data["sentiment"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["sentiment"],
    )

    model = build_model()
    model.fit(train_text, train_label)

    predictions = model.predict(test_text)
    accuracy = accuracy_score(test_label, predictions)
    report = classification_report(
        test_label,
        predictions,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_label, predictions, labels=[0, 1])

    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as file:
        pickle.dump(model, file)

    normalized_path = save_normalized_csv(data_path)
    evaluation = {
        "total_reviews": int(len(data)),
        "train_reviews": int(len(train_text)),
        "test_reviews": int(len(test_text)),
        "label_counts": data["label"].value_counts().to_dict(),
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": pd.DataFrame(
            matrix,
            index=["actual_negative", "actual_positive"],
            columns=["predicted_negative", "predicted_positive"],
        ),
        "model_path": str(model_output),
        "normalized_data_path": str(normalized_path),
    }
    return evaluation


def load_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python train.py` first."
        )
    with path.open("rb") as file:
        return pickle.load(file)


def predict_sentiment(text: str, model_path: str | Path = MODEL_PATH) -> dict[str, object]:
    model = load_model(model_path)
    prediction = int(model.predict([text])[0])
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))

    return {
        "text": text,
        "sentiment": "positive" if prediction == 1 else "negative",
        "label": prediction,
        "confidence": confidence,
    }

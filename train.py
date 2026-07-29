from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sentiment analysis model.")
    parser.add_argument("--data", default="data.csv", help="Path to the source dataset.")
    parser.add_argument(
        "--model",
        default="models/sentiment_model.pkl",
        help="Where to save the reusable trained model.",
    )
    parser.add_argument(
        "--metrics",
        default="reports/metrics.json",
        help="Where to save evaluation metrics.",
    )
    args = parser.parse_args()

    evaluation = train_and_evaluate(args.data, args.model)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        key: value
        for key, value in evaluation.items()
        if key != "confusion_matrix"
    }
    serializable["confusion_matrix"] = evaluation["confusion_matrix"].to_dict()
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print("Sentiment model trained successfully.")
    print(f"Total reviews: {evaluation['total_reviews']}")
    print(f"Train/Test split: {evaluation['train_reviews']}/{evaluation['test_reviews']}")
    print(f"Accuracy: {evaluation['accuracy']:.2%}")
    print("\nConfusion matrix:")
    print(evaluation["confusion_matrix"])
    print(f"\nModel saved to: {evaluation['model_path']}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Normalized dataset saved to: {evaluation['normalized_data_path']}")


if __name__ == "__main__":
    main()

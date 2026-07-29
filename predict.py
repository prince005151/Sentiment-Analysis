from __future__ import annotations

import argparse

from src.model import predict_sentiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for text.")
    parser.add_argument("text", help="Review text to classify.")
    parser.add_argument(
        "--model",
        default="models/sentiment_model.pkl",
        help="Path to the trained model.",
    )
    args = parser.parse_args()

    result = predict_sentiment(args.text, args.model)
    confidence = result["confidence"]
    confidence_text = (
        f" ({confidence:.2%} confidence)" if confidence is not None else ""
    )
    print(f"{result['sentiment']}{confidence_text}")


if __name__ == "__main__":
    main()

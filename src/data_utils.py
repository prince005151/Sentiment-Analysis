from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


def _extract_named_list(source: str, name: str) -> list[str]:
    tree = ast.parse(source)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        target_names = [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ]
        if name in target_names:
            value = ast.literal_eval(node.value)
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list.")
            return [str(item) for item in value]

    raise ValueError(f"Could not find {name} in the dataset file.")


def load_sentiment_data(path: str | Path = "data.csv") -> pd.DataFrame:
    """Load the supplied review-list dataset into review/sentiment columns."""
    dataset_path = Path(path)
    source = dataset_path.read_text(encoding="utf-8")

    positive_reviews = _extract_named_list(source, "positive_reviews")
    negative_reviews = _extract_named_list(source, "negative_reviews")

    data = pd.DataFrame(
        {
            "review": positive_reviews + negative_reviews,
            "sentiment": [1] * len(positive_reviews) + [0] * len(negative_reviews),
        }
    )
    data["review"] = data["review"].str.strip()
    data = data.drop_duplicates(subset=["review"]).reset_index(drop=True)
    data["label"] = data["sentiment"].map({1: "positive", 0: "negative"})
    return data


def save_normalized_csv(
    input_path: str | Path = "data.csv",
    output_path: str | Path = "reports/normalized_sentiment_data.csv",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = load_sentiment_data(input_path)
    data.to_csv(output, index=False)
    return output

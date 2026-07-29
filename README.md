# Sentiment Analysis Project

This project trains a reusable binary sentiment classifier from the supplied review dataset.

## What It Does

- Loads and explores the provided data.
- Converts text into numerical TF-IDF features.
- Splits data into train and test sets.
- Trains a Logistic Regression sentiment model.
- Evaluates accuracy, precision, recall, F1-score, and confusion matrix.
- Saves the trained model in reusable pickle format.
- Provides both command-line prediction and a Streamlit app.

## Project Files

- `data.csv`: Provided source data.
- `src/data_utils.py`: Data loading and normalization helpers.
- `src/model.py`: Model training, evaluation, saving, loading, and prediction.
- `train.py`: Train and evaluate the model.
- `predict.py`: Predict sentiment from the command line.
- `app.py`: Streamlit interface.
- `models/sentiment_model.pkl`: Saved pickle model after training.
- `reports/metrics.json`: Saved evaluation metrics after training.
- `reports/normalized_sentiment_data.csv`: Clean CSV version of the dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Predict From Command Line

```bash
python predict.py "The movie was wonderful and exciting"
```

## Run The App

```bash
streamlit run app.py
```

## Deploy On Streamlit Cloud

Use this as the main file path:

```text
app.py
```

Make sure `models/sentiment_model.pkl` is included in the uploaded repository. The
app only predicts with this saved model and does not train from the UI.

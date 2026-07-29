from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.model import MODEL_PATH, predict_sentiment


st.set_page_config(page_title="Sentiment Analysis", page_icon="SA", layout="centered")

st.title("Sentiment Analysis")
st.caption("Enter a sentence or review to predict its sentiment.")

model_exists = Path(MODEL_PATH).exists()

if not model_exists:
    st.error("Trained model not found. Run `python train.py` once in the terminal.")
    st.stop()

review = st.text_area(
    "Review text",
    placeholder="Type a movie or product review here...",
    height=140,
)

if st.button("Predict sentiment", type="primary"):
    if not review.strip():
        st.warning("Enter review text first.")
    else:
        result = predict_sentiment(review)
        confidence = result["confidence"]
        st.subheader(result["sentiment"].title())
        if confidence is not None:
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.2%}")

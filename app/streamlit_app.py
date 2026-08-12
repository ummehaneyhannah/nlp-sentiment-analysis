import streamlit as st
import joblib
import re

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Compare a classical ML baseline against a fine-tuned transformer.")

# Model selector
model_choice = st.radio(
    "Choose a model:",
    ["Baseline (TF-IDF + Logistic Regression)", "DistilBERT (Transformer)"]
)

# ---- Load baseline ----
@st.cache_resource
def load_baseline():
    clf = joblib.load("models/baseline_logreg.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return clf, vectorizer

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---- Load DistilBERT (only when selected, since it's heavier) ----
@st.cache_resource
def load_distilbert():
    from transformers import pipeline
    return pipeline("text-classification", model="ummeyhaney/distilbert-imdb-sentiment")

# ---- UI ----
user_input = st.text_area(
    "Your review:",
    height=150,
    placeholder="e.g. This movie was absolutely fantastic, great acting and story!"
)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        if model_choice == "Baseline (TF-IDF + Logistic Regression)":
            clf, vectorizer = load_baseline()
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            prediction = clf.predict(vec)[0]
            probability = clf.predict_proba(vec)[0]

            if prediction == 1:
                st.success(f"**Positive** 😊 (confidence: {probability[1]*100:.1f}%)")
            else:
                st.error(f"**Negative** 😞 (confidence: {probability[0]*100:.1f}%)")

        else:
            with st.spinner("Loading DistilBERT model (first run may take a minute)..."):
                clf = load_distilbert()
            result = clf(user_input)[0]
            label = result["label"]
            score = result["score"] * 100

            if label == "LABEL_1" or label.upper() == "POSITIVE":
                st.success(f"**Positive** 😊 (confidence: {score:.1f}%)")
            else:
                st.error(f"**Negative** 😞 (confidence: {score:.1f}%)")

st.markdown("---")
st.caption(
    "Baseline: TF-IDF + Logistic Regression (Test Accuracy: 88.97%) | "
    "DistilBERT: fine-tuned transformer (Test Accuracy: 91.05%)"
)
import streamlit as st
import joblib
import re

# Page setup
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review below and the model will predict if it's Positive or Negative.")

# Load model and vectorizer (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    clf = joblib.load("models/baseline_logreg.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return clf, vectorizer

clf, vectorizer = load_model()

# Same cleaning function used during training
def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# User input
user_input = st.text_area("Your review:", height=150, placeholder="e.g. This movie was absolutely fantastic, great acting and story!")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = clf.predict(vec)[0]
        probability = clf.predict_proba(vec)[0]

        if prediction == 1:
            st.success(f"**Positive** 😊 (confidence: {probability[1]*100:.1f}%)")
        else:
            st.error(f"**Negative** 😞 (confidence: {probability[0]*100:.1f}%)")

st.markdown("---")
st.caption("Model: TF-IDF + Logistic Regression, trained on the IMDB movie review dataset (25,000 reviews, ~89% test accuracy).")
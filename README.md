# Movie Review Sentiment Analysis

A sentiment analysis system that classifies movie reviews as Positive or Negative, comparing a classical ML baseline (TF-IDF + Logistic Regression) against a fine-tuned transformer (DistilBERT), with a deployed Streamlit demo app.

## Demo

Enter any movie review in the app and get an instant Positive/Negative prediction with confidence score.

## Dataset

[IMDB Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) — 25,000 labeled training reviews, 25,000 test reviews, perfectly balanced between positive and negative classes.

## Approach

1. **EDA** — checked class balance, review length distribution
2. **Preprocessing** — removed HTML tags, lowercased, removed punctuation (for classical model); lighter cleaning for the transformer since its tokenizer handles casing/punctuation
3. **Baseline model** — TF-IDF (unigrams + bigrams, 20k features) + Logistic Regression
4. **Transformer model** — fine-tuned `distilbert-base-uncased` for 2 epochs
5. **Evaluation** — compared both models on a held-out test set
6. **Deployment** — built a Streamlit app for real-time predictions

## Results

| Model | Test Accuracy | Test F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 88.97% | 0.890 |
| DistilBERT (fine-tuned) | 90.82% | 0.908 |

## Key Findings

- The transformer outperforms the classical baseline by ~1.85%, but the gap is modest — TF-IDF + Logistic Regression remains a strong, fast, and lightweight option for this task.
- Error analysis showed both models struggle most with **mixed-sentiment reviews** — e.g. reviews that use positive language ("entertaining," "enjoyed") while delivering an overall negative verdict.

## How to Run

```bash
git clone https://github.com/<your-username>/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Project Structure
## Future Improvements

- Handle sarcasm and mixed-sentiment reviews better
- Add SHAP/LIME explainability
- Extend to multi-class emotion detection
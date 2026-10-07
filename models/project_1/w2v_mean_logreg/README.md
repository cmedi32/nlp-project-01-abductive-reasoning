# Word2Vec mean pooling + Logistic Regression

Static word embeddings, mean-pooled per sentence, with a logistic regression classifier for ART (Project 1).

## Links

| Field | Value |
|---|---|
| GitHub reference | `<https://github.com/<user>/<repo>/tree/main/models/project_1/w2v_mean_logreg>` |
| Weights & Biases profile | `<https://wandb.ai/<username>>` |
| Weights & Biases run / reference | `<https://wandb.ai/<team>/<project>/runs/<run-id>>` |

## Contents

| File | Description |
|---|---|
| `w2v_mean_logreg.ipynb` | Training and evaluation notebook — @todo frozen snapshot of `notebooks/project_1/static/w2v_mean_logreg.ipynb` |
| `weights/logreg.joblib` | Trained classifier (not tracked in git) |

## Setup

- **Input format:** every sentence is tokenized with `nltk.word_tokenize` (punctuation removed, stopwords kept, no lowercasing, no stemming/lemmatization, no truncation), embedded as the mean of its word vectors, and the four field vectors (O1, O2, H1, H2) are mean-pooled into one 300-dim example vector
- **Embeddings:** `word2vec-google-news-300` via gensim downloader, 3M words, 300 dims, pretrained and frozen. Lookup: exact token, then lowercased token, otherwise skipped; a sentence with no hit at all becomes a zero vector
- **Architecture:** `sklearn.linear_model.LogisticRegression` on the 300-dim example vector
- **Hyperparameters:** `max_iter=1000`, otherwise scikit-learn defaults (L2 penalty, `C=1.0`, `lbfgs`)

## Results

| Split | Accuracy |
|---|---|
| Train | 54.89% |
| Validation | 52.70% |
| Test | **51.57%** |

Majority-class baseline on test: 50.98% (on validation: 50.80%)

Per class on test (f1): hypothesis_1 0.534, hypothesis_2 0.496. Confusion matrix `[[425, 356], [386, 365]]`.

Expectation (see `EXPECTATIONS.md`): 60–70% for the simple embedding models. Measured 51.57% — not reached, see Notes.

## Weights

```python
# Load the weights
import joblib
classifier = joblib.load("weights/logreg.joblib")
classifier.predict_proba(vector.reshape(1, -1))
```

## Notes

- **The example vector does not depend on the label.** Mean-pooling O1, O2, H1 and H2 into
  one vector gives a bit-identical result when the two hypotheses are swapped, so the
  classifier cannot tell which hypothesis is the right one and mostly learns the label
  prior. The 52.70% against a 50.80% baseline comes from the observations alone. A pair
  representation that keeps the two hypotheses apart is the next step.
- **Out of vocabulary:** 11.7% of all tokens, but 97% of those occurrences are `to`, `a`,
  `and`, `of` and `'s`, which are genuinely missing from the Google News vectors. The real
  OOV rate is ~0.33% (numbers, typos such as `didnt`, British spellings such as
  `cancelled`, hyphenated forms such as `co-worker`). 6 sentences across all splits resolve
  to no vector at all and fall back to zeros.
- **The lowercase fallback rescues 18 tokens** in the whole dataset, all sentence-initial or
  mistyped names (`Muffles`, `CHuck`, `GIna`). It costs nothing but changes nothing.
- **Error analysis:** the confusion matrix is close to uniform and the model predicts 811
  vs. 721 where the truth is 781 vs. 751 — it guesses along the label prior. The 3.3 points
  between train (54.89%) and test (51.57%) are fitted to the training split, not signal.

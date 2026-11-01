"""TF-IDF: one sparse tf-idf vector per sentence, the four of an example concatenated (O1, O2, H1, H2)."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data import TEXT_FIELDS


def fit_vectorizer(train_tokens):
    # vocabulary: every single sentence (O1, O2, H1, H2) of the training data, already tokenized
    all_sentences = []
    for example in train_tokens:
        for field in TEXT_FIELDS:
            all_sentences.append(example[field])

    # the sentences are already token lists, so the analyzer just hands them through unchanged
    vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens)
    vectorizer.fit(all_sentences)
    print(f"vocabulary: {len(vectorizer.vocabulary_)} terms")
    return vectorizer


def build_features(token_split, vectorizer):
    """One tf-idf matrix per field, glued side by side -> X: (examples, 4 * vocabulary) csr matrix, y: targets."""
    X = sparse.hstack([vectorizer.transform(token_split[field]) for field in TEXT_FIELDS], format="csr")
    y = np.array(token_split["target"])
    return X, y


def vectorizer_summary(vectorizer, X_train):
    return {
        "tfidf/vocabulary_size": len(vectorizer.vocabulary_),
        "tfidf/input_dim": X_train.shape[1],
    }

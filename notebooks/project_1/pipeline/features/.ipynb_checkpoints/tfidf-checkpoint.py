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
    """Feature matrix X (examples, 4 * vocabulary) and targets y of a tokenized split."""
    # For each of the four sentences, transform turns every token list into one row with one number per
    # vocabulary word: 0 if the word is missing, otherwise its weight, rare words more ("panda" 0.61) than common ones ("the" 0.13).
    # That gives four matrices, each with one row per example and 16,189 columns.
    # hstack glues them side by side, so every example becomes one long row O1 | O2 | H1 | H2 with 4 * 16,189 = 64,756 numbers.
    # This row is the input of the MLP, y holds the right answer (0 = hypothesis_1, 1 = hypothesis_2).
    matrices = []
    for field in TEXT_FIELDS:
        matrices.append(vectorizer.transform(token_split[field]))

    X = sparse.hstack(matrices, format="csr")
    y = np.array(token_split["target"])
    return X, y


def vectorizer_summary(vectorizer, X_train):
    return {
        "tfidf/vocabulary_size": len(vectorizer.vocabulary_),
        "tfidf/input_dim": X_train.shape[1],
    }

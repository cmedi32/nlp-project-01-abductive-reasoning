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


def idf_weights(vectorizer, train_tokens):
    """idf of every vocabulary word as dict word -> idf, plus the idf for words never seen in training."""
    n_sentences = len(train_tokens) * len(TEXT_FIELDS)  # the documents fit_vectorizer counted

    # vectorizer.idf_: (16189,), one idf per vocabulary term -> dict term -> idf
    idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    idf_unseen = np.log(1 + n_sentences) + 1  # df = 0 in the smoothed formula -> highest possible idf

    print(f"tf-idf fitted on {n_sentences} training sentences, vocabulary: {len(idf)} terms")
    print(f"idf range: {vectorizer.idf_.min():.2f} – {vectorizer.idf_.max():.2f}, words never seen in train: {idf_unseen:.2f}")
    return idf, idf_unseen


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

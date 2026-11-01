"""Shared preprocessing for the OOV evals: w2v_oov, glove_oov and fasttext_oov.

Edit preprocess() here, then run summary.ipynb: all three notebooks import this function,
so every embedding is measured on exactly the same tokens.
"""
import nltk

nltk.download("punkt_tab", quiet=True)


def preprocess(text):
    tokens = nltk.word_tokenize(text)a
    return [token for token in tokens if any(character.isalnum() for character in token)]

"""Same as nltk_alnum, plus lowercasing (for uncased vocabularies: glove-wiki-gigaword-300, tf-idf)."""
from . import nltk_alnum

CONFIG = {**nltk_alnum.CONFIG, "lowercase": True}

def preprocess(text):
    return [token.lower() for token in nltk_alnum.preprocess(text)] 

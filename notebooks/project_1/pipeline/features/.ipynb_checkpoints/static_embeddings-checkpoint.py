"""Static word embeddings: tokens -> sentence vector (mean pooling) -> example vector (mean or concat).

The vectors are only a lookup table, they are turned into numpy features before training starts.
"""
from collections import Counter

import gensim.downloader as api
import numpy as np

from ..data import TEXT_FIELDS


def load_vectors(name):
    """gensim downloader model id, e.g. "glove-wiki-gigaword-300" or "word2vec-google-news-300"."""
    wv = api.load(name)
    print(f"vocabulary: {len(wv.index_to_key)} words, {wv.vector_size} dimensions")
    return wv


class MeanEmbedder:
    """Sentence vector = mean of the word vectors of its tokens. OOV tokens are skipped and counted.

    lowercase_fallback: an unknown token is looked up a second time in lowercase
    (useful for cased vocabularies like word2vec, pointless for already lowercased tokens).
    """

    def __init__(self, wv, lowercase_fallback=False):
        self.wv = wv
        self.lowercase_fallback = lowercase_fallback
        self.oov_counts = Counter()  # oov token -> how often it occurred
        self.stats = Counter()  # "tokens", "oov", "empty_sentences"

    def lookup(self, token):
        if token in self.wv:
            return self.wv[token]
        if self.lowercase_fallback and token.lower() in self.wv:
            return self.wv[token.lower()]
        return None

    def __call__(self, tokens):
        vectors = []
        for token in tokens:
            self.stats["tokens"] += 1
            vector = self.lookup(token)
            if vector is None:
                self.stats["oov"] += 1
                self.oov_counts[token] += 1
            else:
                vectors.append(vector)
        if not vectors:
            self.stats["empty_sentences"] += 1
            return np.zeros(self.wv.vector_size)
        return np.mean(vectors, axis=0)

    def summary(self):
        """OOV counters of every sentence embedded so far, keys as logged to W&B."""
        return {
            "oov/tokens": self.stats["tokens"],
            "oov/occurrences": self.stats["oov"],
            "oov/rate": self.stats["oov"] / self.stats["tokens"],
            "oov/unique_types": len(self.oov_counts),
            "oov/empty_sentences": self.stats["empty_sentences"],
        }


def combine_mean(vectors):
    """O1, O2, H1, H2 sentence vectors -> their mean (300 dims), the order of the sentences is lost. used for early models."""
    return np.mean(vectors, axis=0)


def combine_concat(vectors):
    """O1, O2, H1, H2 sentence vectors -> glued in this fixed order (4 * 300 dims), swapping H1 and H2 changes the input, order contains all information."""
    return np.concatenate(vectors)

def combine_concat_candidate_framing(vectors)
    """seperate canidates for improved reasoning in bidirectional rnns."""
    # @todo candidate framing [O1] <sep> [h1] <sep> [O2]


def concat_add_interaction_features(vectors)
    """add some sort of salt and pepper."""

def embed_example(example, embed, combine):
    vectors = [embed(example[field]) for field in TEXT_FIELDS]
    return combine(vectors)

# build a feature from tokens
def build_features(token_split, embed, combine):
    """Feature matrix X (examples, dims) and targets y of a tokenized split."""
    
    # one feature vector per example: embed its four sentences, then combine them (mean or concat)
    rows = []
    for example in token_split:
        rows.append(embed_example(example, embed, combine))

    X = np.stack(rows)  # list of vectors -> matrix (examples, dims)
    y = np.array(token_split["target"])
    return X, y

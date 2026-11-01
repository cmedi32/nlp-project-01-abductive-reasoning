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
        vectors, weights = [], []
        for token in tokens:
            self.stats["tokens"] += 1
            vector = self.lookup(token)
            if vector is None:
                self.stats["oov"] += 1
                self.oov_counts[token] += 1
            else:
                vectors.append(vector)
                weights.append(self.weight(token))
        if not vectors:
            self.stats["empty_sentences"] += 1
            return self.empty_vector()
        return self.pool(vectors, weights)

    def weight(self, token):
        return 1.0  # plain mean: every word counts the same

    def pool(self, vectors, weights):
        return np.mean(vectors, axis=0)  # all weights are 1.0, so the plain mean

    def empty_vector(self):
        return np.zeros(self.wv.vector_size)  # sentence without any known word

    def summary(self):
        """OOV counters of every sentence embedded so far, keys as logged to W&B."""
        return {
            "oov/tokens": self.stats["tokens"],
            "oov/occurrences": self.stats["oov"],
            "oov/rate": self.stats["oov"] / self.stats["tokens"],
            "oov/unique_types": len(self.oov_counts),
            "oov/empty_sentences": self.stats["empty_sentences"],
        }


class TfidfMeanEmbedder(MeanEmbedder):
    """Sentence vector = tf-idf weighted mean of its word vectors: rare words count more.

    idf: dict word -> idf, idf_unseen: idf for words never seen in training (both from tfidf.idf_weights).
    The idf weights are fixed numpy numbers computed before training, not model parameters: backprop never
    reaches them, so the model cannot learn which words matter (learnable weights in this average = attention pooling).
    """

    def __init__(self, wv, idf, idf_unseen, lowercase_fallback=False):
        super().__init__(wv, lowercase_fallback)
        self.idf = idf
        self.idf_unseen = idf_unseen

    def weight(self, token):
        if token not in self.idf:
            self.stats["tfidf_unseen"] += 1  # in the word vectors, but in no training sentence
        return self.idf.get(token, self.idf_unseen)

    def pool(self, vectors, weights):
        # per dimension sum(weight * vector) / sum(weights): (300,)
        return np.average(vectors, axis=0, weights=weights).astype(np.float32)

    def empty_vector(self):
        return np.zeros(self.wv.vector_size, dtype=np.float32)

    def summary(self):
        return {
            **super().summary(),
            "tfidf/vocabulary_size": len(self.idf),
            "tfidf/unseen_tokens": self.stats["tfidf_unseen"],
        }


def combine_mean(vectors):
    """O1, O2, H1, H2 sentence vectors -> their mean (300 dims), the order of the sentences is lost. used for early models."""
    return np.mean(vectors, axis=0)


def combine_concat(vectors):
    """O1, O2, H1, H2 sentence vectors -> glued in this fixed order (4 * 300 dims), swapping H1 and H2 changes the input, order contains all information."""
    return np.concatenate(vectors)

def combine_concat_candidate_framing(vectors):
    """seperate canidates for improved reasoning in bidirectional rnns.

    O1, O2, H1, H2 sentence vectors -> two candidate stories glued together (6 * 300 dims):
        A = O1, H1, O2   does H1 bridge the two observations?
        B = O1, H2, O2   does H2?
    The model reads A and B as two sequences of 3 time steps. No <sep> token needed: one time step is
    exactly one sentence, so the sentence boundaries are already there.
    """
    observation_1, observation_2, hypothesis_1, hypothesis_2 = vectors  # TEXT_FIELDS order
    return np.concatenate([observation_1, hypothesis_1, observation_2, observation_1, hypothesis_2, observation_2])

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

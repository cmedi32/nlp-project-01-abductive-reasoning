"""Feature stage: token lists -> one feature row per example, plus the target vector.

    static_embeddings.py   frozen pretrained word vectors, mean pooling per sentence, mean or concat per example
    tfidf.py               tf-idf vector per sentence, concatenated per example (sparse)

Every variant ends in build_features(token_split, ...) -> (X, y), so the loaders and models don't care
where the features come from. New variant: add a file with its own build_features() and import it.
"""

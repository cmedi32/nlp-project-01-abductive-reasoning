"""Shared pipeline stages of the project 1 notebooks in static/ and rnn/.

Every notebook runs the same stages in the same order and only picks one variant per stage this is basically whats coverd by the presentation mainly:

    tracking.py        .env + W&B login, run init, upload of the best checkpoint
    data.py            dataset splits, TEXT_FIELDS
    preprocessing/     raw text -> tokens, one file per variant (preprocess() + CONFIG for W&B)
    features/          tokens -> feature matrix: static embeddings (mean pooling) or tf-idf
    loaders.py         dense or sparse feature matrix, only relevant for tfidf -> DataLoader
    training.py        trainer, early stopping, best checkpoint, run summary, hyper parameter application
    evaluation.py      majority baseline, probability per hypothesis, wandb reporting
"""

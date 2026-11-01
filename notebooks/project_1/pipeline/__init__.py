"""Shared pipeline stages of the project 1 notebooks in static/ and rnn/.

Every notebook runs the same stages in the same order and only picks one variant per stage:

    tracking.py        .env + W&B login, run init, upload of the best checkpoint
    data.py            dataset splits, TEXT_FIELDS
    preprocessing/     raw text -> tokens, one file per variant (preprocess() + CONFIG for W&B)
    features/          tokens -> feature matrix: static embeddings (mean pooling) or tf-idf
    loaders.py         dense or sparse feature matrix -> DataLoader
    training.py        trainer, early stopping, best checkpoint, run summary
    evaluation.py      majority baseline, probability per hypothesis

The model classes are not part of this package: every notebook defines its own LightningModule completely.

New variant of a stage: add a file next to the existing ones with the same interface and change
the import in the notebook, e.g. preprocessing/nltk_alnum_lower.py -> preprocessing/my_variant.py.

The notebooks in notebooks/project_1/{static,rnn}/ find this package via sys.path.insert(0, "..").
"""

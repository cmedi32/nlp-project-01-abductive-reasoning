"""Preprocessing stage: raw sentences -> token lists.

One file per variant, each provides
    CONFIG       the preprocessing decisions, merged into the W&B run config of the notebook
    preprocess   function: raw sentence (str) -> tokens (list[str])

New variant: copy one of the files, change preprocess() and CONFIG, and import it in the notebook:
    from pipeline.preprocessing import my_variant as preprocessing
"""
from ..data import TEXT_FIELDS


def tokenize_split(split, preprocess):
    """Applies preprocess() to the four sentences of every example and adds the 0-based target."""

    def preprocess_example(example):
        processed = {field: preprocess(example[field]) for field in TEXT_FIELDS}
        processed["target"] = example["label"] - 1 # we subtract 1 - so the classifier does not need to calculate that, the training data already indicates if its the first or the second hypothesis
        return processed

    # no cache: preprocess() lives in a module now, datasets would not notice an edit in that file and reuse the old tokens
    return split.map(preprocess_example, load_from_cache_file=False)

"""Dataset: the course version of ART with its own train / validation / test split."""
from datasets import load_dataset

DATASET = "amarfurt/art-nlp-hs26"

# the four sentences of one example, always in this order
TEXT_FIELDS = ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]


def load_splits(name=DATASET):
    dataset = load_dataset(name)

    train = dataset["train"]
    valid = dataset["validation"]
    test = dataset["test"]

    print(f"train: {len(train)}, valid: {len(valid)}, test: {len(test)}")
    return train, valid, test

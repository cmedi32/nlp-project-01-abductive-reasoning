"""Evaluation helpers: majority baseline and the probability per hypothesis."""
import numpy as np
import torch

from .data import TEXT_FIELDS


def majority_accuracy(y):
    """Accuracy of always answering the most frequent target."""
    return np.bincount(y).max() / len(y)


def predict_hypotheses(trainer, model, test_loader, test):
    """Best checkpoint, softmax over the 2 logits -> one probability distribution per test example."""
    probabilities = torch.cat(trainer.predict(model, dataloaders=test_loader, ckpt_path="best")).numpy()

    results = test.to_pandas()[TEXT_FIELDS + ["label"]]  # test_loader is not shuffled, same order as test
    results["p_hypothesis_1"] = probabilities[:, 0]
    results["p_hypothesis_2"] = probabilities[:, 1]
    results["answer"] = probabilities.argmax(axis=1) + 1  # same numbering as label: 1 -> hypothesis_1, 2 -> hypothesis_2

    print(f"accuracy from the answers: {(results['answer'] == results['label']).mean():.4f}")
    return results

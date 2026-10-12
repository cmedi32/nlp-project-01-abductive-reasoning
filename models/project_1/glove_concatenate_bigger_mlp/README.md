# GloVe mean pooling, concatenated + bigger MLP

Static GloVe embeddings, mean-pooled per sentence, the four sentence vectors concatenated in fixed order, with a two-hidden-layer MLP trained with Lightning for ART (Project 1).

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/glove_concatenate_bigger_mlp |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/ugjbs8fm (run `glove-concatenate-bigger-mlp`, model artifact `glove-concatenate-bigger-mlp:best`) |

## Contents

| File | Description |
|---|---|
| `glove_concatenate_bigger_mlp.ipynb` | Training and evaluation notebook — @todo frozen snapshot of `notebooks/project_1/static/glove_concatenate_bigger_mlp.ipynb` |
| `weights/best-epoch=3.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `glove-concatenate-bigger-mlp:best`) |

## Setup

- **Input format:** every sentence is tokenized with `nltk.word_tokenize` (punctuation removed, **lowercased after tokenizing** because GloVe is uncased, stopwords kept, no stemming/lemmatization, no truncation) and embedded as the mean of its word vectors. The four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one 1200-dim example vector
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader (Wikipedia 2014 + Gigaword 5, 6B tokens), 400k words, 300 dims, uncased, pretrained and frozen. Lookup: token, otherwise skipped; a sentence with no hit at all becomes a zero vector in its slot
- **Architecture:** LightningModule `MLP`: `Linear(1200, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 2)`, 373k parameters, `CrossEntropyLoss` on the two logits
- **Training:** identical to `w2v_mean_mlp`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42. Stopped after 9 epochs (2,367 steps), best epoch 3 (0-based). ~80 s on an RTX 2070 SUPER

## Results

| Split | Accuracy | Loss | Majority baseline |
|---|---|---|---|
| Train (last epoch) | 76.36% | 0.471 | — |
| Validation | 59.80% | 0.6528 | 50.80% |
| Test | **51.70%** | 0.7795 | 50.98% |

Skill over the majority baseline, `(accuracy − baseline) / (1 − baseline)`: validation 0.18, test 0.01.
Pseudo-R², `1 − loss / ln 2`: validation 0.06, test −0.12.

Shortcut rules without any model, evaluated on all examples without a tie:

| Rule | Train | Validation | Test |
|---|---|---|---|
| Pick the hypothesis that shares more words with O1 and O2 | 63.4% | 63.5% | 51.9% |
| Pick the shorter hypothesis | 56.5% | 57.1% | 49.9% |

Expectation (see `EXPECTATIONS.md`, variant 3): ~75%. Estimate before the run: 53–57%. Validation
came in above (59.80%), test below (51.70%) — see Notes.

## Weights

```python
# Load the weights; the MLP class from the notebook must be defined
model = MLP.load_from_checkpoint("weights/best-epoch=3.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 1200-dim concatenation O1, O2, H1, H2 built as in the notebook.

## Notes

- **Concatenation made learning possible.** Swapping H1 and H2 now changes the input, and the
  validation loss drops to 0.653, below ln 2 = 0.693. `w2v_mean_mlp` stayed at 0.6926.
- **Validation and test disagree, and it is not leakage.** No observation pair from validation or
  test appears in train (0 of 1,000 and 0 of 1,532). Test is built differently: the shortcut rules
  above work on train and validation but are at chance on test. Test is very likely the original
  ART dev set (exactly 1,532 examples), which was filtered against such cues; train and validation
  come from the same pool.
- **What the model learned is the shortcut.** Averaged word vectors measure shared topic and
  vocabulary, so the MLP learned "the hypothesis that sounds more like the story is correct". The
  word-overlap rule alone reaches 63.5% on validation, more than the model (59.80%). On test that
  cue does not hold, which leaves 51.70%.
- **The test loss of 0.7795 is worse than guessing** (ln 2 = 0.6931). The model is confident and
  wrong about half the time; pseudo-R² on test is −0.12.
- **Validation is a poor stand-in for test on this dataset.** Train accuracy climbs to 76.36% by
  the last epoch while the best checkpoint is epoch 3. Early stopping and checkpoint selection run
  on validation and therefore favour the shortcut. Test is the number to report.
- **The changes were not isolated.** GloVe, lowercasing, concatenation and the second hidden layer
  changed together. Only the concatenation can explain leaving ln 2, because the averaged input is
  symmetric; how much GloVe or the extra layer contribute is unknown.
- **Word order within a sentence is still ignored.** Each sentence is still a mean, so "He was
  never lost" and "He was lost" get nearly the same vector. Only the order of the four sentences
  matters.
- **Out of vocabulary:** 0.17% of all tokens (1,144 of 670,395; 850 types), 7 empty sentences,
  against 11.74% for word2vec. GloVe has `to`, `a`, `and`, `of` and `'s`. Lowercasing merges case
  variants such as `mTurk`, `Mturk`, `mturk`, which is why there are 850 OOV types here against 861
  in `w2v_glove_oov_comparison.ipynb`. What remains are rare names, `selfie` and typos.
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-3
  checkpoint. The real value is 8.

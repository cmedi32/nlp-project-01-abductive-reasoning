# TF-IDF, concatenated + small MLP

Sparse tf-idf vectors per sentence (sklearn `TfidfVectorizer`), the four sentence vectors concatenated in fixed order, with a one-hidden-layer MLP trained with Lightning for ART (Project 1). No pretrained embeddings.

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/tfidf_concatenate_small_mlp |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/726nrg9t (run `tfidf-concatenate-small-mlp`, model artifact `tfidf-concatenate-small-mlp:best`) |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/static/tfidf_concatenate_small_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here) |
| `weights/best-epoch=0.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `tfidf-concatenate-small-mlp:best`) |

## Setup

- **Input format and vocabulary:** identical to `tfidf_concatenate_mlp`. Same preprocessing as the GloVe notebooks, `TfidfVectorizer` fitted on every single training sentence (4 × 16,801), **16,189 terms**, default sklearn settings. The four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one sparse vector of **64,756 dims**, densified per example in the `SparseDataset`
- **Architecture:** LightningModule `MLP`: `Linear(64756, 256) → ReLU → Linear(256, 2)`, 16.58M parameters (16,578,306, all but 514 in the first layer), `CrossEntropyLoss` on the two logits. One hidden layer less than `tfidf_concatenate_mlp`
- **Training:** same as `tfidf_concatenate_mlp`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42. Stopped after 6 epochs (1,578 steps), best epoch 0 (0-based). ~36 s on an RTX 2070 SUPER

## Results

| Split | Accuracy | Loss | Majority baseline |
|---|---|---|---|
| Train (last epoch) | 99.99% | 0.0050 | — |
| Validation | 57.40% | 0.6696 | 50.80% |
| Test | **51.63%** | 0.7077 | 50.98% |

Skill over the majority baseline, `(accuracy − baseline) / (1 − baseline)`: validation 0.13, test 0.01.
Pseudo-R², `1 − loss / ln 2`: validation 0.03, test −0.02.

Comparison on the same splits (single seed each, test has 1,532 examples, so ±2.5 percentage points (95%) of sampling error):

| Model | Valid acc | Test acc | Test loss |
|---|---|---|---|
| `w2v_mean_logreg` | 52.70% | 51.57% | — |
| `w2v_mean_mlp` | 50.80% | 50.46% | 0.6927 |
| `glove_concatenate_small_mlp` | 61.70% | 50.98% | 0.8308 |
| `glove_concatenate_bigger_mlp` | 59.80% | 51.70% | 0.7795 |
| `glove_tfidf_mlp` | 62.20% | 51.96% | 0.8015 |
| **`tfidf_concatenate_small_mlp`** | 57.40% | 51.63% | 0.7077 |
| `tfidf_concatenate_mlp` | 57.60% | 50.46% | 0.7046 |
| `glove_concatenate_rnn` | 61.80% | 51.37% | 0.8940 |
| Majority baseline | 50.80% | 50.98% | — |

Expectation (see `EXPECTATIONS.md`, variant 4): 52–53%. Result 51.63%, just below the estimate.

## Weights

```python
# Load the weights; the MLP class from the notebook must be defined
model = MLP.load_from_checkpoint("weights/best-epoch=0.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 64,756-dim concatenation O1, O2, H1, H2 built as in the notebook. **The fitted vectorizer is not stored with the weights.** It has to be refitted on the training sentences as in the notebook; sklearn sorts the vocabulary alphabetically, so the column order is the same on every refit.

## Notes

- **Same picture as `tfidf_concatenate_mlp`.** Best checkpoint after the first epoch, then five
  epochs without improvement while train accuracy goes to 99.99%. With 16.6M parameters against
  16,801 training examples the model memorises the training data.
- **The second hidden layer did not matter.** 50.46% (two hidden layers) against 51.63% (one) on
  test is inside the sampling error; nearly all parameters sit in the first layer either way.
- **Validation is optimistic,** 57.40% against 51.63% on test, for the same reason as in the other
  static models (see `glove_concatenate_bigger_mlp`, Notes). Test is the number to report.
- **The test loss of 0.7077 is worse than guessing** (ln 2 = 0.6931), pseudo-R² −0.02.
- **No OOV statistic.** Tokens outside the training vocabulary are silently dropped when
  transforming validation and test; their rate was not measured.
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-0
  checkpoint. The real value is 5 (epochs 1–5 without improvement).

# GloVe mean pooling, concatenated + small MLP

Static GloVe embeddings, mean-pooled per sentence, the four sentence vectors concatenated in fixed order, with a one-hidden-layer MLP trained with Lightning for ART (Project 1).

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/glove_concatenate_small_mlp |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/m218oxoo (run `glove-concatenate-small-mlp`, model artifact `glove-concatenate-small-mlp:best`) |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/static/glove_concatenate_small_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `weights/best-epoch=6.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `glove-concatenate-small-mlp:best`) |

## Setup

- **Input format:** identical to `glove_concatenate_bigger_mlp`. `nltk.word_tokenize`, punctuation removed, lowercased after tokenizing, stopwords kept, no stemming/lemmatization, no truncation. Every sentence is the mean of its word vectors, the four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one 1200-dim example vector
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader, 400k words, 300 dims, uncased, pretrained and frozen. Lookup: token, otherwise skipped; a sentence with no hit at all becomes a zero vector in its slot
- **Architecture:** LightningModule `MLP`: `Linear(1200, 256) → ReLU → Linear(256, 2)`, 307,970 parameters, `CrossEntropyLoss` on the two logits. Same head as `w2v_mean_mlp`, one hidden layer less than `glove_concatenate_bigger_mlp`
- **Training:** same as `glove_concatenate_bigger_mlp`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42. Stopped after 12 epochs (3,156 steps), best epoch 6 (0-based). ~78 s on an RTX 2070 SUPER

## Results

| Split | Accuracy | Loss | Majority baseline |
|---|---|---|---|
| Train (last epoch) | 81.72% | 0.407 | — |
| Validation | 61.70% | 0.6513 | 50.80% |
| Test | **50.98%** | 0.8308 | 50.98% |

Skill over the majority baseline, `(accuracy − baseline) / (1 − baseline)`: validation 0.22, test 0.00.
Pseudo-R², `1 − loss / ln 2`: validation 0.06, test −0.20.

Comparison on the same splits (single seed each, test has 1,532 examples, so ±2.5 percentage points (95%) of sampling error):

| Model | Valid acc | Test acc | Test loss |
|---|---|---|---|
| `w2v_mean_logreg` | 52.70% | 51.57% | — |
| `w2v_mean_mlp` | 50.80% | 50.46% | 0.6927 |
| **`glove_concatenate_small_mlp`** | 61.70% | 50.98% | 0.8308 |
| `glove_concatenate_bigger_mlp` | 59.80% | 51.70% | 0.7795 |
| `glove_tfidf_mlp` (final static model) | 59.60% | 52.09% | 0.7559 |
| `tfidf_concatenate_small_mlp` | 57.40% | 51.63% | 0.7077 |
| `tfidf_concatenate_mlp` | 57.60% | 50.46% | 0.7046 |
| `glove_concatenate_rnn` | 61.80% | 51.37% | 0.8940 |
| `glove_concatenate_gru_unidirectional` | 64.30% | 50.98% | 0.7968 |
| `glove_concatenate_gru_bidirectional` | 65.80% | 51.57% | 0.7861 |
| `glove_concatenate_lstm_unidirectional` | 64.70% | 50.59% | 0.8101 |
| `glove_concatenate_lstm_bidirectional` | 62.80% | 51.70% | 0.8616 |
| `glove_lstm_bidirectional_improved` (final RNN model) | 69.80% | 51.83% | 0.8854 |
| Majority baseline | 50.80% | 50.98% | — |

The shortcut rules without any model are the same as in `glove_concatenate_bigger_mlp` (word overlap with O1 and O2: 63.5% validation, 51.9% test; shorter hypothesis: 57.1% validation, 49.9% test).

Expectation (see `EXPECTATIONS.md`, variant 3): ~75%. Validation 61.70%, test 50.98% — not reached.

## Weights

```python
# Load the weights; the MLP class from the notebook must be defined
model = MLP.load_from_checkpoint("weights/best-epoch=6.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 1200-dim concatenation O1, O2, H1, H2 built as in the notebook.

## Notes

- **Meets the project constraint** of a 2-layer classifier with a ReLU (`glove_concatenate_bigger_mlp`
  has 3 linear layers). It is the plain-mean baseline for `glove_tfidf_mlp`.
- **Test accuracy equals the majority baseline** (781 of 1,532). Validation is the best of
  all static models, test is at chance — the same validation/test gap as in
  `glove_concatenate_bigger_mlp` (see Notes there): the model learns the word-overlap shortcut,
  which holds on validation and not on test.
- **The second hidden layer did not matter.** 51.70% (bigger) against 50.98% (small) on test is
  inside the sampling error. The bottleneck is the input, not the classifier.
- **A rerun moves by about a point.** The same setup inside `glove_tfidf_mlp.ipynb` with
  `sentence_weighting = "none"` (run `glove-mean-mlp`) gave 60.80% validation and 51.11% test
  (before dropout was added to that notebook; the run is not in the W&B project).
- **The test loss of 0.8308 is worse than guessing** (ln 2 = 0.6931), pseudo-R² −0.20. Train
  accuracy reaches 81.72% while the best checkpoint is epoch 6.
- **Out of vocabulary:** identical to `glove_concatenate_bigger_mlp` (0.17% of all tokens, 850 types,
  7 empty sentences).
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-6
  checkpoint. The real value is 11 (epochs 7–11 without improvement).

# Word2Vec mean pooling + MLP

Static word embeddings, mean-pooled per sentence and across the four fields, with a one-hidden-layer MLP trained with Lightning for ART (Project 1).

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/w2v_mean_mlp |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/ltmkjfol (run `w2v-mean-mlp`, model artifact `w2v-mean-mlp:best`) |

## Contents

| File | Description |
|---|---|
| `w2v_mean_mlp.ipynb` | Training and evaluation notebook — @todo frozen snapshot of `notebooks/project_1/static/w2v_mean_mlp.ipynb` |
| `weights/best-epoch=3.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `w2v-mean-mlp:best`) |

## Setup

- **Input format:** identical to `w2v_mean_logreg`. Every sentence is tokenized with `nltk.word_tokenize` (punctuation removed, stopwords kept, no lowercasing, no stemming/lemmatization, no truncation), embedded as the mean of its word vectors, and the four field vectors (O1, O2, H1, H2) are mean-pooled into one 300-dim example vector
- **Embeddings:** identical to `w2v_mean_logreg`. `word2vec-google-news-300` via gensim downloader, 3M words, 300 dims, pretrained and frozen. Lookup: exact token, then lowercased token, otherwise skipped; a sentence with no hit at all becomes a zero vector
- **Architecture:** LightningModule `MLP`: `Linear(300, 256) → ReLU → Linear(256, 2)`, 77.6k parameters, `CrossEntropyLoss` on the two logits
- **Training:** Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42. Stopped after 9 epochs (2,367 steps), best epoch 3 (0-based). ~50 s on an RTX 2070 SUPER

## Results

| Split | Accuracy | Loss | Majority baseline |
|---|---|---|---|
| Train (last epoch) | 55.40% | 0.684 | — |
| Validation | 50.80% | 0.6926 | 50.80% |
| Test | **50.46%** | 0.6927 | 50.98% |

Skill over the majority baseline, `(accuracy − baseline) / (1 − baseline)`: validation 0.00, test −0.01.
Pseudo-R², `1 − loss / ln 2`: 0.00 on validation and test.

Expectation (see `EXPECTATIONS.md`, variant 2): up to ~70%. Measured 50.46% — not reached, see Notes.

## Weights

```python
# Load the weights; the MLP class from the notebook must be defined
model = MLP.load_from_checkpoint("weights/best-epoch=3.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

## Notes

- **The model learned nothing.** A validation loss of 0.6926 ≈ ln 2 = 0.6931 means it answers
  ~50/50 for every example. Validation accuracy equals the majority baseline, test accuracy is
  below its baseline. The cause is the same as for `w2v_mean_logreg`: the mean of O1, O2, H1
  and H2 is identical when the two hypotheses are swapped, so the input carries no information
  about which one is correct. More model capacity cannot recover what the input does not contain.
- **Train accuracy of 55.40%** at the last epoch is fitted to the training split; validation and
  test stay at chance.
- **Compared with `w2v_mean_logreg`** (52.70% validation, 51.57% test): the differences are within
  the noise of a single run, ±3.1 points (95%) on 1,000 validation examples and ±2.5 points on
  1,532 test examples.
- **Out of vocabulary:** identical to `w2v_mean_logreg` (11.74% of all tokens, 6 empty sentences).
- **Logging bug:** the W&B summary shows `early_stopping/stopped_epoch` = 0. `trainer.validate(ckpt_path="best")`
  restores the callback states from the epoch-3 checkpoint, which resets the value before the
  summary reads it. The real value is 8 (2,367 steps = 9 epochs × 263 batches).

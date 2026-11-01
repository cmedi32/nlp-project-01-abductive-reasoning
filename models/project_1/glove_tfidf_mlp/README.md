# GloVe TF-IDF weighted mean, concatenated + 2-layer MLP

Static GloVe embeddings, combined per sentence as a TF-IDF weighted mean, the four sentence vectors concatenated in fixed order, with a 2-layer MLP trained with Lightning for ART (Project 1). Final static model of Project 1.

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/glove_tfidf_mlp |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/nhxmvokh (run `glove-tfidf-mlp`, model artifact `glove-tfidf-mlp:best`) |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/static/final_presentable_notebook/glove_tfidf_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here) |
| `notebooks/project_1/static/final_presentable_notebook/LEARNINGS.md` | What the static runs taught, design decisions of the TF-IDF weighting, next iterations |
| `weights/best-epoch=3.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `glove-tfidf-mlp:best`) |

## Setup

- **Project constraint:** static, pretrained word embeddings that are never trained, and a classifier with 2 layers and a ReLU. GloVe is only a lookup table; the sentence vectors are numpy features built before training, so no gradient reaches the embeddings
- **Input format:** same preprocessing as `glove_concatenate_bigger_mlp` (`nltk.word_tokenize`, punctuation removed, lowercased, stopwords kept, no stemming/lemmatization, no truncation). The four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one 1200-dim example vector
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader, 400k words, 300 dims, uncased, pretrained and frozen. A token not in GloVe is skipped; a sentence with no hit at all becomes a zero vector in its slot
- **Sentence vector:** **TF-IDF weighted mean** of the GloVe vectors, `s = Σ idf(t) · v(t) / Σ idf(t)` over the tokens GloVe knows; a word that appears twice counts twice
  - idf from sklearn `TfidfVectorizer` (smoothed: `ln((1 + n) / (1 + df)) + 1`), fitted on the 67,204 single training sentences (O1, O2, H1, H2), 16,189 terms, idf range 2.06–11.42
  - a word in GloVe but never seen in train gets the highest idf, 12.12 (`df = 0`); 1,417 such tokens
  - `sentence_weighting = "none"` in the config switches back to the plain mean (run `glove-mean-mlp`) for the ablation
- **Architecture:** LightningModule `MLP`: `Linear(1200, 256) → ReLU → Linear(256, 2)`, 307,970 trainable parameters (all in the MLP), `CrossEntropyLoss` on the two logits
- **Training:** Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42. Stopped after 9 epochs (2,367 steps), best epoch 3 (0-based). ~132 s on an RTX 2070 SUPER

## Results

| Split | Accuracy | Loss | Majority baseline |
|---|---|---|---|
| Train (last epoch) | 92.27% | 0.234 | — |
| Validation | 62.20% | 0.6443 | 50.80% |
| Test | **51.96%** | 0.8015 | 50.98% |

Skill over the majority baseline, `(accuracy − baseline) / (1 − baseline)`: validation 0.23, test 0.02.
Pseudo-R², `1 − loss / ln 2`: validation 0.07, test −0.16.

Ablation with the same code and seed (from `LEARNINGS.md`):

| Sentence vector | Valid acc | Test acc | Test loss |
|---|---|---|---|
| Plain mean (`glove-mean-mlp`) | 60.80% | 51.11% | 0.84 |
| **TF-IDF weighted mean** | 62.20% | 51.96% | 0.80 |

Comparison on the same splits (single seed each, test has 1,532 examples, so ±2.5 percentage points (95%) of sampling error):

| Model | Valid acc | Test acc | Test loss |
|---|---|---|---|
| `w2v_mean_logreg` | 52.70% | 51.57% | — |
| `w2v_mean_mlp` | 50.80% | 50.46% | 0.6927 |
| `glove_concatenate_small_mlp` | 61.70% | 50.98% | 0.8308 |
| `glove_concatenate_bigger_mlp` | 59.80% | 51.70% | 0.7795 |
| **`glove_tfidf_mlp`** | 62.20% | 51.96% | 0.8015 |
| `tfidf_concatenate_small_mlp` | 57.40% | 51.63% | 0.7077 |
| `tfidf_concatenate_mlp` | 57.60% | 50.46% | 0.7046 |
| `glove_concatenate_rnn` | 61.80% | 51.37% | 0.8940 |
| `glove_concatenate_gru_unidirectional` | 64.30% | 50.98% | 0.7968 |
| `glove_concatenate_gru_bidirectional` | 65.80% | 51.57% | 0.7861 |
| `glove_concatenate_lstm_unidirectional` | 64.70% | 50.59% | 0.8101 |
| `glove_concatenate_lstm_bidirectional` | 62.80% | 51.70% | 0.8616 |
| Majority baseline | 50.80% | 50.98% | — |

The shortcut rules without any model are the same as in `glove_concatenate_bigger_mlp` (word overlap with O1 and O2: 63.5% validation, 51.9% test; shorter hypothesis: 57.1% validation, 49.9% test).

Expectation (see `EXPECTATIONS.md`, variant 3, "weighting better using tfidf"): ~75%. Validation 62.20%, test 51.96% — not reached.

## Weights

```python
# Load the weights; the MLP class from the notebook must be defined
model = MLP.load_from_checkpoint("weights/best-epoch=3.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 1200-dim concatenation O1, O2, H1, H2 built as in the notebook. **The idf weights are not stored with the weights.** They have to be refitted on the training sentences as in the notebook; the fit is deterministic.

## Notes

- **Best static model on validation and test, but inside the noise.** 62.20% / 51.96% is the highest
  of all static runs on both splits, yet every test difference between the static models is
  smaller than the ±2.5 point sampling error.
- **TF-IDF does not hurt and shows no measurable effect on test.** +1.4 points on validation and
  +0.85 points on test against the plain mean. Weighting content words makes the topic-overlap cue
  clearer; that cue works on validation and not on test (see `glove_concatenate_bigger_mlp`, Notes).
- **The test loss of 0.8015 is worse than guessing** (ln 2 = 0.6931), pseudo-R² −0.16. The model
  overfits fast: 92.27% train accuracy at the last epoch, best checkpoint at epoch 3.
- **Word order within a sentence is still ignored.** Only the order of the four sentences matters.
- **Out of vocabulary:** identical to `glove_concatenate_bigger_mlp` (0.17% of all tokens, 850 types,
  7 empty sentences).
- **No logging bug here.** The notebook reads `early_stopping.stopped_epoch` right after `fit`, so the
  W&B summary shows the real value 8.
- **Single seed.** Next iterations (seeds, hard validation metric, swap augmentation,
  regularisation) are listed in `LEARNINGS.md`.

# TF-IDF, concatenated + two-hidden-layer MLP

Sparse tf-idf vectors per sentence (sklearn `TfidfVectorizer`), the four sentence vectors concatenated in fixed order, with a two-hidden-layer MLP trained with Lightning for ART (Project 1). No pretrained embeddings.

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/tfidf_concatenate_mlp |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/a12r6lo9 (run `tfidf-concatenate-mlp`, model artifact `tfidf-concatenate-mlp:best`) |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/static/tfidf_bigger_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here) |
| `weights/best-epoch=0.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `tfidf-concatenate-mlp:best`) |

## Setup

- **Input format:** same preprocessing as `glove_concatenate_bigger_mlp`: `nltk.word_tokenize`, punctuation removed, lowercased, stopwords kept, no stemming/lemmatization, no truncation. The `TfidfVectorizer` receives the token lists directly (`analyzer` passes them through)
- **Vocabulary:** fitted on every single sentence (O1, O2, H1, H2) of the training data, 4 × 16,801 sentences, giving **16,189 terms**. Each sentence is one document for the idf. Default sklearn settings (smooth idf, sublinear tf off, L2 norm per sentence vector). Terms that only occur in validation or test are dropped
- **Example vector:** the four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one sparse vector of 4 × 16,189 = **64,756 dims**. The rows are densified per example in the `Dataset`, never as a whole matrix
- **Architecture:** LightningModule `MLP`: `Linear(64756, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 2)`, 16.64M parameters (16.58M in the first layer), `CrossEntropyLoss` on the two logits
- **Training:** same as `glove_concatenate_bigger_mlp`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42. Stopped after 6 epochs (1,578 steps), best epoch 0 (0-based). ~38 s on an RTX 2070 SUPER

## Results

| Split | Accuracy | Loss | Majority baseline |
|---|---|---|---|
| Train (last epoch) | 100.00% | 0.00013 | — |
| Validation | 57.60% | 0.6687 | 50.80% |
| Test | **50.46%** | 0.7046 | 50.98% |

Skill over the majority baseline, `(accuracy − baseline) / (1 − baseline)`: validation 0.14, test −0.01.
Pseudo-R², `1 − loss / ln 2`: validation 0.04, test −0.02.

Comparison on the same splits (single seed each, test has 1,532 examples, so ±1.3 percentage points of sampling error):

| Model | Valid acc | Test acc | Test loss |
|---|---|---|---|
| `w2v_mean_logreg` | 52.70% | 51.57% | — |
| `w2v_mean_mlp` | 50.80% | 50.46% | 0.6927 |
| `glove_concatenate_bigger_mlp` | 59.80% | 51.70% | 0.7795 |
| **`tfidf_concatenate_mlp`** | 57.60% | 50.46% | 0.7046 |
| Majority baseline | 50.80% | 50.98% | — |

The shortcut rules without any model are the same as in `glove_concatenate_bigger_mlp` (word overlap with O1 and O2: 63.5% validation, 51.9% test; shorter hypothesis: 57.1% validation, 49.9% test).

Expectation (see `EXPECTATIONS.md`, variant 4): 52–53%. Result 50.46%, below the estimate and below the majority baseline.

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

- **No model is clearly better than chance on test.** TF-IDF reaches 50.46%, GloVe concat 51.70%, the majority baseline 50.98%. The 1.2 point gap between TF-IDF and GloVe is inside the sampling error of the test set, so this run does not show that one representation beats the other. The initial expectation "TF-IDF is much worse" is not supported by the numbers either.
- **The model memorises the training data.** Train accuracy is 100% with a loss of 0.00013 after six epochs, while the validation loss was lowest after the first epoch (0.6687) and never improved in the five epochs after. With 64,756 inputs and 16.6M parameters against 16,801 training examples, most weights belong to words seen only a few times.
- **Validation is optimistic.** 57.60% on validation against 50.46% on test. Checkpoint selection and early stopping run on validation, and validation comes from the same pool as train, while test is built differently (see `glove_concatenate_bigger_mlp`, Notes). Test is the number to report.
- **The test loss of 0.7046 is worse than guessing** (ln 2 = 0.6931), pseudo-R² −0.02.
- **Why tf-idf has no edge here:** unrelated words get unrelated dimensions (`sad` and `upset` share nothing), so nothing learned for one word carries over to another. H1 and H2 have separate weights even though they are the same kind of sentence. Words missing from the training vocabulary have no input at all. Cues that tf-idf can pick up (negations, words typical for one hypothesis position) seem to be the same shortcuts that do not hold on test.
- **Word order is ignored,** as in the averaged GloVe vectors. Only the order of the four sentences matters.
- **No OOV statistic.** Unlike the GloVe notebooks, the summary has no `oov/*` entries. Tokens outside the training vocabulary are silently dropped when transforming validation and test; their rate was not measured.
- **Single seed.** No variance estimate. Differences of 1–2 points between the static models should not be interpreted.
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0 because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-0 checkpoint. The real value is 5 (epochs 1–5 without improvement).

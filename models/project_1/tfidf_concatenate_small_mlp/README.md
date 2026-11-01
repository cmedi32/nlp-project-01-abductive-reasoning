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
| `notebooks/project_1/static/tfidf_concatenate_small_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `weights/best-epoch=0.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `tfidf-concatenate-small-mlp:best`) |

## Setup

- **Input format and vocabulary:** identical to `tfidf_concatenate_mlp`. Same preprocessing as the GloVe notebooks, `TfidfVectorizer` fitted on every single training sentence (4 × 16,801), **16,189 terms**, default sklearn settings. The four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one sparse vector of **64,756 dims**, densified per example in the `SparseDataset`
- **Architecture:** LightningModule `MLP`: `Linear(64756, 256) → ReLU → Linear(256, 2)`, 16.58M parameters (16,578,306, all but 514 in the first layer), `CrossEntropyLoss` on the two logits. One hidden layer less than `tfidf_concatenate_mlp`
- **Training:** same as `tfidf_concatenate_mlp`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

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

- **No OOV statistic.** Tokens outside the training vocabulary are silently dropped when
  transforming validation and test; their rate was not measured.
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-0
  checkpoint.

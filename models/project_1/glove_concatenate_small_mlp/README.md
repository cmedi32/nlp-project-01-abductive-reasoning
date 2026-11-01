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
- **Training:** same as `glove_concatenate_bigger_mlp`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

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
- **Out of vocabulary:** identical to `glove_concatenate_bigger_mlp` (0.17% of all tokens, 850 types,
  7 empty sentences).
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-6
  checkpoint.

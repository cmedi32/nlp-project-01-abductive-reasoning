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
| `notebooks/project_1/static/w2v_mean_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `weights/best-epoch=3.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `w2v-mean-mlp:best`) |

## Setup

- **Input format:** identical to `w2v_mean_logreg`. Every sentence is tokenized with `nltk.word_tokenize` (punctuation removed, stopwords kept, no lowercasing, no stemming/lemmatization, no truncation), embedded as the mean of its word vectors, and the four field vectors (O1, O2, H1, H2) are mean-pooled into one 300-dim example vector
- **Embeddings:** identical to `w2v_mean_logreg`. `word2vec-google-news-300` via gensim downloader, 3M words, 300 dims, pretrained and frozen. Lookup: exact token, then lowercased token, otherwise skipped; a sentence with no hit at all becomes a zero vector
- **Architecture:** LightningModule `MLP`: `Linear(300, 256) → ReLU → Linear(256, 2)`, 77.6k parameters, `CrossEntropyLoss` on the two logits
- **Training:** Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

## Weights

```python
# Load the weights; the MLP class from the notebook must be defined
model = MLP.load_from_checkpoint("weights/best-epoch=3.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

## Notes

- **The input cannot tell the hypotheses apart.** As in `w2v_mean_logreg`, the mean of O1, O2, H1 and H2 is
  identical when the two hypotheses are swapped, so the input carries no information about which one is
  correct. More model capacity cannot recover what the input does not contain.
- **Out of vocabulary:** identical to `w2v_mean_logreg` (11.74% of all tokens, 6 empty sentences).
- **Logging bug:** the W&B summary shows `early_stopping/stopped_epoch` = 0. `trainer.validate(ckpt_path="best")`
  restores the callback states from the epoch-3 checkpoint, which resets the value before the
  summary reads it.

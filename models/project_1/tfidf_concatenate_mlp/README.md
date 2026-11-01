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
| `notebooks/project_1/static/tfidf_bigger_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `weights/best-epoch=0.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `tfidf-concatenate-mlp:best`) |

## Setup

- **Input format:** same preprocessing as `glove_concatenate_bigger_mlp`: `nltk.word_tokenize`, punctuation removed, lowercased, stopwords kept, no stemming/lemmatization, no truncation. The `TfidfVectorizer` receives the token lists directly (`analyzer` passes them through)
- **Vocabulary:** fitted on every single sentence (O1, O2, H1, H2) of the training data, 4 × 16,801 sentences, giving **16,189 terms**. Each sentence is one document for the idf. Default sklearn settings (smooth idf, sublinear tf off, L2 norm per sentence vector). Terms that only occur in validation or test are dropped
- **Example vector:** the four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one sparse vector of 4 × 16,189 = **64,756 dims**. The rows are densified per example in the `Dataset`, never as a whole matrix
- **Architecture:** LightningModule `MLP`: `Linear(64756, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 2)`, 16.64M parameters (16.58M in the first layer), `CrossEntropyLoss` on the two logits
- **Training:** same as `glove_concatenate_bigger_mlp`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

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

- **No similarity between words:** unrelated words get unrelated dimensions (`sad` and `upset` share
  nothing), so nothing learned for one word carries over to another. H1 and H2 have separate weights even
  though they are the same kind of sentence. Words missing from the training vocabulary have no input at all.
- **Word order is ignored,** as in the averaged GloVe vectors. Only the order of the four sentences matters.
- **No OOV statistic.** Unlike the GloVe notebooks, the summary has no `oov/*` entries. Tokens outside the training vocabulary are silently dropped when transforming validation and test; their rate was not measured.
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0 because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-0 checkpoint.

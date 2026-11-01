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
| `notebooks/project_1/static/glove_concatenate_bigger_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `weights/best-epoch=3.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `glove-concatenate-bigger-mlp:best`) |

## Setup

- **Input format:** every sentence is tokenized with `nltk.word_tokenize` (punctuation removed, **lowercased after tokenizing** because GloVe is uncased, stopwords kept, no stemming/lemmatization, no truncation) and embedded as the mean of its word vectors. The four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one 1200-dim example vector
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader (Wikipedia 2014 + Gigaword 5, 6B tokens), 400k words, 300 dims, uncased, pretrained and frozen. Lookup: token, otherwise skipped; a sentence with no hit at all becomes a zero vector in its slot
- **Architecture:** LightningModule `MLP`: `Linear(1200, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 2)`, 373k parameters, `CrossEntropyLoss` on the two logits
- **Training:** identical to `w2v_mean_mlp`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

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

- **Concatenation keeps the hypotheses apart.** Swapping H1 and H2 changes the input; the mean of all four
  sentences (`w2v_mean_mlp`) did not.
- **Differs from `w2v_mean_mlp` in four places at once:** GloVe instead of word2vec, lowercasing,
  concatenation instead of the mean, a second hidden layer.
- **Does not meet the project constraint** of a 2-layer classifier (3 linear layers here).
  `glove_concatenate_small_mlp` and the final static model `glove_tfidf_mlp` use 2 layers.
- **Word order within a sentence is still ignored.** Each sentence is still a mean, so "He was
  never lost" and "He was lost" get nearly the same vector. Only the order of the four sentences
  matters.
- **Out of vocabulary:** 0.17% of all tokens (1,144 of 670,395; 850 types), 7 empty sentences,
  against 11.74% for word2vec. GloVe has `to`, `a`, `and`, `of` and `'s`. Lowercasing merges case
  variants such as `mTurk`, `Mturk`, `mturk`, which is why there are 850 OOV types here against 861
  in the word2vec/GloVe OOV comparison (`notebooks/project_1/static/eval/`). What remains are rare names, `selfie` and typos.
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-3
  checkpoint.

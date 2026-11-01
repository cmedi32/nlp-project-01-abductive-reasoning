# GloVe TF-IDF weighted mean, concatenated + 2-layer MLP

Static GloVe embeddings, combined per sentence as a TF-IDF weighted mean, the four sentence vectors concatenated in fixed order, with a 2-layer MLP (dropout 0.1) trained with Lightning for ART (Project 1). **Final static model of Project 1.**

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/glove_tfidf_mlp |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/p8h4wp5i (run `glove-tfidf-mlp`, model artifact `glove-tfidf-mlp:v2`, alias `best`) |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/static/final_presentable_notebook/glove_tfidf_mlp.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `notebooks/project_1/static/final_presentable_notebook/sweep.yaml` | W&B grid sweep over learning rate, hidden size and dropout, see [Hyperparameter sweep](#hyperparameter-sweep) |
| `notebooks/project_1/sweep/sweep_run.py` | Sweep wrapper shared by both final notebooks: runs the unchanged notebook once per grid point with the sweep values |
| `weights/best-epoch=3-v2.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `glove-tfidf-mlp:v2`) |

## Setup

- **Project constraint:** static, pretrained word embeddings that are never trained, and a classifier with 2 layers and a ReLU. GloVe is only a lookup table; the sentence vectors are numpy features built before training, so no gradient reaches the embeddings
- **Input format:** same preprocessing as `glove_concatenate_bigger_mlp` (`nltk.word_tokenize`, punctuation removed, lowercased, stopwords kept, no stemming/lemmatization, no truncation). The four sentence vectors are **concatenated in fixed order O1, O2, H1, H2** into one 1200-dim example vector
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader, 400k words, 300 dims, uncased, pretrained and frozen. A token not in GloVe is skipped; a sentence with no hit at all becomes a zero vector in its slot
- **Sentence vector:** **TF-IDF weighted mean** of the GloVe vectors, `s = Σ idf(t) · v(t) / Σ idf(t)` over the tokens GloVe knows; a word that appears twice counts twice
  - idf from sklearn `TfidfVectorizer` (smoothed: `ln((1 + n) / (1 + df)) + 1`), fitted on the 67,204 single training sentences (O1, O2, H1, H2), 16,189 terms, idf range 2.06–11.42
  - a word in GloVe but never seen in train gets the highest idf, 12.12 (`df = 0`); 1,417 such tokens
  - lowest idf in train: `the` 2.06, `to` 2.20, `a` 2.24, `was` 2.47, `and` 2.66, `her` 3.07, `his` 3.17, `he` 3.20. In "he was never lost" the plain mean gives every word 25%, the TF-IDF mean gives `he` 19.3%, `was` 14.9%, `never` 30.7%, `lost` 35.1%
  - stopwords are kept: TF-IDF gives them a low weight instead of dropping them (NLTK's stopword list would also remove `not`, `no`, `nor`)
  - `sentence_weighting = "none"` in the config switches back to the plain mean (run `glove-mean-mlp`) for the ablation
- **Architecture:** LightningModule `MLP`: `Dropout(0.1) → Linear(1200, 256) → ReLU → Dropout(0.1) → Linear(256, 2)`, 307,970 trainable parameters (all in the MLP, dropout has none), `CrossEntropyLoss` on the two logits
- **Config:** `hidden_dim: 256`, `dropout: 0.1`, `learning_rate: 1e-3`, `weight_decay: 0.0`, `gradient_clip_val: 0.0`, `batch_size: 64`, `max_epochs: 100`, `patience: 5`, `seed: 42`
- **Training:** AdamW with weight decay 0 (identical to Adam), learning rate `1e-3`, no gradient clipping, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

## Hyperparameter sweep

`sweep.yaml` next to the notebook defines a W&B grid sweep (Grid Search Sweep Plan, Project 1). Not run yet.

| Parameter (notebook config key) | Values |
|---|---|
| `learning_rate` | 1e-3, 1e-4, 1e-5 |
| `hidden_dim` | 128, 256, 512 |
| `dropout` | 0.1, 0.3 |

3 × 3 × 2 = 18 runs, metric `accuracy/valid` (maximize): the validation accuracy of each run's best checkpoint by `loss/valid`. Everything else stays at the notebook's config.

```bash
cd notebooks/project_1/static/final_presentable_notebook
wandb sweep sweep.yaml
wandb agent <sweep path printed by the line above>
```

`notebooks/project_1/sweep/sweep_run.py` runs the notebook unchanged and only differs for sweep runs: the sweep values are written into the
notebook's `config`, runs are named after them (e.g. `glove-tfidf-mlp-lr0.0001-h256-d0.1`), checkpoints go to
`weights/sweep/<sweep id>/<run name>/`, the test set is not evaluated and no checkpoint is uploaded. The test number
of the winner comes from one normal run of the notebook with the best values.

## Weights

```python
# Load the weights; the MLP class from the notebook must be defined
model = MLP.load_from_checkpoint("weights/best-epoch=3-v2.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 1200-dim concatenation O1, O2, H1, H2 built as in the notebook. **The idf weights are not stored with the weights.** They have to be refitted on the training sentences as in the notebook; the fit is deterministic.

## Notes

- **Word order within a sentence is still ignored.** Only the order of the four sentences matters.
- **Out of vocabulary:** identical to `glove_concatenate_bigger_mlp` (0.17% of all tokens, 850 types,
  7 empty sentences).
- **No logging bug here.** The notebook reads `early_stopping.stopped_epoch` right after `fit`, so the
  W&B summary shows the real value.

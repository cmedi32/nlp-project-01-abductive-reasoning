# GloVe mean pooling, concatenated + simple RNN + MLP

Static GloVe sentence vectors (mean pooling) read as a sequence of four time steps O1 → O2 → H1 → H2 by a one-layer, unidirectional vanilla RNN, with an MLP head, trained with Lightning for ART (Project 1). First recurrent model.

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/glove_concatenate_rnn |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/jay1wc0v (run `glove-concatenate-rnn`, model artifact `glove-concatenate-rnn:best`); first run with identical metrics: https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/auk12qid |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/rnn/rnn_basic.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `weights/best-epoch=9-v1.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` of run `jay1wc0v` (not tracked in git, uploaded to W&B as `glove-concatenate-rnn:best`). `-v1` because the first run left `best-epoch=9.ckpt` in the same folder |
| `weights/best-epoch=9.ckpt` | Checkpoint of the first run `auk12qid`, same epoch and same `loss/valid` |

## Setup

- **Input format:** identical to `glove_concatenate_bigger_mlp`. `nltk.word_tokenize`, punctuation removed, lowercased, stopwords kept, no stemming/lemmatization, no truncation. Every sentence is the mean of its GloVe vectors; the 1200-dim concatenation O1, O2, H1, H2 is reshaped to **(4, 300): one time step per sentence**
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader, 400k words, 300 dims, uncased, pretrained and frozen. Lookup: token, otherwise skipped; a sentence with no hit at all becomes a zero vector in its slot
- **Architecture:** LightningModule `RNNClassifier`
  - encoder: `nn.RNN(300, 256, batch_first=True)`, 1 layer, unidirectional, tanh, 142,848 parameters. The last hidden state (after reading H2) summarizes the example
  - head: `Linear(256, 256) → ReLU → Linear(256, 2)`, 66,306 parameters
  - 209,154 parameters in total, `CrossEntropyLoss` on the two logits, softmax over the logits gives p(H1) and p(H2)
- **Training:** same as the static models. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

## Weights

```python
# Load the weights; the RNNClassifier class from the notebook must be defined
model = RNNClassifier.load_from_checkpoint("weights/best-epoch=9-v1.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 1200-dim concatenation O1, O2, H1, H2 built as in the notebook; `forward` reshapes it into 4 time steps.

## Notes

- **The RNN reads the same input as the concatenated MLPs.** Each time step is still a mean
  sentence vector, so word order inside a sentence is ignored. The recurrence only orders the four
  sentences, which the fixed concatenation already did. What changes is weight sharing: all four
  sentences go through the same input weights (`300 × 256`), where the MLP has separate weights per
  slot (`1200 × 256`).
- **Unidirectional, last hidden state.** The state the head sees is computed right after H2; H1 and
  the observations reach it through one to three more tanh steps. The GRU and LSTM variants
  (`glove_concatenate_{gru,lstm}_{unidirectional,bidirectional}`) keep everything else fixed.
- **`num_layers` in the config is not wired.** The config logs `num_layers: 1`, but the value is not
  passed to `nn.RNN` (default 1). Changing it in the config has no effect in this notebook.
- **Out of vocabulary:** identical to `glove_concatenate_bigger_mlp` (0.17% of all tokens, 850 types,
  7 empty sentences).
- **Logging bug:** same as `w2v_mean_mlp`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the epoch-9
  checkpoint.

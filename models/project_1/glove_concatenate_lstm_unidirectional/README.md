# GloVe mean pooling, concatenated + unidirectional LSTM + MLP

Static GloVe sentence vectors (mean pooling) read as a sequence of four time steps O1 → O2 → H1 → H2 by a one-layer, unidirectional LSTM, with an MLP head, trained with Lightning for ART (Project 1).

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/glove_concatenate_lstm_unidirectional |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/zxaopklh (run `glove-concatenate-lstm-unidirectional`, model artifact `glove-concatenate-lstm-unidirectional:best`; the notebook output of this run was not saved, so the upload is not verified locally) |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/rnn/glove_lstm_unidirectional.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `weights/best-epoch=6.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `glove-concatenate-lstm-unidirectional:best`) |
| `weights/best-epoch={3,4,76}.ckpt` | Checkpoints of the earlier runs `ikuq3gj9` (`1e-3`), `wn51gxps` and `cuballoa` (`1e-5`), see Notes |

## Setup

- **Input format:** identical to `glove_concatenate_rnn`. `nltk.word_tokenize`, punctuation removed, lowercased, stopwords kept, no stemming/lemmatization, no truncation. Every sentence is the mean of its GloVe vectors; the 1200-dim concatenation O1, O2, H1, H2 is reshaped to **(4, 300): one time step per sentence**
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader, 400k words, 300 dims, uncased, pretrained and frozen. Lookup: token, otherwise skipped; a sentence with no hit at all becomes a zero vector in its slot
- **Architecture:** LightningModule `LSTMClassifier`
  - encoder: `nn.LSTM(300, 256, num_layers=1, bidirectional=False, batch_first=True)`, 571,392 parameters, read with `_, (hidden, _) = self.rnn(sequence)`
  - LSTM cell: a separate cell state c runs next to the hidden state h, input, forget and output gates control what is written, kept and exposed; 4 weight blocks per layer instead of 1. The head only gets h, c is not used
  - reading: one pass over the 4 time steps O1 → O2 → H1 → H2. The features are the last layer's final hidden state (after H2), `hidden[-1]` → 256 dims
  - head: `Linear(256, 256) → ReLU → Linear(256, 2)`, 66,306 parameters
  - 637,698 parameters in total, `CrossEntropyLoss` on the two logits, softmax over the logits gives p(H1) and p(H2)
- **Config:** `encoder: "lstm"`, `bidirectional: False`, `num_layers: 1`, `rnn_hidden_dim: 256`, `hidden_dim: 256`. `bidirectional` and `num_layers` are passed to `nn.LSTM`; the head input grows with the number of directions, so both can be changed in the config without touching the code
- **Training:** same as `glove_concatenate_rnn` except the learning rate. Adam, learning rate `1e-4` (the other recurrent models use `1e-3`, see Notes), batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

Recurrent variants, everything except cell and direction fixed (input 300, hidden 256, 1 layer, same head):

| Model | Cell | Direction | Encoder params | Head params | Total |
|---|---|---|---|---|---|
| `glove_concatenate_rnn` | RNN (tanh) | unidirectional | 142,848 | 66,306 | 209,154 |
| `glove_concatenate_gru_unidirectional` | GRU | unidirectional | 428,544 | 66,306 | 494,850 |
| `glove_concatenate_gru_bidirectional` | GRU | bidirectional | 857,088 | 131,842 | 988,930 |
| **`glove_concatenate_lstm_unidirectional`** | LSTM | unidirectional | 571,392 | 66,306 | 637,698 |
| `glove_concatenate_lstm_bidirectional` | LSTM | bidirectional | 1,142,784 | 131,842 | 1,274,626 |

## Weights

```python
# Load the weights; the LSTMClassifier class from the notebook must be defined
model = LSTMClassifier.load_from_checkpoint("weights/best-epoch=6.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 1200-dim concatenation O1, O2, H1, H2 built as in the notebook; `forward` reshapes it into 4 time steps.

## Notes

- **Only the recurrent layer and the learning rate differ from `glove_concatenate_rnn`.** Preprocessing,
  embeddings, sequence of four mean sentence vectors, head, the rest of the training and logging are
  the same.
- **Unidirectional:** as in `glove_concatenate_rnn`, the head sees the state right after H2; H1 and the observations reach it through one to three more recurrent steps.
- **The sequence has only 4 steps.** Vanishing gradients over long sequences, the problem the gates
  were built for, do not occur here; the gates mainly change how the four sentence vectors are mixed.
- **Word order within a sentence is still ignored,** each time step is a mean sentence vector.
- **Learning rate differs from the other recurrent models.** The notebook config uses `1e-4` instead of
  `1e-3`, so for this model the learning rate changed together with the cell.
- **Logging bug:** same as `glove_concatenate_rnn`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the best checkpoint.

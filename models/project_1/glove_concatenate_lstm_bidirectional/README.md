# GloVe mean pooling, concatenated + bidirectional LSTM + MLP

Static GloVe sentence vectors (mean pooling) read as a sequence of four time steps O1 → O2 → H1 → H2 by a one-layer, bidirectional LSTM, with an MLP head, trained with Lightning for ART (Project 1).

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/glove_concatenate_lstm_bidirectional |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/s54mav5x (run `glove-concatenate-lstm-bidirectional`, model artifact `glove-concatenate-lstm-bidirectional:best`) |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/rnn/glove_lstm_bidirectional.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `weights/best-epoch=4.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `glove-concatenate-lstm-bidirectional:best`) |

## Setup

- **Input format:** identical to `glove_concatenate_rnn`. `nltk.word_tokenize`, punctuation removed, lowercased, stopwords kept, no stemming/lemmatization, no truncation. Every sentence is the mean of its GloVe vectors; the 1200-dim concatenation O1, O2, H1, H2 is reshaped to **(4, 300): one time step per sentence**
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader, 400k words, 300 dims, uncased, pretrained and frozen. Lookup: token, otherwise skipped; a sentence with no hit at all becomes a zero vector in its slot
- **Architecture:** LightningModule `LSTMClassifier`
  - encoder: `nn.LSTM(300, 256, num_layers=1, bidirectional=True, batch_first=True)`, 1,142,784 parameters, read with `_, (hidden, _) = self.rnn(sequence)`
  - LSTM cell: a separate cell state c runs next to the hidden state h, input, forget and output gates control what is written, kept and exposed; 4 weight blocks per layer instead of 1. The head only gets h, c is not used
  - reading: two passes over the 4 time steps, forward O1 → O2 → H1 → H2 and backward H2 → H1 → O2 → O1. The features are the last layer's forward final state (after H2) concatenated with its backward final state (after O1), `torch.cat([hidden[-2], hidden[-1]], dim=1)` → 512 dims
  - head: `Linear(512, 256) → ReLU → Linear(256, 2)`, 131,842 parameters
  - 1,274,626 parameters in total, `CrossEntropyLoss` on the two logits, softmax over the logits gives p(H1) and p(H2)
- **Config:** `encoder: "lstm"`, `bidirectional: True`, `num_layers: 1`, `rnn_hidden_dim: 256`, `hidden_dim: 256`. `bidirectional` and `num_layers` are passed to `nn.LSTM`; the head input grows with the number of directions, so both can be changed in the config without touching the code
- **Training:** same as `glove_concatenate_rnn`. Adam, learning rate `1e-3`, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42. Stopped after 10 epochs (2,630 steps), best epoch 4 (0-based). ~74 s on an RTX 2070 SUPER

Recurrent variants, everything except cell and direction fixed (input 300, hidden 256, 1 layer, same head):

| Model | Cell | Direction | Encoder params | Head params | Total |
|---|---|---|---|---|---|
| `glove_concatenate_rnn` | RNN (tanh) | unidirectional | 142,848 | 66,306 | 209,154 |
| `glove_concatenate_gru_unidirectional` | GRU | unidirectional | 428,544 | 66,306 | 494,850 |
| `glove_concatenate_gru_bidirectional` | GRU | bidirectional | 857,088 | 131,842 | 988,930 |
| `glove_concatenate_lstm_unidirectional` | LSTM | unidirectional | 571,392 | 66,306 | 637,698 |
| **`glove_concatenate_lstm_bidirectional`** | LSTM | bidirectional | 1,142,784 | 131,842 | 1,274,626 |

## Results

| Split | Accuracy | Loss | Majority baseline |
|---|---|---|---|
| Train (last epoch) | 86.95% | 0.284 | — |
| Validation | 62.80% | 0.6331 | 50.80% |
| Test | **51.70%** | 0.8616 | 50.98% |

Skill over the majority baseline, `(accuracy − baseline) / (1 − baseline)`: validation 0.24, test 0.01.
Pseudo-R², `1 − loss / ln 2`: validation 0.09, test −0.24.

Recurrent variants on the same splits (single seed each, test has 1,532 examples, so ±2.5 percentage points (95%) of sampling error):

| Model | Epochs | Valid acc | Valid loss | Test acc | Test loss |
|---|---|---|---|---|---|
| `glove_concatenate_rnn` | 15 | 61.80% | 0.6335 | 51.37% | 0.8940 |
| `glove_concatenate_gru_unidirectional` | 9 | 64.30% | 0.6314 | 50.98% | 0.7968 |
| `glove_concatenate_gru_bidirectional` | 8 | 65.80% | 0.6362 | 51.57% | 0.7861 |
| `glove_concatenate_lstm_unidirectional` (lr `1e-4`) | 12 | 64.70% | 0.6407 | 50.59% | 0.8101 |
| **`glove_concatenate_lstm_bidirectional`** | 10 | 62.80% | 0.6331 | 51.70% | 0.8616 |
| `glove_tfidf_mlp` (best static model) | 9 | 62.20% | 0.6443 | 51.96% | 0.8015 |
| Majority baseline | — | 50.80% | — | 50.98% | — |

Expectation (see `EXPECTATIONS.md`, RNN roadmap step 3, `Unidirectional → Bidirectional` — Adv 8–14). Overall target for the recurrent models: ~60%. Validation 62.80% is above the ~60% target, test 51.70% is not.

## Weights

```python
# Load the weights; the LSTMClassifier class from the notebook must be defined
model = LSTMClassifier.load_from_checkpoint("weights/best-epoch=4.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 1200-dim concatenation O1, O2, H1, H2 built as in the notebook; `forward` reshapes it into 4 time steps.

## Notes

- **Only the recurrent layer differs from `glove_concatenate_rnn`.** Preprocessing, embeddings,
  sequence of four mean sentence vectors, head, training and logging are the same, so a
  difference in the results comes from the cell type and the direction, or from seed noise
  (single seed: ±2.5 points (95%) on test).
- **Bidirectional:** the forward pass ends on H2, the backward pass ends on O1, so both ends of the story reach the head in one step. It doubles the encoder and the first head layer.
- **The sequence has only 4 steps.** Vanishing gradients over long sequences, the problem the gates
  were built for, do not occur here; the gates mainly change how the four sentence vectors are mixed.
- **Word order within a sentence is still ignored,** each time step is a mean sentence vector.
- **Fits the training data hardest, best recurrent test accuracy.** 86.95% train accuracy in the
  last epoch, the most of all recurrent runs, and the highest test loss of the GRU/LSTM runs (0.8616).
  Test 51.70% is the best of the current recurrent runs, 0.26 points below `glove_tfidf_mlp`.
- **Validation gain does not reach test.** The five recurrent runs in the table above land between 61.80% and 65.80% on
  validation and between 50.59% and 51.70% on test, the same validation/test gap as in the static
  models (see `glove_concatenate_bigger_mlp`, Notes). Every test difference between them is smaller
  than the ±2.5 point sampling error.
- **Logging bug:** same as `glove_concatenate_rnn`. The W&B summary shows `early_stopping/stopped_epoch` = 0
  because `trainer.validate(ckpt_path="best")` restores the callback states from the best checkpoint.
  The real value is 9 (epochs 5–9 without improvement).

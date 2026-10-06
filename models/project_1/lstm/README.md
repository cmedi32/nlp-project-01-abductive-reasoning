# LSTM

Word embeddings with an LSTM encoder for ART (Project 1).

## Links

| Field | Value |
|---|---|
| GitHub reference | `<https://github.com/<user>/<repo>/tree/main/models/lstm>` |
| Weights & Biases profile | `<https://wandb.ai/<username>>` |
| Weights & Biases run / reference | `<https://wandb.ai/<team>/<project>/runs/<run-id>>` |

## Contents 

| File | Description |
|---|---|
| `lstm.ipynb` | Training and evaluation notebook |
| `weights/` | Trained weights (not tracked in git) |

## Setup @todo

- **Input format:** `<e.g. O1 [SEP] H1 [SEP] O2 and O1 [SEP] H2 [SEP] O2, scored separately>`
- **Embeddings:** `<e.g. GloVe 300d, frozen/fine-tuned>`
- **Architecture:** `<e.g. 1-layer BiLSTM, hidden size 256, linear classifier>`
- **Hyperparameters:** `<learning rate, batch size, epochs, dropout, seed>`

## Results @todo

| Split | Accuracy |
|---|---|
| Validation | `<...>` |
| Test | `<...>` |

Expectation (see `EXPECTATIONS.md`): `<...>`

## Weights @todo

```python
# Example: load the weights
model.load_state_dict(torch.load("weights/lstm_best.pt"))
```

## Notes 

`<observations, limitations, error analysis>`

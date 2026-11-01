# GloVe TF-IDF weighted mean, candidate framing + shared bidirectional LSTM + MLP

Static GloVe sentence vectors (TF-IDF weighted mean) arranged as two candidate stories, O1 → H1 → O2 and O1 → H2 → O2, each read by the same one-layer bidirectional LSTM, with an MLP head (dropout 0.3) that sees both stories side by side, trained with Lightning for ART (Project 1). **Final RNN model of Project 1.**

## Links

| Field | Value |
|---|---|
| GitHub reference | https://github.com/cmedi32/nlp-project-01-abductive-reasoning/tree/main/models/project_1/glove_lstm_bidirectional_improved |
| Weights & Biases profile | https://wandb.ai/cmedi32 |
| Weights & Biases run / reference | https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning/runs/03na0g0x (run `glove-lstm-bidirectional-improved`, model artifact `glove-lstm-bidirectional-improved:v2`, alias `best`) |

## Contents

| File | Description |
|---|---|
| `notebooks/project_1/rnn/final_presentable_notebook/glove_lstm_bidirectional_improved.ipynb` | Training and evaluation notebook (lives in the notebooks folder, no frozen copy here; preprocessing, features, loaders and training come from `notebooks/project_1/pipeline/`, the model class is defined in the notebook) |
| `notebooks/project_1/rnn/final_presentable_notebook/sweep.yaml` | W&B grid sweep over learning rate, LSTM state size and dropout, see [Hyperparameter sweep](#hyperparameter-sweep) |
| `notebooks/project_1/sweep/sweep_run.py` | Sweep wrapper shared by both final notebooks: runs the unchanged notebook once per grid point with the sweep values |
| `notebooks/project_1/rnn/final_presentable_notebook/parameter-architectural-decisions.txt` | Open architecture and hyperparameter decisions |
| `weights/best-epoch=3-v1.ckpt` | Lightning checkpoint of the best epoch by `loss/valid` (not tracked in git, uploaded to W&B as `glove-lstm-bidirectional-improved:v2`) |

## Setup

- **Input format:** same preprocessing as `glove_concatenate_rnn` (`nltk.word_tokenize`, punctuation removed, lowercased, stopwords kept, no stemming/lemmatization, no truncation)
- **Embeddings:** `glove-wiki-gigaword-300` via gensim downloader, 400k words, 300 dims, uncased, pretrained and frozen. A token not in GloVe is skipped; a sentence with no hit at all becomes a zero vector
- **Sentence vector:** TF-IDF weighted mean of the GloVe vectors, the same weighting as in `glove_tfidf_mlp` (idf fitted on the 67,204 training sentences, 16,189 terms, 1,417 tokens in GloVe but never seen in train). The idf weights are fixed numbers, not learned
- **Candidate framing:** every example becomes two stories, **A = O1 → H1 → O2** and **B = O1 → H2 → O2**, 3 time steps of 300 dims each. The LSTM reads each hypothesis between the two observations, so it can model whether the hypothesis bridges them. One row is A followed by B, 6 × 300 = 1800 dims (`combine_concat_candidate_framing`)
- **Architecture:** LightningModule `CandidateLSTMClassifier`
  - encoder: `nn.LSTM(300, 256, num_layers=1, bidirectional=True, batch_first=True)`, 1,142,784 parameters, **shared**: the same weights encode story A and story B
  - per story: the last layer's forward final state (after O2) concatenated with its backward final state (after O1), `torch.cat([hidden[-2], hidden[-1]], dim=1)` → 512 dims; the cell state c is not used
  - head input: both stories side by side, `[h_A ; h_B]` → 1024 dims
  - head: `Dropout(0.3) → Linear(1024, 256) → ReLU → Dropout(0.3) → Linear(256, 2)`, 262,914 parameters
  - 1,405,698 parameters in total, `CrossEntropyLoss` on the two logits, softmax over the logits gives p(H1) and p(H2)
- **Config:** `encoder: "lstm"`, `bidirectional: True`, `num_layers: 1`, `rnn_hidden_dim: 256`, `hidden_dim: 256`, `dropout: 0.3`, `learning_rate: 1e-3`, `weight_decay: 0.0`, `gradient_clip_val: 0.0`, `batch_size: 64`, `max_epochs: 100`, `patience: 5`, `seed: 42`. `bidirectional`, `num_layers` and `rnn_hidden_dim` are passed to `nn.LSTM` and the head input follows them; `encoder` is only logged, the cell is always an LSTM. With `num_layers > 1` the dropout also acts between the LSTM layers
- **Training:** AdamW with weight decay 0 (identical to Adam), learning rate `1e-3`, no gradient clipping, batch size 64 (train shuffled), max 100 epochs, early stopping on `loss/valid` with patience 5, best checkpoint by `loss/valid`, seed 42

## Hyperparameter sweep

`sweep.yaml` next to the notebook defines a W&B grid sweep (Grid Search Sweep Plan, Project 1). Not run yet.

| Parameter (notebook config key) | Values |
|---|---|
| `learning_rate` | 1e-3, 1e-4, 1e-5 |
| `rnn_hidden_dim` (the plan's `hidden_size`, LSTM state per direction) | 128, 256 |
| `dropout` | 0.1, 0.3 |

3 × 2 × 2 = 12 runs, metric `accuracy/valid` (maximize): the validation accuracy of each run's best checkpoint by
`loss/valid`. The head's `hidden_dim` and everything else stay at the notebook's config.

```bash
cd notebooks/project_1/rnn/final_presentable_notebook
wandb sweep sweep.yaml
wandb agent <sweep path printed by the line above>
```

`notebooks/project_1/sweep/sweep_run.py` runs the notebook unchanged and only differs for sweep runs: the sweep values are written into the
notebook's `config`, runs are named after them (e.g. `glove-lstm-bidirectional-improved-lr0.0001-rnn256-d0.3`),
checkpoints go to `weights/sweep/<sweep id>/<run name>/`, the test set is not evaluated (no `trainer.test`, no
predictions) and no checkpoint is uploaded. The test number of the winner comes from one normal run of the notebook
with the best values.

## Weights

```python
# Load the weights; the CandidateLSTMClassifier class from the notebook must be defined
model = CandidateLSTMClassifier.load_from_checkpoint("weights/best-epoch=3-v1.ckpt", map_location="cpu")
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)), dim=1)
```

`vector` is the 1800-dim row story A (O1, H1, O2) followed by story B (O1, H2, O2) built as in the notebook; `forward`
reshapes it into 2 stories of 3 time steps. **The idf weights are not stored with the weights**, they have to be
refitted on the training sentences as in the notebook; the fit is deterministic.

## Notes

- **Shared encoder, no interaction features.** Swapping H1 and H2 swaps h_A and h_B, so the head sees
  both candidates in a fixed order. Comparison features such as `|h_A − h_B|` or `h_A ⊙ h_B` are not
  implemented.
- **Word order within a sentence is still ignored,** each time step is a sentence vector.
- **Out of vocabulary:** identical to `glove_concatenate_bigger_mlp` (0.17% of all tokens, 850 types,
  7 empty sentences).
- **Logging bug:** the W&B summary shows `early_stopping/stopped_epoch` = 0 because the notebook reads it
  after `trainer.validate(ckpt_path="best")`, which restores the callback states from the best
  checkpoint.

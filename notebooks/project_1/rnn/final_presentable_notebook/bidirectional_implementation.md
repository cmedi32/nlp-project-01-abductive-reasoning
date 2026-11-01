# BiLSTM Implementation Guide — ART Task

## Task Framing

Given O1, O2, h1, h2 → classify which hypothesis (1 or 2) is more plausible.
Create **two candidate sequences** per example, one per hypothesis:

- Sequence A: `[O1] <sep> [h1] <sep> [O2]`
- Sequence B: `[O1] <sep> [h2] <sep> [O2]`

Concatenating O1 + hypothesis + O2 lets the BiLSTM model the narrative flow — does this hypothesis bridge the two observations?

---

## Pipeline

### 1. Tokenize & Vocab

- Word-level tokenization (split or spaCy), lowercased
- Build vocab from training set only
- Special tokens: `<pad>` (idx 0), `<unk>`, `<sep>`
- Pad/truncate both sequences to the same fixed length per batch

### 2. Embedding Layer

- `nn.Embedding` initialized with **pretrained GloVe** (e.g. `glove.6B.300d`)
- OOV words → random or zero vector
- Experiment: freeze vs. fine-tune embeddings
- Output: `(batch_size, seq_len, 300)`

### 3. Shared BiLSTM Encoder

- Same BiLSTM encodes both Sequence A and Sequence B (shared weights)
- Final hidden state = `[h_forward ; h_backward]` → size `2 * hidden_dim`
- Produces per example:
  - `h_A` from (O1 + h1 + O2)
  - `h_B` from (O1 + h2 + O2)

### 4. MLP Classifier Input

Concatenate representations + interaction features:

```
mlp_input = [ h_A ; h_B ; |h_A - h_B| ; h_A * h_B ]
```

- `h_A, h_B` — raw representations of each candidate
- `|h_A - h_B|` — what differs between the two candidates
- `h_A * h_B` — where they align
- Total size: `4 × 2 × hidden_dim`

### 5. MLP Head

- 1–2 hidden layers, ReLU activation, dropout
- Final linear layer → 2 outputs
- Loss: CrossEntropyLoss

---

## Why This Framing Works

Encoding O1+h+O2 as a single sequence means the LSTM reads the observation, then the hypothesis, then the second observation — hidden states carry cross-segment context. Encoding the four texts separately throws away that sequential interaction.

## Key Hyperparameters to Experiment With

| Parameter | Try |
|---|---|
| hidden_dim | 128, 256 |
| num_layers | 1, 2 |
| dropout | 0.3, 0.5 |
| embedding | frozen vs. fine-tuned |
| learning rate | 1e-3, 5e-4 |
| max_seq_len | based on data distribution |

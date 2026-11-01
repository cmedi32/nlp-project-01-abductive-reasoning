# Learnings: static word embeddings + 2-layer MLP (Project 1)

Working folder for the final static model. `glove_tfidf_mlp.ipynb` is the notebook. This file
collects what the static runs taught us. It also tracks the next iterations and the @todos
still open for the presentation.

## Project constraint

- **Static word embeddings:** pretrained, frozen, never trained (`glove-wiki-gigaword-300`).
- **Classifier:** 2 layers with a ReLU non-linearity, `Linear(1200, 256) → ReLU → Linear(256, 2)`.
- **Only the classifier is trained.** GloVe is used as a lookup table, and the sentence vectors are
  built as numpy features before training, so no gradient reaches the embeddings. The notebook
  prints the trainable parameters: 307,970, all in the MLP.
- `glove_concatenate_bigger_mlp` had 3 linear layers (2 hidden) and does not meet the 2-layer
  requirement. The final notebook goes back to 2 layers.

## Learnings

### 1. Averaging the four sentences destroys the task

The mean of O1, O2, H1, H2 does not change when H1 and H2 swap, so the input carries no
information about which hypothesis is correct. `w2v_mean_mlp` stayed at a validation loss of
0.6926 ≈ ln 2 = 0.6931, which means it answers 50/50 for every example. More capacity cannot recover what the input
does not contain. 

**Concatenating** the sentence vectors in the fixed order O1, O2, H1, H2 fixed it: the loss drops below ln 2.

### 2. Validation and test measure different things

- The word-overlap rule reaches 63.5% on validation and 51.9% on test. Choosing the shorter hypothesis works the
  same way: 57.1% on validation, 49.9% on test.
- Test is very likely the original ART dev set: exactly 1,532 examples, adversarially filtered
  against such cues. Train and validation come from the same pool. It is not leakage: no observation
  pair from validation or test appears in train.
- Early stopping and checkpoint selection run on validation, so they favour the shortcut.
  **Test is the number to report.**

### 3. What the model learns is the shortcut

Averaged word vectors measure shared topic and vocabulary, so the MLP learns "the hypothesis that sounds more like the story is correct". 

The word-overlap rule alone (63.5%) beats every model on validation. On test that cue does not hold, which leaves about 51%.

### 4. The model overfits fast and is overconfident

- `glove_concatenate_bigger_mlp`: 76% train accuracy against 60% validation; best epoch 3 of 9.
- Every GloVe run has a **test loss above ln 2** (0.78–0.84), so it is worse than always answering 50/50. The model is
  confident and wrong about half the time.

### 5. Single-seed differences are mostly noise

- ±2.5 points (95%) on test and ±3.1 points on validation. No difference between the static models in the
  table above is larger than that.
- The same setup gave 61.70% / 50.98% (`glove_concatenate_small_mlp`) and later 60.80% / 51.11%
  (`glove_mean_mlp`). Even with the same seed, a rerun moves by about a point because of GPU
  non-determinism and the float32 cast.
- Model comparisons need 3–5 seeds per variant, reported as mean ± std.

### 6. More layers did not help

The 3-layer MLP was not better than the 2-layer one (51.70% vs. 50.98% on test, inside the noise).
The bottleneck is the input, not the classifier. Since the project asks for 2 layers, the final
notebook uses 2.

### 7. The plain mean is dominated by frequent words → TF-IDF weighting

**Problem.** "the", "to", "a", "was", "he" appear in O1, O2, H1 and H2 alike. In the plain mean
they get the same weight as the words that matter, so the four sentence vectors look alike.

**Approach.** Sentence vector = TF-IDF weighted mean of the GloVe vectors:

- `idf(w) = ln((1 + n) / (1 + df(w))) + 1`, the scikit-learn default `smooth_idf=True`
- `s = Σ_t idf(t) · v(t) / Σ_t idf(t)` over the tokens of the sentence that GloVe knows; a word that
  appears twice counts twice (the "tf" part)

**Design decisions.**

| Decision | Choice | Why |
|---|---|---|
| What is a document for the idf | one sentence (O1, O2, H1 or H2) | the weights should separate sentences from each other |
| Fitted on | training sentences only (67,204) | validation and test must not leak into the weights |
| Word in GloVe but never in train | highest idf (12.12, `df = 0`) | consistent with the smoothed formula: unseen = rarest; 1,417 tokens |
| Stopwords | kept | tf-idf gives them a low weight instead of dropping them; NLTK's stopword list would also remove `not`, `no`, `nor` |
| Normalisation | divide by the sum of the weights | keeps the vector on the scale of a mean, independent of sentence length |

**Effect on one sentence** ("he was never lost"):

| Token | Plain mean | TF-IDF |
|---|---|---|
| he | 25.0% | 19.3% |
| was | 25.0% | 14.9% |
| never | 25.0% | 30.7% |
| lost | 25.0% | 35.1% |

Lowest idf in train: `the` 2.06, `to` 2.20, `a` 2.24, `was` 2.47, `and` 2.66, `her` 3.07, `his` 3.17, `he` 3.20.

**Difference from the tf-idf-only notebooks.** There the tf-idf vector *is* the input
(4 × 16,189 = 64,756 dims, 16.6M parameters, 100% train accuracy). Here tf-idf only decides how
much each GloVe vector counts, and the input stays 1200 dims. `sad` and `upset` stay close because
GloVe puts them close.

**Result.** Compared with the plain mean, under the same code and seed: +1.4 points on validation (62.20% vs. 60.80%),
+0.85 points on test (51.96% vs. 51.11%), test loss 0.80 vs. 0.84. **Inside the noise: TF-IDF does
not hurt and may help slightly on validation, but it shows no measurable effect on test.** As
expected, weighting content words makes the topic-overlap cue clearer. That cue works on validation
and not on test.

## Next iterations (same model, not applied yet)

Ordered by how much they are expected to tell us:

- [ ] **Seeds:** 3–5 runs per variant, mean ± std. This is needed before any further comparison is meaningful.
- [ ] **Hard validation metric:** accuracy on the validation examples where the word-overlap rule
  is wrong or tied. It shows whether a change learns more than the shortcut, without touching test.

- [ ] **Swap augmentation:** add (O1, O2, H2, H1, 1 − y) for every training example. This doubles the data and
  removes the position bias. At test time, average `p(x)` with the flipped `p(swap(x))`.
- [ ] **Regularisation:** `AdamW` with weight decay 1e-4…1e-2, a lower learning rate (3e-4, 1e-4), and
  `CrossEntropyLoss(label_smoothing=0.1)` against the overconfidence (test loss > ln 2).

- [ ] **Standardise the features:** fit mean and std on train only (after augmentation).
- [ ] **Comparison features:** `H1 − H2`, `|H1 − H2|`, `O ⊙ H1`, `O ⊙ H2` with `O = (O1 + O2) / 2`.
  A ReLU MLP learns products between input slots poorly. Watch out: this also makes the overlap shortcut easier.
- [ ] **Mean + max pooling** per sentence (600 dims per sentence)

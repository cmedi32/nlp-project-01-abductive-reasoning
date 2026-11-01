# Abductive Commonsense Reasoning (ART): NLP Course Project HS26

Course project for the NLP course at HSLU (Information Technology), HS26, lecturer: Andreas Marfurt.
The task is **Abductive Reasoning in Narrative Text (ART)**: given two observations O₁ and O₂, select the more plausible of two hypotheses that explains what happened in between.

## Disclaimer: AI-assisted code

> **Most of the code in this repository was written with agentic AI support** (coding agents). I have reviewed it and am responsible for it, but it may contain errors, inefficiencies or questionable design choices.
>
> Progress and known weak spots of the AI-generated code are documented in each model's `README.md` under `models/<model>/`

The tools used are listed under [Tools and attribution](#tools-and-attribution).

## Project links

| Field | Value |
|---|---|
| Author | `Cédric Müller` |
| GitHub repository | <https://github.com/cmedi32/nlp-project-01-abductive-reasoning> |
| Weights & Biases profile | <https://forge.coreweave.com/profile/cmedi32> |
| Weights & Biases project | <https://wandb.ai/cmedi32-hochschule-luzern/nlp-project-01-abductive-reasoning> |

## Projects

| Project | Methods | Submission |
|---|---|---|
| 1 | Word embeddings, word embeddings with RNN | Week 8 (Ilias) |
| 2 | Transformers: randomly initialized, pretrained, LLM | Week 14 (Ilias) |

## Project 1 results

Single seed per model. Test has 1,532 examples, so every test accuracy carries about ±2.5 percentage points (95%) of
sampling error; validation (1,000 examples) about ±3.1. Each model card has the details, the W&B run and the notes.

| Model card | Method | Valid acc | Test acc |
|---|---|---|---|
| [`w2v_mean_logreg`](models/project_1/w2v_mean_logreg/README.md) | word2vec mean per sentence, the four sentences averaged, logistic regression | 52.70% | 51.57% |
| [`w2v_mean_mlp`](models/project_1/w2v_mean_mlp/README.md) | word2vec mean per sentence, the four sentences averaged, 2-layer MLP | 50.80% | 50.46% |
| [`tfidf_concatenate_mlp`](models/project_1/tfidf_concatenate_mlp/README.md) | TF-IDF vectors, concatenated, 3-layer MLP | 57.60% | 50.46% |
| [`tfidf_concatenate_small_mlp`](models/project_1/tfidf_concatenate_small_mlp/README.md) | TF-IDF vectors, concatenated, 2-layer MLP | 57.40% | 51.63% |
| [`glove_concatenate_bigger_mlp`](models/project_1/glove_concatenate_bigger_mlp/README.md) | GloVe mean per sentence, concatenated, 3-layer MLP | 59.80% | 51.70% |
| [`glove_concatenate_small_mlp`](models/project_1/glove_concatenate_small_mlp/README.md) | GloVe mean per sentence, concatenated, 2-layer MLP | 61.70% | 50.98% |
| **[`glove_tfidf_mlp`](models/project_1/glove_tfidf_mlp/README.md)** | **Final static model:** GloVe TF-IDF weighted mean, concatenated, 2-layer MLP, dropout 0.1 | 59.60% | **52.09%** |
| [`glove_concatenate_rnn`](models/project_1/glove_concatenate_rnn/README.md) | GloVe sentence vectors as 4 time steps, unidirectional RNN | 61.80% | 51.37% |
| [`glove_concatenate_gru_unidirectional`](models/project_1/glove_concatenate_gru_unidirectional/README.md) | same, unidirectional GRU | 64.30% | 50.98% |
| [`glove_concatenate_gru_bidirectional`](models/project_1/glove_concatenate_gru_bidirectional/README.md) | same, bidirectional GRU | 65.80% | 51.57% |
| [`glove_concatenate_lstm_unidirectional`](models/project_1/glove_concatenate_lstm_unidirectional/README.md) | same, unidirectional LSTM | 64.70% | 50.59% |
| [`glove_concatenate_lstm_bidirectional`](models/project_1/glove_concatenate_lstm_bidirectional/README.md) | same, bidirectional LSTM | 62.80% | 51.70% |
| **[`glove_lstm_bidirectional_improved`](models/project_1/glove_lstm_bidirectional_improved/README.md)** | **Final RNN model:** GloVe TF-IDF sentence vectors, candidate stories O1 → H → O2, shared bidirectional LSTM, dropout 0.3 | **69.80%** | 51.83% |
| Majority baseline | always the more frequent label | 50.80% | 50.98% |

**What the numbers say.** Validation and test disagree on this dataset. A word-overlap rule without any model reaches
63.5% on validation and 51.9% on test: test is very likely the adversarially filtered ART dev set, train and
validation come from the same pool. The models learn the overlap shortcut, so validation keeps improving while every
test accuracy stays within the sampling error of the majority baseline. Test is the number to report. The full
analysis is in the notes of [`glove_concatenate_bigger_mlp`](models/project_1/glove_concatenate_bigger_mlp/README.md).

## Repository structure

```
.
├── README.md
├── EXPECTATIONS.md                        # Expectations before and during Project 1
├── requirements.txt
├── Libraries.pdf                          # Libraries referenced in the course slides
├── class_solution/                        # Course solutions (HSLU), reference for the coding agents only
├── notebooks/project_1/
│   ├── pipeline/                          # Shared stages imported by every notebook (see below)
│   ├── sweep/                             # sweep_run.py: W&B sweep wrapper for the final notebooks
│   ├── static/                            # Static word embeddings + classifier, one notebook per experiment
│   │   ├── eval/                          # OOV comparison of word2vec, GloVe and fastText
│   │   └── final_presentable_notebook/    # Final static model + W&B sweep
│   └── rnn/                               # Word embeddings + RNN, one notebook per experiment
│       └── final_presentable_notebook/    # Final RNN model + W&B sweep
└── models/project_1/
    └── <model>/
        ├── README.md                      # Model card: setup, hyperparameters, results, W&B run, notes
        └── weights/                       # Checkpoints (not tracked in git, uploaded to W&B as artifacts)
```

### `notebooks/project_1/`

One Jupyter notebook per experiment. Each notebook covers preprocessing, input/output format, model architecture,
training and evaluation, and logs its run to Weights & Biases. The model class (the complete LightningModule) lives
inside each notebook; everything else comes from the shared `pipeline/` package:

| Module | Stage |
|---|---|
| `tracking.py` | `.env` + W&B login, run init, upload of the best checkpoint |
| `data.py` | dataset splits, `TEXT_FIELDS` |
| `preprocessing/` | raw text → tokens, one file per variant (`preprocess()` + `CONFIG` for W&B) |
| `features/` | tokens → feature matrix: static embeddings (mean or TF-IDF weighted mean pooling) or TF-IDF |
| `loaders.py` | dense or sparse feature matrix → DataLoader |
| `training.py` | Lightning trainer, early stopping on `loss/valid`, best checkpoint, run summary |
| `evaluation.py` | majority baseline, probability per hypothesis |

| Notebook | Model card |
|---|---|
| `static/w2v_mean_logreg.ipynb` | `w2v_mean_logreg` |
| `static/w2v_mean_mlp.ipynb` | `w2v_mean_mlp` |
| `static/tfidf_bigger_mlp.ipynb` | `tfidf_concatenate_mlp` |
| `static/tfidf_concatenate_small_mlp.ipynb` | `tfidf_concatenate_small_mlp` |
| `static/glove_concatenate_bigger_mlp.ipynb` | `glove_concatenate_bigger_mlp` |
| `static/glove_concatenate_small_mlp.ipynb` | `glove_concatenate_small_mlp` |
| `static/final_presentable_notebook/glove_tfidf_mlp.ipynb` | `glove_tfidf_mlp` (final static model) |
| `rnn/rnn_basic.ipynb` | `glove_concatenate_rnn` |
| `rnn/glove_gru_unidirectional.ipynb` | `glove_concatenate_gru_unidirectional` |
| `rnn/glove_gru_bidirectional.ipynb` | `glove_concatenate_gru_bidirectional` |
| `rnn/glove_lstm_unidirectional.ipynb` | `glove_concatenate_lstm_unidirectional` |
| `rnn/glove_lstm_bidirectional.ipynb` | `glove_concatenate_lstm_bidirectional` |
| `rnn/final_presentable_notebook/glove_lstm_bidirectional_improved.ipynb` | `glove_lstm_bidirectional_improved` (final RNN model) |

### `models/project_1/`

One subfolder per trained model, named after its W&B run (`glove-tfidf-mlp` → `glove_tfidf_mlp`). Each subfolder contains:

| Item | Purpose |
|---|---|
| `README.md` | Model card: architecture, hyperparameters, validation and test results, link to the W&B run, notes |
| `weights/` | Best checkpoint by `loss/valid` (not tracked in git; the same file is the W&B model artifact `<run name>:best`) |

The notebook that produced the weights stays in `notebooks/project_1/`, the model card links to it.

## Hyperparameter sweeps

Both final notebooks have a W&B grid sweep next to them (`sweep.yaml`). Both point to one wrapper,
`notebooks/project_1/sweep/sweep_run.py`, that runs the **unchanged** notebook once per grid point with the sweep values
written into its `config`:

| Final model | Swept config keys | Runs |
|---|---|---|
| `glove_tfidf_mlp` | `learning_rate` 1e-3, 1e-4, 1e-5 · `hidden_dim` 128, 256, 512 · `dropout` 0.1, 0.3 | 18 |
| `glove_lstm_bidirectional_improved` | `learning_rate` 1e-3, 1e-4, 1e-5 · `rnn_hidden_dim` 128, 256 · `dropout` 0.1, 0.3 | 12 |

```bash
cd notebooks/project_1/static/final_presentable_notebook   # or rnn/final_presentable_notebook
wandb sweep sweep.yaml
wandb agent <sweep path printed by the line above>
```

The sweep maximizes `accuracy/valid` of each run's best checkpoint. Sweep runs never touch the test set and upload no
checkpoints; the winner gets its test number from one normal run of the notebook with the best values. Details in the
two final model cards.

## Data

- Source dataset:
[`allenai/art`](https://huggingface.co/datasets/allenai/art) (Bhagavatula et al., 2020), based on ROCStories.
- Course version (duplicates removed, own splits):
 [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)


Each example has two observations, two hypotheses and a label (1 or 2) for the more plausible hypothesis.
Splits: 16,801 train, 1,000 validation, 1,532 test.

## Setup

```bash
git clone https://github.com/cmedi32/nlp-project-01-abductive-reasoning.git
cd nlp-project-01-abductive-reasoning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "WANDB_API_KEY=<your key>" > .env
jupyter lab
```

The notebooks were run on an NVIDIA RTX 2070 SUPER.
GloVe (`glove-wiki-gigaword-300`) and word2vec (`word2vec-google-news-300`) are downloaded by gensim on first use.

### Environment variables

Credentials and host settings are read from `.env` in the project root. The file is ignored by git.

| Variable | Default | Purpose |
|---|---|---|
| `WANDB_API_KEY` | – | W&B API key, see <https://wandb.ai/authorize> (or the `/authorize` page of your host). The only value without a default. |
| `WANDB_BASE_URL` | `https://api.wandb.ai` | W&B host. Change it when logging to a self-hosted / CoreWeave instance. |
| `WANDB_ENTITY` | `cmedi32-hochschule-luzern` | W&B user or team the runs belong to. |
| `WANDB_PROJECT` | `nlp-project-01-abductive-reasoning` | W&B project all runs are logged to. |
| `HF_TOKEN` | – | Optional, only for higher Hugging Face rate limits or private datasets. |

Variables that are already set in the shell take precedence over `.env`, so a machine
where `wandb login` was already run works without one. The defaults live in `notebooks/project_1/pipeline/tracking.py`.

## Reproducing results

1. Open the notebook of the model you want in `notebooks/project_1/` (the table above maps notebooks to model cards).
2. Run all cells. The run is logged to the W&B project linked above, the best checkpoint is written to
   `models/project_1/<model>/weights/` and uploaded as the model artifact `<run name>:best`.
3. To use trained weights without training, download that artifact from W&B; the model card shows how to load it.

## Tools and attribution

- **Tools used:** `Claude Code - Libraries Referenced in Course Slides`
- **Prior work and external code/models:** `No AI/ML Specific Course Work at all`

## References

- Everything under class_solution belongs to hslu and the lecturer and is only present in this repo to guide coding agents NLP Class ref: https://elearning.hslu.ch/ilias/ilias.php?baseClass=ilrepositorygui&cmdNode=yb:n6&cmdClass=ilobjcoursegui&ref_id=7206312&item_ref_id=0

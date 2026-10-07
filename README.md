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
| GitHub repository | `<https://github.com/cmedi32/nlp-project-01-abductive-reasoning>` |
| Weights & Biases profile | `<https://forge.coreweave.com/profile/cmedi32>` |

## Repository structure

```
.
├── EXPECTATIONS.md            # Expected best test score per model
├── notebooks/project_x                 # is the working notebook for experiments, tuning and logging many W&B runs.
│   ├── ngram.ipynb                          
└── models/project_x                    # One folder per trained model
    └── lstm/                  # Example
        ├── README.md          # Short model card: setup, hyperparameters, results, W&B run
        └── weights/           # Stored weights (not tracked in git, see below)
```

### `notebooks/`

Training examples, one self-contained Jupyter notebook per method (for example `lstm.ipynb`, `ngram.ipynb`). Each notebook covers preprocessing, input/output format, model architecture, training and evaluation, and logs its runs to Weights & Biases.

### `models/`

One subfolder per trained model. Each subfolder contains:

| Item | Purpose |
|---|---|
| `README.md` | Quick model card: architecture, hyperparameters, validation and test results, link to the W&B run |
| `<model>.ipynb` | The notebook that produced the weights, for reproducibility |
| `weights/` | Storage for the trained weights (checkpoints) |

## Data

- Source dataset: 
[`allenai/art`](https://huggingface.co/datasets/allenai/art) (Bhagavatula et al., 2020), based on ROCStories.
- Course version (duplicates removed, own splits):
 [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)


Each example has two observations, two hypotheses and a label (1 or 2) for the more plausible hypothesis.

**Split usage:** train for fitting, validation for model selection and hyperparameter tuning, test only for the final comparison. Never tune on the test set.

## Projects

| Project | Methods | Submission |
|---|---|---|
| 1 | Word embeddings, word embeddings with RNN | Week 8 (Ilias) |
| 2 | Transformers: randomly initialized, pretrained, LLM | Week 14 (Ilias) |

## Deliverables per project

| Project | Repo Link | Experiment Tracking Link | Presentation Slides as PDF |
|---|---|---|---|
| 1 | SS | Tracking Link | SwissTransfer Link |

## Setup

```bash
git clone <https://github.com/<user>/<repo>>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in WANDB_API_KEY
jupyter lab
```

### Environment variables

Credentials and host settings are read from `.env` in the project root. The file is
ignored by git; `.env.example` documents the variables and their defaults.

| Variable | Default | Purpose |
|---|---|---|
| `WANDB_API_KEY` | – | W&B API key, see `<https://wandb.ai/authorize>` (or the `/authorize` page of your host). The only value without a default. |
| `WANDB_BASE_URL` | `https://api.wandb.ai` | W&B host. Change it when logging to a self-hosted / CoreWeave instance. |
| `WANDB_ENTITY` | `cmedi32` | W&B user or team the runs belong to. |
| `WANDB_PROJECT` | `nlp-project-01-abductive-reasoning` | W&B project all runs are logged to. |
| `HF_TOKEN` | – | Optional, only for higher Hugging Face rate limits or private datasets. |

Variables that are already set in the shell take precedence over `.env`, so a machine
where `wandb login` was already run works without one.

## Reproducing results

1. Open the notebook of the model you want in `notebooks/` (or in its `models/<model>/` folder).
2. Run all cells. Runs are logged to the W&B project linked above.
3. Pretrained weights, if provided, are loaded from `models/<model>/weights/`. See the model's `README.md` for the download location.

## Tools and attribution

- **Tools used:** `Claude Code - Libraries Referenced in Course Slides`
- **Prior work and external code/models:** `No AI/ML Specific Course Work at all`

## References

- Everything under class_solution belongs to hslu and the lecturer and is only present in this repo to guide coding agents NLP Class ref: https://elearning.hslu.ch/ilias/ilias.php?baseClass=ilrepositorygui&cmdNode=yb:n6&cmdClass=ilobjcoursegui&ref_id=7206312&item_ref_id=0

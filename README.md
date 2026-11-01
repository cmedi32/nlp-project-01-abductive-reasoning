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

## Repository structure

```
.
├── EXPECTATIONS.md                        # Expectations before and during Project 1
├── requirements.txt
├── class_solution/                        # Course solutions (HSLU), reference for the coding agents only
├── notebooks/project_1/
│   ├── pipeline/                          # Shared stages imported by every notebook
│   ├── sweep/                             # W&B sweep wrapper for the final notebooks
│   ├── static/                            # Static word embeddings, one notebook per experiment
│   │   └── final_presentable_notebook/    # Final static model + sweep.yaml
│   └── rnn/                               # Word embeddings with RNN, one notebook per experiment
│       └── final_presentable_notebook/    # Final RNN model + sweep.yaml
└── models/project_1/<model>/
    ├── README.md                          # Model card: setup, results, W&B run
    └── weights/                           # Checkpoints (not tracked in git)
```

## Data

- Source dataset:
[`allenai/art`](https://huggingface.co/datasets/allenai/art) (Bhagavatula et al., 2020), based on ROCStories.
- Course version (duplicates removed, own splits):
 [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)


Each example has two observations, two hypotheses and a label (1 or 2) for the more plausible hypothesis.

## Setup

```bash
git clone https://github.com/cmedi32/nlp-project-01-abductive-reasoning.git
cd nlp-project-01-abductive-reasoning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "WANDB_API_KEY=<your key>" > .env
jupyter lab
```

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
where `wandb login` was already run works without one.

## Reproducing results

1. Open the notebook of the model you want in `notebooks/project_1/`; its model card in `models/project_1/<model>/` names it.
2. Run all cells. The run is logged to the W&B project linked above, the best checkpoint goes to `models/project_1/<model>/weights/`.
3. Hyperparameter sweep of a final model: `cd` into its `final_presentable_notebook/`, then `wandb sweep sweep.yaml` and `wandb agent <sweep path printed by the first command>`.

## Tools and attribution

- **Tools used:** `Claude Code - Libraries Referenced in Course Slides`
- **Prior work and external code/models:** `No AI/ML Specific Course Work at all`

## References

- Everything under class_solution belongs to hslu and the lecturer and is only present in this repo to guide coding agents NLP Class ref: https://elearning.hslu.ch/ilias/ilias.php?baseClass=ilrepositorygui&cmdNode=yb:n6&cmdClass=ilobjcoursegui&ref_id=7206312&item_ref_id=0

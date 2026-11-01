"""Weights & Biases: credentials from .env, run init, checkpoint artifact."""
import os

import wandb
from dotenv import find_dotenv, load_dotenv

# used when neither the shell nor the .env file sets the variable
DEFAULTS = {
    "WANDB_BASE_URL": "https://api.wandb.ai",
    "WANDB_ENTITY": "cmedi32-hochschule-luzern",
    "WANDB_PROJECT": "nlp-project-01-abductive-reasoning",
}


def login():
    # AI generated snipped reused from now on, ensuring i do not leak my keys
    # Read the .env file in the project root. find_dotenv walks up from the notebook directory,
    # load_dotenv keeps variables that are already set in the shell.
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path)
    print(f"environment loaded from: {dotenv_path or '<no .env found, using the defaults below>'}")

    for name, default in DEFAULTS.items():
        os.environ[name] = os.getenv(name, default)

    wandb.login()


def init_run(name, config):
    return wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        name=name,
        config=config,
    )


def log_best_checkpoint(run, name, checkpoint):
    """Uploads the best checkpoint of a ModelCheckpoint callback as model artifact with alias "best"."""
    artifact = wandb.Artifact(name, type="model", metadata={"loss/valid": checkpoint.best_model_score.item()})
    artifact.add_file(checkpoint.best_model_path)
    run.log_artifact(artifact, aliases=["best"])

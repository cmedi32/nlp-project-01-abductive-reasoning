"""W&B sweep run for the final notebooks: executes an unchanged notebook with the values of one grid point.

Every final notebook has a sweep.yaml next to it that points here and names the notebook. Start from that folder,
the agent then starts this script once per grid point:

    cd notebooks/project_1/static/final_presentable_notebook   # or rnn/final_presentable_notebook
    wandb sweep sweep.yaml
    wandb agent <sweep path printed by the line above>

The code cells of the notebook run in order (the %pip cell is skipped), five pipeline functions are swapped for sweep
runs only:

- tracking.init_run: after wandb.init the sweep values are written into the notebook's config dict,
  the run gets a descriptive name, e.g. glove-tfidf-mlp-lr0.0001-h256-d0.1
- training.weights_dir: checkpoints go to models/project_1/<model>/weights/sweep/<sweep id>/<run name>,
  the weights of the final model stay untouched
- training.evaluate_best: validation only, the test set is evaluated once, by the final run with the best values
- evaluation.predict_hypotheses: skipped, it would print the test accuracy into every sweep run
- tracking.log_best_checkpoint: no upload, the "best" artifact keeps pointing to the final run
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import wandb
import yaml

if "WANDB_SWEEP_ID" not in os.environ or len(sys.argv) != 2:
    sys.exit("start this script with `wandb agent` from the folder of a final notebook, see the docstring")

NOTEBOOK = Path(sys.argv[1]).resolve()  # sweep.yaml passes the file name, the agent runs in the notebook's folder
HERE = NOTEBOOK.parent
PARAMETERS = list(yaml.safe_load((HERE / "sweep.yaml").read_text())["parameters"])  # config keys of the notebook
SHORT = {"learning_rate": "lr", "hidden_dim": "h", "rnn_hidden_dim": "rnn", "dropout": "d", "sentence_weighting": "w"}  # for the run name

# the agent passes the sweep values as a file path relative to its working directory, keep it valid after the chdir
if "WANDB_SWEEP_PARAM_PATH" in os.environ:
    os.environ["WANDB_SWEEP_PARAM_PATH"] = os.path.abspath(os.environ["WANDB_SWEEP_PARAM_PATH"])
os.chdir(HERE)  # the notebook resolves ../.. (pipeline) and wandb/ relative to its own folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/project_1, the same path the notebook adds

from pipeline import evaluation, tracking, training  # noqa: E402  swapped below, before the notebook imports from them

init_run = tracking.init_run
weights_dir = training.weights_dir


def sweep_init_run(name, config):
    missing = [key for key in PARAMETERS if key not in config]
    if missing:
        raise KeyError(f"sweep.yaml parameters {missing} are no keys of the notebook config")
    run = init_run(name, config)  # in a sweep, run.config holds the sweep values instead of the notebook defaults
    for key in PARAMETERS:
        config[key] = run.config[key]  # the notebook reads the plain dict, not run.config
    run.name = "-".join([name] + [f"{SHORT.get(key, key)}{config[key]}" for key in PARAMETERS])
    return run


def sweep_weights_dir(run_name):
    return weights_dir(run_name) / "sweep" / wandb.run.sweep_id / wandb.run.name


def validate_best(trainer, model, valid_loader, test_loader):
    # logs loss/valid + accuracy/valid of the best checkpoint as the last values of the run: accuracy/valid = sweep metric
    trainer.validate(model, dataloaders=valid_loader, ckpt_path="best")


def skip_predictions(trainer, model, test_loader, test):
    return pd.DataFrame()  # empty, so results.head(10) in the notebook still works


def keep_checkpoint_local(run, name, checkpoint):
    print(f"sweep run, checkpoint not uploaded: {checkpoint.best_model_path}")


tracking.init_run = sweep_init_run
tracking.log_best_checkpoint = keep_checkpoint_local
training.weights_dir = sweep_weights_dir
training.evaluate_best = validate_best
evaluation.predict_hypotheses = skip_predictions

namespace = {"__name__": "__main__"}
for index, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
    source = "".join(cell["source"])
    if cell["cell_type"] != "code" or source.lstrip().startswith(("%", "!")):  # markdown, %pip install
        continue
    exec(compile(source, f"{NOTEBOOK.name} cell {index}", "exec"), namespace)

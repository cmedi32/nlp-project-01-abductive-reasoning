"""Training: Lightning trainer with early stopping on loss/valid, keeps the best checkpoint."""
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .evaluation import majority_accuracy

# <repo>/models/project_1, this file is <repo>/notebooks/project_1/pipeline/training.py
MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "project_1"


def weights_dir(run_name):
    """W&B run name -> its weights folder, e.g. "glove-concatenate-rnn" -> models/project_1/glove_concatenate_rnn/weights."""
    return MODELS_DIR / run_name.replace("-", "_") / "weights"


def fit(model, train_loader, valid_loader, checkpoint_dir, max_epochs, patience):
    """Fits the model and returns the trainer, its callbacks: trainer.early_stopping_callback, trainer.checkpoint_callback."""
    early_stopping = EarlyStopping(monitor="loss/valid", mode="min", patience=patience, verbose=True) # @todo check against documentation and understand it
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename="best-{epoch}", monitor="loss/valid", mode="min", save_top_k=1) # @todo same here but this is largely library oriented

    trainer = L.Trainer( # train
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=WandbLogger(),  # reuses the run from wandb.init
        callbacks=[early_stopping, checkpoint], # ensure it stops when performance degrades
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return trainer


def evaluate_best(trainer, model, valid_loader, test_loader):
    """Validation and test metrics of the best checkpoint (lowest loss/valid), logged to W&B by the trainer."""
    trainer.validate(model, dataloaders=valid_loader, ckpt_path="best")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


def training_summary(trainer, y_test):
    checkpoint = trainer.checkpoint_callback
    return {
        "accuracy/majority": majority_accuracy(y_test),
        "best/loss_valid": checkpoint.best_model_score.item(),
        "best/checkpoint": Path(checkpoint.best_model_path).name,
        "early_stopping/stopped_epoch": trainer.early_stopping_callback.stopped_epoch,
    }

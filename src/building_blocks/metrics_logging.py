import torch
from typing import Literal
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import auc, f1_score, precision_score, recall_score, accuracy_score

from src.utils.cuda_utils import tensor_to_numpy

class ValidationPrintCallback(Callback):
    def __init__(self, logger=None):
        self.validation_losses = []
        self.best_loss = float('inf')
        self.logger = logger

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        print(f"Training started. Logs will be saved to {self.logger.log_dir if self.logger else 'nowhere'}")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        # Access the logged validation loss
        val_loss = trainer.callback_metrics.get('val_loss')
        print(f"Epoch {trainer.current_epoch}: Validation Loss = {val_loss:.4f}")

        self.validation_losses.append(val_loss)
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            print("New best loss!")
        else:
            print("No improvement in loss")

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        print(f"Training finished. Validation losses: {self.validation_losses}, Best loss: {self.best_loss}")

def compute_classification_metrics(y: torch.tensor, y_hat: torch.tensor, phase: Literal["train", "val", "test"] = "train") -> dict:
    y_np = tensor_to_numpy(y)
    y_hat_np = tensor_to_numpy(y_hat)#.round() # check if rounding is necessary

    acc = accuracy_score(y_np, y_hat_np)
    f1 = f1_score(y_np, y_hat_np)
    # precision = precision_score(y_np, y_hat_np)
    # recall = recall_score(y_np, y_hat_np)
    # auc_score = auc(y_np, y_hat_np)

    metrics = {
        f"{phase}_accuracy": acc,
        # f"{phase}_f1": f1,
        # f"{phase}_precision": precision,
        # f"{phase}_recall": recall,
        # f"{phase}_auc": auc_score
    }

    return {k: round(v, 4) for k, v in metrics.items()}

class MetricsComputation:

    def __init__(self, y: torch.tensor, y_hat: torch.tensor):
        self.y = tensor_to_numpy(y)
        self.y_hat = tensor_to_numpy(y_hat)

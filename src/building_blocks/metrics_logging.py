import torch
from typing import Literal
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from src.data_management.data_set import NakoSingleFeatureDataset
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
        metrics = trainer.callback_metrics        
        val_loss = metrics.get('val_loss')

        if val_loss is not None:
            print(f"Epoch {trainer.current_epoch}: Validation Loss = {val_loss:.4f}")
            self.validation_losses.append(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print("New best loss!")
            # else:
            #     print("No improvement in loss")
        else:
            print("No validation loss logged")

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        print(f"Training finished. Validation losses: {self.validation_losses}, Best loss: {self.best_loss}")

class ClassificationMetrics:
    def __init__(self, phase: Literal["train", "val", "test"] = "train"):
        self.phase = phase

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> dict:
        """Compute classification metrics for given predictions and labels."""
        # Convert to numpy
        y = tensor_to_numpy(y)
        y_hat = tensor_to_numpy(y_hat)
        
        return self.compute_classification_metrics(y, y_hat)

    def compute_classification_metrics(self, y: torch.tensor, y_hat: torch.tensor) -> dict:
        """Compute classification metrics using stored predictions and labels."""
        y_hat_thresh = (y_hat > 0.5).astype(int)  # threshold at 0.5

        metrics = {
            f"{self.phase}_accuracy": accuracy_score(y, y_hat_thresh),
            f"{self.phase}_f1": f1_score(y, y_hat_thresh),
            f"{self.phase}_precision": precision_score(y, y_hat_thresh, zero_division=0),
            f"{self.phase}_recall": recall_score(y, y_hat_thresh),
            f"{self.phase}_auc": roc_auc_score(y, y_hat)  # use raw probabilities for AUC
        }

        return {k: round(v, 4) for k, v in metrics.items()}
    

def process_metrics_file(csv_path: Path, output_path: Path = None) -> pd.DataFrame:
    """Process metrics CSV file to combine matching epochs and round values.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save processed CSV
    
    Returns:
        Processed DataFrame
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Group by epoch and step, combining all metrics
    df = df.groupby(['epoch', 'step']).first().reset_index()
    df = df.drop(columns=['step'])
    
    # Round all numeric columns to 4 decimals
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df

class ExperimentTrackingCallback(Callback):
    def __init__(self, logger=None, train_set: NakoSingleFeatureDataset=None, val_set:NakoSingleFeatureDataset=None, test_set:NakoSingleFeatureDataset=None):
        self.logger = logger
        self.start_time = None
        self.epoch_times = []
        self.last_epoch_time = None
        self.validation_losses = []
        self.best_loss = float('inf')
        
        # Store dataset indices
        self.dataset_info = {
            "train_indices": train_set.subject_ids if train_set else None,
            "val_indices": val_set.subject_ids if val_set else None,
            "test_indices": test_set.subject_ids if test_set else None
        }

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.start_time = datetime.now()
        print(f"Training started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.last_epoch_time = datetime.now()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if self.last_epoch_time:
            epoch_duration = datetime.now() - self.last_epoch_time
            self.epoch_times.append(epoch_duration.total_seconds())

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')       

        if val_loss is not None:
            self.validation_losses.append(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
            # else:
            #     print("No improvement in loss")
        else:
            print(f"No validation loss logged at epoch {trainer.current_epoch}")

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        experiment_info = {
            "timing": {
                "start_time": self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration_seconds": duration.total_seconds(),
                "total_duration_minutes": duration.total_seconds() / 60,
                "average_epoch_time_seconds": sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
                "epochs_completed": len(self.epoch_times)
            },
            "dataset_splits": self.dataset_info,
            "metrics": {
                "validation_losses": self.validation_losses,
                "best_validation_loss": float(self.best_loss)
            }
        }

        if self.logger and self.logger.log_dir:
            log_file = Path(self.logger.log_dir) / "experiment_info.json"
            with open(log_file, 'w') as f:
                json.dump(experiment_info, f, indent=4)
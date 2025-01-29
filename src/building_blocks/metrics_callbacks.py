from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from src.data_management.data_set import NakoSingleFeatureDataset

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
            "modality": train_set.modalitiy.value if train_set else None,
            "feature_set": train_set.feature_set.value if train_set else None,
            "len_train": len(train_set) if train_set else None,
            "len_val": len(val_set) if val_set else None,
            "len_test": len(test_set) if test_set else None,
            "train_indices": train_set.subject_ids if train_set else None,
            "val_indices": val_set.subject_ids if val_set else None,
            "test_indices": test_set.subject_ids if test_set else None,
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
                "total_duration_seconds": round(duration.total_seconds(), 2),
                "total_duration_minutes": round(duration.total_seconds() / 60, 2),
                "average_epoch_time_seconds": round(sum(self.epoch_times) / len(self.epoch_times), 2) if self.epoch_times else 0,
                "epochs_completed": len(self.epoch_times)
            },
            "metrics": {
                "validation_losses": [tensor.item() for tensor in self.validation_losses],
                "best_loss": self.best_loss.item() if self.best_loss != float('inf') else None
            },
            "dataset_info": self.dataset_info,
        }

        if self.logger and self.logger.log_dir:
            log_file = Path(self.logger.log_dir) / "experiment_info.json"
            with open(log_file, 'w') as f:
                json.dump(experiment_info, f, indent=4)
                
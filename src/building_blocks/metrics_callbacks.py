from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
from datetime import datetime
import json

from src.data_management.data_set import NakoSingleFeatureDataset

class ExperimentStartCallback(Callback):
    """Logs the start of the training process with the provided parameters in console"""
    def __init__(self, logger=None, logging_dict={}):
        self.logger = logger
        self.logging_dict = logging_dict

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        print("")
        print("Training started with the following parameters:")
        for key, value in self.logging_dict.items():
            print(f"{key}: {value}")
        print("")


class ValidationPrintCallback(Callback):
    """Prints validation loss"""
    def __init__(self, logger=None):
        self.validation_losses = []
        self.best_loss = float("inf")
        self.logger = logger

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        print(
            f"Training started. Logs will be saved to {self.logger.log_dir if self.logger else 'nowhere'}"
        )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")

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
        print(
            f"Training finished. Validation losses: {self.validation_losses}, Best loss: {self.best_loss}"
        )


class ExperimentSetupLogger(Callback):
    """Logs initial experiment configuration and setup parameters to JSON"""
    def __init__(
        self,
        logger=None,
        train_set: NakoSingleFeatureDataset = None,
        val_set: NakoSingleFeatureDataset = None,
        test_set: NakoSingleFeatureDataset = None,
        print_collection_dict={},
    ):
        self.logger = logger
        self.print_collection_dict = print_collection_dict
        
        # Store dataset indices
        self.dataset_info = {
            "modality": train_set.feature_map.modality_label if train_set else None,
            "feature_map": train_set.feature_map.label if train_set else None,
            "len_train": len(train_set) if train_set else None,
            "len_val": len(val_set) if val_set else None,
            "len_test": len(test_set) if test_set else None,
            "train_indices": train_set.subject_ids if train_set else None,
            "val_indices": val_set.subject_ids if val_set else None,
            "test_indices": test_set.subject_ids if test_set else None,
        }

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Log experiment setup parameters before training starts"""
        # initial_lr = trainer.optimizers[0].param_groups[0]["lr"]
        
        setup_info = {
            "training_params": {
                **self.print_collection_dict,
            },
            "model_info": {
                "lightning_wrapper": pl_module.__class__.__name__,
                "model_architecture": pl_module.model.__class__.__name__,
                "model_params": sum(p.numel() for p in pl_module.model.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
            },
            "dataset_info": self.dataset_info
        }

        if self.logger and self.logger.log_dir:
            log_file = Path(self.logger.log_dir) / "experiment_setup.json"
            with open(log_file, "w") as f:
                json.dump(setup_info, f, indent=4)


class TrainingProgressTracker(Callback):
    """Tracks and logs training progress metrics. Notably run parameters, epoch times, validation losses and best loss"""
    def __init__(self, logger=None):
        self.logger = logger
        self.start_time = None
        self.epoch_times = []
        self.last_epoch_time = None
        self.validation_losses = []
        self.best_loss = float("inf")

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
        # Log validation loss
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")

        if val_loss is not None:
            self.validation_losses.append(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        end_time = datetime.now()
        duration = end_time - self.start_time

        training_info = {
            "timing": {
                "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": round(duration.total_seconds(), 2),
                "total_duration_minutes": round(duration.total_seconds() / 60, 2),
                "total_duration_hours": round(duration.total_seconds() / 3600, 2),
                "average_epoch_time_seconds": round(sum(self.epoch_times) / len(self.epoch_times), 2) if self.epoch_times else 0,
                "epoch_times": self.epoch_times
            },
            "metrics": {
                "validation_losses": [tensor.item() for tensor in self.validation_losses],
                "best_loss": self.best_loss.item() if self.best_loss != float("inf") else None,
                "final_metrics": {k: v.item() if hasattr(v, 'item') else v 
                                for k, v in trainer.callback_metrics.items()}
            }
        }

        if self.logger and self.logger.log_dir:
            log_file = Path(self.logger.log_dir) / "training_progress.json"
            with open(log_file, "w") as f:
                json.dump(training_info, f, indent=4)
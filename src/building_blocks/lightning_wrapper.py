# https://lightning.ai/docs/pytorch/stable/starter/introduction.html
import lightning as L
from typing import Literal
from torch import optim, nn
import torch

from src.building_blocks.custom_model import BaseConvBranch2d, BaseConvBranch3d
from src.building_blocks.resnet18 import ResNet18Base2d
from src.building_blocks.torchmetrics import MetricsFactory


class LightningWrapperCnn(L.LightningModule):
    def __init__(
        self, model: ResNet18Base2d | BaseConvBranch2d | BaseConvBranch3d, task: Literal["classification", "regression"], learning_rate: float = 1e-3
    ):
        super().__init__()
        self.task = task
        self.save_hyperparameters(ignore=['model'])

        # Training building blocks
        self.model = model
        self.loss_func = nn.BCELoss() if task == "classification" else nn.MSELoss()
        self.learning_rate = learning_rate

        # Initialize training and validation metrics
        self.train_metrics = MetricsFactory.create_metrics(task, "train")
        self.val_metrics = MetricsFactory.create_metrics(task, "val")


        # Training set up
        self.logging_params = {
            "sync_dist": True,
            "on_epoch": True,
            "on_step": False,
        }

        # TODO: Remove later
        self.epoch_samples = {"train": 0, "val": 0}
        self.total_samples = {"train": 0, "val": 0}

    # TODO: Remove later
    def on_train_epoch_start(self):
        self.epoch_samples["train"] = 0

    # TODO: Remove later
    def on_validation_epoch_start(self):
        self.epoch_samples["val"] = 0

    def setup(self, stage: str) -> None:
        """Called on every device."""
        # Move metrics to the correct device
        for metric in self.train_metrics.metrics.values():
            metric.to(self.device)
        for metric in self.val_metrics.metrics.values():
            metric.to(self.device)

    def forward(self, x):
        return self.model(x)
    
    # TODO: Remove later
    def on_train_end(self):
        super().on_train_end()
        print(f"\nGPU {self.device} Statistics:")
        print(f"Training samples per epoch: {self.epoch_samples['train']}")
        print(f"Validation samples per epoch: {self.epoch_samples['val']}")
        
        if self.trainer is not None:
            total_train = self.trainer.world_size * self.epoch_samples['train']
            total_val = self.trainer.world_size * self.epoch_samples['val']
            print(f"\nAcross all {self.trainer.world_size} GPUs:")
            print(f"Total training samples per epoch: {total_train}")
            print(f"Total validation samples per epoch: {total_val}")
            print(f"Expected train set size: 7680")
            print(f"Expected val set size: 1280")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        # Validate labels for binary classification
        if self.task == "classification" and not ((y == 0) | (y == 1)).all():
            raise ValueError(f"Labels must be 0 or 1, got values: {y.unique()}")

        loss = self.loss_func(y_hat, y)

        # Log all metrics
        metrics_dict = {
            "train_loss": loss,
            **self.train_metrics(y, y_hat)
        }
        self.log_dict(metrics_dict, **self.logging_params)

        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, **self.logging_params)

        # TODO: Remove later
        self.epoch_samples["train"] += len(y)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)

        # Log all metrics
        metrics_dict = {
            "val_loss": loss,
            **self.val_metrics(y, y_hat)
        }
        self.log_dict(metrics_dict, **self.logging_params)

        # TODO: Remove later
        self.epoch_samples["val"] += len(y)
        
        return loss

    def test_step(self, batch):
        # test_step defines the test loop.
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y, y_hat)

        # self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,  # every epoch
            },
        }

    def _debug_shapes(self, x, y):
        """Debugging function to check the shapes of the input and labels."""
        print("input_shape", x.shape)
        print("label_shape", y.shape)

    def _debug_prediction_shapes(self, y, y_hat):
        """Debugging function to check the shapes of the predictions."""
        print("label_shape", y.shape)
        print("prediction_shape", y_hat.shape)

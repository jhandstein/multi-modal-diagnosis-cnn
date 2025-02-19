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

    def setup(self, stage: str) -> None:
        """Called on every device."""
        # Move metrics to the correct device
        for metric in self.train_metrics.metrics.values():
            metric.to(self.device)
        for metric in self.val_metrics.metrics.values():
            metric.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        if self.task == "classification" and not ((y == 0) | (y == 1)).all():
            raise ValueError(f"Labels must be 0 or 1, got values: {y.unique()}")

        loss = self.loss_func(y_hat, y)

        # Debug info for metrics
        if batch_idx == 0:  # First batch of epoch
            self._prediction_count = 0
            self._total_predictions = []
            self._total_labels = []
        
        # Accumulate predictions and labels
        self._prediction_count += len(y)
        self._total_predictions.append(y_hat.detach().cpu())
        self._total_labels.append(y.detach().cpu())        
        
        # Update metrics on each step
        metrics_dict = {
            "train_loss": loss,
            **self.train_metrics(y, y_hat)
        }

        # Print debug info every N batches
        # TODO: Make this actually print
        if batch_idx % 2 == 0:
            print(f"\nBatch {batch_idx}:")
            print(f"Processed predictions so far: {self._prediction_count}")
            print(f"Current batch metrics: {metrics_dict}")
            
            # Verify metrics manually for classification
            if self.task == "classification":
                all_preds = torch.cat(self._total_predictions)
                all_labels = torch.cat(self._total_labels)
                print(f"Total predictions shape: {all_preds.shape}")
                print(f"Prediction distribution: {(all_preds > 0.5).float().mean():.3f}")
                print(f"Label distribution: {all_labels.float().mean():.3f}")
    
        self.log_dict(metrics_dict, **self.logging_params)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)

        # Update metrics on each step
        metrics_dict = {
            "val_loss": loss,
            **self.val_metrics(y, y_hat)
        }
        self.log_dict(metrics_dict, **self.logging_params)

        # TODO: Add back learning rate
        
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

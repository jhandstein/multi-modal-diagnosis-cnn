# https://lightning.ai/docs/pytorch/stable/starter/introduction.html
import lightning as L
from torch import optim, nn
import torch

from src.building_blocks.metrics_logging import ClassificationMetrics
from src.building_blocks.layers import ConvBranch2d
 
class LightningWrapper2dCnnClassification(L.LightningModule):
    def __init__(self, input_shape: tuple):
        super().__init__()
        self.model = ConvBranch2d(input_shape)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.logging_params = {
            "sync_dist": True,
            "on_epoch": True,
            "on_step": False,
        }

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch):
        # Unpacking and forward pass
        x, y = batch
        y_hat = self.model(x)
                
        # Validate labels for binary classification
        if not ((y == 0) | (y == 1)).all():
            raise ValueError(f"Labels must be 0 or 1, got values: {y.unique()}")
                
        # Compute loss
        loss = nn.functional.binary_cross_entropy(y_hat, y)

        # Store predictions for later metric computation
        self.train_step_outputs.append({
                "y": y,
                "y_hat": y_hat,
            })
        
        return loss
    
    def on_train_epoch_end(self):
        # Gather predictions from all GPUs
        y = torch.cat([batch_out["y"] for batch_out in self.train_step_outputs])
        y_hat = torch.cat([batch_out["y_hat"] for batch_out in self.train_step_outputs])
        
        # Gather across GPUs
        y = self.all_gather(y)
        y_hat = self.all_gather(y_hat)
        
        # Reshape if needed (all_gather adds new dimension)
        y, y_hat = self._flatten_predictions(y, y_hat)
        
        # Compute metrics for whole epoch
        epoch_loss = nn.functional.binary_cross_entropy(y_hat, y)
        # metrics = compute_classification_metrics(y, y_hat, phase="train")
        metrics = ClassificationMetrics(phase="train")
        metrics_dict = {
            "train_loss": epoch_loss, 
            **metrics(y, y_hat), 
            }
        self.log_dict(metrics_dict, **self.logging_params)
        
        # Clear saved outputs
        self.train_step_outputs.clear()
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy(y_hat, y)

        # Store predictions for later metric computation
        self.validation_step_outputs.append({
                "y": y,
                "y_hat": y_hat,
            })

        return loss
    
    def on_validation_epoch_end(self):
        # Gather predictions from all GPUs
        y = torch.cat([x["y"] for x in self.validation_step_outputs])
        y_hat = torch.cat([x["y_hat"] for x in self.validation_step_outputs])
        
        # Gather across GPUs
        y = self.all_gather(y)
        y_hat = self.all_gather(y_hat)
        
        # Reshape if needed (all_gather adds new dimension)
        y, y_hat = self._flatten_predictions(y, y_hat)

        # Compute metrics for whole epoch
        # metrics = compute_classification_metrics(y, y_hat, phase="val")
        metrics = ClassificationMetrics(phase="val")
        metrics_dict = {
            "val_loss": nn.functional.binary_cross_entropy(y_hat, y),
            **metrics(y, y_hat),
        }
        self.log_dict(metrics_dict, **self.logging_params)
                
        # Clear saved outputs
        self.validation_step_outputs.clear()
        pass

    def test_step(self, batch):
        # test_step defines the test loop.
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy(y, y_hat)
        
        # self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": 
                {
                    "scheduler": scheduler,
                    "interval": "epoch", 
                    "frequency": 1, # every epoch
                }}
    
    def _debug_shapes(self, x, y):
        """Debugging function to check the shapes of the input and labels."""
        print("input_shape", x.shape)
        print("label_shape", y.shape)

    def _debug_prediction_shapes(self, y, y_hat):
        """Debugging function to check the shapes of the predictions."""
        print("label_shape", y.shape)
        print("prediction_shape", y_hat.shape)

    def _flatten_predictions(self, y, y_hat):
        """Flatten the tensors if they have more than one dimension."""
        if len(y.shape) > 1:
            y = y.reshape(-1)
            y_hat = y_hat.reshape(-1)

        # self._debug_prediction_shapes(y, y_hat)
        return y, y_hat
        
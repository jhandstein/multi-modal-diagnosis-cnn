# https://lightning.ai/docs/pytorch/stable/starter/introduction.html
import lightning as L
from torch import optim, nn
import torch

from src.building_blocks.metrics_logging import compute_classification_metrics
from src.building_blocks.layers import ConvBranch2d
 
class BinaryClassificationCnn2d(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvBranch2d()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        # print(self.device)

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch):
        x, y = batch
        
        # Debug shapes and types
        # self._debug_shapes(x, y)
        
        # Forward pass
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()
        
        # Convert and validate labels
        y = y.float()
        if not ((y == 0) | (y == 1)).all():
            raise ValueError(f"Labels must be 0 or 1, got values: {y.unique()}")
        
        # TODO: check for y_hat values??
        
        # Logging the metrics
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        
        self.log("train_loss_for_testing", loss, on_epoch=True, on_step=False)

        # Store predictions for later use
        self.train_step_outputs.append(
            {
                "y": y,
                "y_hat": y_hat,
            }
        )
        
        return loss
    
    def on_train_epoch_end(self):
        # Gather predictions from all GPUs
        y = torch.cat([batch_out["y"] for batch_out in self.train_step_outputs])
        y_hat = torch.cat([batch_out["y_hat"] for batch_out in self.train_step_outputs])
        
        # Gather across GPUs
        y = self.all_gather(y)
        y_hat = self.all_gather(y_hat)
        
        # Reshape if needed (all_gather adds new dimension)
        if len(y.shape) > 1:
            y = y.reshape(-1)
            y_hat = y_hat.reshape(-1)
        print("Prediction shapes", y.shape, y_hat.shape)
        
        # Compute metrics on CPU
        metrics = compute_classification_metrics(y, y_hat, phase="train")
        recompute_loss = {"recomputed_train_loss": nn.functional.binary_cross_entropy(y_hat, y)}
        inferred_loss = {"train_loss": self.trainer.callback_metrics["train_loss_for_testing"]}
        metrics = {**metrics, **recompute_loss, **inferred_loss}
        self.log_dict(metrics, sync_dist=True)
        
        # Clear saved outputs
        self.train_step_outputs.clear()
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()
        loss = nn.functional.binary_cross_entropy(y_hat, y)

        self.log("val_loss", loss)

        return loss
    
    def on_validation_epoch_end(self):
        # Gather predictions from all GPUs
        # y = torch.cat([x["y"] for x in self.validation_step_outputs])
        # y_hat = torch.cat([x["y_hat"] for x in self.validation_step_outputs])
        
        # # Gather across GPUs
        # y = self.all_gather(y)
        # y_hat = self.all_gather(y_hat)
        
        # # Reshape if needed (all_gather adds new dimension)
        # if len(y.shape) > 1:
        #     y = y.reshape(-1)
        #     y_hat = y_hat.reshape(-1)
        
        # # Compute metrics on CPU
        # metrics = compute_classification_metrics(y, y_hat, phase="val")
        # self.log_dict(metrics)
        
        # # Clear saved outputs
        # self.validation_step_outputs.clear()
        pass

    def test_step(self, batch):
        # test_step defines the test loop.
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()
        loss = nn.functional.binary_cross_entropy(y, y_hat)
        # Logging to TensorBoard (if installed) by default
        # self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def _debug_shapes(self, x, y):
        print("input_shape", x.shape)
        print("label_shape", y.shape)
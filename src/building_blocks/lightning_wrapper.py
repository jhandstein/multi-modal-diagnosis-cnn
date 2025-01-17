# https://lightning.ai/docs/pytorch/stable/starter/introduction.html
import lightning as L
from torch import optim, nn

from src.building_blocks.layers import ConvBranch2d
 
class BinaryClassificationCnn2d(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvBranch2d()
        # print(self.device)

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch):
        x, y = batch
        # Debug shapes and types
        # print("input_shape", x.shape)
        # print("label_shape", y.shape)
        
        # Forward pass
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()
        
        # Convert and validate labels
        y = y.float()
        if not ((y == 0) | (y == 1)).all():
            raise ValueError(f"Labels must be 0 or 1, got values: {y.unique()}")
        
        # TODO: check for y_hat values??
        
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        # self.log("val_loss", loss)
        return loss

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
    
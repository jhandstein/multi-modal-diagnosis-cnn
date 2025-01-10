# https://lightning.ai/docs/pytorch/stable/starter/introduction.html
import lightning as L
from torch import optim, nn, utils, Tensor

from src.building_blocks.layers import ConvBranch2d
 
# define the LightningModule
class MultiModalCnn2d(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvBranch2d()
        
    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y, y_hat)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

network = MultiModalCnn2d()

# setup data
# dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, batch_size=16)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=network, train_dataloaders=train_loader)
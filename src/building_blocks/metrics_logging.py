from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

class ValidationPrintCallback(Callback):
    def __init__(self):
        self.validation_losses = []
        self.best_loss = float('inf')

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
        
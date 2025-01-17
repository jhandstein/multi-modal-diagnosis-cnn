import torch
from torch.utils.data import Dataset
import lightning as L
from lightning.pytorch import seed_everything

from src.data_management.data_set import prepare_standard_data_sets
from src.building_blocks.metrics_logging import ValidationPrintCallback
from src.data_management.data_loader import prepare_standard_data_loaders
from src.building_blocks.lightning_wrapper import BinaryClassificationCnn2d
from src.utils.check_cuda import check_cuda



def train_model():#
    """Handles all the logic for training the model."""

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Declare lightning wrapper model
    lightning_model = BinaryClassificationCnn2d()

    # Set parameters for training
    num_gpus = torch.cuda.device_count()
    batch_size = 8
    epochs = 5

    train_set, val_set, test_set = prepare_standard_data_sets()
    train_loader = prepare_standard_data_loaders(train_set, batch_size=batch_size, num_gpus=num_gpus)
    val_loader = prepare_standard_data_loaders(val_set, batch_size=2, num_gpus=num_gpus)


    callback = ValidationPrintCallback()

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=num_gpus,
        max_epochs=epochs,
        strategy="ddp",
        sync_batchnorm=True,
        deterministic=True, # works together with seed_everything to ensure reproducibility across runs
        benchmark=True, # best algorithm for your hardware, only works with homogenous input sizes
        default_root_dir="models",
        log_every_n_steps=1,
        callbacks=[callback],
    )


    trainer.fit(
        model=lightning_model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
   
    # TODO: Test model and log train loss
    


if __name__ == "__main__":
    
    # check_cuda()
    train_model()

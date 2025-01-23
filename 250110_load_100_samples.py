from pathlib import Path
import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers

from src.data_management.data_set import prepare_standard_data_sets
from src.building_blocks.metrics_logging import ValidationPrintCallback, process_metrics_file
from src.data_management.data_loader import prepare_standard_data_loaders
from src.building_blocks.lightning_wrapper import BinaryClassificationCnn2d
from src.utils.cuda_utils import check_cuda



def train_model():
    """Handles all the logic for training the model."""

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Declare lightning wrapper model
    lightning_model = BinaryClassificationCnn2d()

    # Set parameters for training
    num_gpus = torch.cuda.device_count()
    batch_size = 8
    epochs = 100

    train_set, val_set, test_set = prepare_standard_data_sets(n_samples=256)
    train_loader = prepare_standard_data_loaders(train_set, batch_size=batch_size, num_gpus=num_gpus)
    val_loader = prepare_standard_data_loaders(val_set, batch_size=2, num_gpus=num_gpus)

    # Logging and callbacks
    log_dir = Path("models")
    logger = pl_loggers.CSVLogger(log_dir, name="2D_CNN")
    callback = ValidationPrintCallback(logger=logger)

    gpu_params = {
        "accelerator": "gpu" if torch.cuda.is_available() else None,
        "devices": num_gpus,
        "strategy": "ddp",
        "sync_batchnorm": True,
        "benchmark": True, # best algorithm for your hardware, only works with homogenous input sizes
    }

    training_params = {
        "deterministic": True, # works together with seed_everything to ensure reproducibility across runs
        "max_epochs": epochs,
    }

    logging_params = {
        "log_every_n_steps": 1,
        "callbacks": [callback],
        "logger": logger,
    }

    trainer = L.Trainer(
        **gpu_params,
        **training_params,
        **logging_params
    )

    trainer.fit(
        model=lightning_model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    # Process metrics
    metrics_file = Path(logger.log_dir, "metrics.csv")
    processed_file = Path(logger.log_dir, "metrics_processed.csv")
    process_metrics_file(metrics_file, processed_file)


    # TODO: Test model
    


if __name__ == "__main__":
    
    # check_cuda()
    train_model()

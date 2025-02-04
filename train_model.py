import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import seed_everything

from src.building_blocks.lightning_wrapper import LightningWrapper2dCnn
from src.building_blocks.metrics_callbacks import (
    ExperimentTrackingCallback,
    ValidationPrintCallback,
)
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_loader import prepare_standard_data_loaders
from src.data_management.data_set import sample_standard_data_sets
from src.data_management.data_set_factory import DataSetConfig, DataSetFactory
from src.plots.save_training_plot import plot_mae_mse, plot_training_metrics
from src.utils.config import (
    AGE_SEX_BALANCED_1K_PATH,
    AGE_SEX_BALANCED_10K_PATH,
    FeatureMapType,
)
from src.utils.cuda_utils import check_cuda
from src.utils.process_metrics import format_metrics_file
from src.utils.file_path_helper import construct_model_name


def train_model():
    """Handles all the logic for training the model."""

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Set parameters for training
    num_gpus = torch.cuda.device_count()
    batch_size = 64  # should be maximum val_set size / num_gpus?
    epochs = 3
    task = "classification"
    feature_map = FeatureMapType.GM
    target = "sex" if task == "classification" else "age"
    experiment_notes = {"notes": "First time ResNet18."}

    # Experiment setup
    if epochs > 10:
        log_dir = Path("models")
        data_split_path = AGE_SEX_BALANCED_10K_PATH
    else:
        log_dir = Path("models_test")
        data_split_path = AGE_SEX_BALANCED_1K_PATH

    # Prepare data sets and loaders
    ds_details = {
        "feature_map": feature_map,
        "target": target,
        "middle_slice": True,
    }

    # Create data sets and loaders
    data_split = DataSplitFile(data_split_path).load_data_splits_from_file()
    ds_config = DataSetConfig(**ds_details)
    train_set, val_set, test_set = DataSetFactory(
        data_split["train"], data_split["val"], data_split["test"], ds_config
    ).create_data_sets()

    train_loader = prepare_standard_data_loaders(
        train_set, batch_size=batch_size, num_gpus=num_gpus
    )
    val_loader = prepare_standard_data_loaders(val_set, batch_size=2, num_gpus=num_gpus)

    # Declare lightning wrapper model
    lightning_wrapper = LightningWrapper2dCnn(
        train_set.data_shape, task=task
    )

    # Model name for logging
    model_name = construct_model_name(lightning_wrapper.model, train_set, task=task, dim="2D")

    # Logging and callbacks
    logger = pl_loggers.CSVLogger(log_dir, name=model_name)
    print_callback = ValidationPrintCallback(logger=logger)
    json_callback = ExperimentTrackingCallback(
        logger=logger,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        batch_size=batch_size,
        notes=experiment_notes,
    )

    gpu_params = {
        "accelerator": "gpu" if torch.cuda.is_available() else None,
        "devices": num_gpus,
        "strategy": "ddp",
        "sync_batchnorm": True,
        # benchmark interferes with reproducibility from seed_everything and deterministic
        "benchmark": False,  # best algorithm for your hardware, only works with homogenous input sizes
    }

    training_params = {
        "deterministic": True,  # works together with seed_everything to ensure reproducibility across runs
        "max_epochs": epochs,
    }

    logging_params = {
        "log_every_n_steps": 1,
        "callbacks": [print_callback, json_callback],
        "logger": logger,
    }

    trainer = L.Trainer(**gpu_params, **training_params, **logging_params)

    trainer.fit(
        model=lightning_wrapper,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Process metrics
    metrics_file = Path(logger.log_dir, "metrics.csv")
    formatted_file = Path(logger.log_dir, "metrics_formatted.csv")
    format_metrics_file(metrics_file, formatted_file)

    # Plot training metrics (after some time to allow for file writing)
    time.sleep(2)
    plot_training_metrics(formatted_file, task=task)
    if task == "regression":
        plot_mae_mse(formatted_file)

    # TODO: Test model
    def test_model():
        checkpoint_path = Path(
            "models/CNN_2D_anat_WM/version_0/checkpoints/epoch=99-step=22400.ckpt"
        )
        lightning_model = LightningWrapper2dCnn.load_from_checkpoint(
            checkpoint_path
        )

        # Set model into evaluation mode
        lightning_model.eval()


def print_ds_indices():
    train_set, val_set, test_set = sample_standard_data_sets(n_samples=1024)
    print("Data set indices:")
    print(train_set.subject_ids)
    print(val_set.subject_ids)
    print(test_set.subject_ids)


if __name__ == "__main__":
    # check_cuda()

    train_model()

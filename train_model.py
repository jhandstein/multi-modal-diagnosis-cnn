import time
from pathlib import Path

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import seed_everything
import torch

from src.building_blocks.model_factory import ModelFactory
from src.building_blocks.lightning_trainer_config import LightningTrainerConfig
from src.building_blocks.lightning_wrapper import LightningWrapper2dCnn
from src.building_blocks.metrics_callbacks import (
    ExperimentSetupLogger,
    TrainingProgressTracker,
    ValidationPrintCallback,
)
from src.data_management.data_set import DataSetConfig
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_loader import prepare_standard_data_loaders
from src.data_management.data_set_factory import DataSetFactory
from src.plots.plot_metrics import plot_all_metrics
from src.utils.config import (
    AGE_SEX_BALANCED_1K_PATH,
    AGE_SEX_BALANCED_10K_PATH,
    FeatureMapType,
)
from src.utils.cuda_utils import check_cuda, print_model_size
from src.utils.process_metrics import format_metrics_file
from src.utils.file_path_helper import construct_model_name


def train_model():
    """Handles all the logic for training the model."""

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Set parameters for training
    task = "classification"
    dim = "2D"
    feature_map = FeatureMapType.GM
    target = "sex" if task == "classification" else "age"
    model_type = "ConvBranch" # "ResNet18"

    num_gpus = 1 #torch.cuda.device_count()
    batch_size = 64 if dim == "2D" else 1 # should be maximum val_set size / num_gpus?
    print(f"Batch size: {batch_size}")
    epochs = 3
    learning_rate = 1e-3
    experiment_notes = {"notes": f"Automatic learning rate not actually used due to model instability. New lr schedualer (plateau) implemented."}

    # Experiment setup
    if epochs > 20:
        log_dir = Path("models")
        data_split_path = AGE_SEX_BALANCED_10K_PATH
        print("Training on 10k data set.")
    else:
        log_dir = Path("models_test")
        data_split_path = AGE_SEX_BALANCED_1K_PATH
        print("Training on 1k data set.")

    # Prepare data sets and loaders
    ds_config = DataSetConfig(
        feature_map=feature_map,
        target=target,
        middle_slice=True if dim == "2D" else False,
    )
    data_split = DataSplitFile(data_split_path).load_data_splits_from_file()
    
    train_set, val_set, test_set = DataSetFactory(
        data_split["train"], data_split["val"], data_split["test"], ds_config
    ).create_data_sets()

    train_loader = prepare_standard_data_loaders(
        train_set, batch_size=batch_size, num_gpus=num_gpus
    )
    val_loader = prepare_standard_data_loaders(val_set, batch_size=1, num_gpus=num_gpus)

    # Setup model
    if model_type == "ConvBranch":
        model = ModelFactory(task=task, dim=dim).create_conv_branch(input_shape=train_set.data_shape)
    elif model_type == "ResNet18":
        model = ModelFactory(task=task, dim=dim).create_resnet18(in_channels=1)
    else:
        raise ValueError("Model type not supported. Check the model_type argument.")

    print_model_size(model)

    # Setup lightning wrapper
    lightning_wrapper = LightningWrapper2dCnn(
        model=model, task=task, learning_rate=learning_rate
    )

    # Model name for logging
    model_name = construct_model_name(lightning_wrapper.model, train_set, task=task, dim=dim)

    # Logging and callbacks
    logger = pl_loggers.CSVLogger(log_dir, name=model_name)
    print_callback = ValidationPrintCallback(logger=logger)

    setup_logger = ExperimentSetupLogger(
        logger=logger,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        batch_size=batch_size,
        num_gpus=num_gpus,
        notes=experiment_notes,
    )
    progress_logger = TrainingProgressTracker(
        logger=logger,
    )

    # Trainer setup
    trainer_config = LightningTrainerConfig(
        devices=num_gpus,
        max_epochs=epochs,
        # TODO: Check if these are needed
        precision="32-true" if dim == "2D" else "16-mixed",
        # accumulate_grad_batches=2,  # Accumulate gradients over 2 batches
        # gradient_clip_val=0.5,  # Add gradient clipping
    )
    trainer = L.Trainer(**trainer_config.dict(), callbacks=[print_callback, setup_logger, progress_logger], logger=logger)

    trainer.fit(
        model=lightning_wrapper,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Process metrics
    metrics_file = Path(logger.log_dir, "metrics.csv")
    format_metrics_file(metrics_file)

    # Plot training metrics (after some time to allow for file writing)
    time.sleep(2)
    formatted_file = Path(metrics_file.parent, f"{metrics_file.stem}_formatted.csv")
    plot_all_metrics(formatted_file, task=task, splits=["train", "val"])

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


if __name__ == "__main__":
    # check_cuda()

    train_model()

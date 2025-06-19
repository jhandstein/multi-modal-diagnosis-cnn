import argparse
import time
from pathlib import Path
from typing import Literal

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
import torch


from src.building_blocks.model_factory import ModelFactory
from src.building_blocks.lightning_configs import LightningTrainerConfig
from src.building_blocks.lightning_wrapper import MultiModalityWrapper, OneCycleWrapper
from src.building_blocks.metrics_callbacks import (
    ExperimentSetupLogger,
    ExperimentStartCallback,
    TestingProgressTracker,
    TrainingProgressTracker,
    ValidationPrintCallback,
)
from src.data_management.data_set import BaseDataSetConfig, BaseNakoDataset
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_loader import (
    infer_batch_size,
    infer_gpu_count,
    prepare_standard_data_loaders,
)
from src.data_management.data_set_factory import DataSetFactory
from src.plots.plot_metrics import plot_all_metrics
from src.utils.config import (
    AGE_SEX_BALANCED_10K_PATH,
    HIGH_QUALITY_IDS,
    LOW_QUALITY_IDS,
    MEDIUM_QUALITY_IDS,
    FeatureMapType,
)
from src.utils.cuda_utils import allocated_free_gpus, check_cuda, calculate_model_size
from src.utils.process_metrics import format_metrics_file
from src.utils.file_path_helper import construct_model_name

# Global variables for training parameters
TASK = "regression"  # "classification" "regression"
DATA_SUBSET = "high"  # "big_sample", "low", "medium", "high"
DIM = "2D"

ANAT_FEATURE_MAPS: list[FeatureMapType] = [
    FeatureMapType.GM,
    FeatureMapType.WM,
    FeatureMapType.CSF,
    FeatureMapType.T1,
]
FUNC_FEATURE_MAPS: list[FeatureMapType] = [
    FeatureMapType.REHO,
    FeatureMapType.BOLD,
]
DUAL_MODALITY = (
    True if len(ANAT_FEATURE_MAPS) > 0 and len(FUNC_FEATURE_MAPS) > 0 else False
)
FEATURE_MAPS = ANAT_FEATURE_MAPS + FUNC_FEATURE_MAPS
TARGET = "sex" if TASK == "classification" else "age"
TEMPORAL_PROCESS = ["mean", "variance", "tsnr"]  # "mean", "variance", "tsnr", None

MODEL_TYPE = "ResNet18"  # "ResNet18" "ConvBranch"
EPOCHS = 40
LEARNING_RATE = 1e-3  # mr_lr = lr * 25
EXPERIMENT = "quality_separation"
EXPERIMENT_NOTES = {
    "notes": f"Showing the effects of data with different quality on the model performance. {DATA_SUBSET} quality data subset.",
}


def train_model(num_gpus: int = None, compute_node: str = None, prefix: str = None):
    """Handles all the logic for training the model."""

    print("Training model...")

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Infer batch size and GPUs
    # global BATCH_SIZE, ACCUMULATE_GRAD_BATCHES, LOG_DIR
    batch_size, acc_grad_batches = infer_batch_size(
        compute_node, DIM, MODEL_TYPE
    )
    num_gpus = infer_gpu_count(compute_node, num_gpus)
    used_gpus = allocated_free_gpus(num_gpus)
    log_dir = Path("models") if EPOCHS > 20 else Path("models_test")

    print_collection_dict = {
        "Compute Node": compute_node,
        "Experiment": EXPERIMENT,
        "Run Prefix": prefix,
        "Model Type": MODEL_TYPE,
        "Data Subset": DATA_SUBSET,
        "Data Dimension": DIM,
        "Feature Maps": [fm.label for fm in FEATURE_MAPS],
        "Temporal Processing": TEMPORAL_PROCESS,
        "Target": TARGET,
        "Task": TASK,
        "Epochs": EPOCHS,
        "Batch Size": batch_size,
        "Accumulate Gradient Batches": acc_grad_batches,
        "Num GPUs": num_gpus,
        "Used GPUs": used_gpus,
        "Initial Learning Rate": LEARNING_RATE,
        "Experiment Notes": EXPERIMENT_NOTES,
    }

    # Prepare data sets and loaders
    data_set_path = select_data_set(DATA_SUBSET)
    data_split = DataSplitFile(data_set_path).load_data_splits_from_file()

    base_config = BaseDataSetConfig(
        target=TARGET,
        middle_slice=True if DIM == "2D" else False,
        slice_dim=0 if DIM == "2D" else None,
        temporal_processes=TEMPORAL_PROCESS,
    )

    train_set, val_set, test_set = DataSetFactory(
        train_ids=data_split["train"],
        val_ids=data_split["val"],
        test_ids=data_split["test"],
        base_config=base_config,
        anat_feature_maps=ANAT_FEATURE_MAPS,
        func_feature_maps=FUNC_FEATURE_MAPS,
    ).create_data_sets()

    train_loader = prepare_standard_data_loaders(train_set, batch_size=batch_size)
    val_loader = prepare_standard_data_loaders(val_set, batch_size=batch_size)

    # Derive the amount of channels for the feature maps
    anat_channels = len(ANAT_FEATURE_MAPS) if ANAT_FEATURE_MAPS else 0
    func_channels = (
        len(FUNC_FEATURE_MAPS) + len(TEMPORAL_PROCESS) - 1
        if FeatureMapType.BOLD in FUNC_FEATURE_MAPS
        else len(FUNC_FEATURE_MAPS)
    )

    # Create model according to the feature maps
    if DUAL_MODALITY:
        model = ModelFactory(task=TASK, dim=DIM).create_resnet_multi_modal(
            anat_channels=anat_channels,
            func_channels=func_channels,
        )
    else:
        num_channels = anat_channels or func_channels
        model = ModelFactory(task=TASK, dim=DIM).create_resnet18(
            in_channels=num_channels
        )

    print_collection_dict["Model Size"] = calculate_model_size(model)

    # Setup lightning wrapper
    wrapper_class = MultiModalityWrapper if DUAL_MODALITY else OneCycleWrapper
    lightning_wrapper = wrapper_class(
        model=model, task=TASK, learning_rate=LEARNING_RATE
    )

    # Model name for logging
    model_name = construct_model_name(
        lightning_wrapper.model,
        train_set,
        experiment=EXPERIMENT,
        compute_node=compute_node,
    )

    # Logging and callbacks
    logger = pl_loggers.CSVLogger(log_dir, name=model_name)
    start_info_callback = ExperimentStartCallback(
        logger=logger, logging_dict=print_collection_dict
    )
    print_callback = ValidationPrintCallback(logger=logger)

    setup_logger = ExperimentSetupLogger(
        logger=logger,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        print_collection_dict=print_collection_dict,
    )
    progress_logger = TrainingProgressTracker(
        logger=logger,
    )
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer setup
    trainer_config = LightningTrainerConfig(
        devices=used_gpus,
        max_epochs=EPOCHS,
        deterministic=(
            True if DIM == "2D" else False
        ),  # maxpool3d has no deterministic implementation
        accumulate_grad_batches=acc_grad_batches,
    )
    trainer = L.Trainer(
        **trainer_config.dict(),
        callbacks=[
            start_info_callback,
            print_callback,
            setup_logger,
            progress_logger,
            learning_rate_monitor,
        ],
        logger=logger,
    )

    trainer.fit(
        model=lightning_wrapper,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Clean up the trainer and CUDA operations
    del trainer  # Explicitly delete the trainer object
    torch.cuda.empty_cache()  # Clear CUDA memory cache

    # test_model()
    test_trainer_config = LightningTrainerConfig(
        devices=1,  # Use the same number of GPUs for testing
        max_epochs=1,  # No need for multiple epochs in testing
        deterministic=True,  # Ensure deterministic behavior for testing
        strategy="auto",
        accumulate_grad_batches=acc_grad_batches,
    )
    test_trainer = L.Trainer(
        **test_trainer_config.dict(),
        callbacks=[ValidationPrintCallback(logger=logger), TestingProgressTracker(logger=logger)],
        logger=logger,
    )

    # Prepare the test data loader
    test_loader = prepare_standard_data_loaders(test_set, batch_size=batch_size)
    # Load the model checkpoint
    wrapper_class = MultiModalityWrapper if DUAL_MODALITY else OneCycleWrapper
    model_checkpoint_dir=Path(logger.log_dir, "checkpoints")
    checkpoint_file = list(model_checkpoint_dir.glob("*.ckpt"))[0] # Assuming there's only one checkpoint file

    lightning_wrapper = wrapper_class.load_from_checkpoint(
        checkpoint_path=checkpoint_file,
        model=model,
        task=TASK,
        learning_rate=LEARNING_RATE,
    )
    # Run the test step
    test_trainer.test(
        model=lightning_wrapper,
        dataloaders=test_loader,
    )

    # Process metrics
    metrics_file = Path(logger.log_dir, "metrics.csv")
    format_metrics_file(metrics_file)

    # Plot training metrics (after some time to allow for file writing)
    time.sleep(2)
    formatted_file = Path(metrics_file.parent, f"{metrics_file.stem}_formatted.csv")
    plot_all_metrics(formatted_file, task=TASK, splits=["train", "val"])


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_model", description="Train a model on the MRI data set."
    )

    parser.add_argument("prefix", type=str, help="The prefix/tag for this training run")
    parser.add_argument(
        "-c",
        "--compute_node",
        type=str,
        required=True,
        choices=["cuda01", "cuda02"],
        help="The computing node to use for training.",
    )
    parser.add_argument(
        "-g",
        "--num_gpus",
        type=int,
        default=None,
        help="The number of GPUs to use for training.",
    )
    return parser


def select_data_set(split: Literal["big_sample", "low", "medium", "high"]) -> Path:
    """Selects the data set based on the split type."""
    if split == "big_sample":
        return AGE_SEX_BALANCED_10K_PATH
    elif split == "low":
        return LOW_QUALITY_IDS
    elif split == "medium":
        return MEDIUM_QUALITY_IDS
    elif split == "high":
        return HIGH_QUALITY_IDS
    else:
        raise ValueError(
            f"Unknown split type: {split}. Choose from 'big_sample', 'low', 'medium', or 'high'."
        )


if __name__ == "__main__":
    """
    The script can be run either directly with Python or via the run_training.sh script.
    If using the latter, the first argument is an optional prefix for the output file.

    Direct Python execution:
    - python train_model.py -c cuda02
    - python train_model.py -c cuda02 -g 4
    - python train_model.py --compute_node cuda02 --num_gpus 2

    Via run_training.sh (which handles nohup output):
    - bash run_training.sh test -c cuda02
    - bash run_training.sh experiment1 -c cuda02 -g 4
    - bash run_training.sh gpu_test -c cuda02 --num_gpus 2

    Note:
    - -c/--compute_node is required in both cases
    - -g/--num_gpus is optional
    - When using run_training.sh, the first argument is an optional prefix for the output file
    """

    # check_cuda()
    parser = setup_parser()
    args = parser.parse_args()
    train_model(args.num_gpus, args.compute_node, args.prefix)

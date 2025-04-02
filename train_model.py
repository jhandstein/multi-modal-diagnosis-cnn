import argparse
import time
from pathlib import Path

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor


from src.building_blocks.model_factory import ModelFactory
from src.building_blocks.lightning_configs import LightningTrainerConfig
from src.building_blocks.lightning_wrapper import MultiModalityWrapper, OneCycleWrapper
from src.building_blocks.metrics_callbacks import (
    ExperimentSetupLogger,
    ExperimentStartCallback,
    TrainingProgressTracker,
    ValidationPrintCallback,
)
from src.data_management.data_set import BaseDataSetConfig
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_loader import infer_batch_size, infer_gpu_count, prepare_standard_data_loaders
from src.data_management.data_set_factory import DataSetFactory
from src.plots.plot_metrics import plot_all_metrics
from src.utils.config import (
    AGE_SEX_BALANCED_10K_PATH,
    FeatureMapType,
)
from src.utils.cuda_utils import allocated_free_gpus, check_cuda, calculate_model_size
from src.utils.process_metrics import format_metrics_file
from src.utils.file_path_helper import construct_model_name


def train_model(num_gpus: int = None, compute_node: str = None, prefix: str = None):
    """Handles all the logic for training the model."""

    print("Training model...")

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Set parameters for training
    task = "regression" # "classification" "regression"
    dim = "2D"
    anat_feature_maps = [
        # FeatureMapType.GM, 
        # FeatureMapType.WM, 
        FeatureMapType.CSF,
        # FeatureMapType.SMRI
    ]
    func_feature_maps = [
        FeatureMapType.REHO,
    ]
    dual_modality = True if len(anat_feature_maps) > 0 and len(func_feature_maps) > 0 else False
    feature_maps = anat_feature_maps + func_feature_maps
    target = "sex" if task == "classification" else "age"
    model_type = "ResNet18" # "ResNet18" "ConvBranch"

    epochs = 40
    batch_size, accumulate_grad_batches = infer_batch_size(compute_node, dim, model_type)
    # todo: derive learning rate dynamically from dict / utility function
    learning_rate = 1e-3 # mr_lr = lr * 25
    num_gpus = infer_gpu_count(compute_node, num_gpus)
    used_gpus = allocated_free_gpus(num_gpus)
    experiment = "dual_modality"
    experiment_notes = {"notes": f"Testing dual modality for the first time. FMs: {[fm.label for fm in feature_maps]}."}

    print_collection_dict = {
        "Compute Node": compute_node,
        "Experiment": experiment,
        "Run Prefix": prefix,
        "Model Type": model_type,
        "Data Dimension": dim,
        "Feature Maps": [fm.label for fm in feature_maps],
        "Target": target,
        "Task": task,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Accumulate Gradient Batches": accumulate_grad_batches,
        "Num GPUs": num_gpus,
        "Used GPUs": used_gpus,
        "Initial Learning Rate": learning_rate,
        "Experiment Notes": experiment_notes,
    }

    # Experiment setup
    if epochs > 20:
        log_dir = Path("models")
    else:
        log_dir = Path("models_test")

    # Prepare data sets and loaders
    data_split = DataSplitFile(AGE_SEX_BALANCED_10K_PATH).load_data_splits_from_file()

    base_config = BaseDataSetConfig(
        target=target,
        middle_slice=True if dim == "2D" else False,
        slice_dim=0 if dim == "2D" else None,    
        )
    
    train_set, val_set, test_set = DataSetFactory(
        train_ids=data_split["train"], 
        val_ids=data_split["val"], 
        test_ids=data_split["test"],
        base_config=base_config,
        anat_feature_maps=anat_feature_maps,
        func_feature_maps=func_feature_maps
    ).create_data_sets()

    train_loader = prepare_standard_data_loaders(
        train_set, batch_size=batch_size
    )
    val_loader = prepare_standard_data_loaders(val_set, batch_size=batch_size)

    #! ConvBranch is not supported for now
    # Create model according to the feature maps
    if dual_modality:
        model = ModelFactory(task=task, dim=dim).create_resnet_multi_modal(
            anat_channels=len(anat_feature_maps),
            func_channels=len(func_feature_maps)
        )
    else:
        feature_maps = anat_feature_maps or func_feature_maps
        model = ModelFactory(task=task, dim=dim).create_resnet18(in_channels=len(feature_maps))

    print_collection_dict["Model Size"] = calculate_model_size(model)

    # Setup lightning wrapper
    if dual_modality:
        wrapper_class = MultiModalityWrapper
    else:
        wrapper_class = OneCycleWrapper
    lightning_wrapper = wrapper_class(
        model=model, task=task, learning_rate=learning_rate
    )

    # Model name for logging
    model_name = construct_model_name(lightning_wrapper.model, train_set, experiment=experiment, compute_node=compute_node)

    # Logging and callbacks
    logger = pl_loggers.CSVLogger(log_dir, name=model_name)
    start_info_callback = ExperimentStartCallback(logger=logger, logging_dict=print_collection_dict)
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
        max_epochs=epochs,
        deterministic=True if dim == "2D" else False, # maxpool3d has no deterministic implementation
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = L.Trainer(
        **trainer_config.dict(), 
        callbacks=[start_info_callback, print_callback, setup_logger, progress_logger, learning_rate_monitor], 
        logger=logger,
        )

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
        # checkpoint_path = Path(
        #     "models/CNN_2D_anat_WM/version_0/checkpoints/epoch=99-step=22400.ckpt"
        # )
        # lightning_model = LightningWrapperCnn.load_from_checkpoint(
        #     checkpoint_path
        # )

        # # Set model into evaluation mode
        # lightning_model.eval()
        pass


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_model",
        description="Train a model on the MRI data set."
        )
    
    parser.add_argument(
        "prefix",
        type=str,
        help="The prefix/tag for this training run"
    )
    parser.add_argument(
        "-c", "--compute_node",
        type=str,
        required=True,
        choices=["cuda01", "cuda02"],
        help="The computing node to use for training."
    )
    parser.add_argument(
        "-g", "--num_gpus",
        type=int,
        default=None,
        help="The number of GPUs to use for training."
    )
    return parser

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

    # print(f"Batch size and accumulate batches: {infer_batch_size(args.compute_node, '3D', 'ConvBranch')}")
    # print(f"Number of GPUs: {infer_gpu_count(args.compute_node, args.num_gpus)}")

    train_model(args.num_gpus, args.compute_node, args.prefix)


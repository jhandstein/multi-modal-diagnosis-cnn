from pathlib import Path
import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers

from src.plots.save_training_plot import plot_training_metrics
from src.data_management.feature_map_files import FeatureMapFile
from src.data_management.data_set import prepare_standard_data_sets
from src.building_blocks.metrics_logging import ExperimentTrackingCallback, ValidationPrintCallback, process_metrics_file
from src.data_management.data_loader import prepare_standard_data_loaders
from src.building_blocks.lightning_wrapper import BinaryClassificationCnn2d
from src.utils.cuda_utils import check_cuda
from src.utils.config import FeatureType, ModalityType



def train_model():
    """Handles all the logic for training the model."""

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Declare lightning wrapper model
    lightning_model = BinaryClassificationCnn2d()

    # Set parameters for training
    num_gpus = torch.cuda.device_count()
    batch_size = 8 # should be maximum val_set size / num_gpus
    epochs = 100
    sample_size = 16384
    # sample_size = 256

    # Prepare data sets and loaders
    # TODO: replace with k-fold cross-validation? 4 folds in publication
    train_set, val_set, test_set = prepare_standard_data_sets(n_samples=sample_size, val_test_frac=1/16)
    train_loader = prepare_standard_data_loaders(train_set, batch_size=batch_size, num_gpus=num_gpus)
    val_loader = prepare_standard_data_loaders(val_set, batch_size=2, num_gpus=num_gpus)
    
    # Experiment setup
    if epochs > 10:
        log_dir = Path("models")
    else:
        log_dir = Path("models_test")

    dim = "2D"
    modality = train_set.modalitiy.value
    feature_map = train_set.feature_set.value
    model_name = f"CNN_{dim}_{modality}_{feature_map}"


    # Logging and callbacks
    logger = pl_loggers.CSVLogger(log_dir, name=model_name)
    print_callback = ValidationPrintCallback(logger=logger)
    json_callback = ExperimentTrackingCallback(logger=logger, train_set=train_set, val_set=val_set, test_set=test_set)

    gpu_params = {
        "accelerator": "gpu" if torch.cuda.is_available() else None,
        "devices": num_gpus,
        "strategy": "ddp",
        "sync_batchnorm": True,
        # benchmark interferes with reproducibility from seed_everything and deterministic
        "benchmark": False, # best algorithm for your hardware, only works with homogenous input sizes
    }

    training_params = {
        "deterministic": True, # works together with seed_everything to ensure reproducibility across runs
        "max_epochs": epochs,
    }

    logging_params = {
        "log_every_n_steps": 1,
        "callbacks": [print_callback, json_callback],
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
    
def print_ds_indices():
    train_set, val_set, test_set = prepare_standard_data_sets(n_samples=1024)
    print("Data set indices:")
    print(train_set.subject_ids)
    print(val_set.subject_ids)
    print(test_set.subject_ids)

if __name__ == "__main__":
    
    # check_cuda()

    # compare data dimensions for raw and feature maps
    # fm = FeatureMapFile(130926, ModalityType.RAW, FeatureType.SMRI)
    # print(fm.get_path())
    # print(fm.print_stats())
    # fm2 = FeatureMapFile(130926, ModalityType.ANAT, FeatureType.GM)
    # print(fm2.get_path())
    # print(fm2.print_stats())

    # train_model()
    plot_training_metrics(Path("models/CNN_2D_sMRI_GM/version_0/metrics_processed.csv"))

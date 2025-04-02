import matplotlib.pyplot as plt
import lightning as L

from lightning.pytorch.tuner import Tuner
import torch

from src.building_blocks.model_factory import ModelFactory
from src.building_blocks.lightning_wrapper import LightningWrapperCnn
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_loader import infer_batch_size, prepare_standard_data_loaders
from src.data_management.data_set import SingleModalityDataSetConfig
from src.data_management.data_set_factory import DataSetFactory
from src.utils.config import AGE_SEX_BALANCED_10K_PATH, PLOTS_PATH, FeatureMapType
from src.utils.cuda_utils import calculate_model_size


def estimate_initial_learning_rate():
    """
    Implements learning rate finder using PyTorch Lightning
    """
    num_gpus = 8
    task = "classification"
    dim = "3D"
    target = "sex" if task == "classification" else "age"
    model_type = "ConvBranch"

    batch_size, accumulate_batches = infer_batch_size(dim, model_type)

    data_split_path = AGE_SEX_BALANCED_10K_PATH
    # Prepare data sets and loaders
    #? does it already work with multi-modality?
    ds_config = SingleModalityDataSetConfig(
        feature_maps=[FeatureMapType.GM],
        target=target,
        middle_slice=False
    )
    data_split = DataSplitFile(data_split_path).load_data_splits_from_file()
    
    train_set, val_set, test_set = DataSetFactory(
        data_split["train"], 
        data_split["val"], 
        data_split["test"], 
        ds_config,
        anat_feature_maps=[FeatureMapType.GM],
    ).create_data_sets()

    train_loader = prepare_standard_data_loaders(
        train_set, batch_size=batch_size
    )
    val_loader = prepare_standard_data_loaders(
        val_set, batch_size=batch_size
        )

    # Setup model
    if model_type == "ConvBranch":
        model = ModelFactory(task=task, dim=dim).create_conv_branch(input_shape=train_set.data_shape)
    elif model_type == "ResNet18":
        model = ModelFactory(task=task, dim=dim).create_resnet18(in_channels=1)
    else:
        raise ValueError("Model type not supported. Check the model_type argument.")

    # Declare lightning wrapper model
    lightning_wrapper = LightningWrapperCnn(
        model=model, task=task, learning_rate=1e-3
    )

    # Create trainer with auto learning rate finder
    trainer = L.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=num_gpus,  # Add this line for automatic device selection
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_batches,
    )

    # Run learning rate finder
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(lightning_wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Get suggestion and adjust it
    suggested_lr = lr_finder.suggestion()
    adjusted_lr = suggested_lr / 10  # Use 1/10th of the suggested value, recommended by the authors?
    print(f"Suggested LR: {suggested_lr}")
    print(f"Adjusted LR: {adjusted_lr}")

    model_name = f"{lightning_wrapper.model.__class__.__name__}_batch_size_{batch_size}"
    plot_lr_finder_results(lr_finder, model_name)

    return suggested_lr

def plot_lr_finder_results(lr_finder, model_name: str):
    """
    Plots the learning rate finder results
    """
    fig, ax = plt.subplots()
        # Plot loss vs learning rate
    ax.plot(lr_finder.results['lr'], lr_finder.results['loss'])
    
    # Add suggested learning rate vertical line
    suggested_lr = lr_finder.suggestion()
    ax.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
    
    # Format plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Rate Finder Results')
    ax.grid(True)
    ax.legend()
    
    # Save plot
    save_path = PLOTS_PATH / "lr_finder" / f"{model_name}_lr_finder_results.png"
    plt.savefig(save_path)
    plt.close()
    
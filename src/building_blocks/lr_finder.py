import matplotlib.pyplot as plt
import lightning as L

from lightning.pytorch.tuner import Tuner

from src.building_blocks.lightning_wrapper import LightningWrapper2dCnn
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_loader import prepare_standard_data_loaders
from src.data_management.data_set import DataSetConfig
from src.data_management.data_set_factory import DataSetFactory
from src.utils.config import AGE_SEX_BALANCED_10K_PATH, PLOTS_PATH, FeatureMapType


def estimate_initial_learning_rate():
    """
    Implements learning rate finder using PyTorch Lightning
    """
    batch_size = 64
    num_gpus = 8
    task = "classification"
    target = "sex" if task == "classification" else "age"

    data_split_path = AGE_SEX_BALANCED_10K_PATH
    # Prepare data sets and loaders
    ds_config = DataSetConfig(
        feature_map=FeatureMapType.GM,
        target=target,
        middle_slice=True
    )
    data_split = DataSplitFile(data_split_path).load_data_splits_from_file()
    
    train_set, val_set, test_set = DataSetFactory(
        data_split["train"], data_split["val"], data_split["test"], ds_config
    ).create_data_sets()

    train_loader = prepare_standard_data_loaders(
        train_set, batch_size=batch_size, num_gpus=num_gpus
    )
    val_loader = prepare_standard_data_loaders(
        val_set, batch_size=batch_size, num_gpus=num_gpus
    )

    # Declare lightning wrapper model
    lightning_wrapper = LightningWrapper2dCnn(
        train_set.data_shape, task=task, learning_rate=1e-3
    )

    # Create trainer with auto learning rate finder
    trainer = L.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=num_gpus,  # Add this line for automatic device selection
        log_every_n_steps=1
    )

    # Run learning rate finder
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(lightning_wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # TODO: Add batch size finder

    # Get suggestion
    suggested_lr = lr_finder.suggestion()
    print(suggested_lr)
    model_name = lightning_wrapper.model.__class__.__name__
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

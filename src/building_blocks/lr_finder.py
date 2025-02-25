import matplotlib.pyplot as plt
import lightning as L

from lightning.pytorch.tuner import Tuner
from pytorch_lightning.callbacks import LearningRateMonitor
import torch

from src.building_blocks.model_factory import ModelFactory
from src.building_blocks.lightning_wrapper import LightningWrapperCnn
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_loader import prepare_standard_data_loaders
from src.data_management.data_set import DataSetConfig
from src.data_management.data_set_factory import DataSetFactory
from src.utils.config import AGE_SEX_BALANCED_10K_PATH, PLOTS_PATH, FeatureMapType
from utils.cuda_utils import calculate_model_size


def estimate_initial_learning_rate():
    """
    Implements learning rate finder using PyTorch Lightning
    """
    batch_size = 8 #64
    num_gpus = 8
    task = "regression"
    dim = "3D"
    target = "sex" if task == "classification" else "age"
    model_type = "ConvBranch"

    data_split_path = AGE_SEX_BALANCED_10K_PATH
    # Prepare data sets and loaders
    ds_config = DataSetConfig(
        feature_map=FeatureMapType.GM,
        target=target,
        middle_slice=False
    )
    data_split = DataSplitFile(data_split_path).load_data_splits_from_file()
    
    train_set, val_set, test_set = DataSetFactory(
        data_split["train"], data_split["val"], data_split["test"], ds_config
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
        log_every_n_steps=1
    )

    # Run learning rate finder
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(lightning_wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # TODO: Add batch size finder

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


def test_lr_monitor_one_cylce_policy():
    """Test the OneCycleLR policy"""

    # https://github.com/Lightning-AI/pytorch-lightning/discussions/13236
    # https://github.com/Lightning-AI/pytorch-lightning/discussions/9601

    class TestWrapper(LightningWrapperCnn):

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=0.1,
                    steps_per_epoch=64,
                    epochs=10,
                    three_phase=True
                )
            return [optimizer], [lr_scheduler]
        
    # Set parameters for training
    task = "regression" # "classification" "regression"
    dim = "3D"
    feature_map = FeatureMapType.GM
    target = "sex" if task == "classification" else "age"
    model_type = "ResNet18" # "ResNet18" "ConvBranch"

    num_gpus = torch.cuda.device_count()
    batch_size = 64 if dim == "2D" else 2 # should be maximum val_set size / num_gpus?
    learning_rate = 1e-3
    experiment_notes = {"notes": f"ResNet183D with accumulate batches=8, batch_size=2. Running classification with Adam default LR. Scaling of 0.5 for image inputs."}

    print_collection_dict = {
        "Task": task,
        "Batch Size": batch_size,
        "Num GPUs": num_gpus,
        "Learning Rate": learning_rate,
        "Experiment Notes": experiment_notes,
    }

    # Prepare data sets and loaders
    ds_config = DataSetConfig(
        feature_map=feature_map,
        target=target,
        middle_slice=True if dim == "2D" else False,
    )
    data_split = DataSplitFile(AGE_SEX_BALANCED_10K_PATH).load_data_splits_from_file()
    
    train_set, val_set, test_set = DataSetFactory(
        data_split["train"], data_split["val"], data_split["test"], ds_config
    ).create_data_sets()

    train_loader = prepare_standard_data_loaders(
        train_set, batch_size=batch_size
    )
    val_loader = prepare_standard_data_loaders(val_set, batch_size=batch_size)

    # Setup model
    if model_type == "ConvBranch":
        model = ModelFactory(task=task, dim=dim).create_conv_branch(input_shape=train_set.data_shape)
    elif model_type == "ResNet18":
        model = ModelFactory(task=task, dim=dim).create_resnet18(in_channels=1)
    else:
        raise ValueError("Model type not supported. Check the model_type argument.")

    print_collection_dict["Model Size"] = calculate_model_size(model)

    wrapper = TestWrapper(
        model=model,
        task=task,
        learning_rate=learning_rate
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(
        default_root_dir="one_cycle_test",
        max_epochs=10,
        callbacks=[lr_monitor]
    )
    trainer.fit(wrapper, train_loader, val_loader)
    breakpoint()
from time import time

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet18

from src.building_blocks.resnet18 import FlexibleResNet
from src.plots.save_training_plot import plot_mae_mse, plot_training_metrics
from src.plots.plot_age_range import plot_age_range
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_set import NakoSingleFeatureDataset
from src.data_management.data_set_factory import DataSetConfig, DataSetFactory
from src.utils.config import (
    AGE_SEX_BALANCED_10K_PATH,
    AGE_SEX_BALANCED_1K_PATH,
    FeatureMapType,
)
from src.testing._250131_first_data_splits import create_balanced_samples
from src.utils.file_path_helper import construct_model_name


def test_data_set_factory():
    ds_details = {
        "feature_map": FeatureMapType.GM,
        "target": "sex",
        "middle_slice": True,
    }

    # start timer
    start = time()

    ds_config = DataSetConfig(**ds_details)
    ds_factory = DataSetFactory(
        [100000, 100005], [100006, 100010], [100011, 100015], ds_config
    )
    train_set, val_set, test_set = ds_factory.create_data_sets()
    print("Shape:", train_set.data_shape)

    # end timer
    end = time()
    print(f"Time elapsed: {end - start} seconds")
    
def test_resnet():
        # Example with smaller input dimensions
    batch_size = 4
    channels = 1
    height = 96  # Can be any size that works with the architecture
    width = 96
    
    # Create model instance
    model = FlexibleResNet(
        input_channels=channels,
        num_classes=2  # For binary classification (e.g., sex)
    )
    
    # Create dummy input
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return model


if __name__ == "__main__":
    print("Hello from test.py")
    # test_data_set_factory()


    test_resnet()

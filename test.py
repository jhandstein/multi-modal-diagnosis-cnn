from time import time

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet18

from src.data_management.mri_image_files import MriImageFile
from src.plots.plot_mri_slice import plot_mri_slice
from src.plots.plot_metrics import plot_all_metrics
from src.building_blocks.lightning_wrapper import LightningWrapper2dCnn
from src.data_management.data_loader import prepare_standard_data_loaders
from src.building_blocks.lr_finder import estimate_initial_learning_rate
from src.plots.plot_age_range import plot_age_range
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_set import DataSetConfig, NakoSingleFeatureDataset
from src.data_management.data_set_factory import DataSetFactory
from src.utils.config import (
    AGE_SEX_BALANCED_10K_PATH,
    AGE_SEX_BALANCED_1K_PATH,
    FeatureMapType,
)
from src.testing._250131_first_data_splits import create_balanced_samples
from src.utils.file_path_helper import construct_model_name
from src.utils.process_metrics import format_metrics_file


def test_data_set_factory():

    # start timer
    start = time()

    ds_config = DataSetConfig(
        feature_map=FeatureMapType.GM,
        target="sex",
        middle_slice=True
        )
    ds_factory = DataSetFactory(
        [100000, 100005], 
        [100006, 100010], 
        [100011, 100015], 
        ds_config
    )
    train_set, val_set, test_set = ds_factory.create_data_sets()
    print("Shape:", train_set.data_shape)

    # end timer
    end = time()
    print(f"Time elapsed: {end - start} seconds")

def find_lr():
    sug_rate = estimate_initial_learning_rate()
    print(sug_rate)

if __name__ == "__main__":
    print("Hello from test.py")
    # test_data_set_factory()

    # find_lr()
    # format_metrics_file(Path("/home/julius/repositories/ccn_code/models_test/ConvBranch2dRegression_2D_anat_GM_regression_age/version_0/metrics.csv"))
    # metrics_path = Path("/home/julius/repositories/ccn_code/models/ResNet18Regression_2D_anat_GM_regression_age/version_0/metrics_formatted.csv")
    # plot_all_metrics(metrics_path, task="regression", splits=["train", "val"])
    # plot_mri_slice(100008, slice_dim=2)

    # for dim in [0, 1, 2]:
    #     for fm in [FeatureMapType.GM, FeatureMapType.WM, FeatureMapType.CSF, FeatureMapType.REHO, FeatureMapType.SMRI, FeatureMapType.FMRI]:
    #         plot_mri_slice(100008, slice_dim=dim, feature_map=fm)

    mri_image_file = MriImageFile(100008, FeatureMapType.GM)
    print(mri_image_file.get_size())
from time import time

from pathlib import Path

from src.building_blocks.model_factory import ModelFactory
from src.data_management.mri_image_files import MriImageFile
from src.plots.plot_mri_slice import plot_mri_slice
from src.plots.plot_metrics import plot_all_metrics
from src.building_blocks.lightning_wrapper import LightningWrapperCnn
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
from src.utils.cuda_utils import allocated_free_gpus, calculate_tensor_size


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

def check_file_size():
    """Create MRI image file and check size."""
    mri_image_file = MriImageFile(100008, FeatureMapType.GM, middle_slice=False)
    size = calculate_tensor_size(mri_image_file.load_as_tensor(), batch_size=16)
    print(f"Size in MB: {size}")

def plot_metrics_when_failed_during_training():
    # Setup
    task = "regression"

    # Format metrics file
    # metrics_path = Path("/home/julius/repositories/ccn_code/models_test/250228_ConvBranch2dBinary_anat_GM_sex/version_1/metrics.csv")
    # format_metrics_file(metrics_path)

    # Plot metrics
    metrics_path = Path("/home/julius/repositories/ccn_code/models/250302_ConvBranch3dRegression_anat_GM_age/version_0/metrics_formatted.csv")
    plot_all_metrics(metrics_path, task=task, splits=["train", "val"])


def plot_mri_slices():
    """Plot MRI slices for different dimensions and feature maps."""
    for dim in [0, 1, 2]:
        for fm in [FeatureMapType.GM, FeatureMapType.WM, FeatureMapType.CSF, FeatureMapType.REHO, FeatureMapType.SMRI, FeatureMapType.FMRI]:
            plot_mri_slice(100010, slice_dim=dim, feature_map=fm)

def cache_tensor():
    """Cache tensor."""
    mri_image_file = MriImageFile(100008, FeatureMapType.GM, middle_slice=True, slice_dim=0)
    tensor = mri_image_file.load_as_tensor()
    print(tensor.shape)
    print(mri_image_file.cache_path)

    tensor2 = mri_image_file.load_as_tensor()
    print(tensor2.shape)

def time_tensor_loading():
    for i in range(10):
        mri_file = MriImageFile(100010, FeatureMapType.GM, middle_slice=True, slice_dim=0)
        start = time()
        tensor = mri_file.load_as_tensor()
        end = time()
        print(f"Time elapsed for loading from .nii.gz: {end - start} seconds")

        file_2 = MriImageFile(100008, FeatureMapType.GM, middle_slice=True, slice_dim=0)
        start = time()
        tensor_2 = file_2.load_as_tensor()
        end = time()
        print(f"Time elapsed for loading from cache: {end - start} seconds", "\n")
    
if __name__ == "__main__":
    print("Hello from test.py")
    # test_data_set_factory()

    # find_lr()
    # check_file_size()
    # plot_metrics_when_failed_during_training()

    # mri_image_file = MriImageFile(100008, FeatureMapType.GM, middle_slice=False)
    # print(mri_image_file.get_size())
    # mri_image_file._num_params()

    # plot_mri_slices()

    # print(allocated_free_gpus(2, max_usage_ratio=0.1))

    # cache_tensor()
    time_tensor_loading()

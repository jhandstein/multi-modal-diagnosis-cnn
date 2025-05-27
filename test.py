import json
from pathlib import Path

from src.data_management.data_quality_separation import QualitySampler
from src.building_blocks.model_factory import ModelFactory
from src.data_management.mri_image_files import MriImageFile
from src.data_management.normalization import MriImageNormalizer
from src.plots.plot_mri_slice import plot_mri_slice
from src.plots.plot_metrics import plot_all_metrics
from src.data_management.data_loader import prepare_standard_data_loaders
from src.building_blocks.lr_finder import estimate_initial_learning_rate
from src.plots.plot_age_range import plot_age_range
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_set import SingleModalityDataSetConfig
from src.data_management.data_set_factory import DataSetFactory
from src.utils.config import (
    AGE_SEX_BALANCED_10K_PATH,
    HIGH_QUALITY_IDS,
    LOW_QUALITY_IDS,
    MEDIUM_QUALITY_IDS,
    QUALITY_SPLITS_PATH,
    FeatureMapType,
)
from src.utils.cuda_utils import allocated_free_gpus, calculate_tensor_size
from src.utils.performance_evaluation import calc_loss_based_on_target_mean
from src.utils.cache_data_set import cache_data_set


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
        # for fm in [FeatureMapType.GM, FeatureMapType.WM, FeatureMapType.CSF, FeatureMapType.REHO, FeatureMapType.T1, FeatureMapType.FMRI]:
        #     plot_mri_slice(100010, slice_dim=dim, feature_map=fm)
        plot_mri_slice(100010, slice_dim=dim, feature_map=FeatureMapType.CSF)


def check_mri_intensities():
    mri_file = MriImageFile(100010, FeatureMapType.T1, middle_slice=True, slice_dim=0)
    tensor = mri_file.load_as_tensor()
    print(tensor.min(), tensor.max())
    
if __name__ == "__main__":
    print("Hello from test.py")


    # split_results = DataSplitFile(QUALITY_SPLITS_PATH).load_data_splits_from_file()
    # # print(split_results)

    # sampler = QualitySampler(split_results)
    # sampler.resample_faulty_subjects("medium")
    



    # for path in [AGE_SEX_BALANCED_10K_PATH]:
    for path in [MEDIUM_QUALITY_IDS]: # LOW_QUALITY_IDS, MEDIUM_QUALITY_IDS, HIGH_QUALITY_IDS
        data_split = DataSplitFile(path).load_data_splits_from_file()

        print(f"Caching data split from {path.stem}:")
        cache_data_set(data_split["train"], data_split["val"], data_split["test"], 
                       batch_size=8, 
                       num_workers=8)
import json
from pathlib import Path

from src.data_splitting.create_data_split import create_balanced_sample
from src.data_management.mri_image_files import MriImageFile
from src.data_management.normalization import MriImageNormalizer
from src.plots.plot_mri_slice import plot_mri_slice
from src.plots.plot_metrics import plot_all_metrics, plot_metric
from src.data_management.data_loader import prepare_standard_data_loaders
from src.building_blocks.lr_finder import estimate_initial_learning_rate
from src.plots.plot_age_range import plot_age_range, plot_age_range_with_comparison
from src.data_splitting.create_data_split import DataSplitFile
from src.data_management.data_set import BaseDataSetConfig, SingleModalityDataSetConfig
from src.data_management.data_set_factory import DataSetFactory
from src.utils.config import (
    AGE_SEX_BALANCED_10K_PATH,
    HIGH_QUALITY_IDS,
    LOW_QUALITY_IDS,
    MEDIUM_QUALITY_IDS,
    PHQ9_CUTOFF_SPLIT_PATH,
    GAD7_CUTOFF_SPLIT_PATH,
    QUALITY_SPLITS_PATH,
    FeatureMapType,
    TrainingMetric,
)
from src.utils.cuda_utils import allocated_free_gpus, calculate_tensor_size


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
        plot_mri_slice(100008, slice_dim=dim, feature_map=FeatureMapType.BOLD)


def check_mri_intensities():
    mri_file = MriImageFile(100010, FeatureMapType.T1, middle_slice=True, slice_dim=0)
    tensor = mri_file.load_as_tensor()
    print(tensor.min(), tensor.max())
    
if __name__ == "__main__":
    print("Hello from test.py")

    # for tp in [None, "mean", "variance", "tsnr"]:
    #     print(f"Checking value range for temporal process: {tp}")
    #     check_fm_value_range(
    #         feature_map=FeatureMapType.BOLD,
    #         temporal_process=tp
    #     )

    # data_split = DataSplitFile(GAD7_CUTOFF_SPLIT_PATH).load_data_splits_from_file()

    # cache_data_set("2D", data_split["train"], data_split["val"], data_split["test"], num_workers=8)

    # advanced_phenotype = "gad7_cutoff"
    # sampler = PhenotypeSampler(advanced_phenotype)
    # subjects = sampler.sample_binary_dataset()
    # sampler.check_sample_age_sex_distribution(subjects)
    # sampler.split_and_save_binary_sample(subjects, GAD7_CUTOFF_SPLIT_PATH)

    # train, val, test = generate_sample_data_sets(label=advanced_phenotype, middle_slice=True)
    # print(train.target, train.labels.value_counts())

    # phenotype_split_file = DataSplitFile(GAD7_CUTOFF_SPLIT_PATH)
    # split = phenotype_split_file.load_data_splits_from_file()

    # print(f"Train set size: {len(split['train'])}")
    # print(f"Validation set size: {len(split['val'])}")
    # print(f"Test set size: {len(split['test'])}")

    # calc_loss_based_on_target_mean(label="phq9_sum", data_path=PHQ9_CUTOFF_SPLIT_PATH)
    # plot_metric(
    #     file_path=Path("/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_250712_advanced_phenotypes/gad7/_3D/1312/Regression3d_gad7_sum_anat_GM_WM_CSF/version_0/metrics_formatted.csv"),
    #     metric=TrainingMetric.MAE,
    #     splits=["train", "val"],
    # )
    plot_age_range_with_comparison()

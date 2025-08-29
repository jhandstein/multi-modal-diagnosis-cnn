import matplotlib.pyplot as plt

from src.data_splitting.load_targets import extract_targets
from src.data_splitting.subject_selection import load_subject_ids_from_file
from src.data_splitting.create_data_split import DataSplitFile
from src.data_management.data_set import NakoSingleModalityDataset, SingleModalityDataSetConfig
from src.utils.config import (
    AGE_SEX_BALANCED_10K_PATH,
    AGE_SEX_BALANCED_1K_PATH,
    HIGH_QUALITY_IDS,
    LOW_QUALITY_IDS,
    MEDIUM_QUALITY_IDS,
    PLOTS_PATH,
    FeatureMapType,
)


def plot_age_range():
    ds_config = SingleModalityDataSetConfig(
        feature_maps=[FeatureMapType.GM],
        target="age",
        middle_slice=True
        )

    for path in [AGE_SEX_BALANCED_10K_PATH]:

        split_path = path
        data_split = DataSplitFile(split_path).load_data_splits_from_file()
        full_set = NakoSingleModalityDataset(
            data_split["train"] + data_split["val"] + data_split["test"], ds_config
        )

        plt.figure(figsize=(10, 6))
        plt.hist(full_set.labels, bins=50)
        plt.title("Distribution of Age Values")
        plt.xlabel("Age (in years)")
        plt.ylabel("Count")
        file_path = (
            PLOTS_PATH / "variable_distributions" / f"distribution_{path.stem}.png"
        )
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        plt.savefig(fname=file_path)

def plot_age_range_with_comparison():
    ds_config = SingleModalityDataSetConfig(
        feature_maps=[FeatureMapType.GM],
        target="age",
        middle_slice=True
    )

    # Load split data ages
    split_path = AGE_SEX_BALANCED_10K_PATH
    data_split = DataSplitFile(split_path).load_data_splits_from_file()
    full_set = NakoSingleModalityDataset(
        data_split["train"] + data_split["val"] + data_split["test"], ds_config
    )
    split_ages = full_set.labels

    # Load all available subject IDs
    available_ids = load_subject_ids_from_file()   
    nako_full_ages = extract_targets("age", available_ids)



    plt.figure(figsize=(10, 6))
    plt.hist(nako_full_ages, bins=50, alpha=0.5, label="All NAKO MRI samples", density=True, color="tab:blue")
    plt.hist(split_ages, bins=50, alpha=0.5, label="Study data subset", density=True, color="tab:orange")
    plt.title("Distribution of Age Values in the NAKO")
    plt.xlabel("Age (in years)")
    plt.ylabel("Relative Frequency")
    plt.legend()
    file_path = (
        PLOTS_PATH / "variable_distributions" / f"distribution_{split_path.stem}_vs_full_nako.png"
    )
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    plt.savefig(fname=file_path)
    plt.close()
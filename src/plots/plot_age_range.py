import matplotlib.pyplot as plt

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

    for path in [AGE_SEX_BALANCED_10K_PATH, AGE_SEX_BALANCED_1K_PATH]:

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

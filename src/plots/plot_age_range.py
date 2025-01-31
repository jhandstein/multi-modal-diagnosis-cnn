import matplotlib.pyplot as plt

from data_management.create_data_split import DataSplitFile
from data_management.data_set import NakoSingleFeatureDataset
from utils.config import AGE_SEX_BALANCED_10K_PATH, AGE_SEX_BALANCED_1K_PATH, FeatureType, ModalityType


def plot_age_range():
    ds_details = {
        "modality": ModalityType.ANAT,
        "feature_set": FeatureType.GM,
        "target": "age",
        "middle_slice": True
    }

    for path in [AGE_SEX_BALANCED_1K_PATH, AGE_SEX_BALANCED_10K_PATH]:

        split_path = path
        data_split = DataSplitFile(split_path).load_data_splits_from_file()
        full_set = NakoSingleFeatureDataset(data_split["train"] + data_split["val"] + data_split["test"], **ds_details)

        plt.figure(figsize=(10, 6))
        plt.hist(full_set.labels, bins=50)
        plt.title('Distribution of Labels')
        plt.xlabel('Label Value')
        plt.ylabel('Count')
        plt.savefig(fname=f"distribution_{path.stem}.png")
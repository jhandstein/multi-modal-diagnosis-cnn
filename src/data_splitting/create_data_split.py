from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplitFile:
    """Data class for storing and loading data split information as JSON files"""

    file_path: Path

    def save_data_splits_to_file(self, data_splits: dict) -> None:
        """Saves the pre-defined data splits to a JSON file"""
        with open(self.file_path, "w") as file:
            json.dump(data_splits, file, indent=4)

    def load_data_splits_from_file(self) -> dict:
        """Loads the pre-defined data splits from a JSON file"""
        with open(self.file_path, "r") as file:
            return json.load(file)


def create_balanced_sample(labels: pd.Series, sample_size: int) -> pd.Series:
    """
    Create balanced sample from binary labels

    Args:
        labels: Series with subject IDs as index and binary labels as values
        sample_size: Total size of desired sample (should be even)

    Returns:
        Series of balanced samples with subject IDs as index and binary labels as values
    """
    if sample_size % 2 != 0:
        sample_size -= 1
        print(f"Sample size must be even for balanced sampling. Adjusted to {sample_size}")
    n_per_class = sample_size // 2

    # Group by label and sample from each group
    balanced_sample = (
        labels.groupby(labels).sample(n=n_per_class).sample(frac=1)
    )  # shuffle

    return balanced_sample


def sub_sample_data_split(
    balanced_sample: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Subsample the balanced sample into training, validation and test set. Length of the balanced sample should be a multiple of 8."""
    n = len(balanced_sample)
    n_train = n * 6 // 8
    n_val = n // 8

    # Create balanced sample for training set
    train_set = create_balanced_sample(balanced_sample, n_train)
    remaining = balanced_sample.drop(train_set.index)
    # Create balanced sample for validation and test set
    val_set = create_balanced_sample(remaining, n_val)
    test_set = remaining.drop(val_set.index)

    # print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}, Test set size: {len(test_set)}")

    return train_set, val_set, test_set

def find_min_minority_class_count(labels_series: list[pd.Series]) -> int:
    """
    Find the minimum count of the minority class across multiple pandas Series
    with binary labels.
    
    Args:
        labels_series: List of pandas Series with binary values
    
    Returns:
        int: Minimum count of minority class across all series
    """
    min_counts = []
    
    for series in labels_series:
        # Count values in the series
        value_counts = series.value_counts()
        
        # Find the minority class (the one with fewer instances)
        minority_count = value_counts.min()
        min_counts.append(minority_count)
        
    return min(min_counts)


def check_split_results(train: pd.Series, val: pd.Series, test: pd.Series) -> None:
    """Check the split results for balanced samples."""
    print(f"Set sizes: train {len(train)}, val {len(val)}, test {len(test)}")
    print(f"Label distribution (training): \n{train.value_counts()}")
    print(f"Label distribution (validation): \n{val.value_counts()}")
    print(f"Label distribution (test): \n{test.value_counts()}")

    # Check label balance within each set
    for name, dataset in [("train", train), ("val", val), ("test", test)]:
        value_counts = dataset.value_counts()
        if len(set(value_counts)) != 1:  # All classes should have same count
            raise ValueError(f"Unbalanced classes in {name} set: {value_counts}")

    # Check label proportions across sets. Convert to lsit to get raw values
    train_props = list(train.value_counts(normalize=True))
    val_props = list(val.value_counts(normalize=True))
    test_props = list(test.value_counts(normalize=True))

    if not (train_props == val_props and train_props == test_props):
        raise ValueError("Label proportions differ between splits")


if __name__ == "__main__":

    pass

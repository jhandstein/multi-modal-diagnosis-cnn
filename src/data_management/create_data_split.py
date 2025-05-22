from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplitFile:
    """Data class for storing data split information"""

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
        sample_size: Total size of desired sample (must be even)

    Returns:
        Series of balanced samples with subject IDs as index and binary labels as values
    """
    if sample_size % 2 != 0:
        raise ValueError("Sample size must be even for balanced sampling")
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

    # create balanced sample for training set
    train_set = create_balanced_sample(balanced_sample, n_train)
    remaining = balanced_sample.drop(train_set.index)
    # create balanced sample for validation and test set
    val_set = create_balanced_sample(remaining, n_val)
    test_set = remaining.drop(val_set.index)

    return train_set, val_set, test_set


def sample_max_minority_class(
    labels_series: pd.Series,
    num_samples: int,
    random_seed: int = 42,
) -> pd.Series:
    """
    Sample the maximum number of samples from the minority class.

    Args:
        labels_series: Series with subject IDs as index and binary labels as values
        num_samples: Total size of desired sample (must be even)
        random_state: Random state for reproducibility
    Returns:
        Series of balanced samples with subject IDs as index and binary labels as values
    """

    # Get the counts of each label
    label_counts = labels_series.value_counts()
    minority_label, majority_label = label_counts.idxmin(), label_counts.idxmax()
    minority_count = label_counts[minority_label]

    # Take all samples from the minority class
    minority_samples = labels_series[labels_series == minority_label]

    # Calculate how many samples we need from majority class to reach num_samples
    samples_needed = num_samples - minority_count

    # Sample from majority class
    majority_samples = labels_series[labels_series == majority_label].sample(
        samples_needed, random_state=random_seed
    )
    print(
        f"Minority samples: {len(minority_samples)}, Majority samples: {len(majority_samples)}"
    )

    # Combine samples
    return pd.concat([minority_samples, majority_samples])

def stratified_binary_split(series_labels: pd.Series, split_ratio: float = 0.8, 
                           random_seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """
    Split a series of labels into two sets (train and test) while preserving label distribution.

    """
    # Get subject IDs and labels
    subject_ids = series_labels.index.tolist()
    labels = series_labels.values
    
    # Split into train and test
    train_ids, test_ids, _, _ = train_test_split(
        subject_ids, labels, 
        train_size=split_ratio, 
        stratify=labels,
        random_state=random_seed
    )
    # Convert back to Series with subject IDs as index and labels as values
    train_ids = pd.Series(train_ids, index=train_ids)
    test_ids = pd.Series(test_ids, index=test_ids)
    # Assign labels to the Series
    train_ids = train_ids.map(series_labels)
    test_ids = test_ids.map(series_labels)

    return train_ids, test_ids


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

    # Check label proportions across sets
    train_props = train.value_counts(normalize=True)
    val_props = val.value_counts(normalize=True)
    test_props = test.value_counts(normalize=True)

    if not (train_props.equals(val_props) and train_props.equals(test_props)):
        raise ValueError("Label proportions differ between splits")


if __name__ == "__main__":

    pass

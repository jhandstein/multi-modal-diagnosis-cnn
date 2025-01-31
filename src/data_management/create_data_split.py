from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

@dataclass
class DataSplit:
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
    balanced_sample = (labels.groupby(labels)
                           .sample(n=n_per_class)
                           .sample(frac=1))  # shuffle
    
    return balanced_sample

def sub_sample_data_split(balanced_sample: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
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

    
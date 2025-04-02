import pandas as pd

from src.data_management.create_data_split import (
    DataSplitFile,
    create_balanced_sample,
    sub_sample_data_split,
)
from src.utils.config import (
    AGE_SEX_BALANCED_10K_PATH,
    AGE_SEX_BALANCED_1K_PATH,
    HIGH_QUALITY_FMRI_IDS,
    HIGH_QUALITY_T1_IDS,
    LOW_QUALITY_FMRI_IDS,
    NUM_SAMPLES_10K,
    NUM_SAMPLES_1K,
)
from src.utils.load_targets import extract_targets
from src.utils.subject_selection import load_subject_ids_from_file


def create_balanced_samples():
    hq_fmri_ids = load_subject_ids_from_file(HIGH_QUALITY_FMRI_IDS)
    hq_fmri_targets = extract_targets("sex", hq_fmri_ids)

    # Create balanced sample of 1k subjects
    balanced_ids_1k = create_balanced_sample(hq_fmri_targets, NUM_SAMPLES_1K)

    # Split into training, validation and test set
    train, val, test = sub_sample_data_split(balanced_ids_1k)
    check_split_results(train, val, test)

    # Save to file
    split_1k = DataSplitFile(AGE_SEX_BALANCED_1K_PATH)
    split_1k.save_data_splits_to_file(
        {
            "train": train.index.tolist(),
            "val": val.index.tolist(),
            "test": test.index.tolist(),
        }
    )

    # For the 10k sample, we have don't have enough subjects (only 3604 male subjects)
    # Therefore we draw 3604*2 = 7208 subjects from the HQ data and the rest from the non-HQ data
    n_hq_samples = 3604 * 2
    partial_hq_draw = create_balanced_sample(hq_fmri_targets, n_hq_samples)

    # load the high quality sMRI indices and the low quality fMRI indices
    hq_smri_ids = load_subject_ids_from_file(HIGH_QUALITY_T1_IDS)
    hq_smri_targets = extract_targets("sex", hq_smri_ids)
    lq_fmri_ids = load_subject_ids_from_file(LOW_QUALITY_FMRI_IDS)
    lq_fmri_targets = extract_targets("sex", lq_fmri_ids)
    # create overlap between the high quality sMRI and low quality fMRI
    overlap = get_overlapping_subjects(hq_smri_targets, lq_fmri_targets)

    # remove the subjects that are already in the hq_draw
    overlap = overlap[~overlap.index.isin(partial_hq_draw.index)]
    second_draw = create_balanced_sample(overlap, NUM_SAMPLES_10K - n_hq_samples)
    balanced_ids_10k = pd.concat([partial_hq_draw, second_draw])

    # Split into training, validation and test set
    train, val, test = sub_sample_data_split(balanced_ids_10k)
    check_split_results(train, val, test)

    # Save to file
    split_10k = DataSplitFile(AGE_SEX_BALANCED_10K_PATH)
    split_10k.save_data_splits_to_file(
        {
            "train": train.index.tolist(),
            "val": val.index.tolist(),
            "test": test.index.tolist(),
        }
    )


def get_overlapping_subjects(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Get subjects that appear in both series with matching labels."""
    # Find common indices
    common_indices = series1.index.intersection(series2.index)
    
    # Get values for common indices
    s1_overlap = series1[common_indices]
    s2_overlap = series2[common_indices]
    
    # Verify labels match
    if not (s1_overlap == s2_overlap).all():
        raise ValueError("Labels don't match for overlapping subjects")
        
    return s1_overlap


def check_split_results(train: pd.Series, val: pd.Series, test: pd.Series) -> None:
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
    create_balanced_samples()

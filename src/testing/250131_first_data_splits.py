import pandas as pd

from src.data_management.create_data_split import DataSplitFile, create_balanced_sample, sub_sample_data_split
from src.utils.config import AGE_SEX_BALANCED_10K_PATH, AGE_SEX_BALANCED_1K_PATH, HIGH_QUALITY_FMRI_IDS, HIGH_QUALITY_SMRI_IDS, NUM_SAMPLES_10K, NUM_SAMPLES_1K
from src.utils.load_targets import extract_target
from src.utils.subject_selection import load_subject_ids_from_file


def create_balanced_samples():
    hq_ids = load_subject_ids_from_file(HIGH_QUALITY_FMRI_IDS)
    hq_targets = extract_target("sex", hq_ids)

    # Create balanced sample of 1k subjects
    balanced_ids_1k = create_balanced_sample(hq_targets, NUM_SAMPLES_1K)

    # Split into training, validation and test set
    train, val, test = sub_sample_data_split(balanced_ids_1k)
    print_split_results(train, val, test)

    # Save to file
    split_1k = DataSplitFile(AGE_SEX_BALANCED_1K_PATH)
    split_1k.save_data_splits_to_file({
        "train": train.index.tolist(),
        "val": val.index.tolist(),
        "test": test.index.tolist()
    })

    # For the 10k sample, we have don't have enough subjects (only 3604 male subjects)
    # Therefore we draw 3604*2 = 7208 subjects from the HQ data and the rest from the non-HQ data
    n_hq_samples = 3604 * 2
    other_ids = load_subject_ids_from_file(HIGH_QUALITY_SMRI_IDS)
    other_targets = extract_target("sex", other_ids)
    hq_draw = create_balanced_sample(hq_targets, n_hq_samples)

    # remove the subjects that are already in the hq_draw
    other_targets = other_targets[~other_targets.index.isin(hq_draw.index)]
    other_draw = create_balanced_sample(other_targets, NUM_SAMPLES_10K - n_hq_samples)
    balanced_ids_10k = pd.concat([hq_draw, other_draw])

    # Split into training, validation and test set
    train, val, test = sub_sample_data_split(balanced_ids_10k)
    print_split_results(train, val, test)

    # Save to file
    split_10k = DataSplitFile(AGE_SEX_BALANCED_10K_PATH)
    split_10k.save_data_splits_to_file({
        "train": train.index.tolist(),
        "val": val.index.tolist(),
        "test": test.index.tolist()
    })


def print_split_results(train: pd.Series, val: pd.Series, test: pd.Series) -> None:
    print(f"Set sizes: train {len(train)}, val {len(val)}, test {len(test)}")
    print(f"Label distribution (training): \n{train.value_counts()}")
    print(f"Label distribution (validation): \n{val.value_counts()}")
    print(f"Label distribution (test): \n{test.value_counts()}")

if __name__ == "__main__":
    create_balanced_samples()
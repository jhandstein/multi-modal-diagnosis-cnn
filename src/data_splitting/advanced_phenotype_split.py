from pathlib import Path
import pandas as pd
from src.data_splitting.create_data_split import (
    DataSplitFile,
    create_balanced_sample,
    sub_sample_data_split,
)
from src.data_splitting.subject_selection import load_subject_ids_from_file
from src.data_splitting.load_targets import extract_targets
from src.utils.config import FAULTY_SAMPLE_IDS


class PhenotypeSampler:

    def __init__(self, target: str):

        self.target = target
        self.subject_ids = load_subject_ids_from_file()
        self.target_values = extract_targets(target, self.subject_ids)
        self._strip_nan()

    def _strip_nan(self):
        """Strip NaN values and faulty samples from the target values."""
        # 7777 / 8888 / 9999 is used as a placeholder for NaN values in the NAKO table
        self.target_values = self.target_values[
            ~self.target_values.isin([7777, 8888, 9999])
        ]
        # remove faulty sample IDs
        self.target_values = self.target_values[
            ~self.target_values.index.isin(FAULTY_SAMPLE_IDS)
        ]
        return self

    def sample_binary_dataset(self):
        """Sample a balanced dataset for binary classification tasks."""
        minority_class_count = self.target_values.value_counts().min()
        subject_series = create_balanced_sample(
            self.target_values, 2 * minority_class_count
        )
        print(
            f"Sampled {len(subject_series)} subjects for target '{self.target}' with balanced classes."
        )
        print(subject_series.value_counts())
        return subject_series

    def split_and_save_binary_sample(
        self, sample_subjects: pd.Series, save_path: Path
    ) -> None:
        """Subsample the balanced sample into training, validation, and test sets, and save the splits to a file."""
        train_subjects, val_subjects, test_subjects = sub_sample_data_split(
            sample_subjects
        )
        print(f"Train set size: {len(train_subjects)}")
        print(f"Validation set size: {len(val_subjects)}")
        print(f"Test set size: {len(test_subjects)}")
        # Save the splits to files or return them as needed

        data_split_file = DataSplitFile(save_path)
        data_split_file.save_data_splits_to_file(
            {
                "train": train_subjects.index.tolist(),
                "val": val_subjects.index.tolist(),
                "test": test_subjects.index.tolist(),
            }
        )

    def check_sample_age_sex_distribution(self, sample_subjects: pd.Series):
        """Check the age and the sex distribution of the sampled subjects."""
        sample_ids = sample_subjects.index.tolist()
        age_series = extract_targets("age", sample_ids)
        sex_series = extract_targets("sex", sample_ids)
        # Bin ages into decades
        bins = range(10, 81, 10)  # Adjust start/end as needed
        age_binned = pd.cut(age_series, bins=bins, right=False)
        print(
            f"Sampled age distribution (binned):\n{age_binned.value_counts().sort_index()}"
        )
        print(f"Sampled sex distribution:\n{sex_series.value_counts()}")


if __name__ == "__main__":
    print("Hello from advanced_phenotype_split.py")

    # Example usage of extract_targets
    subject_ids = load_subject_ids_from_file()
    phq9_cutoff_counts = extract_targets("phq9_cutoff", subject_ids).value_counts()
    print(phq9_cutoff_counts)

    gad7_cutoff_counts = extract_targets("gad7_cutoff", subject_ids).value_counts()
    print(gad7_cutoff_counts)

    # Further implementation of PhenotypeSampler would go here

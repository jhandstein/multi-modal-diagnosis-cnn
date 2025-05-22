import json
from pathlib import Path
import pandas as pd

from src.data_management.create_data_split import (
    sample_max_minority_class,
    stratified_binary_split,
    DataSplitFile
)
from src.utils.config import HIGH_QUALITY_IDS, LOW_QUALITY_IDS, MEDIUM_QUALITY_IDS, METRICS_CSV_PATH, QUALITY_SPLITS_PATH
from src.utils.load_targets import extract_targets

# https://mriqc.readthedocs.io/en/latest/iqms/t1w.html
# https://mriqc.readthedocs.io/en/latest/iqms/bold.html


class QualityAnalyzer:
    """
    Analyzes MRI quality metrics and separates subjects into groups based on their quality scores.

    Args:
        metrics_csv_path (Path): Path to the CSV file containing quality metrics.
        metrics_to_process (list[tuple[str, bool]]): List of tuples where each tuple contains a metric key and a boolean indicating if higher values are better.
    """

    QC_METRICS_MAP = {
        "euler_number": "freesurfer_euler_total",
        "snr": "mriqc_T1w_snr_total",
        "cnr": "freesurfer_cnr_total_mean",  # mriqc_T1w_cnr as alternative
        "fd": "mriqc_bold_fd_mean",  # framewise displacement
        "dvars": "xcpd_dvars_after",
    }

    def __init__(
        self, metrics_csv_path: Path, metrics_to_process: list[tuple[str, bool]] = None
    ):
        self.metrics_to_process = metrics_to_process or []
        self.df = pd.read_csv(metrics_csv_path, index_col=0)
        self.original_df = self.df.copy()  # Keep a copy for final stats

    def _get_metric_col_name(self, metric_key: str) -> str:
        """Helper to get the actual column name for a metric key."""
        return self.QC_METRICS_MAP[metric_key]

    def print_statistics(self, metrics_keys: list[str]):
        """Print descriptive statistics for given metric keys."""
        for metric_key in metrics_keys:
            col_name = self._get_metric_col_name(metric_key)
            print(f"\nStatistics for {metric_key} ({col_name}):")
            print(self.df[col_name].describe(), "\n")
            for p in [0.99, 0.95, 0.90]:
                print(f"  Quantile {p*100:.0f}%: {self.df[col_name].quantile(p)}")

    def remove_outliers(
        self,
        metric_key: str,
        good_is_high: bool = True,
        lower_quantile=0.01,
        upper_quantile=0.99,
    ):
        """
        Remove outliers for a given metric, targeting the worst values.
        If good_is_high is True (high values are good), values below lower_quantile (worst) are removed.
        If good_is_high is False (low values are good), values above upper_quantile (worst) are removed.
        """
        col_name = self._get_metric_col_name(metric_key)
        if good_is_high:
            # High values are good, so remove the worst (lowest) values
            self.df = self.df[
                self.df[col_name] >= self.df[col_name].quantile(lower_quantile)
            ]
        else:
            # Low values are good, so remove the worst (highest) values
            self.df = self.df[
                self.df[col_name] <= self.df[col_name].quantile(upper_quantile)
            ]

    def remove_metric_nans(self, metric_key: str):
        """Remove rows with NaN values for a given metric."""
        col_name = self._get_metric_col_name(metric_key)
        self.df.dropna(subset=[col_name], inplace=True)

    def invert_metric_if_needed(self, metric_key: str, lower_is_better: bool):
        """Invert the metric if lower values are better."""
        if lower_is_better:
            col_name = self._get_metric_col_name(metric_key)
            self.df[col_name] = self.df[col_name].max() - self.df[col_name]

    def normalize_metric_by_rank(self, metric_key: str):
        """Normalize the metric using percentile rank."""
        col_name = self._get_metric_col_name(metric_key)
        self.df[col_name] = self.df[col_name].rank(pct=True)

    def create_combined_metric(
        self,
        metric_keys: list[str],
        weights: list[float] = None,
        combined_metric_name: str = "combined_metric",
    ):
        """
        Create a combined metric by summing the normalized metrics, optionally with weights.
        Assumes metrics are already normalized and inverted if necessary.
        """
        if weights and len(metric_keys) != len(weights):
            raise ValueError("Number of metric keys and weights must match.")

        if not weights:
            weights = [1.0] * len(metric_keys)  # Equal weights if not provided

        self.df[combined_metric_name] = 0
        for i, metric_key in enumerate(metric_keys):
            col_name = self._get_metric_col_name(metric_key)
            self.df[combined_metric_name] += self.df[col_name] * weights[i]

        # Normalize the combined metric itself
        self.df[combined_metric_name] = self.df[combined_metric_name].rank(pct=True)

    def split_data_into_groups(
        self, by_metric_name: str = "combined_metric"
    ) -> dict[str, list]:
        """Split data into three equally sized groups ("low", "medium", "high") based on a specified metric."""
        self.df = self.df.sort_values(by=by_metric_name, ascending=True)

        group_names = ["low", "medium", "high"]
        n_total = len(self.df)
        group_size = n_total // len(group_names)

        split_indices = {}

        for i, group_name in enumerate(group_names):
            start_idx = i * group_size
            # For the last group, include any remainder rows
            end_idx = n_total if i == 2 else (i + 1) * group_size
            split_indices[group_name] = list(self.df.iloc[start_idx:end_idx].index)

        return split_indices

    def processing_pipeline(
        self, combined_metric_name: str = "combined_metric"
    ) -> dict[str, list]:
        """
        Runs the full processing pipeline.

        1. Remove NaNs
        2. Remove outliers with the worst values
        3. Invert metrics if lower is better
        4. Normalize the metrics
        5. Create combined metric
        6. Split data into (three) groups
        """
        # Print initial statistics for relevant metrics
        self.print_statistics([m[0] for m in self.metrics_to_process])
        print("\n--- Initial DataFrame shape ---")
        print(f"Initial shape: {self.df.shape}")

        for metric_key, higher_is_better in self.metrics_to_process:
            print(f"\nProcessing metric: {metric_key}")

            # 1. Remove NaNs
            self.remove_metric_nans(metric_key)
            print(f"Shape after NaN removal for {metric_key}: {self.df.shape}")

            # 2. Remove outliers
            self.remove_outliers(metric_key, good_is_high=higher_is_better)
            print(f"Shape after outlier removal for {metric_key}: {self.df.shape}")

            # 3. Invert metrics if lower is better
            self.invert_metric_if_needed(
                metric_key, lower_is_better=not higher_is_better
            )

            # 4. Normalize the metric
            self.normalize_metric_by_rank(metric_key)

        # 5. Create combined metric
        self.create_combined_metric(
            [m[0] for m in self.metrics_to_process],
            combined_metric_name=combined_metric_name,
        )

        # 6. Split data
        results = self.split_data_into_groups(by_metric_name=combined_metric_name)

        return results

    def get_group_statistics(
        self, split_results: dict[str, list], original_metric_keys: list[str]
    ) -> pd.DataFrame:
        """Calculates statistics for each group based on the original (unprocessed) data and returns a DataFrame."""
        stats_data = []

        # Iterate through each group
        for group_name, subject_ids in split_results.items():
            if not subject_ids:
                continue

            group_df = self.original_df.loc[subject_ids]
            # Iterate through each metric key
            for metric_key in original_metric_keys:
                col_name = self._get_metric_col_name(metric_key)
                if col_name in group_df.columns:
                    # Get descriptive statistics
                    desc_stats = group_df[col_name].describe().to_dict()

                    # Create a single row for this group and metric
                    row_data = {
                        "group": group_name,
                        "metric": metric_key,
                        "column": col_name,
                        "n_subjects": len(subject_ids),
                        "mean": desc_stats.get("mean"),
                        "std": desc_stats.get("std"),
                        "min": desc_stats.get("min"),
                        "q1": desc_stats.get("25%"),
                        "median": desc_stats.get("50%"),
                        "q3": desc_stats.get("75%"),
                        "max": desc_stats.get("max"),
                    }
                    # Round float values to 4 decimal places
                    for key, val in row_data.items():
                        if isinstance(val, float):
                            row_data[key] = round(val, 4)
                    stats_data.append(row_data)

        # Create DataFrame from collected statistics
        stats_df = pd.DataFrame(stats_data)
        # Order the DataFrame by metric and group
        stats_df = stats_df.sort_values(by=["metric"])
        return stats_df

    def save_ids_to_json(self, split_results: dict[str, list]):
        """Saves the subject IDs for each group to a JSON file."""
        # print(list(split_results.keys()))
        output_path = Path(
            "/home/julius/repositories/ccn_code/src/data_management",
            f"quality_split_{'_'.join([m[0] for m in self.metrics_to_process])}.json",
        )
        with open(output_path, "w") as f:
            json.dump(split_results, f, indent=4)
        print(f"Group IDs saved to {output_path}")


class QualitySampler:
    """
    Class to sample subjects from the previously determined quality groups.
    """

    def __init__(self, quality_split_results: dict[str, list]):
        self.quality_split_results = quality_split_results
        self.lq_labels: pd.Series = extract_targets("sex", quality_split_results["low"])
        self.mq_labels: pd.Series = extract_targets(
            "sex", quality_split_results["medium"]
        )
        self.hq_labels: pd.Series = extract_targets(
            "sex", quality_split_results["high"]
        )
        self.final_split_results = {}

    def balance_quality_groups(self) -> dict[str, list]:
        """
        Balance the quality groups by sampling the same number of subjects from each group.
        The number of samples is determined by the minimum number of samples across all groups.
        The sampled subjects are then split into training, validation, and test sets in the ratio of 6:1:1.
        Labels are as balanced as possible.
        """

        RANDOM_SEED = 42

        # Find minimum number of samples across all groups and then find the highest number divisible by 8
        num_samples = min(len(self.lq_labels), len(self.mq_labels), len(self.hq_labels))
        num_samples = num_samples - (num_samples % 8)
        print(f"Number of samples per group: {num_samples}")

        # Create a dictionary to hold the sampled IDs for each group

        # Sample from each group
        for group_name, label_series in zip(
            ["low", "medium", "high"], [self.lq_labels, self.mq_labels, self.hq_labels]
        ):

            sampled_ids = sample_max_minority_class(label_series, num_samples, random_seed=RANDOM_SEED)

            print(f"Sampled {len(sampled_ids)} subjects from {group_name} quality group")

            train, val_test = stratified_binary_split(sampled_ids, split_ratio=6 / 8, random_seed=RANDOM_SEED)
            val, test = stratified_binary_split(val_test, split_ratio=0.5, random_seed=RANDOM_SEED)

            print(f"  {len(train)} train, {len(val)} val, {len(test)} test samples")

            # Store the sampled IDs in the final split results
            self.final_split_results[group_name] = {
                "train": train.tolist(),
                "val": val.tolist(),
                "test": test.tolist(),
            }

    def save_data_splits_to_file(self):
        """Saves the pre-defined data splits to multiple JSON files."""
        if not self.final_split_results:
            raise ValueError("No data splits available. Please run balance_quality_groups() first.")
        for group_name, splits in self.final_split_results.items():
            if group_name == "low":
                file_path = LOW_QUALITY_IDS
            elif group_name == "medium":
                file_path = MEDIUM_QUALITY_IDS
            elif group_name == "high":
                file_path = HIGH_QUALITY_IDS
            else:
                raise ValueError(f"Unknown group name: {group_name}")
            
            file = DataSplitFile(file_path)
            file.save_data_splits_to_file(splits)



if __name__ == "__main__":
    # Works only when run from the root directory of the project

    # Define metrics to process: (metric_key, higher_is_better)
    # For 'framewise_displacement': good quality is low, so for outlier removal, we treat 'good' as 'low' (i.e., good_is_high=False)
    # and it needs inversion because lower values are better.
    # For 'cnr': good quality is high, so for outlier removal, 'good' is 'high' (good_is_high=True) and it does not need inversion.
    metrics_config = [
        (
            "fd",
            False,
        ),  # good_is_high=False means we remove low outliers (0.01 percentile)
        (
            "cnr",
            True,
        ),  # good_is_high=True means we remove high outliers (0.99 percentile)
    ]
    analyzer = QualityAnalyzer(
        metrics_csv_path=METRICS_CSV_PATH, metrics_to_process=metrics_config
    )

    split_group_ids = analyzer.processing_pipeline()

    # Print statistics for the groups using the original, unprocessed values of the metrics
    results_df = analyzer.get_group_statistics(
        split_results=split_group_ids,
        original_metric_keys=[m[0] for m in metrics_config],
    )
    print("\n--- Group Statistics DataFrame ---")
    print(results_df)

    # Save the group IDs to a JSON file
    analyzer.save_ids_to_json(split_results=split_group_ids)

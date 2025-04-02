from pathlib import Path
import pandas as pd

from src.data_management.data_set import NakoMultiModalityDataset, NakoSingleModalityDataset


def format_metrics_file(csv_path: Path) -> pd.DataFrame:
    """Process metrics CSV file to combine matching epochs and round values.

    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save processed CSV

    Returns:
        Processed DataFrame
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Group by epoch and step, combining all metrics
    df = df.groupby(["epoch", "step"]).first().reset_index()
    df = df.drop(columns=["step"])

    # Round all numeric columns to 4 decimals
    numeric_cols = df.select_dtypes(include=["float64"]).columns
    # exclude columns that are not meant to be rounded like learning_rate
    numeric_cols = numeric_cols[~numeric_cols.str.contains("learning_rate")]
    df[numeric_cols] = df[numeric_cols].round(4)
    df['learning_rate'] = df['learning_rate'].apply(lambda x: f'{x:.3e}')

    # Save if output path provided
    out_path = csv_path.parent / f"{csv_path.stem}_formatted.csv"
    df.to_csv(out_path, index=False)

    return df


def get_modality_and_features(dataset: NakoSingleModalityDataset | NakoMultiModalityDataset):
    """Extract modality and feature information from both single and multi-modality datasets.
    
    Args:
        dataset: Either SingleModalityDataset or MultiModalityDataset
        
    Returns:
        tuple: (modality_label, feature_dict)
        - modality_label: str - either specific modality or "anat-func"
        - feature_dict: dict - contains feature maps based on dataset type
    """
    if hasattr(dataset, "feature_maps"):  # SingleModalityDataset
        return (
            dataset.feature_maps[0].modality_label,
            {"feature_maps": [fm.label for fm in dataset.feature_maps]}
        )
    else:  # MultiModalityDataset
        return (
            "anat-func",
            {
                "anatomical_maps": [fm.label for fm in dataset.anatomical_maps],
                "functional_maps": [fm.label for fm in dataset.functional_maps]
            }
        )

from pathlib import Path
import pandas as pd


def process_metrics_file(csv_path: Path, output_path: Path = None) -> pd.DataFrame:
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
    df = df.groupby(['epoch', 'step']).first().reset_index()
    df = df.drop(columns=['step'])
    
    # Round all numeric columns to 4 decimals
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df
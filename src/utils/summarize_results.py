from pathlib import Path
import json
from typing import Literal

import pandas as pd


# EXPERIMENTS_DIR = Path("/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_250710_final_results/27")

PARENT_FOLDER_2D = Path(
    "/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_250710_final_results"
)

PARENT_FOLDER_3D = Path(
    "/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_250710_final_results/_3D"
)


def main():
    # Specify the seeds you want to aggregate
    seeds = [42, 404, 1312] if "3D" in str(PARENT_FOLDER_2D) else [27, 42, 404, 1312, 1984]
    all_seed_results = []
    for seed in seeds:
        print(f"Loading results for seed {seed}...")
        all_seed_results.append(load_seed_data(seed))

    # Combine results for classification and regression
    classification_df = combine_seed_results(all_seed_results, "classification")
    regression_df = combine_seed_results(all_seed_results, "regression")

    # Print or save the results
    print("Classification Results:")
    print(classification_df)
    print("\nRegression Results:")
    print(regression_df)

    # Summarize by feature map
    classification_summary = summarize_by_feature_map(classification_df)
    regression_summary = summarize_by_feature_map(regression_df)
    # Prettify the DataFrames
    classification_summary = prettify_df(classification_summary, "classification")
    regression_summary = prettify_df(regression_summary, "regression")

    print("\nClassification Summary by Feature Map:")
    print(classification_summary)
    print("\nRegression Summary by Feature Map:")
    print(regression_summary)

    # Optionally save to CSV:
    # classification_df.to_csv("classification_results.csv", index=False)
    # regression_df.to_csv("regression_results.csv", index=False)

def merge_2d_3d_tables(summary_2d: pd.DataFrame, summary_3d: pd.DataFrame) -> pd.DataFrame:
    """Merge 2D and 3D summary tables side by side with MultiIndex columns."""
    summary_2d.columns = pd.MultiIndex.from_product([["2D"], summary_2d.columns])
    summary_3d.columns = pd.MultiIndex.from_product([["3D"], summary_3d.columns])
    merged = pd.concat([summary_2d, summary_3d], axis=1)
    return merged

def print_latex_table(merged_df: pd.DataFrame, caption: str, label: str):
    """Prints a LaTeX table from the merged DataFrame."""
    latex = merged_df.to_latex(
        multicolumn=True,
        multirow=False,
        caption=caption,
        label=label,
        escape=False,
        column_format="l|" + "c" * merged_df.shape[1],
        bold_rows=True,
        na_rep="--"
    )
    print(latex)

def prettify_df(
    df: pd.DataFrame, task_type: Literal["classification", "regression"]
) -> pd.DataFrame:
    """Prettify the DataFrame by rounding and formatting."""
    # Round all float columns to three decimals
    df = df.round(3)

    # # If task_type is regression, round "test_loss" to two decimals
    # if task_type == "regression":
    #     # Round and remove trailing zeros for the 'MSELoss' row
    #     df.loc["test_loss"] = df.loc["test_loss"].apply(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else x)

    index_labels = {
        "test_loss": "MSELoss" if task_type == "regression" else "BCELoss",
        "test_accuracy": "Accuracy",
        "test_auc": "AUC",
        "test_f1": "F1",
        "test_mae": "MAE",
        "test_r2": "R2",
        "test_spearman": "Spearman",
    }

    column_labels = {
        "T1": "T1",
        "GM_WM_CSF": "GM/WM/CSF",
        "bold": "BOLD",
        "GM_WM_CSF_bold": "GM/WM/CSF/BOLD",

    }

    if "phq9" in str(PARENT_FOLDER_2D) or "gad7" in str(PARENT_FOLDER_2D):
        # Add additional feature maps for PHQ-9 or GAD-7
        column_labels.update({
        "reho": "ReHo",
        "T1_bold": "T1/BOLD",
        })

    # Rename columns for better readability
    df = df.rename(index=index_labels)
    df = df.rename(columns=column_labels)

    index_order = (
        ["BCELoss", "Accuracy", "AUC", "F1"]
        if task_type == "classification"
        else ["MSELoss", "MAE", "R2", "Spearman"]
    )
    df = df.reindex(index_order)

    #  # Only format the MSELoss row after renaming
    # if task_type == "regression":
    #     df.loc["MSELoss"] = df.loc["MSELoss"].apply(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else x)
    return df


def summarize_by_feature_map(df: pd.DataFrame) -> pd.DataFrame:
    """Group by 'feature_maps' and average all numeric columns."""
    # Exclude 'seed' column from averaging if present
    cols_to_average = [col for col in df.columns if col not in ["feature_maps", "seed"]]
    summary = (
        df.groupby("feature_maps", as_index=False)[cols_to_average].mean()
        # .round(3)
    )
    # Set "feature_maps" as columns and metrics as rows
    summary = summary.set_index("feature_maps").T
    summary.columns.name = "Metric"  # <-- Add this line to remove 'feature_maps' label
    # Sort the columns as follows:
    order = ["T1", "GM_WM_CSF", "bold", "GM_WM_CSF_bold"]
    if "phq9" in str(PARENT_FOLDER_2D) or "gad7" in str(PARENT_FOLDER_2D):
        # Add additional feature maps for PHQ-9 or GAD-7
        order.extend(["reho", "T1_bold"])
    summary = summary[order]
    return summary


def combine_seed_results(
    seed_results_list: list, task_type: Literal["classification", "regression"]
) -> pd.DataFrame:
    """Combine results from all seeds for a specific task type into a single DataFrame."""
    combined_results = []

    for seed_results in seed_results_list:
        # seed_results[task_type] is a dict: {seed: {fm_string: DataFrame}}
        for seed, experiments in seed_results.get(task_type, {}).items():
            for fm_string, results in experiments.items():
                # Convert the results DataFrame to a dictionary and add seed and feature map string
                result_dict = results.to_dict(orient="records")[0]
                result_dict["seed"] = seed
                result_dict["feature_maps"] = fm_string
                combined_results.append(result_dict)

    # # Every column should be rounded to 3 decimal places except for "test_loss" for regression, which should be rounded to 2 decimal places
    # if task_type == "regression":
    #     for result in combined_results:
    #         if "test_loss" in result:
    #             result["test_loss"] = round(result["test_loss"], 2)
    #         for key in result:
    #             if key.startswith("test_") and key != "test_loss":
    #                 result[key] = round(result[key], 3)
    # else:
    #     for result in combined_results:
    #         for key in result:
    #             if key.startswith("test_"):
    #                 result[key] = round(result[key], 3)

    return pd.DataFrame(combined_results).sort_values(by=["feature_maps", "seed"])


def load_seed_data(seed: int) -> dict:
    """Load all experiment results for a given seed from the parent folder."""
    seed_folder_path = PARENT_FOLDER_2D / str(seed)
    if not seed_folder_path.exists():
        raise FileNotFoundError(f"Seed folder {seed_folder_path} does not exist.")

    seed_results = {}

    for exp_folder in seed_folder_path.iterdir():
        if not exp_folder.is_dir():
            continue
        setup, test_results = load_test_results(exp_folder)

        task_type = setup.get("training_params", {}).get("Task", "Unknown").lower()
        feature_maps = setup.get("training_params", {}).get("Feature Maps", [])
        fm_string = "_".join(feature_maps) if feature_maps else "default"

        seed_results.setdefault(task_type, {}).setdefault(seed, {})[
            fm_string
        ] = test_results

    return seed_results


def load_test_results(experiment_folder: Path) -> dict:
    """Load the experiment setup and test results from the given experiment folder."""
    setup_path = experiment_folder / "version_0" / "experiment_setup.json"
    if setup_path.exists():
        with open(setup_path, "r") as f:
            setup = json.load(f)
            task_type = setup.get("training_params", {}).get("Task", "Unknown")
            # train_set_len = setup.get("dataset_info", {}).get("len_train", "Unknown")
            # val_set_len = setup.get("dataset_info", {}).get("len_val", "Unknown")
            # print(f"Train set length: {train_set_len}, Validation set length: {val_set_len}")
    else:
        raise FileNotFoundError(f"Experiment setup file {setup_path} does not exist.")

    results_path = experiment_folder / "version_0" / "metrics.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        filtered_df = summarize_results(df, task_type)

        # Throw an error if the test_results DataFrame is empty
        if filtered_df.empty:
            raise ValueError(
                f"No test results found for {task_type} task in {experiment_folder}."
            )
        elif len(filtered_df) > 1:
            print(
                f"Warning: More than one row of test results found for {task_type} task in {experiment_folder}. Using the last row."
            )

    return setup, filtered_df


def summarize_results(
    results_df: pd.DataFrame, task_type: Literal["classification", "regression"]
) -> pd.DataFrame:
    """Summarize the results DataFrame for a specific task type."""
    if task_type == "classification":
        metrics = ["loss", "accuracy", "auc", "f1"]
    elif task_type == "regression":
        metrics = ["loss", "mae", "r2", "spearman"]
    else:
        raise ValueError("Invalid task type. Use 'classification' or 'regression'.")

    # Filter the DataFrame for the relevant columns
    filtered_df = results_df.filter(regex=f"test_({'|'.join(metrics)})")
    filtered_df = filtered_df.dropna(how="all")  # Remove rows where all columns are NaN

    return filtered_df


if __name__ == "__main__":
    main()

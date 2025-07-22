from pathlib import Path
import json
from typing import Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PARENT_FOLDER_2D = Path(
    "/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_250710_final_results"
)
PARENT_FOLDER_3D = Path(
    "/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_250710_final_results/_3D"
)

class ResultsSummarizer:
    def __init__(self, parent_folder_2d=PARENT_FOLDER_2D, parent_folder_3d=PARENT_FOLDER_3D):
        self.parent_folder_2d = parent_folder_2d
        self.parent_folder_3d = parent_folder_3d

        self.class_2d = None
        self.reg_2d = None
        self.class_3d = None
        self.reg_3d = None

        self.merged_class = None
        self.merged_reg = None


    def run(self, process_both=True):
        if process_both:
            self.class_2d, self.reg_2d = self.process_parent_folder(self.parent_folder_2d)
            self.class_3d, self.reg_3d = self.process_parent_folder(self.parent_folder_3d)
            self.merged_class = self.merge_2d_3d_tables(self.class_2d, self.class_3d)
            self.merged_reg = self.merge_2d_3d_tables(self.reg_2d, self.reg_3d)
            print("\nMerged Classification Summary (2D vs 3D):")
            print(self.merged_class)
            print("\nMerged Regression Summary (2D vs 3D):")
            print(self.merged_reg)
            print("\nLaTeX Classification Table:")
            self.print_latex_table(
                self.merged_class,
                caption="Classification Summary by Feature Map and Dimensionality",
                label="tab:classification_summary"
            )
            print("\nLaTeX Regression Table:")
            self.print_latex_table(
                self.merged_reg,
                caption="Regression Summary by Feature Map and Dimensionality",
                label="tab:regression_summary"
            )
           
            self.plot_merged_classification_bar(self.merged_class)
        else:
            self.class_2d, self.reg_2d = self.process_parent_folder(self.parent_folder_2d)
            print("Classification Summary:")
            print(self.class_2d)
            print("\nRegression Summary:")
            print(self.reg_2d)
            print("\nLaTeX Classification Table:")
            self.print_latex_table(
                self.class_2d,
                caption="Classification Summary by Feature Map",
                label="tab:classification_summary"
            )
            print("\nLaTeX Regression Table:")
            self.print_latex_table(
                self.reg_2d,
                caption="Regression Summary by Feature Map",
                label="tab:regression_summary"
            )

    def process_parent_folder(self, parent_folder: Path):
        seeds = [42, 404, 1312] if "3D" in str(parent_folder) else [27, 42, 404, 1312, 1984]
        all_seed_results = []
        for seed in seeds:
            print(f"Loading results for seed {seed} in {parent_folder}...")
            all_seed_results.append(self.load_seed_data(seed, parent_folder))
        self.classification_all_seeds = self.combine_seed_results(all_seed_results, "classification")
        self.regression_all_seeds = self.combine_seed_results(all_seed_results, "regression")
        print("\nCombined Results:")
        print("Classification Results:")
        print(self.classification_all_seeds)
        print("\nRegression Results:")
        print(self.regression_all_seeds)
        classification_summary = self.summarize_by_feature_map(self.classification_all_seeds, parent_folder)
        regression_summary = self.summarize_by_feature_map(self.regression_all_seeds, parent_folder)
        classification_summary = self.prettify_df(classification_summary, "classification", parent_folder)
        regression_summary = self.prettify_df(regression_summary, "regression", parent_folder)
        return classification_summary, regression_summary

    def plot_merged_classification_bar(self, merged_class, std_2d=None, std_3d=None):
        metric = "Accuracy"
        feature_maps = merged_class['2D'].columns.tolist()
        n_feature_maps = len(feature_maps)
        bar_width = 0.35
        x = np.arange(n_feature_maps)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        acc_2d = merged_class['2D'].loc[metric].values
        acc_3d = merged_class['3D'].loc[metric].values
        err_2d = std_2d if std_2d is not None else np.zeros_like(acc_2d)
        err_3d = std_3d if std_3d is not None else np.zeros_like(acc_3d)
        ax.bar(x - bar_width/2, acc_2d, width=bar_width, label='2D', color=colors[:n_feature_maps], yerr=err_2d, capsize=5)
        ax.bar(x + bar_width/2, acc_3d, width=bar_width, label='3D', color=colors[:n_feature_maps], alpha=0.6, yerr=err_3d, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_maps)
        ax.set_ylabel('Accuracy')
        ax.set_title('2D vs 3D Classification Accuracy by Feature Map')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def get_metric_std_by_feature_map(self, df: pd.DataFrame, metric: str) -> pd.Series:
        """
        Returns std of the specified metric for each feature map.
        Example: metric="test_accuracy" for classification, metric="test_mae" for regression.
        """
        return df.groupby("feature_maps")[metric].std()

    def merge_2d_3d_tables(self, summary_2d: pd.DataFrame, summary_3d: pd.DataFrame) -> pd.DataFrame:
        summary_2d.columns = pd.MultiIndex.from_product([["2D"], summary_2d.columns])
        summary_3d.columns = pd.MultiIndex.from_product([["3D"], summary_3d.columns])
        merged = pd.concat([summary_2d, summary_3d], axis=1)
        return merged

    def print_latex_table(self, merged_df: pd.DataFrame, caption: str, label: str):
        latex = merged_df.to_latex(
            multicolumn=True,
            multirow=False,
            caption=caption,
            label=label,
            escape=False,
            column_format="l|" + "r" * int(merged_df.shape[1] * 0.5) + "|" + "r" * int(merged_df.shape[1] * 0.5),
            na_rep="--",
            float_format=lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if isinstance(x, float) else x
        )
        print(latex)

    def prettify_df(
        self, df: pd.DataFrame, task_type: Literal["classification", "regression"], parent_folder: Path = PARENT_FOLDER_2D
    ) -> pd.DataFrame:
        df = df.round(3)
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
            "GM_WM_CSF": "MORPH",
            "bold": "BOLD",
            "GM_WM_CSF_bold": "MORPH/BOLD",
        }
        if "phq9" in str(parent_folder) or "gad7" in str(parent_folder):
            column_labels.update({
                "reho": "ReHo",
                "T1_bold": "T1/BOLD",
            })
        df = df.rename(index=index_labels)
        df = df.rename(columns=column_labels)
        index_order = (
            ["BCELoss", "Accuracy", "AUC", "F1"]
            if task_type == "classification"
            else ["MSELoss", "MAE", "R2", "Spearman"]
        )
        df = df.reindex(index_order)
        return df

    def summarize_by_feature_map(self, df: pd.DataFrame, parent_folder: Path = PARENT_FOLDER_2D) -> pd.DataFrame:
        cols_to_average = [col for col in df.columns if col not in ["feature_maps", "seed"]]
        summary = (
            df.groupby("feature_maps", as_index=False)[cols_to_average].mean()
        )
        summary = summary.set_index("feature_maps").T
        summary.columns.name = "Metric"
        order = ["T1", "GM_WM_CSF", "bold", "GM_WM_CSF_bold"]
        if "phq9" in str(parent_folder) or "gad7" in str(parent_folder):
            order.extend(["reho", "T1_bold"])
        summary = summary[order]
        return summary

    def combine_seed_results(
        self, seed_results_list: list, task_type: Literal["classification", "regression"]
    ) -> pd.DataFrame:
        combined_results = []
        for seed_results in seed_results_list:
            for seed, experiments in seed_results.get(task_type, {}).items():
                for fm_string, results in experiments.items():
                    result_dict = results.to_dict(orient="records")[0]
                    result_dict["seed"] = seed
                    result_dict["feature_maps"] = fm_string
                    combined_results.append(result_dict)
        return pd.DataFrame(combined_results).sort_values(by=["feature_maps", "seed"])

    def load_seed_data(self, seed: int, parent_folder: Path = PARENT_FOLDER_2D) -> dict:
        seed_folder_path = parent_folder / str(seed)
        if not seed_folder_path.exists():
            raise FileNotFoundError(f"Seed folder {seed_folder_path} does not exist.")
        seed_results = {}
        for exp_folder in seed_folder_path.iterdir():
            if not exp_folder.is_dir():
                continue
            setup, test_results = self.load_test_results(exp_folder)
            task_type = setup.get("training_params", {}).get("Task", "Unknown").lower()
            feature_maps = setup.get("training_params", {}).get("Feature Maps", [])
            fm_string = "_".join(feature_maps) if feature_maps else "default"
            seed_results.setdefault(task_type, {}).setdefault(seed, {})[
                fm_string
            ] = test_results
        return seed_results

    def load_test_results(self, experiment_folder: Path) -> dict:
        setup_path = experiment_folder / "version_0" / "experiment_setup.json"
        taining_time_path = experiment_folder / "version_0" / "training_progress.json"
        if setup_path.exists():
            with open(setup_path, "r") as f:
                setup = json.load(f)
                task_type = setup.get("training_params", {}).get("Task", "Unknown")
                node = setup.get("training_params", {}).get("Compute Node", "Unknown")
                num_gpus = setup.get("training_params", {}).get("Num GPUs", "Unknown")
                dim = setup.get("training_params", {}).get("Data Dimension", "Unknown")
                fms = setup.get("training_params", {}).get("Feature Maps", [])
            with open(taining_time_path, "r") as f:
                training_time = json.load(f)
                training_time = training_time.get("timing", "Unknown").get("total_duration_hours", "Unknown")
        else:
            raise FileNotFoundError(f"Experiment setup file {setup_path} does not exist.")
        results_path = experiment_folder / "version_0" / "metrics.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            filtered_df = self.summarize_results(df, task_type)
            if filtered_df.empty:
                raise ValueError(
                    f"No test results found for {task_type} task in {experiment_folder}."
                )
            elif len(filtered_df) > 1:
                print(
                    f"Warning: More than one row of test results found for {task_type} task in {experiment_folder}. Using the last row."
                )
        print(f"Training time on {num_gpus} GPUs on node {node}: {training_time}. (Dimensionality: {dim}, Feature Maps: {fms})")
        return setup, filtered_df

    def summarize_results(
        self, results_df: pd.DataFrame, task_type: Literal["classification", "regression"]
    ) -> pd.DataFrame:
        if task_type == "classification":
            metrics = ["loss", "accuracy", "auc", "f1"]
        elif task_type == "regression":
            metrics = ["loss", "mae", "r2", "spearman"]
        else:
            raise ValueError("Invalid task type. Use 'classification' or 'regression'.")
        filtered_df = results_df.filter(regex=f"test_({'|'.join(metrics)})")
        filtered_df = filtered_df.dropna(how="all")
        return filtered_df

if __name__ == "__main__":
    summarizer = ResultsSummarizer()
    summarizer.run(process_both=True)

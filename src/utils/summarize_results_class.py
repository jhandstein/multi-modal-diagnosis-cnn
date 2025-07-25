from pathlib import Path
import json
from typing import Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # <-- add this import

PARENT_FOLDER_2D = Path(
    "/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_250710_final_results"
)
PARENT_FOLDER_3D = Path(
    "/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_250710_final_results/_3D"
)

PLOT_FOLDER = Path(
    "/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/ccn-code/plots"
)

class ResultsSummarizer:
    def __init__(self, parent_folder_2d=PARENT_FOLDER_2D, parent_folder_3d=PARENT_FOLDER_3D):
        self.parent_folder_2d = parent_folder_2d
        self.parent_folder_3d = parent_folder_3d

        # Ensure plot folder exists
        PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

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

            # Calculate and plot standard deviations for 2D and 3D results
            # generic_title = f"(2D vs. 3D)"

            # std_acc_2d = self.get_metric_std_by_feature_map(self.classification_all_seeds_2d, "test_accuracy")
            # std_acc_3d = self.get_metric_std_by_feature_map(self.classification_all_seeds_3d, "test_accuracy")
            # print("\nStandard Deviation of Test Accuracy by Feature Map:")
            # print("2D:", std_acc_2d)
            # print("3D:", std_acc_3d)
            # self.plot_merged_bar(self.merged_class, metric="Accuracy", std_2d=std_acc_2d, std_3d=std_acc_3d, ylabel="Accuracy", title=f"Final results for Sex Classification {generic_title}")

            # std_mae_2d = self.get_metric_std_by_feature_map(self.regression_all_seeds_2d, "test_mae")
            # std_mae_3d = self.get_metric_std_by_feature_map(self.regression_all_seeds_3d, "test_mae")
            # print("\nStandard Deviation of Test MAE by Feature Map:")
            # print("2D:", std_mae_2d)
            # print("3D:", std_mae_3d)
            # self.plot_merged_bar(self.merged_reg, metric="MAE", std_2d=std_mae_2d, std_3d=std_mae_3d, ylabel="MAE", title=f"Final results for Age Regression {generic_title}")

            # Print std of all metrics by feature map
            print("\nStandard Deviation of All Classification Metrics by Feature Map (2D):")
            std_2d = self.get_all_metrics_std_by_feature_map(self.classification_all_seeds_2d)
            print(std_2d)
            print("\nStandard Deviation of All Classification Metrics by Feature Map (3D):")
            std_3d = self.get_all_metrics_std_by_feature_map(self.classification_all_seeds_3d)
            print(std_3d)

            print("\nStandard Deviation of All Regression Metrics by Feature Map (2D):")
            std_2d_reg = self.get_all_metrics_std_by_feature_map(self.regression_all_seeds_2d)
            print(std_2d_reg)
            print("\nStandard Deviation of All Regression Metrics by Feature Map (3D):")
            std_3d_reg = self.get_all_metrics_std_by_feature_map(self.regression_all_seeds_3d)
            print(std_3d_reg)



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
        dim = "3D" if "3D" in str(parent_folder) else "2D"
        seeds = [42, 404, 1312] if dim in str(parent_folder) else [27, 42, 404, 1312, 1984]
        all_seed_results = []
        for seed in seeds:
            print(f"Loading results for seed {seed} in {parent_folder}...")
            all_seed_results.append(self.load_seed_data(seed, parent_folder))
        classification_all_seeds = self.combine_seed_results(all_seed_results, "classification")
        regression_all_seeds = self.combine_seed_results(all_seed_results, "regression")

        if dim == "2D":
            self.classification_all_seeds_2d = classification_all_seeds
            self.regression_all_seeds_2d = regression_all_seeds
        else:
            self.classification_all_seeds_3d = classification_all_seeds
            self.regression_all_seeds_3d = regression_all_seeds
        print("\nCombined Results:")
        print("Classification Results:")
        print(classification_all_seeds)
        print("\nRegression Results:")
        print(regression_all_seeds)
        classification_summary = self.summarize_by_feature_map(classification_all_seeds, parent_folder)
        regression_summary = self.summarize_by_feature_map(regression_all_seeds, parent_folder)
        classification_summary = self.prettify_df(classification_summary, "classification", parent_folder)
        regression_summary = self.prettify_df(regression_summary, "regression", parent_folder)
        return classification_summary, regression_summary

    def plot_merged_bar(self, merged_df, metric="Accuracy", std_2d=None, std_3d=None, ylabel="Accuracy", title="2D vs 3D by Feature Map"):
        feature_maps = merged_df['2D'].columns.tolist()
        n_feature_maps = len(feature_maps)
        bar_width = 0.35
        x = np.arange(n_feature_maps)
        fig, ax = plt.subplots(figsize=(8, 5))
        # Use Dark2 colormap for colors
        dark2 = cm.get_cmap('Dark2')
        colors = [dark2(i) for i in range(n_feature_maps)]
        values_2d = merged_df['2D'].loc[metric].values
        values_3d = merged_df['3D'].loc[metric].values
        err_2d = std_2d if std_2d is not None else np.zeros_like(values_2d)
        err_3d = std_3d if std_3d is not None else np.zeros_like(values_3d)
        ax.bar(x - bar_width/2, values_2d, width=bar_width, label='2D', color=colors, yerr=err_2d, capsize=5)
        ax.bar(x + bar_width/2, values_3d, width=bar_width, label='3D', color=colors, alpha=0.6, yerr=err_3d, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_maps)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        # Save plot to PLOT_FOLDER
        filename = f"{title.replace(' ', '_')}_{metric}.png"
        plt.savefig(PLOT_FOLDER / filename)
        plt.close(fig)

    def get_metric_std_by_feature_map(self, df: pd.DataFrame, metric: str) -> pd.Series:
        """
        Returns std of the specified metric for each feature map.
        Example: metric="test_accuracy" for classification, metric="test_mae" for regression.
        """
        return df.groupby("feature_maps")[metric].std()
    
    def get_all_metrics_std_by_feature_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns std of all metrics for each feature map.
        Output: DataFrame with metrics as rows and feature maps as columns.
        """
        metrics = [col for col in df.columns if col not in ["feature_maps", "seed"]]
        std_df = df.groupby("feature_maps")[metrics].std().T
        std_df.columns.name = "Feature Map"
        return std_df

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

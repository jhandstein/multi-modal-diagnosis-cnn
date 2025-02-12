from typing import Literal, List
from matplotlib.path import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import TrainingMetric


# Define color schemes for different metrics
METRIC_COLORS = {
    # Main metrics
    TrainingMetric.LOSS: ["crimson", "orangered", "tab:pink"],
    TrainingMetric.ACCURACY: ["royalblue", "deepskyblue", "teal"],
    TrainingMetric.R2: ["royalblue", "deepskyblue", "teal"],
    # Binary classification metrics
    TrainingMetric.AUC: ["darkcyan", "turquoise", "lightgreen"],
    TrainingMetric.F1: ["tab:brown", "peru", "tab:orange"],
    TrainingMetric.PRECISION: ["olive", "lightgreen", "darkgreen"],
    TrainingMetric.RECALL: ["indigo", "mediumorchid", "indianred"],
    # Regression metrics
    TrainingMetric.MAE: ["mediumorchid", "indigo", "tab:pink"],
    TrainingMetric.MSE: ["lightgreen", "darkgreen", "olive"],
    TrainingMetric.RMSE: ["tab:brown", "peru", "tab:orange"],
}

def plot_metric(
    file_path: Path,
    metric: TrainingMetric,
    splits: List[Literal["train", "val", "test"]] = ["train", "val"]
) -> None:
    """
    Plot a single metric across different data splits.
    
    Args:
        file_path: Path to the CSV file containing the metrics
        metric: The metric to plot (from TrainingMetric enum)
        splits: List of data splits to include in the plot
    """
    df = pd.read_csv(file_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get proper metric name and formatting
    metric_name = metric.value.upper() if metric.value in ["auc", "mae", "mse", "rmse"] else metric.value.capitalize()
    
    # Plot each split
    for split, color in zip(splits, METRIC_COLORS[metric]):
        column_name = f"{split}_{metric.value}"
        if column_name in df.columns:
            ax.plot(
                df["epoch"],
                df[column_name],
                color=color,
                label=f"{split.capitalize()} {metric_name}"
            )
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Over Training")
    ax.legend(loc="center right")
    
    plt.tight_layout()
    plot_folder_path = file_path.parent / "plots"
    if not plot_folder_path.exists():
        plot_folder_path.mkdir()
    save_path = plot_folder_path / f"{metric.value}_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_main_metrics(
    file_path: Path,
    task: Literal["classification", "regression"],
    splits: List[Literal["train", "val", "test"]] = ["train", "val"]
) -> None:
    """
    Plot main training metrics (loss + accuracy/r2) in one figure with two y-axes.
    """
    df = pd.read_csv(file_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot losses on primary y-axis (left)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=METRIC_COLORS[TrainingMetric.LOSS][0])
    
    for split, color in zip(splits, METRIC_COLORS[TrainingMetric.LOSS]):
        ax1.plot(
            df["epoch"], 
            df[f"{split}_loss"],
            color=color,
            label=f"{split.capitalize()} Loss"
        )
    ax1.tick_params(axis="y", labelcolor=METRIC_COLORS[TrainingMetric.LOSS][0])
    
    # Plot accuracy/r2 on secondary y-axis (right)
    ax2 = ax1.twinx()
    second_metric = TrainingMetric.ACCURACY if task == "classification" else TrainingMetric.R2
    metric_name = second_metric.value.capitalize()
    
    ax2.set_ylabel(metric_name, color=METRIC_COLORS[second_metric][0])
    for split, color in zip(splits, METRIC_COLORS[second_metric]):
        ax2.plot(
            df["epoch"],
            df[f"{split}_{second_metric.value}"],
            color=color,
            label=f"{split.capitalize()} {metric_name}"
        )
    ax2.tick_params(axis="y", labelcolor=METRIC_COLORS[second_metric][0])
    
    # Add title and legend
    plt.title("Training and Validation Metrics")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    
    # Save plot
    plt.tight_layout()
    plot_folder_path = file_path.parent / "plots"
    plot_folder_path.mkdir(exist_ok=True)
    save_path = plot_folder_path / "_main_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_task_metrics(
    file_path: Path,
    task: Literal["classification", "regression"],
    splits: List[Literal["train", "val", "test"]] = ["train", "val"]
) -> None:
    """
    Plot task-specific metrics (classification: f1/precision/recall/auc, regression: mae/mse/rmse).
    """
    classification_metrics = [
        TrainingMetric.F1,
        TrainingMetric.PRECISION,
        TrainingMetric.RECALL,
        TrainingMetric.AUC
    ]
    regression_metrics = [
        TrainingMetric.MAE,
        TrainingMetric.MSE,
        TrainingMetric.RMSE
    ]
    
    metrics_to_plot = classification_metrics if task == "classification" else regression_metrics
    
    for metric in metrics_to_plot:
        plot_metric(file_path, metric, splits)

def plot_learning_rate(file_path: Path) -> None:
    """
    Plot the learning rate schedule over epochs.
    
    Args:
        file_path: Path to the CSV file containing the metrics
    """
    df = pd.read_csv(file_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning rate on log scale
    ax.semilogy(df["epoch"], df["learning_rate"], color="darkslateblue", label="Learning Rate")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc="center right")
    
    plt.tight_layout()
    plot_folder_path = file_path.parent / "plots"
    plot_folder_path.mkdir(exist_ok=True)
    save_path = plot_folder_path / "learning_rate.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_all_metrics(
    file_path: Path,
    task: Literal["classification", "regression"],
    splits: List[Literal["train", "val", "test"]] = ["train", "val"]
) -> None:
    """
    Plot both main metrics and task-specific metrics.
    """
    plot_main_metrics(file_path, task, splits)
    plot_task_metrics(file_path, task, splits)
    plot_learning_rate(file_path)


from typing import Literal
from matplotlib.path import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_metrics(
    file_path: Path, task: Literal["classification", "regression"]
):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create figure and axis objects with shared x-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot losses on primary y-axis (left)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(df["epoch"], df["train_loss"], color="tab:red", label="Train Loss")
    ax1.plot(df["epoch"], df["val_loss"], color="tab:orange", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Create second y-axis that shares x-axis
    ax2 = ax1.twinx()

    # Plot accuracies on secondary y-axis (right)
    if task == "classification":
        ax2.set_ylabel("Accuracy", color="tab:blue")
        ax2.plot(
            df["epoch"], df["train_accuracy"], color="tab:blue", label="Train Accuracy"
        )
        ax2.plot(
            df["epoch"], df["val_accuracy"], color="tab:cyan", label="Val Accuracy"
        )
        ax2.tick_params(axis="y", labelcolor="tab:blue")
    elif task == "regression":
        ax2.set_ylabel("R squared", color="tab:blue")
        ax2.plot(df["epoch"], df["train_r2"], color="tab:blue", label="Train R squared")
        ax2.plot(df["epoch"], df["val_r2"], color="tab:cyan", label="Val R squared")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Set y-axis limits for accuracies
    ax1.set_ylim(0, 1000)
    ax2.set_ylim(0, 1)

    # Add title and legend
    plt.title("Training and Validation Metrics")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Adjust layout and save
    plt.tight_layout()
    save_path = file_path.parent / "training_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_mae_mse(file_path: Path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create figure and axis objects with shared x-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot MAE on primary y-axis (left)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE", color="tab:purple")
    ax1.plot(df["epoch"], df["train_mae"], color="tab:purple", label="Train MAE")
    ax1.plot(df["epoch"], df["val_mae"], color="tab:pink", label="Val MAE")
    ax1.tick_params(axis="y", labelcolor="tab:purple")

    # Create second y-axis that shares x-axis
    ax2 = ax1.twinx()

    # Plot MSE on secondary y-axis (right)
    ax2.set_ylabel("MSE", color="tab:green")
    ax2.plot(df["epoch"], df["train_mse"], color="tab:green", label="Train MSE")
    ax2.plot(df["epoch"], df["val_mse"], color="tab:olive", label="Val MSE")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # Add axis limits
    ax1.set_ylim(0, 30)
    ax2.set_ylim(0, 1000)

    # Add title and legend
    plt.title("Training and Validation Errors")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Adjust layout and save
    plt.tight_layout()
    save_path = file_path.parent / "training_errors.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
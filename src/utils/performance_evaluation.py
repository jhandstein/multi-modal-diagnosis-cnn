from torch import nn
import torch

from src.building_blocks.torchmetrics import RegressionMetrics
from src.data_management.create_data_split import DataSplitFile
from src.utils.load_targets import extract_targets
from src.utils.config import AGE_SEX_BALANCED_10K_PATH
from src.utils.create_sample_data import generate_sample_data_sets

def calc_loss_based_on_target_mean(label: str = "age"):
    """Calculate the MSE loss between targets and their mean (training set) for a regression dataset."""
    data_split = DataSplitFile(AGE_SEX_BALANCED_10K_PATH).load_data_splits_from_file()

    # Convert targets to tensors immediately after extraction
    train_targets = torch.tensor(extract_targets(label, data_split["train"]).values, dtype=torch.float32)
    val_targets = torch.tensor(extract_targets(label, data_split["val"]).values, dtype=torch.float32)
    test_targets = torch.tensor(extract_targets(label, data_split["test"]).values, dtype=torch.float32)

    # Calculate means
    train_mean = train_targets.mean()
    val_mean = val_targets.mean()
    test_mean = test_targets.mean()
    
    print(f"Means - Train: {train_mean:.2f}, Val: {val_mean:.2f}, Test: {test_mean:.2f}")

    # Create tensor of same length as targets filled with mean value
    train_mean_tensor = torch.full_like(train_targets, train_mean.item())
    val_test_mean_tensor = torch.full_like(val_targets, train_mean.item())
    
    # Compute loss only with the train mean tensor
    loss_func = nn.MSELoss()
    train_loss = loss_func(train_targets, train_mean_tensor)
    val_loss = loss_func(val_targets, val_test_mean_tensor)
    test_loss = loss_func(test_targets, val_test_mean_tensor)
    print(f"Losses - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")

    # Return regression metrics
    train_metrics = RegressionMetrics(phase="train")
    val_metrics = RegressionMetrics(phase="val")
    test_metrics = RegressionMetrics(phase="test")

    # Compute metrics
    train_results = train_metrics(train_targets, train_mean_tensor)
    val_results = val_metrics(val_targets, val_test_mean_tensor)
    test_results = test_metrics(test_targets, val_test_mean_tensor)
    print(f"Metrics - Train: {train_results}, Val: {val_results}, Test: {test_results}")

        


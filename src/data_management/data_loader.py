from typing import Literal
from torch.utils.data import DataLoader, Dataset

BATCH_PARAMS_CUDA01 = {
    ("2D", "ConvBranch"): (64, 1),
    ("2D", "ResNet18"): (64, 1),
    ("3D", "ConvBranch"): (4, 4),
    ("3D", "ResNet18"): (4, 4), # TODO: Figure out why GPU runs of of memory after having processed 5 (!) batches with 8 samples each
}


def infer_batch_size(dim: Literal["2D", "3D"], model_type: Literal["ConvBranch", "ResNet18"]) -> tuple[int, int]:
    """
    Infer the batch size and number of accumulated batches based on the model type and dimensionality. Just a wrapper around the BATCH_PARAMS dictionary.
    """
    return BATCH_PARAMS_CUDA01[(dim, model_type)]


def prepare_standard_data_loaders(
    data_set: Dataset, batch_size: int = 8, num_workers: int = 4
) -> DataLoader:
    """
    Prepare standard data loaders for training and validation.
    
    Args:
        data_set (Dataset): The data set to load.
        batch_size (int): The batch size to use.
        num_workers (int): The number of workers (threads) to use.
    """
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        # Efficiency properties
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True, # disable if issues with the server
        # Data properties
        shuffle=False,
        drop_last=True,
    )

    return data_loader
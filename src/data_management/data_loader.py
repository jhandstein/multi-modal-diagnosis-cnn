from typing import Literal
from torch.utils.data import DataLoader, Dataset

# todo: rewrite into class
BATCH_PARAMS_CUDA01 = {
    ("2D", "ConvBranch"): (64, 1),
    ("2D", "ResNet18"): (64, 1),
    ("3D", "ConvBranch"): (4, 8),
    ("3D", "ResNet18"): (4, 8), # TODO: Figure out why GPU runs of of memory after having processed 5 (!) batches with 8 samples each on cuda01
}

BATCH_PARAMS_CUDA02 = {
    ("2D", "ConvBranch"): (64, 1),
    ("2D", "ResNet18"): (64, 1),
    ("3D", "ConvBranch"): (8, 4),
    ("3D", "ResNet18"): (8, 4),
}


def infer_gpu_count(compute_node: Literal["cuda01", "cuda02"], num_gpus: int | None = None) -> int:
    """
    Infer the number of GPUs from the node if no default value is provided. Validate the number of GPUs based on the node.
    """
    if num_gpus is None:
        return 4 if compute_node == "cuda01" else 1
    
    if compute_node == "cuda02" and num_gpus > 2:
        raise ValueError("Dirty boy! You can't use more than 2 GPUs on cuda02.")
    
    if compute_node == "cuda01" and num_gpus > 8:
        print("Warning: You can't use more than 8 GPUs on cuda01. Using 8 GPUs.")
        num_gpus = 8
        
    return num_gpus


def infer_batch_size(
        compute_node: Literal["cuda01", "cuda02"], 
        dim: Literal["2D", "3D"], 
        model_type: Literal["ConvBranch", "ResNet18"]
        ) -> tuple[int, int]:
    """
    Infer the batch size and number of accumulated batches based on the model type and dimensionality. Just a wrapper around the BATCH_PARAMS dictionaries.

    Args:
        computing_node (Literal["cuda01", "cuda02"]): The computing node to use.
        dim (Literal["2D", "3D"]): The dimensionality of the data.
        model_type (Literal["ConvBranch", "ResNet18"]): The model type.

    Returns:
        tuple[int, int]: The batch size and number of accumulated batches.
    """
    if compute_node == "cuda01":
        return BATCH_PARAMS_CUDA01[(dim, model_type)]
    elif compute_node == "cuda02":
        return BATCH_PARAMS_CUDA02[(dim, model_type)]
    else:
        raise ValueError("Invalid computing node.")



def prepare_standard_data_loaders(
    data_set: Dataset, batch_size: int = 8, num_workers: int = 4, drop_last: bool = True
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
        drop_last=drop_last,
    )

    return data_loader
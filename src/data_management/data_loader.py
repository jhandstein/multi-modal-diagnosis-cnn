from torch.utils.data import DataLoader, Dataset


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

from torch.utils.data import DataLoader, Dataset


def prepare_standard_data_loaders(
    data_set: Dataset, batch_size: int = 8, num_gpus: int = 8, test_flag: bool = False
) -> DataLoader:
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_gpus if not test_flag else 0,
        drop_last=True,
    )

    return data_loader

from torch.utils.data import DataLoader
from tqdm import tqdm


from src.data_management.data_set_factory import DataSetFactory
from src.data_management.data_set import BaseDataSetConfig, NakoSingleModalityDataset
from src.data_management.create_data_split import DataSplitFile
from src.data_management.mri_image_files import MriImageFile
from src.utils.config import AGE_SEX_BALANCED_10K_PATH, FeatureMapType, ModalityType


def check_fm_value_range(
    feature_map: FeatureMapType, temporal_process: str | None = None
):
    """
    Checks the value range of a feature map for a given data split.
    """

    split_file = DataSplitFile(AGE_SEX_BALANCED_10K_PATH)
    data_split = split_file.load_data_splits_from_file()

    base_config = BaseDataSetConfig(
        target="age",
        middle_slice=False,
        slice_dim=0,
        temporal_processes=[temporal_process] if temporal_process else None,
    )

    anat_feature_maps = (
        [feature_map]
        if feature_map.modality == ModalityType.ANAT or feature_map == FeatureMapType.T1
        else []
    )
    func_feature_maps = (
        [feature_map]
        if feature_map.modality == ModalityType.FUNC
        or feature_map == FeatureMapType.BOLD
        else []
    )

    ds_factory = DataSetFactory(
        data_split["train"],
        data_split["val"],
        data_split["test"],
        base_config,
        anat_feature_maps=anat_feature_maps,
        func_feature_maps=func_feature_maps,
    )

    train_set, val_set, test_set = ds_factory.create_data_sets()

    print("Computing statistics for train set...")
    train_stats = compute_dataset_statistics(train_set)
    print("Train set:", train_stats)

    print("Computing statistics for val set...")
    val_stats = compute_dataset_statistics(val_set)
    print("Val set:", val_stats)

    print("Computing statistics for test set...")
    test_stats = compute_dataset_statistics(test_set)
    print("Test set:", test_stats)


def compute_dataset_statistics(dataset, batch_size=8, num_workers=4):

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    count = 0
    running_mean = 0.0
    running_var = 0.0
    running_min = None
    running_max = None

    for batch in tqdm(data_loader, desc="Computing statistics"):
        features = batch[0]  # features, label
        arr = features.detach().cpu().numpy().reshape(-1)

        # Update min/max
        arr_min = arr.min()
        arr_max = arr.max()
        running_min = arr_min if running_min is None else min(running_min, arr_min)
        running_max = arr_max if running_max is None else max(running_max, arr_max)

        # Update mean/variance (Welford's algorithm)
        for x in arr:
            count += 1
            delta = x - running_mean
            running_mean += delta / count
            running_var += delta * (x - running_mean)

    variance = running_var / count if count > 1 else 0.0
    stats = {
        "min": running_min,
        "max": running_max,
        "mean": running_mean,
        "var": variance,
    }
    return stats

from src.data_management.data_set_factory import DataSetFactory
from src.data_management.data_set import BaseDataSetConfig
from src.utils.config import FeatureMapType
from src.data_management.data_loader import prepare_standard_data_loaders
from tqdm import tqdm


def cache_data_set(        
        train_ids: list[int], 
        val_ids: list[int], 
        test_ids: list[int],
        batch_size: int = 8,
        num_workers: int = 4 
        ):
    """
    Caches the data set for the given subject IDs using data loaders for efficient memory usage
    
    Args:
        train_ids (list[int]): List of training subject IDs
        val_ids (list[int]): List of validation subject IDs
        test_ids (list[int]): List of test subject IDs
        batch_size (int): Batch size for the data loader
        num_workers (int): Number of worker processes for data loading
    """
    # Define the base configuration for the dataset
    base_config = BaseDataSetConfig(
        target="age",
        temporal_processes=["mean", "variance", "tsnr"],
    )

    # Create the dataset factory
    dataset_factory = DataSetFactory(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        base_config=base_config,
        anat_feature_maps=[
            FeatureMapType.GM, 
            FeatureMapType.WM,
            FeatureMapType.CSF,
            FeatureMapType.T1
            ],
        func_feature_maps=[
            FeatureMapType.REHO,
            FeatureMapType.BOLD,
        ]
    )

    train_set, val_set, test_set = dataset_factory.create_data_sets()
    
    # Cache the datasets using data loaders
    for dataset in [train_set, val_set, test_set]:
        data_loader = prepare_standard_data_loaders(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        total_batches = len(data_loader)
        with tqdm(total=total_batches, desc=f"Caching {dataset.__class__.__name__}") as pbar:
            for _ in data_loader:
                pbar.update(1)
                
        print(f"Successfully cached {len(dataset)} samples for {dataset.__class__.__name__} and {base_config.temporal_processes} temporal process(es).")
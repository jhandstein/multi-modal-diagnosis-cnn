from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_set import BaseDataSetConfig, NakoSingleModalityDataset
from src.data_management.data_set_factory import DataSetFactory
from src.utils.config import AGE_SEX_BALANCED_10K_PATH, FeatureMapType


def generate_sample_data_sets(label: str = "age", middle_slice: bool = True) -> tuple[
    NakoSingleModalityDataset,
    NakoSingleModalityDataset,
    NakoSingleModalityDataset,
]:
    """Generate a simple dataset for testing purposes."""
    base_config = BaseDataSetConfig(
        target=label,
        middle_slice=middle_slice,
    )
    anat_feature_maps = [
        FeatureMapType.GM,
    ]

    split = DataSplitFile(AGE_SEX_BALANCED_10K_PATH).load_data_splits_from_file()
    
    return DataSetFactory(
        train_ids=split["train"],
        val_ids=split["val"],
        test_ids=split["test"],
        base_config=base_config,
        anat_feature_maps=anat_feature_maps,
    ).create_data_sets()

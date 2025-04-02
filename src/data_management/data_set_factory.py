from dataclasses import asdict

from src.data_management.data_set import (
    BaseDataSetConfig,
    SingleModalityDataSetConfig,
    MultiModalityDataSetConfig,
    NakoSingleModalityDataset,
    NakoMultiModalityDataset,
)
from src.utils.config import FeatureMapType

class DataSetFactory:
    def __init__(
        self, 
        train_ids: list[int], 
        val_ids: list[int], 
        test_ids: list[int], 
        base_config: BaseDataSetConfig,
        anat_feature_maps: list[FeatureMapType] = None,
        func_feature_maps: list[FeatureMapType] = None,
    ):
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.base_config = base_config
        self.anat_feature_maps = anat_feature_maps or []
        self.func_feature_maps = func_feature_maps or []
        
        self.ds_config = self._create_dataset_config()

    def _create_dataset_config(self) -> SingleModalityDataSetConfig | MultiModalityDataSetConfig:
        """Creates the appropriate dataset config based on the feature maps."""
        if not self.anat_feature_maps or not self.func_feature_maps:
            # Single modality case
            return SingleModalityDataSetConfig(
                **asdict(self.base_config),
                feature_maps=self.anat_feature_maps + self.func_feature_maps,
            )
        else:
            # Multi modality case
            return MultiModalityDataSetConfig(
                **asdict(self.base_config),
                anatomical_maps=self.anat_feature_maps,
                functional_maps=self.func_feature_maps,
            )

    def create_data_sets(self):
        """Creates the train, validation and test data sets"""
        train_set = self.create_set(self.train_ids)
        val_set = self.create_set(self.val_ids)
        test_set = self.create_set(self.test_ids)
        return train_set, val_set, test_set

    def create_set(self, ids) -> NakoSingleModalityDataset | NakoMultiModalityDataset:
        """Creates the appropriate dataset type based on the config."""
        if isinstance(self.ds_config, SingleModalityDataSetConfig):
            return NakoSingleModalityDataset(ids, self.ds_config)
        return NakoMultiModalityDataset(ids, self.ds_config)
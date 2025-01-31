from src.data_management.data_set_factory import DataSetConfig, DataSetFactory
from src.utils.config import FeatureType, ModalityType


if __name__ == "__main__":

    ds_details = {
        "modality": ModalityType.ANAT,
        "feature_set": FeatureType.GM,
        "target":   "sex",
        "middle_slice": True
    }

    ds_config = DataSetConfig(**ds_details)
    ds_factory = DataSetFactory([100000, 100005], [100006, 100010], [100011, 100015], ds_config)
    train_set, val_set, test_set = ds_factory.create_data_sets()
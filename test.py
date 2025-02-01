import matplotlib.pyplot as plt
from src.data_management.create_data_split import DataSplitFile
from src.data_management.data_set import NakoSingleFeatureDataset
from src.data_management.data_set_factory import DataSetConfig, DataSetFactory
from src.utils.config import AGE_SEX_BALANCED_10K_PATH, AGE_SEX_BALANCED_1K_PATH, FeatureMapType

def test_data_set_factory():
    ds_details = {
        "feature_map": FeatureMapType.GM,
        "target": "sex",
        "middle_slice": True
    }

    ds_config = DataSetConfig(**ds_details)
    ds_factory = DataSetFactory([100000, 100005], [100006, 100010], [100011, 100015], ds_config)
    train_set, val_set, test_set = ds_factory.create_data_sets()

if __name__ == "__main__":
    print("Hello from test.py")

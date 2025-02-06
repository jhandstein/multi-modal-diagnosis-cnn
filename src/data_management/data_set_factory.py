from src.data_management.data_set import DataSetConfig, NakoSingleFeatureDataset


class DataSetFactory:
    def __init__(self, train_ids, val_ids, test_ids, config: DataSetConfig):
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.config = config

    def create_data_sets(
        self,
    ) -> tuple[
        NakoSingleFeatureDataset, NakoSingleFeatureDataset, NakoSingleFeatureDataset
    ]:
        """Creates the train, validation and test data sets"""
        train_set = self.create_set(self.train_ids)
        val_set = self.create_set(self.val_ids)
        test_set = self.create_set(self.test_ids)

        return train_set, val_set, test_set

    def create_set(self, ids) -> NakoSingleFeatureDataset:
        return NakoSingleFeatureDataset(ids, self.config)

import torch
from torch.utils.data import Dataset

from src.data_management.feature_map_files import FeatureMapFile
from src.utils.config import FeatureType, ModalityType

class NakoSingleFeatureDataset(Dataset):
    def __init__(self, subject_ids: int, label_dict: dict, modality: ModalityType, feature_set: FeatureType):
        self.subject_ids = subject_ids
        self.label_dict = label_dict
        self.modalities = modality
        self.feature_sets = feature_set

    def __len__(self):
        return int(len(self.subject_ids))

    def __getitem__(self, subject_id: int):
        feature_file = FeatureMapFile(subject_id, self.modalities, self.feature_sets)
        feature_array = feature_file.load_array()
        feature_tensor = torch.from_numpy(feature_array)
        return feature_tensor, self.label_dict[subject_id]
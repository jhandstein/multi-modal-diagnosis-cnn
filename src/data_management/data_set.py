import torch
from torch.utils.data import Dataset

from src.data_management.feature_map_files import FeatureMapFile
from src.utils.config import FeatureType, ModalityType
from src.utils.load_targets import extract_target

class NakoSingleFeatureDataset(Dataset):
    def __init__(self, subject_ids: int, modality: ModalityType, feature_set: FeatureType, target: str):
        self.subject_ids = subject_ids
        self.labels = extract_target(target, subject_ids)
        self.modalities = modality
        self.feature_sets = feature_set
        

    def __len__(self):
        return int(len(self.subject_ids))

    def __getitem__(self, idx: int):
        subject_id = self.subject_ids[idx]
        feature_file = FeatureMapFile(subject_id, self.modalities, self.feature_sets)
        feature_array = feature_file.load_array()
        feature_tensor = torch.from_numpy(feature_array)
        return feature_tensor, torch.tensor(self.labels[subject_id])
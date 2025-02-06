from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureMapType
from src.utils.load_targets import extract_target
from src.utils.subject_selection import sample_subject_ids

@dataclass
class DataSetConfig:
    feature_map: FeatureMapType = FeatureMapType.GM
    target: str = "sex"
    middle_slice: bool = True
    
class NakoSingleFeatureDataset(Dataset):
    """PyTorch Dataset class for the NAKO dataset and its MRI feature maps"""

    def __init__(
        self,
        subject_ids: int,
        ds_config: DataSetConfig
    ):
        self.subject_ids = subject_ids
        self.target = ds_config.target
        self.labels = extract_target(ds_config.target, subject_ids)
        self.feature_map = ds_config.feature_map
        self.middle_slice = ds_config.middle_slice

        # Get shape of one sample (without channel dimension)
        self.data_shape = MriImageFile(self.subject_ids[0], self.feature_map).get_size()
        if self.middle_slice:
            self.data_shape = (self.data_shape[1:])

    def __len__(self):
        return int(len(self.subject_ids))

    def __getitem__(self, idx: int):
        subject_id = self.subject_ids[idx]
        # Load the feature map as a tensor
        image_file = MriImageFile(subject_id, self.feature_map)
        feature_tensor = image_file.load_as_tensor(middle_slice=self.middle_slice)
        # Load the label as a tensor
        label = torch.tensor(self.labels[subject_id]).float()
        return feature_tensor, label


def print_details(data_set: Dataset):
    for id, label in zip(data_set.subject_ids, data_set.labels):
        print(id, label)
        print(data_set[0][0].shape)
        break

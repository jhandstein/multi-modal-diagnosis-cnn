from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset

from src.data_management.normalization import MriImageNormalizer
from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureMapType
from src.utils.load_targets import extract_target

@dataclass
class DataSetConfig:
    feature_maps: list[FeatureMapType] = field(default_factory=lambda: [FeatureMapType.GM])
    target: str = "sex"
    middle_slice: bool = True
    slice_dim: int | None = 0
    
class NakoSingleModalityDataset(Dataset):
    """PyTorch Dataset class for the NAKO dataset and its MRI feature maps"""

    def __init__(
        self,
        subject_ids: int,
        ds_config: DataSetConfig
    ):
        self.subject_ids = subject_ids
        self.target = ds_config.target
        self.labels = extract_target(ds_config.target, subject_ids)
        self.feature_maps = ds_config.feature_maps
        self.middle_slice = ds_config.middle_slice
        self.slice_dim = ds_config.slice_dim

        # Get shape of one sample (without channel dimension)
        self.data_shape = MriImageFile(self.subject_ids[0], self.feature_maps[0], self.middle_slice, self.slice_dim).get_size()

        if "smri" in [fm.label for fm in ds_config.feature_maps]:
            if not self.middle_slice:
                raise ValueError("Normalization is only supported for 2D data yet!")
            self.normalizer = MriImageNormalizer(data_dim="2D")
            self.normalizer.load_normalization_params()

    def __len__(self):
        return int(len(self.subject_ids))

    def __getitem__(self, idx: int):
        subject_id = self.subject_ids[idx]
        # Load the feature map as a tensor
        feature_tensors = [self._load_feature_tensor(subject_id, fm) for fm in self.feature_maps]
        feature_tensor = torch.cat(feature_tensors, dim=0)
        # Load the label as a tensor
        label = torch.tensor(self.labels[subject_id]).float()
        return feature_tensor, label
    
    def _load_feature_tensor(self, subject_id: int, feature_map: FeatureMapType):
        """Load a feature map as a tensor"""
        image_file = MriImageFile(subject_id, feature_map, self.middle_slice, self.slice_dim)
        feature_tensor = image_file.load_as_tensor()
        if feature_map.label == "smri":
            feature_tensor = self.normalizer.transform(feature_tensor)
        return feature_tensor

# TODO: Remove this or move to normalization.py
def compute_normalization_params(train_dataset):
    """Compute mean and std of the training set after min-max scaling"""
    means = []
    stds = []
    
    for i in range(len(train_dataset)):
        feature, _ = train_dataset[i]
        means.append(feature.mean().item())
        stds.append(feature.std().item())
    
    mean = torch.tensor(means).mean().item()
    std = torch.tensor(stds).mean().item()
    
    return mean, std


def print_details(data_set: Dataset):
    for id, label in zip(data_set.subject_ids, data_set.labels):
        print(id, label)
        print(data_set[0][0].shape)
        break

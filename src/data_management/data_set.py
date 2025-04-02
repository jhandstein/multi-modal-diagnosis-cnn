from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset

from src.data_management.normalization import MriImageNormalizer
from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureMapType
from src.utils.load_targets import extract_targets

@dataclass
class BaseDataSetConfig:
    # feature_maps: list[FeatureMapType] = field(default_factory=lambda: [FeatureMapType.GM])
    target: str = "sex"
    middle_slice: bool = True
    slice_dim: int | None = 0

@dataclass
class SingleModalityDataSetConfig(BaseDataSetConfig):
    feature_maps: list[FeatureMapType] = field(default_factory=lambda: [FeatureMapType.GM])

@dataclass
class MultiModalityDataSetConfig(BaseDataSetConfig):
    anatomical_maps: list[FeatureMapType] = field(default_factory=lambda: [FeatureMapType.GM])
    functional_maps: list[FeatureMapType] = field(default_factory=lambda: [FeatureMapType.REHO])

class BaseNakoDataset(Dataset):
    """Base class for NAKO dataset that an acts as a template for other datasets"""

    def __init__(
        self,
        subject_ids: list[int],
        ds_config: BaseDataSetConfig
    ):
        self.subject_ids = subject_ids
        self.target = ds_config.target
        self.labels = extract_targets(ds_config.target, subject_ids)
        self.middle_slice = ds_config.middle_slice
        self.slice_dim = ds_config.slice_dim


    def __len__(self):
        return int(len(self.subject_ids))

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _load_feature_tensor(self, subject_id: int, feature_map: FeatureMapType):
        """Load a feature map as a tensor"""
        image_file = MriImageFile(subject_id, feature_map, self.middle_slice, self.slice_dim)
        feature_tensor = image_file.load_as_tensor()
        if feature_map.label == "T1":
            feature_tensor = self.normalizer.transform(feature_tensor)
        return feature_tensor
    
    def _initilize_normalizer(self, feature_maps: list[FeatureMapType]):
        """Initialize the normalizer for the dataset if required"""
        if "T1" in [fm.label for fm in feature_maps]:
            if not self.middle_slice:
                raise ValueError("Normalization is only supported for 2D data yet!")
            self.normalizer = MriImageNormalizer(data_dim="2D")
            self.normalizer.load_normalization_params()

class NakoSingleModalityDataset(BaseNakoDataset):
    """PyTorch Dataset class for single modality NAKO data"""

    def __init__(
        self,
        subject_ids: list[int],
        ds_config: SingleModalityDataSetConfig
    ):
        super().__init__(subject_ids, ds_config)
        self.feature_maps = ds_config.feature_maps
        
        # Initialize normalizer if needed
        self._initilize_normalizer(self.feature_maps)

    def __getitem__(self, idx: int):
        subject_id = self.subject_ids[idx]
        # Load the feature maps as tensors
        feature_tensors = [self._load_feature_tensor(subject_id, fm) for fm in self.feature_maps]
        try:
            feature_tensor = torch.cat(feature_tensors, dim=0)
        except RuntimeError:
            raise ValueError("Feature maps have different shapes. Please check the data.")

        # Load the label as a tensor
        label = torch.tensor(self.labels[subject_id]).float()
        return feature_tensor, label


class NakoMultiModalityDataset(BaseNakoDataset):
    """PyTorch Dataset class for multi-modality NAKO data"""

    def __init__(
        self,
        subject_ids: list[int],
        ds_config: MultiModalityDataSetConfig
    ):
        super().__init__(subject_ids, ds_config)
        self.anatomical_maps = ds_config.anatomical_maps
        self.functional_maps = ds_config.functional_maps
        
        # Initialize normalizers if needed
        self._initilize_normalizer(self.anatomical_maps)

    def __getitem__(self, idx: int):
        subject_id = self.subject_ids[idx]
        
        # Load anatomical maps
        anat_tensors = [self._load_feature_tensor(subject_id, fm) for fm in self.anatomical_maps]
        try:
            anat_tensor = torch.cat(anat_tensors, dim=0)
        except RuntimeError:
            raise ValueError("Anatomical maps have different shapes. Please check the data.")

        # Load functional maps
        func_tensors = [self._load_feature_tensor(subject_id, fm) for fm in self.functional_maps]
        try:
            func_tensor = torch.cat(func_tensors, dim=0)
        except RuntimeError:
            raise ValueError("Functional maps have different shapes. Please check the data.")

        # Load the label as a tensor
        label = torch.tensor(self.labels[subject_id]).float()
        return (anat_tensor, func_tensor), label
    

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

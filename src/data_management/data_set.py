from dataclasses import dataclass, field
from typing import Literal
import torch
from torch.utils.data import Dataset

from src.data_management.normalization import MriImageNormalizer
from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureMapType
from src.utils.load_targets import extract_targets


@dataclass
class BaseDataSetConfig:
    target: str = "sex"
    middle_slice: bool = True
    slice_dim: int | None = 0
    temporal_processes: list[Literal["mean", "variance", "tsnr"]] = None


@dataclass
class SingleModalityDataSetConfig(BaseDataSetConfig):
    feature_maps: list[FeatureMapType] = field(
        default_factory=lambda: [FeatureMapType.GM]
    )


@dataclass
class MultiModalityDataSetConfig(BaseDataSetConfig):
    anatomical_maps: list[FeatureMapType] = field(
        default_factory=lambda: [FeatureMapType.GM]
    )
    functional_maps: list[FeatureMapType] = field(
        default_factory=lambda: [FeatureMapType.REHO]
    )


class BaseNakoDataset(Dataset):
    """Base class for NAKO dataset that an acts as a template for other datasets"""

    def __init__(self, subject_ids: list[int], ds_config: BaseDataSetConfig):
        self.subject_ids = subject_ids
        self.target = ds_config.target
        self.labels = extract_targets(ds_config.target, subject_ids)
        self.middle_slice = ds_config.middle_slice
        self.slice_dim = ds_config.slice_dim
        self.temporal_processes = ds_config.temporal_processes

    def __len__(self):
        return int(len(self.subject_ids))

    def __getitem__(self, index):
        """Get a single item from the dataset."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _load_feature_tensor(
        self,
        subject_id: int,
        feature_map: FeatureMapType,
        temporal_process: str | None = None,
    ) -> torch.Tensor:
        """Load a feature map as a tensor"""
        image_file = MriImageFile(
            subject_id, feature_map, self.middle_slice, self.slice_dim, temporal_process
        )
        feature_tensor = image_file.load_as_tensor()
        # TODO: Change this to check for actual enum
        if feature_map.label == "T1":
            # TODO: This value should be derived from the actual dataset if multiple datasets are used
            feature_tensor = self.normalizer.transform(feature_tensor)
        return feature_tensor

    def _initilize_normalizer(self, feature_maps: list[FeatureMapType]):
        """Initialize the normalizer for the dataset if required"""
        if "T1" in [fm.label for fm in feature_maps]:
            if not self.middle_slice:
                raise ValueError("Normalization is only supported for 2D data yet!")
            self.normalizer = MriImageNormalizer(data_dim="2D")
            self.normalizer.load_normalization_params()

    def _create_feature_map_variants(
        self, feature_maps: list[FeatureMapType], temporal_processes: list[str] | None
    ) -> list[tuple[FeatureMapType, str | None]]:
        """Creates variants of feature maps with different temporal processes.

        Args:
            feature_maps: List of feature maps
            temporal_processes: List of temporal processes to apply to BOLD maps

        Returns:
            List of tuples containing (feature_map, temporal_process)
        """
        variants = []
        for fm in feature_maps:
            if fm == FeatureMapType.BOLD and temporal_processes:
                # Create a variant for each temporal process
                for process in temporal_processes:
                    variants.append((fm, process))
            else:
                # Non-BOLD maps don't use temporal processing
                variants.append((fm, None))
        return variants


class NakoSingleModalityDataset(BaseNakoDataset):
    """PyTorch Dataset class for single modality NAKO data"""

    def __init__(self, subject_ids: list[int], ds_config: SingleModalityDataSetConfig):
        super().__init__(subject_ids, ds_config)
        # Initialize feature maps and temporal processes
        self.feature_map_variants = self._create_feature_map_variants(
            ds_config.feature_maps, ds_config.temporal_processes
        )

        # Initialize normalizer if needed
        self._initilize_normalizer([fm for fm, _ in self.feature_map_variants])

    def __getitem__(self, idx: int):
        """Get a single item from the dataset."""
        subject_id = self.subject_ids[idx]
        # Load the feature maps as tensors
        feature_tensors = [
            self._load_feature_tensor(subject_id, fm, temp_proc)
            for fm, temp_proc in self.feature_map_variants
        ]
        try:
            feature_tensor = torch.cat(feature_tensors, dim=0)
        except RuntimeError:
            raise ValueError(
                "Feature maps have different shapes. Please check the data."
            )

        # Load the label as a tensor
        label = torch.tensor(self.labels[subject_id]).float()
        return feature_tensor, label


class NakoMultiModalityDataset(BaseNakoDataset):
    """PyTorch Dataset class for multi-modality NAKO data"""

    def __init__(self, subject_ids: list[int], ds_config: MultiModalityDataSetConfig):
        super().__init__(subject_ids, ds_config)
        self.anat_map_variants = self._create_feature_map_variants(
            ds_config.anatomical_maps,
            None,  # Anatomical maps don't use temporal processing
        )
        self.func_map_variants = self._create_feature_map_variants(
            ds_config.functional_maps, ds_config.temporal_processes
        )

        # Initialize normalizers if needed
        self._initilize_normalizer([fm for fm, _ in self.anat_map_variants])

    def __getitem__(self, idx: int):
        """Get a single item from the dataset."""
        subject_id = self.subject_ids[idx]

        # Load anatomical maps
        anat_tensors = [
            self._load_feature_tensor(subject_id, fm, temp_proc)
            for fm, temp_proc in self.anat_map_variants
        ]
        try:
            anat_tensor = torch.cat(anat_tensors, dim=0)
        except RuntimeError:
            raise ValueError(
                "Anatomical maps have different shapes. Please check the data."
            )

        # Load functional maps with temporal processing
        func_tensors = [
            self._load_feature_tensor(subject_id, fm, temp_proc)
            for fm, temp_proc in self.func_map_variants
        ]
        try:
            #! Re-activate if you want to debug the shapes of functional maps
            # for i, tensor in enumerate(func_tensors):
            #     # print(f"Functional map {i} shape: {tensor.shape}")
            #     if tensor.shape != torch.Size([1, 62, 48]):
            #         print(f"Subject {subject_id} functional map {i} shape: {tensor.shape}")
            func_tensor = torch.cat(func_tensors, dim=0)
        except RuntimeError:
            raise ValueError(
                "Functional maps have different shapes. Please check the data."
            )

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

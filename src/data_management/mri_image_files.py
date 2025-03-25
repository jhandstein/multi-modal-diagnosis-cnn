# https://lukas-snoek.com/NI-edu/fMRI-introduction/week_1/python_for_mri.html
import gc
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path

from src.utils.config import (
    FMRI_PREP_FULL_SAMPLE,
    XCP_D_FULL_SAMPLE,
    DL_CACHE_PATH,
    FeatureMapType,
    ModalityType,
)


class MriImageFile:
    """
    Class to handle the loading of image / feature map files from the NAKO dataset
    """

    def __init__(self, subject_id: int, feature_map: FeatureMapType, middle_slice: bool = True, slice_dim: int | None = 0):
        self.subject_id = subject_id
        self.token = f"sub-{subject_id}"
        self.feature_map = feature_map
        self.middle_slice = middle_slice
        self.slice_dim = slice_dim if middle_slice else None

    @property
    def file_path(self) -> Path:
        """Returns the path to the feature map file"""
        # TODO: refine selection of paths (implement switch case)
        if self.feature_map.modality == ModalityType.ANAT:
            return self._get_anat_path()
        elif self.feature_map.modality == ModalityType.FUNC:
            return self._get_func_path()
        elif self.feature_map.modality == ModalityType.RAW:
            return self._get_smri_path()
        else:
            raise ValueError("Invalid scan type")
        
    @property
    def cache_path(self) -> Path:
        """Returns the path to the cache file. Cached in .npy format for performance"""
        data_dim = "2D" if self.middle_slice else "3D"
        file_suffix = "npy"
        slice_suffix = f"dim_{self.slice_dim}" if self.middle_slice else ""
        file_name = f"{self.feature_map.label}_{slice_suffix}.{file_suffix}" 
        return Path(DL_CACHE_PATH, data_dim, self.token, file_name)

    def load_as_tensor(self) -> torch.Tensor:
        """Loads the feature map file as a torch tensor
        
        Args:
            middle_slice (bool): If True, extracts the middle slice from the specified dimension
        
        Returns:
            torch.tensor: The loaded tensor with shape (1, dim1, dim2) where dim1 and dim2
                        depend on which dimension was sliced or (1, dim1, dim2, dim3) if no slice
        """

        # Check if the tensor is already cached
        if self.cache_path.exists():
            return torch.from_numpy(np.load(self.cache_path))
 
        # Create the cache directory if it does not exist
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the feature map as a float tensor
        img = nib.load(self.file_path)
        t = torch.from_numpy(img.get_fdata(dtype=np.float32))
        
        # Extract the middle slice from the specified dimension
        if self.middle_slice:
            if self.slice_dim not in [0, 1, 2]:
                raise ValueError("slice_dim must be 0, 1, or 2")
            
            # Select the middle slice using indexing for the specified dimension
            slice_idx = t.shape[self.slice_dim] // 2
            if self.slice_dim == 0:
                t = t[slice_idx]
            elif self.slice_dim == 1:
                t = t[:, slice_idx]
            else:  # slice_dim == 2
                t = t[:, :, slice_idx]

        else:
            t = t.unsqueeze_(0).unsqueeze_(0)
            t = F.interpolate(
                t,
                scale_factor=0.5,
                mode='trilinear',
                align_corners=False
            )
            t = t.squeeze_(0).squeeze_(0)

        
        # Add a channel dimension
        t = t.unsqueeze_(0)
        # Cache the tensor
        if self.middle_slice:
            np.save(self.cache_path, t.numpy())
        return t

    def get_size(self) -> tuple:
        """Returns the size of the feature map file"""
        return self.load_as_tensor().shape[1:]
    
    def _num_params(self) -> str:
        num_params = self.load_as_tensor().numel()
        print(f"Number of parameters: {num_params}")
        return num_params

    def print_stats(self):
        """Function to show some basic statistics about the image without loading the whole array"""
        img = nib.load(self.file_path)
        print(img.shape)
        print(img.header.get_zooms())
        print(img.header.get_xyzt_units())

    def _get_anat_path(self) -> Path:
        return Path(
            FMRI_PREP_FULL_SAMPLE,
            f"{self.token}/ses-0/anat/{self.token}_ses-0_space-MNI152NLin2009cAsym_label-{self.feature_map.label}_probseg.nii.gz",
        )

    def _get_func_path(self) -> Path:
        return Path(
            XCP_D_FULL_SAMPLE,
            f"{self.token}/ses-0/func/{self.token}_ses-0_task-rest_space-MNI152NLin2009cAsym_{self.feature_map.label}.nii.gz",
        )

    def _get_smri_path(self) -> Path:
        return Path(
            FMRI_PREP_FULL_SAMPLE,
            f"{self.token}/ses-0/anat/{self.token}_ses-0_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
        )
    
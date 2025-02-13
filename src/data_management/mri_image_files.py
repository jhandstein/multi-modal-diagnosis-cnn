# https://lukas-snoek.com/NI-edu/fMRI-introduction/week_1/python_for_mri.html
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from src.utils.config import (
    FMRI_PREP_FULL_SAMPLE,
    XCP_D_FULL_SAMPLE,
    FeatureMapType,
    ModalityType,
)


class MriImageFile:
    """
    Class to handle the loading of image / feature map files from the NAKO dataset
    """

    def __init__(self, subject_id: int, feature_map: FeatureMapType):
        self.subject_id = subject_id
        self.token = f"sub-{subject_id}"
        self.feature_map = feature_map

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

    def load_as_tensor(self, middle_slice: bool = True, slice_dim: int = 0) -> torch.tensor:
        """Loads the feature map file as a torch tensor
        
        Args:
            middle_slice (bool): If True, extracts the middle slice from the specified dimension
            slice_dim (int): Dimension from which to take the slice (0, 1, or 2)
        
        Returns:
            torch.tensor: The loaded tensor with shape (1, dim1, dim2) where dim1 and dim2
                        depend on which dimension was sliced or (1, dim1, dim2, dim3) if no slice
        """
        # Load the feature map as a float tensor
        t = torch.from_numpy(self.load_array()).float()
        
        # Extract the middle slice from the specified dimension
        if middle_slice:
            if slice_dim not in [0, 1, 2]:
                raise ValueError("slice_dim must be 0, 1, or 2")
            
            # Select the middle slice using indexing for the specified dimension
            slice_idx = t.shape[slice_dim] // 2
            if slice_dim == 0:
                t = t[slice_idx]
            elif slice_dim == 1:
                t = t[:, slice_idx]
            else:  # slice_dim == 2
                t = t[:, :, slice_idx]
        
        # Add a channel dimension
        t = t.unsqueeze(0)
        return t

    def load_array(self) -> np.ndarray:
        """Loads the feature map file as a numpy array"""
        img = nib.load(self.file_path)
        return img.get_fdata()

    def get_size(self) -> tuple:
        """Returns the size of the feature map file"""
        img = nib.load(self.file_path)
        return img.shape

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
    
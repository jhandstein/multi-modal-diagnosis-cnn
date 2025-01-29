# https://lukas-snoek.com/NI-edu/fMRI-introduction/week_1/python_for_mri.html

import torch
import numpy as np
import nibabel as nib 
from pathlib import Path

from src.utils.config import FMRI_PREP_FULL_SAMPLE, FeatureType, ModalityType

class MriImageFile:
    """
    Class to handle the loading of image / feature map files from the NAKO dataset
    """

    def __init__(self, subject_id: int, scan_type: ModalityType, map_type: FeatureType):
        self.subject_id = subject_id
        self.token = f"sub-{subject_id}"
        self.scan_type = scan_type
        self.map_type = map_type

    def get_path(self) -> Path:
        """Returns the path to the feature map file"""
        # TODO: refine selection of paths (implement switch case)
        if self.scan_type == ModalityType.ANAT:
            return self._get_anat_path()
        elif self.scan_type == ModalityType.FUNC:
            return self._get_func_path()
        elif self.scan_type == ModalityType.RAW:
            return self._get_smri_path()
        else:
            raise ValueError("Invalid scan type")
        
    def load_as_tensor(self, middle_slice: bool = True) -> torch.tensor:
        """Loads the feature map file as a torch tensor"""
        # Load the feature map as a float tensor
        t = torch.from_numpy(self.load_array()).float()
        if middle_slice:
            # extract only the middle slice
            t = t[t.shape[0]//2]
            # add a channel dimension
            t = t.unsqueeze(0)
        return t
    
    def load_array(self) -> np.ndarray:
        """Loads the feature map file as a numpy array"""
        img = nib.load(self.get_path())
        return img.get_fdata()
    
    def get_size(self) -> tuple:
        """Returns the size of the feature map file"""
        img = nib.load(self.get_path())
        return img.shape
 
    def print_stats(self):
        """Function to show some basic statistics about the image without loading the whole array"""
        img = nib.load(self.get_path())
        print(img.shape)
        print(img.header.get_zooms())
        print(img.header.get_xyzt_units())
        
    def _get_anat_path(self) -> Path:
        return Path(FMRI_PREP_FULL_SAMPLE, f"{self.token}/ses-0/anat/{self.token}_ses-0_space-MNI152NLin2009cAsym_label-{self.map_type.value}_probseg.nii.gz")

    def _get_func_path(self) -> Path:
        return Path(FMRI_PREP_FULL_SAMPLE, f"{self.token}/ses-0/func/{self.token}_ses-0_task-rest_space-MNI152NLin2009cAsym_{self.map_type.value}.nii.gz")
    
    def _get_smri_path(self) -> Path:
        return Path(FMRI_PREP_FULL_SAMPLE, f"{self.token}/ses-0/anat/{self.token}_ses-0_desc-preproc_T1w.nii.gz")
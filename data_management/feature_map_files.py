# https://lukas-snoek.com/NI-edu/fMRI-introduction/week_1/python_for_mri.html

import numpy as np
import nibabel as nib 
from pathlib import Path
from typing import Literal

from utils.config import NAKO_PATH


FMRIPREP_FOLDER = "derivatives_ses0/fmriprep"


class FeatureMapFile:
    """
    Class to handle the loading of feature map files from the NAKO dataset
    """

    def __init__(self, subject_id: int, scan_type: Literal["anat", "func"], map_type: Literal["GM", "WM", "CSF", "reho", "ALFF", "fALFF", "VMHC"]):
        self.subject_id = subject_id
        self.scan_type = scan_type
        self.map_type = map_type

    def get_path(self) -> Path:
        if self.scan_type == "anat":
            return self._get_anat_path()
        elif self.scan_type == "func":
            return self._get_func_path()
        else:
            raise ValueError("Invalid scan type")
    
    def load_array(self) -> np.ndarray:
        img = nib.load(self.get_path())
        return img.get_fdata()
    
    def load_middle_slice(self) -> np.ndarray:
        img = nib.load(self.get_path())
        return img.get_fdata()[img.shape[0]//2]
    
    def print_stats(self):
        """Function to show some basic statistics about the image without loading the whole array"""
        img = nib.load(self.get_path())
        print(img.shape)
        print(img.header.get_zooms())
        print(img.header.get_xyzt_units())
        
    def _get_anat_path(self) -> Path:
        token = f"sub-{self.subject_id}"
        return Path(NAKO_PATH, FMRIPREP_FOLDER, f"{token}/ses-0/anat/{token}_ses-0_space-MNI152NLin2009cAsym_label-{self.map_type}_probseg.nii.gz")

    def _get_func_path(self) -> Path:
        token = f"sub-{self.subject_id}"
        return Path(NAKO_PATH, FMRIPREP_FOLDER, f"{token}/ses-0/func/{token}_ses-0_task-rest_space-MNI152NLin2009cAsym_{self.map_type}.nii.gz")
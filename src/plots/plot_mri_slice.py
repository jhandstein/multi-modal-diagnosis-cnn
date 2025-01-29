from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureType, ModalityType

def plot_mri_slice(subject_id: int, title: str = None):
    """
    Plots an MRI slice as a 2D image
    """
    mri_slice = MriImageFile(subject_id, ModalityType.RAW, FeatureType.SMRI).load_as_tensor(middle_slice=True)
    mri_slice = mri_slice.numpy().reshape(256, 256)
    plt.imshow(mri_slice, cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.savefig(Path("src/plots","mri_slice.png"))

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.data_management.mri_image_files import MriImageFile
from src.utils.config import PLOTS_PATH, FeatureMapType


def plot_mri_slice(subject_id: int, title: str = None):
    """
    Plots an MRI slice as a 2D image
    """
    mri_slice = MriImageFile(subject_id, FeatureMapType.SMRI).load_as_tensor(
        middle_slice=True
    )
    mri_slice = mri_slice.numpy().reshape(256, 256)
    plt.imshow(mri_slice, cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    file_path = PLOTS_PATH / "mri_slices" / f"slice_{subject_id}.png"
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    plt.savefig(file_path)

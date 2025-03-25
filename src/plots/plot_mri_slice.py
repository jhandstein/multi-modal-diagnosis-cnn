from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.data_management.mri_image_files import MriImageFile
from src.utils.config import PLOTS_PATH, FeatureMapType


def plot_mri_slice(subject_id: int, feature_map = FeatureMapType.GM, slice_dim: int = 0):
    """
    Plots an MRI slice as a 2D image
    """
    
    mri_slice = MriImageFile(subject_id, feature_map, slice_dim=slice_dim).load_as_tensor().squeeze()
    mri_slice = mri_slice.numpy()

    plt.imshow(mri_slice, cmap="gray")
    plt.axis("off")

    folder_path = PLOTS_PATH / "mri_slices" / f"images-{subject_id}"
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path / f"slice_{subject_id}_map_{feature_map.label}_dim{slice_dim}.png"
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    plt.savefig(file_path)

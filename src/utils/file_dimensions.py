from data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureMapType


def raw_and_map_sizes(subject_id: int):
    """Prints the sizes of the raw and mapped feature files for a given subject ID"""
    fm = MriImageFile(subject_id, FeatureMapType.SMRI)
    print(fm.file_path)
    print(fm.print_stats())
    fm2 = MriImageFile(subject_id, FeatureMapType.GM)
    print(fm2.file_path)
    print(fm2.print_stats())

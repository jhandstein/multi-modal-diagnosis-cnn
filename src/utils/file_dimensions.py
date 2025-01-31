from data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureType, ModalityType


def raw_and_map_sizes(subject_id: int):
    """Prints the sizes of the raw and mapped feature files for a given subject ID"""
    fm = MriImageFile(subject_id, ModalityType.RAW, FeatureType.SMRI)
    print(fm.file_path)
    print(fm.print_stats())
    fm2 = MriImageFile(subject_id, ModalityType.ANAT, FeatureType.GM)
    print(fm2.file_path)
    print(fm2.print_stats())
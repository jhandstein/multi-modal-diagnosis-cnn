from src.data_management.feature_map_files import FeatureMapFile
from src.utils.config import FeatureType, ModalityType


def raw_and_map_sizes(subject_id: int):
    """Prints the sizes of the raw and mapped feature files for a given subject ID"""
    fm = FeatureMapFile(subject_id, ModalityType.RAW, FeatureType.SMRI)
    print(fm.get_path())
    print(fm.print_stats())
    fm2 = FeatureMapFile(subject_id, ModalityType.ANAT, FeatureType.GM)
    print(fm2.get_path())
    print(fm2.print_stats())
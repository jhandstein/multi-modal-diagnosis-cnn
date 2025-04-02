from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureMapType


def raw_and_map_sizes(subject_id: int):
    """Prints the sizes of the raw and mapped feature files for a given subject ID"""
    fm = MriImageFile(subject_id, FeatureMapType.T1)
    print(fm.file_path)
    print(fm.print_stats())
    fm2 = MriImageFile(subject_id, FeatureMapType.GM)
    print(fm2.file_path)
    print(fm2.print_stats())
    fm3 = MriImageFile(subject_id, FeatureMapType.REHO)
    print(fm3.file_path)
    print(fm3.print_stats())


def get_folder_size(folder_path):
    """
    Calculate the total size of a folder in megabytes
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        float: Size of the folder in MB
    """
    import os
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Skip if it's a symbolic link
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    
    # Convert bytes to megabytes
    size_in_mb = total_size / (1024 * 1024)
    print(f"Folder size: {round(size_in_mb, 2)} MB")

import os
from pathlib import Path

def get_subject_ids(directory: str) -> list:
    """Extracts all available subject IDs from the given directory"""
    folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return sorted([int(name.split("-")[1]) for name in folder_names if name.startswith("sub-")])

def save_subject_ids(data_directory: Path, file_path: str):
    """Saves all available subject IDs from the given directory to a file"""
    subject_ids = get_subject_ids(data_directory)
    with open(Path(file_path), "w") as file:
        for subject_id in subject_ids:
            file.write(f"{subject_id}\n")
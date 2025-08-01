import os
from pathlib import Path
from random import sample

from src.utils.config import AVAILABLE_SUBJECT_IDS


def get_subject_ids(directory: str) -> list:
    """Extracts all available subject IDs from the given directory of fMRIprep outputs"""
    folder_names = [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
    return sorted(
        [int(name.split("-")[1]) for name in folder_names if name.startswith("sub-")]
    )


def save_subject_ids(data_directory: Path):
    """Saves all available subject IDs from the given directory to a file"""
    subject_ids = get_subject_ids(data_directory)
    with open(Path(AVAILABLE_SUBJECT_IDS), "w") as file:
        for subject_id in subject_ids:
            file.write(f"{subject_id}\n")


def load_subject_ids_from_file(file_path=AVAILABLE_SUBJECT_IDS) -> list[int]:
    """Loads all available subject IDs from a file"""
    with open(Path(file_path), "r") as file:
        return [int(line.strip()) for line in file.readlines()]


def sample_subject_ids(n: int) -> list:
    """Samples n subject IDs from the list of available subject IDs"""
    return sample(load_subject_ids_from_file(), n)

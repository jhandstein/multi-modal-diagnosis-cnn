from enum import Enum
from pathlib import Path

# Sample numbers that are divisible by 128 and therefore suitable for batch processing
NUM_SAMPLES_1K = 1024 
NUM_SAMPLES_10K = 10240
AGE_SEX_BALANCED_1K_PATH = Path("src/data_management/age_sex_split_1k.json")
AGE_SEX_BALANCED_10K_PATH = Path("src/data_management/age_sex_split_10k.json") 

# Important file paths
NAKO_PATH = Path("/ritter/share/data/NAKO")
NAKO_TABLE_PATH = Path(NAKO_PATH, "NAKO_data_processed/NAKO_all_orig_columns_new.csv")
FMRI_PREP_FULL_SAMPLE = Path(NAKO_PATH, "derivatives_ses0/fmriprep")

LOGS_PATH = Path("logs")

# IDs for the subjects that have both sMRI and fMRI data
# AVAILABLE_SUBJECT_IDS = Path("src/utils/subject_ids.txt")
AVAILABLE_SUBJECT_IDS = Path(NAKO_PATH, "derivatives_ses0/info_MRI_availability/list_subjects_NAKO_bothMRI_successfull.txt")

# IDs for the subjects that have high quality sMRI and fMRI data
# High quality fMRI also means that sMRI is high quality
HIGH_QUALITY_SMRI_IDS = Path(NAKO_PATH, "derivatives_ses0/info_MRI_availability/list_subjects_NAKO_sMRI_high_quality.tsv")
HIGH_QUALITY_FMRI_IDS = Path(NAKO_PATH, "derivatives_ses0/info_MRI_availability/list_subjects_NAKO_rsfMRI_high_quality.tsv")

class ModalityType(Enum):
    RAW = "raw"
    ANAT = "anat"
    FUNC = "func"

class FeatureType(Enum):
    # raw data
    SMRI = "smri"
    FMRI = "fmri"

    # feature maps
    GM = "GM"
    WM = "WM"
    CSF = "CSF"
    REHO = "reho"
    ALFF = "alff"
    fALFF = "falff"
    VMHC = "vmhc"
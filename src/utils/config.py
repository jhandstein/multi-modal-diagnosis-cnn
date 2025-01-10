from enum import Enum
from pathlib import Path


NAKO_PATH = Path("/ritter/share/data/NAKO")
NAKO_TABLE_PATH = Path(NAKO_PATH, "NAKO_data_processed/NAKO_all_orig_columns_new.csv")
FMRI_PREP_FULL_SAMPLE = Path(NAKO_PATH, "derivatives_ses0/fmriprep")

LOGS_PATH = Path("logs")
AVAILABLE_SUBJECT_IDS = Path("src/utils/subject_ids.txt")

class ModalityType(Enum):
    ANAT = "anat"
    FUNC = "func"

class FeatureType(Enum):
    GM = "GM"
    WM = "WM"
    CSF = "CSF"
    REHO = "reho"
    ALFF = "alff"
    fALFF = "falff"
    VMHC = "vmhc"
from enum import Enum
from pathlib import Path


NAKO_PATH = Path("/ritter/share/data/NAKO")
FMRI_PREP_FULL_SAMPLE = Path(NAKO_PATH, "derivatives_ses0/fmriprep")

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
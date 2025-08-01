from enum import Enum
from pathlib import Path

# Sample numbers that are divisible by 128 and therefore suitable for batch processing
NUM_SAMPLES_1K = 1024
NUM_SAMPLES_10K = 10240

# Paths for the different dataset splits
SPLIT_PATH = Path("assets/data_splits")
AGE_SEX_BALANCED_1K_PATH = Path(SPLIT_PATH, "age_sex_split_1k.json")
AGE_SEX_BALANCED_10K_PATH = Path(SPLIT_PATH, "age_sex_split_10k.json")

RESHUFFLED_27_PATH = Path(SPLIT_PATH, "reshuffled_27.json")
# RESHUFFLED_42_PATH = Path(SPLIT_PATH, "reshuffled_42.json")
RESHUFFLED_1984_PATH = Path(SPLIT_PATH, "reshuffled_1984.json")
RESHUFFLED_1312_PATH = Path(SPLIT_PATH, "reshuffled_1312.json")
RESHUFFLED_404_PATH = Path(SPLIT_PATH, "reshuffled_404.json")


QUALITY_SPLITS_PATH = Path(SPLIT_PATH, "quality_split_fd_cnr.json")
LOW_QUALITY_IDS = Path(SPLIT_PATH, "quality_split_low_quality_ids.json")
MEDIUM_QUALITY_IDS = Path(SPLIT_PATH, "quality_split_medium_quality_ids.json")
HIGH_QUALITY_IDS = Path(SPLIT_PATH, "quality_split_high_quality_ids.json")

PHQ9_CUTOFF_SPLIT_PATH = Path(SPLIT_PATH, "phq9_cutoff_split.json")
GAD7_CUTOFF_SPLIT_PATH = Path(SPLIT_PATH, "gad7_cutoff_split.json")

# Important file paths
NAKO_PATH = Path("/ritter/share/data/NAKO")
NAKO_TABLE_PATH = Path(NAKO_PATH, "NAKO_data_processed/NAKO_all_orig_columns_v2.csv")
FMRI_PREP_FULL_SAMPLE = Path(NAKO_PATH, "derivatives_ses0/fmriprep")
XCP_D_FULL_SAMPLE = Path(NAKO_PATH, "derivatives_ses0/xcp_d")
DL_CACHE_PATH = Path(NAKO_PATH, "deep_learning_cache")

# Internal paths
LOGS_PATH = Path("logs")
PLOTS_PATH = Path("assets/plots")

# IDs for the subjects that have both sMRI and fMRI data
# AVAILABLE_SUBJECT_IDS = Path("src/utils/subject_ids.txt")
AVAILABLE_SUBJECT_IDS = Path(
    NAKO_PATH,
    "derivatives_ses0/info_MRI_availability/list_subjects_NAKO_bothMRI_successfull.txt",
)

FAULTY_SAMPLE_IDS = [
    127569 # The dimension of the fMRI don't match (1, 62, 54) vs. (1, 62, 48)
]  

# IDs for the subjects that have high quality sMRI and fMRI data
# High quality fMRI also means that sMRI is high quality..?
HIGH_QUALITY_T1_IDS = Path(
    NAKO_PATH,
    "derivatives_ses0/info_MRI_availability/list_subjects_NAKO_sMRI_high_quality.tsv",
)
HIGH_QUALITY_FMRI_IDS = Path(
    NAKO_PATH,
    "derivatives_ses0/info_MRI_availability/list_subjects_NAKO_rsfMRI_high_quality.tsv",
)
LOW_QUALITY_FMRI_IDS = Path(
    NAKO_PATH,
    "derivatives_ses0/info_MRI_availability/list_subjects_NAKO_rsfMRI_low_quality.tsv",
)
METRICS_CSV_PATH = Path("/ritter/share/projects/Marina/DataLab/MRI_quality_metrics/all_quality_metrics_MRI_NAKO_ses0.csv")



class ModalityType(Enum):
    """Enum for the different MRI data types that can be processed"""

    RAW = "raw"
    ANAT = "anat"
    FUNC = "func"


class FeatureMapType(Enum):
    """
    Enum for the different feature map types that can be extracted from the MRI data
    
    fm.label: Label of the feature map
    fm.modality: Modality of the feature map
    fm.modality_label: Modality label of the feature map
    """

    # raw data (RAW modality)
    T1 = ("T1", ModalityType.RAW)
    BOLD = ("bold", ModalityType.RAW)

    # sMRI maps (ANAT modality)
    GM = ("GM", ModalityType.ANAT)
    WM = ("WM", ModalityType.ANAT)
    CSF = ("CSF", ModalityType.ANAT)

    # rs-fMRI maps (FUNC modality)
    REHO = ("reho", ModalityType.FUNC)
    #! The following feature maps are not yet implemented
    # ALFF = ("alff", ModalityType.FUNC)
    # fALFF = ("falff", ModalityType.FUNC)
    # VMHC = ("vmhc", ModalityType.FUNC)

    def __init__(self, feature_name: str, modality: ModalityType):
        self._feature_name = feature_name
        self._modality = modality

    @property
    def label(self) -> str:
        """Returns the label of the feature map"""
        return self._feature_name

    @property
    def modality(self) -> str:
        """Returns the modality of the feature map"""
        return self._modality

    @property
    def modality_label(self) -> str:
        """Returns the modality label of the feature map"""
        return self._modality.value

class TrainingMetric(Enum):
    # Main metrics
    LOSS = "loss"
    ACCURACY = "accuracy"
    R2 = "r2"

    # Binary classification metrics
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"

    # Regression metrics
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    SPEARMAN = "spearman"
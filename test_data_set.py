from pathlib import Path
from src.data_management.data_set import NakoSingleFeatureDataset
from src.utils.config import FMRI_PREP_FULL_SAMPLE, FeatureType, ModalityType
from src.utils.extract_subject_ids import get_subject_ids, save_subject_ids

ds = NakoSingleFeatureDataset([100000, 100001], {100000: 0, 100001: 1}, ModalityType.ANAT, FeatureType.GM)
print(len(ds))

print(ds[100000][1])

# Example usage
# subject_ids = get_subject_ids(FMRI_PREP_FULL_SAMPLE)
# print(subject_ids)
save_subject_ids(FMRI_PREP_FULL_SAMPLE, "src/utils/subject_ids.txt")


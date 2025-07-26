from pathlib import Path
from src.data_management.data_set import NakoSingleModalityDataset
from src.utils.config import FeatureMapType
from src.data_splitting.subject_selection import load_subject_ids_from_file

ds = NakoSingleModalityDataset(
    [100000, 100001], {100000: 0, 100001: 1}, FeatureMapType.GM
)
print(len(ds))

print(ds[100000][1])

# save_subject_ids(FMRI_PREP_FULL_SAMPLE)

print(load_subject_ids_from_file())

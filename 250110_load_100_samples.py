from src.data_management.data_set import NakoSingleFeatureDataset
from src.utils.subject_ids import sample_subject_ids
from src.utils.load_targets import extract_target
from src.utils.config import FeatureType, ModalityType

current_sample = sample_subject_ids(100)

ds = NakoSingleFeatureDataset(current_sample, ModalityType.ANAT, FeatureType.GM, "age")

for id, label in zip(ds.subject_ids, ds.labels):
    print(id, label)
    break

print(ds[0])
import torch
from torch.utils.data import Dataset

from data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureType, ModalityType
from src.utils.load_targets import extract_target
from src.utils.subject_ids import sample_subject_ids

class NakoSingleFeatureDataset(Dataset):
    def __init__(self, subject_ids: int, modality: ModalityType, feature_set: FeatureType, target: str):
        self.subject_ids = subject_ids
        self.labels = extract_target(target, subject_ids)
        self.modalitiy = modality
        self.feature_set = feature_set
        

    def __len__(self):
        return int(len(self.subject_ids))

    def __getitem__(self, idx: int):
        subject_id = self.subject_ids[idx]
        image_file = MriImageFile(subject_id, self.modalitiy, self.feature_set)
        feature_array = image_file.load_array()
        feature_tensor = torch.from_numpy(feature_array).float()
        # extract only the middle slice 
        feature_tensor = feature_tensor[feature_tensor.shape[0]//2]
        feature_tensor = feature_tensor.unsqueeze(0)#.unsqueeze(0)

        label = torch.tensor(self.labels[subject_id]).float()
        
        return feature_tensor, label
    

def prepare_standard_data_sets(n_samples: int = 128, val_test_frac: float = 1/8) -> tuple[NakoSingleFeatureDataset, NakoSingleFeatureDataset, NakoSingleFeatureDataset]:
    current_sample = sample_subject_ids(n_samples)
    train_size = int(n_samples * (1 - (2* val_test_frac)))
    val_size = int(n_samples * val_test_frac)

    # TODO: Device a better way to split the data (eg., k-fold cross-validation and stratified sampling)
    train_idxs = current_sample[:train_size]
    val_idxs = current_sample[train_size:train_size + val_size]
    test_idxs = current_sample[train_size + val_size:]

    train_idxs = sorted(train_idxs)
    val_idxs = sorted(val_idxs)
    test_idxs = sorted(test_idxs)

    params = {
        "modality": ModalityType.ANAT,
        "feature_set": FeatureType.GM,
        "target": "sex",
    }
    
    train_set = NakoSingleFeatureDataset(
        train_idxs, 
        **params
    )

    val_set = NakoSingleFeatureDataset(
        val_idxs, 
        **params
    )

    test_set = NakoSingleFeatureDataset(
        test_idxs, 
        **params
    )

    # Print length of data sets
    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}, Test set size: {len(test_set)}")

    return train_set, val_set, test_set


def print_details(data_set: Dataset):
    for id, label in zip(data_set.subject_ids, data_set.labels):
        print(id, label)
        print(data_set[0][0].shape)
        break
    
import torch
from torch.utils.data import Dataset

from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureType, ModalityType
from src.utils.load_targets import extract_target
from src.utils.subject_ids import sample_subject_ids

class NakoSingleFeatureDataset(Dataset):
    """PyTorch Dataset class for the NAKO dataset and its MRI feature maps"""
    def __init__(self, subject_ids: int, modality: ModalityType, feature_set: FeatureType, target: str, middle_slice: bool = True):
        self.subject_ids = subject_ids
        self.target = target
        self.labels = extract_target(target, subject_ids)
        self.modalitiy = modality
        self.feature_set = feature_set
        self.middle_slice = middle_slice
        # TODO: Remove this hacky approach
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean, self.std = None, None

        # Get shape of one sample (without channel dimension)
        image_file = MriImageFile(self.subject_ids[0], self.modalitiy, self.feature_set)
        sample_tensor = image_file.load_as_tensor(middle_slice=self.middle_slice)
        self.data_shape = tuple(sample_tensor.shape[1:])

        # TODO: Remove this hacky approach
        # Compute statistics
        if self.modalitiy == ModalityType.RAW:
            print("Computing statistics for normalization...")
            self.mean, self.std = self._compute_statistics()

    def __len__(self):
        return int(len(self.subject_ids))

    def __getitem__(self, idx: int):
        subject_id = self.subject_ids[idx]
        # Load the feature map as a tensor
        image_file = MriImageFile(subject_id, self.modalitiy, self.feature_set)
        feature_tensor = image_file.load_as_tensor(middle_slice=self.middle_slice)
        # TODO: Remove this hacky approach
        if self.modalitiy == ModalityType.RAW:
            # Normalize the tensor
            feature_tensor = (feature_tensor - self.mean) / self.std
        # Load the label as a tensor
        label = torch.tensor(self.labels[subject_id]).float()
        return feature_tensor, label
    
    # TODO: Remove this hacky approach
    def _compute_statistics(self):
        # Compute on a subset if dataset is too large
        num_samples = min(1000, len(self.subject_ids))
        sample_indices = torch.randperm(len(self.subject_ids))[:num_samples]
        
        # Process in smaller batches to avoid OOM
        batch_size = 100
        samples_list = []
        
        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i:i+batch_size]
            batch_samples = []
            
            for idx in batch_indices:
                image_file = MriImageFile(self.subject_ids[idx], self.modalitiy, self.feature_set)
                tensor = image_file.load_as_tensor(middle_slice=self.middle_slice)
                batch_samples.append(tensor)
            
            # Move batch to GPU, compute, then back to CPU
            batch_tensor = torch.stack(batch_samples).to(self.device)
            samples_list.append(batch_tensor)
            
        # Compute statistics on GPU
        samples = torch.cat(samples_list)
        mean = torch.mean(samples)
        std = torch.std(samples)
        
        # Return CPU tensors
        return mean.cpu(), std.cpu()
    

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
        "middle_slice": True
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
    
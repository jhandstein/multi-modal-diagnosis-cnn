import json
from pathlib import Path
from typing import Literal
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class MriImageNormalizer:
    """Class to normalize image data using z-score normalization"""
    
    def __init__(self, data_dim: Literal["2D", "3D"] = "2D"):
        self.mean: float | None = None
        self.std: float | None = None
        self.data_dim = data_dim

    def load_normalization_params(self):
        """Load normalization parameters from file"""
        param_path = Path("/home/julius/repositories/ccn_code/src/data_management/10k_split_mean_var.json")
        with open(param_path, "r") as file:
            data = json.load(file)
            self.mean = data[self.data_dim]["train"]["mean"]
            self.std = data[self.data_dim]["train"]["std"]
        
    def fit(self, 
            train_dataset: torch.utils.data.Dataset,
            batch_size: int = 32,
            num_workers: int = 4) -> None:
        """
        Calculate mean and standard deviation from training dataset
        
        Args:
            train_dataset: PyTorch Dataset object
            batch_size: Batch size for computing statistics
            num_workers: Number of workers for data loading
        """
        dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
        )
        
        # Use online/running statistics computation
        mean_accumulator = RunningAverage()
        var_accumulator = RunningAverage()
        
        # Show progress with tqdm
        with tqdm(dataloader, desc="Computing normalization stats") as pbar:
            for i, batch in enumerate(pbar):
                print(f"Processing batch {i}...")
                # Handle both tuple returns and direct tensor returns
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                
                # Ensure float type for accurate statistics
                if not torch.is_floating_point(images):
                    images = images.float()
                
                # Compute batch statistics
                batch_mean = images.mean().item()
                batch_var = images.var(unbiased=True).item()
                batch_size = images.size(0)
                
                # Update running statistics
                mean_accumulator.update(batch_mean, batch_size)
                var_accumulator.update(batch_var, batch_size)
        
        self.mean = mean_accumulator.average
        self.std = torch.sqrt(torch.tensor(var_accumulator.average)).item()
        
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization using computed statistics
        
        Args:
            tensor: Image tensor to normalize
            
        Returns:
            Normalized image tensor
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fit before transform")
            
        return transforms.Normalize(self.mean, self.std)(tensor)
    
    def fit_transform(self, 
                     train_dataset: torch.utils.data.Dataset,
                     batch_size: int = 32,
                     num_workers: int = 4) -> transforms.Normalize:
        """
        Fit to training dataset and return a torchvision transform
        
        Args:
            train_dataset: PyTorch Dataset object
            batch_size: Batch size for computing statistics
            num_workers: Number of workers for data loading
            
        Returns:
            A torchvision Normalize transform with computed statistics
        """
        self.fit(train_dataset, batch_size, num_workers)
        return transforms.Normalize(self.mean, self.std)
    
    @property
    def _is_fitted(self) -> bool:
        """Check if the normalizer has been fitted"""
        return self.mean is not None and self.std is not None
    
    def __repr__(self) -> str:
        """String representation of the normalizer"""
        return f"ImageNormalizer(mean={self.mean:.4f}, std={self.std:.4f})" if self._is_fitted else "ImageNormalizer(unfitted)"


class RunningAverage:
    """Helper class to compute running averages"""
    
    def __init__(self):
        self.total = 0.0
        self.count = 0
        
    def update(self, value: float, count: int = 1) -> None:
        """Update running average with new value and count"""
        self.total += value * count
        self.count += count
        
    @property
    def average(self) -> float:
        """Get current average"""
        if self.count == 0:
            raise ValueError("No values have been added to the running average")
        return self.total / self.count
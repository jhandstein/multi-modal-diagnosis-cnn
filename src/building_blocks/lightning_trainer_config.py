from dataclasses import asdict, dataclass
from typing import Optional

import torch


@dataclass
class LightningTrainerConfig():
    # GPU parameters
    accelerator: Optional[str] = "gpu" if torch.cuda.is_available() else None
    devices: int = 1 # number of GPUs
    strategy: str = "ddp"
    sync_batchnorm: bool = True
    benchmark: bool = False # best algorithm for your hardware, only works with homogenous input sizes

    # Training parameters
    deterministic: bool = True # works together with seed_everything to ensure reproducibility across runs
    max_epochs: int = 3

    # Logging parameters
    log_every_n_steps: int = 1

    def dict(self):
        return asdict(self)
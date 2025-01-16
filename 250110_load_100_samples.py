import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.accelerators import find_usable_cuda_devices

from src.building_blocks.lightning_network import BinaryClassificationCnn2d
from src.building_blocks.layers import ConvBranch2d
from src.data_management.data_set import NakoSingleFeatureDataset
from src.utils.subject_ids import sample_subject_ids
from src.utils.config import FeatureType, ModalityType

seed_everything(42, workers=True)

current_sample = sample_subject_ids(104)
train_set = NakoSingleFeatureDataset(current_sample[:80], ModalityType.ANAT, FeatureType.GM, "sex")
test_set = NakoSingleFeatureDataset(current_sample[80:], ModalityType.ANAT, FeatureType.GM, "sex")
 
lightning_model = BinaryClassificationCnn2d()


num_gpus = torch.cuda.device_count()
batch_size = 8

def print_details():
    for id, label in zip(train_set.subject_ids, train_set.labels):
        print(id, label)
        print(train_set[0][0].shape)
        break

def check_cuda():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")

        # Try to create a tensor on GPU
        try:            
            test_tensor = torch.tensor([1.0]).cuda()
            print(f"Successfully created tensor on {test_tensor.device}")
        except RuntimeError as e:
            print(f"Failed to create tensor on GPU: {e}")

        # List all visible devices
        # for i in range(torch.cuda.device_count()):
        #     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        #     print(f"Device {i} capability: {torch.cuda.get_device_capability(i)}")
    
    else:
        print("No GPU available. Training will run on CPU.")
    
    # Lightning helper function
    find_usable_cuda_devices() # does not work / only shows "cpu"

def print_cuda_version():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

def print_nccl_vars():
    print(f"NCCL_DEBUG: {os.environ.get('NCCL_DEBUG', 'Not set')}")
    print(f"NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE', 'Not set')}")
    print(f"NCCL_P2P_DISABLE: {os.environ.get('NCCL_P2P_DISABLE', 'Not set')}")

def train_model():


    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=False, # false because of trouble shooting during training/testing
        num_workers=num_gpus,
        drop_last=True # ensures that all the GPUs have the same number of batches
    )
    
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=num_gpus,
        max_epochs=3,
        strategy="ddp",
        sync_batchnorm=True,
        deterministic=True, # works together with seed_everything to ensure reproducibility across runs
        default_root_dir="models",
    )

    trainer.fit(
        model=lightning_model, 
        train_dataloaders=train_loader
    )

    # test_loader = DataLoader(
    #     ds_test, 
    #     batch_size=2, 
    #     shuffle=False, # false because of trouble shooting during training/testing
    #     num_workers=num_gpus,
    #     drop_last=True # ensures that all the GPUs have the same number of batches
    # )

    # trainer.test(
    #     model=lightning_model, 
    #     dataloaders=test_loader
    # )

    # for i in range(10):
    #     print(lightning_model.model(ds_train[i][0].unsqueeze(0)))
    
    trainer.test(ckpt_path="best")
    
# def test_model():
#     test_loader = DataLoader(ds_test, batch_size=16, shuffle=True, num_workers=num_gpus)
#     trainer = L.Trainer(devices=num_gpus)
#     results = trainer.test(model=model, dataloaders=test_loader)
#     return results

if __name__ == "__main__":
    # Set environment variables first
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_IB_DISABLE"] = "1"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # print_cuda_version()
    # print_nccl_vars()
    check_cuda()
    train_model()

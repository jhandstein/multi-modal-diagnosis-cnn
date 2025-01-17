import torch
import lightning as L
from lightning.pytorch import seed_everything

from src.data_management.data_loader import prepare_standard_data_loaders
from src.building_blocks.lightning_wrapper import BinaryClassificationCnn2d
from src.data_management.data_set import NakoSingleFeatureDataset, prepare_standard_data_sets
from src.utils.subject_ids import sample_subject_ids
from src.utils.config import FeatureType, ModalityType
from src.utils.check_cuda import check_cuda


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


def train_model():

    train_set, val_set, test_set = prepare_standard_data_sets()
    train_loader = prepare_standard_data_loaders(train_set, batch_size=batch_size, num_gpus=num_gpus)
    val_loader = prepare_standard_data_loaders(val_set, batch_size=batch_size, num_gpus=num_gpus)
    
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=num_gpus,
        max_epochs=3,
        strategy="ddp",
        sync_batchnorm=True,
        deterministic=True, # works together with seed_everything to ensure reproducibility across runs
        default_root_dir="models",
        benchmark=True, # best algorithm for your hardware, only works with homogenous input sizes
    )


    trainer.fit(
        model=lightning_model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

   
    # TODO: Test model and log train loss
    
    
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
    
    check_cuda()
    train_model()

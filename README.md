# multi-modal-diagnosis-cnn
This project provides a flexible framework for predicting neural phenotypes from MRI data using convolutional neural networks (CNNs), specifically ResNet18 architectures. The framework supports multi-modal and multi-dimensional MRI feature maps, enabling both anatomical and functional data to be used for classification or regression tasks (e.g., age, sex, PHQ-9, GAD-7).

- Runs on python 3.12
- Requirements are found in reuqirements.txt


# Talk about execution of train_model.py
## Key Features:
- Flexible Data Handling: Supports different MRI feature maps (e.g., T1, GM, BOLD) and spatial dimensions (2D/3D).
- Multi-Modality: Easily combine anatomical and functional modalities.
- Custom Data Splitting: Advanced phenotype-based and quality-based data splits.
- Training & Evaluation: Modular training pipeline with PyTorch Lightning, including logging, metrics, and plotting.
- Extensible: Modular codebase for easy adaptation to new tasks or data types.

## Excecution of training script
- Files should be executed from the root of the directory to prevent import errors (same for train_model.py and test.py)
- Before the training can be successfully executed, a datasplit for the NAKO has to be created
  - accessed in ` data_split = DataSplitFile(data_set_path).load_data_splits_from_file()` in the train_model.py script
  - data_splits are .json files containing NAKO subject_ids as list for train, val und test sets ("train": [ids], "val": [ids], ...)
  - tools to create a DataSplitFile are located in src/data_splitting/..., most specifically in .../create_data_split.py
- Python file can be executed using `python3 train_model.py run_prefix -c cuda01 -g 1`
- arguments:
  - run_prefix: identifier string for the experiement (required)
  - -c/--compute_node: cuda01/cuda02 (required)
  - -g/--num_gpus: integer, should be divisible by 2 (optional)
  - -s/--seed: integer seed for reproducibility (optional)
- bash script (automatically executes with nohup for background execution) `bash run_training.sh run_prefix -c cuda01 ...`
- as batch for multiple seed `bash run_batch.sh run_prefix -c cuda01 ...`


## Folder Structure
- building_blocks/
Core model components, wrappers, and training utilities (e.g., model factory, Lightning wrappers, metrics callbacks).

- data_management/
Data loading, normalization, dataset configuration, and feature map handling.

- data_splitting/
Advanced subject selection, phenotype-based and quality-based splitting, and split file management.

- plots/
Visualization utilities for MRI slices, metrics, and age distributions.

- utils/
Utility functions for configuration, CUDA management, performance evaluation, and file handling.

## Main Scripts
- train_model.py
Main training script. Handles argument parsing, data preparation, model instantiation, training, and evaluation. Uses PyTorch Lightning for scalable and reproducible training.

- test.py
Utility script for quick testing of data loading, plotting, and metric evaluation.

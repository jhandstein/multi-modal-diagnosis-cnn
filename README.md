# multi-modal-diagnosis-cnn
This project provides a flexible framework for predicting neural phenotypes from MRI data using convolutional neural networks (CNNs), specifically ResNet18 architectures. The framework supports multi-modal and multi-dimensional MRI feature maps, enabling both anatomical and functional data to be used for classification or regression tasks (e.g., age, sex, PHQ-9, GAD-7).

## Key Features:
Flexible Data Handling: Supports different MRI feature maps (e.g., T1, GM, BOLD) and spatial dimensions (2D/3D).
Multi-Modality: Easily combine anatomical and functional modalities.
Custom Data Splitting: Advanced phenotype-based and quality-based data splits.
Training & Evaluation: Modular training pipeline with PyTorch Lightning, including logging, metrics, and plotting.
Extensible: Modular codebase for easy adaptation to new tasks or data types.

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
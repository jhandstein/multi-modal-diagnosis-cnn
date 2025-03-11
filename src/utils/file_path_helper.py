from typing import Literal
from datetime import datetime

from src.building_blocks.resnet18 import ResNet18Base2d
from src.building_blocks.custom_model import BaseConvBranch2d
from src.data_management.data_set import NakoSingleFeatureDataset


def construct_model_name(model: BaseConvBranch2d | ResNet18Base2d, data_set: NakoSingleFeatureDataset, experiment: str, compute_node: str) -> str:
    """Constructs a directory name for the model based on the data set and dimensionality.
    
    Returns format: {date}_{compute_node}_{experiment}_{model}_{target}_{modality}_{feature}
    Example: 240308_gpu01_pretraining_ResNet18Base2d_age_anat_GM
    """
    date_str = datetime.now().strftime("%y%m%d")
    model_tag = model.__class__.__name__
    modality = data_set.feature_map.modality_label
    feature_map = data_set.feature_map.label
    target = data_set.target
    
    return f"{date_str}_{compute_node}_{experiment}_{model_tag}_{target}_{modality}_{feature_map}"

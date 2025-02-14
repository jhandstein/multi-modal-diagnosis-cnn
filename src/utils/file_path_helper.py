from typing import Literal
from datetime import datetime

from src.building_blocks.resnet18 import ResNet18Base2d
from src.building_blocks.custom_model import BaseConvBranch2d
from src.data_management.data_set import NakoSingleFeatureDataset


def construct_model_name(model: BaseConvBranch2d | ResNet18Base2d, data_set: NakoSingleFeatureDataset, task: Literal["classifcation", "regression"], dim: Literal["2D", "3D"] = "2D") -> str:
    """Constructs a name for the model based on the data set and dimensionality."""
    date_str = datetime.now().strftime("%y%m%d")
    model_tag = model.__class__.__name__
    modality = data_set.feature_map.modality_label
    feature_map = data_set.feature_map.label
    target = data_set.target
    return f"{date_str}_{model_tag}_{dim}_{modality}_{feature_map}_{task}_{target}"

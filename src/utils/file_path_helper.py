from typing import Literal
from src.data_management.data_set import NakoSingleFeatureDataset


def construct_modal_name(data_set: NakoSingleFeatureDataset, task: Literal["classifcation", "regression"], dim: Literal["2D", "3D"] = "2D") -> str:
    """Constructs a name for the model based on the data set and dimensionality."""
    modality = data_set.feature_map.modality_label
    feature_map = data_set.feature_map.label
    target = data_set.target
    return f"CNN_{dim}_{modality}_{feature_map}_{task}_{target}"

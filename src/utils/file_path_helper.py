from datetime import datetime

from src.building_blocks.resnet18 import ResNet18Base2d
from src.building_blocks.custom_model import BaseConvBranch2d
from src.data_management.data_set import NakoMultiModalityDataset, NakoSingleModalityDataset
from src.utils.process_metrics import get_modality_and_features


def construct_model_name(model: BaseConvBranch2d | ResNet18Base2d, data_set: NakoSingleModalityDataset | NakoMultiModalityDataset, experiment: str, compute_node: str) -> str:
    """Constructs a directory name for the model based on the data set and dimensionality.
    
    Returns format: {date}_{compute_node}_{experiment}_{model}_{target}_{modality}_{feature}
    Example: 240308_gpu01_pretraining_ResNet18Base2d_age_anat_GM
    """
    date_str = datetime.now().strftime("%y%m%d")
    model_tag = model.__class__.__name__
    target = data_set.target

    modality, features = get_modality_and_features(data_set)
    features_str = "_".join(
        features.get("feature_maps", []) or 
        features.get("anatomical_maps", []) + features.get("functional_maps", [])
    )
    
    return f"{date_str}_{compute_node}_{experiment}_{model_tag}_{target}_{modality}_{features_str}"
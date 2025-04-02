from dataclasses import dataclass
from typing import Literal

from src.building_blocks.custom_model import ConvBranch2dBinary, ConvBranch2dRegression, ConvBranch3dBinary, ConvBranch3dRegression
from src.building_blocks.resnet18 import ResNet18Binary2d, ResNet18Binary2dDualModality, ResNet18Regression2d, ResNet18Binary3d, ResNet18Regression2dDualModality, ResNet18Regression3d


@dataclass
class ModelFactory:
    """Factory class for creating models based on the task and dimensionality."""

    task: Literal["classification", "regression"]
    dim: Literal["2D", "3D"]

    _conv_branch_models = {
        ("2D", "classification"): ConvBranch2dBinary,
        ("2D", "regression"): ConvBranch2dRegression,
        ("3D", "classification"): ConvBranch3dBinary,
        ("3D", "regression"): ConvBranch3dRegression,
    }

    _resnet_models = {
        ("2D", "classification"): ResNet18Binary2d,
        ("2D", "regression"): ResNet18Regression2d,
        ("3D", "classification"): ResNet18Binary3d,
        ("3D", "regression"): ResNet18Regression3d,
    }

    _resnet_dual_modality_models = {
        ("2D", "classification"): ResNet18Binary2dDualModality,
        ("2D", "regression"): ResNet18Regression2dDualModality,
    }

    def create_conv_branch(self, input_shape: tuple):
        model_class = self._conv_branch_models.get((self.dim, self.task))
        if not model_class:
            raise ValueError("Model variant not supported. Check the task and dim arguments.")
        return model_class(input_shape)

    def create_resnet18(self, in_channels: int):
        model_class = self._resnet_models.get((self.dim, self.task))
        if not model_class:
            raise ValueError("Model variant not supported. Check the task and dim arguments.")
        return model_class(in_channels)
    
    def create_resnet_multi_modal(self, anat_channels: int, func_channels: int):
        """Creates a dual modality model with separate branches for anatomical and functional data."""
        if self.dim == "3D":
            raise NotImplementedError("3D fusion models not yet supported")
            
        fusion_model_class = self._resnet_dual_modality_models.get((self.dim, self.task))
        if not fusion_model_class:
            raise ValueError("Fusion model variant not supported. Check the task and dim arguments.")
            
        return fusion_model_class(anat_channels, func_channels)
    
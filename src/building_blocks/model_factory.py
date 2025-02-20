from dataclasses import dataclass
from typing import Literal

from src.building_blocks.custom_model import ConvBranch2dBinary, ConvBranch2dRegression, ConvBranch3dBinary, ConvBranch3dRegression
from src.building_blocks.resnet18 import ResNet18Binary2d, ResNet18Regression2d, ResNet18Binary3d, ResNet18Regression3d


@dataclass
class ModelFactory:
    task: Literal["classification", "regression"]
    dim: Literal["2D", "3D"]

    def create_conv_branch(self, input_shape: tuple):
        if self.dim == "2D":
            if self.task == "classification":
                return ConvBranch2dBinary(input_shape)
            elif self.task == "regression":
                return ConvBranch2dRegression(input_shape)
        elif self.dim == "3D":
            if self.task == "classification":
                return ConvBranch3dBinary(input_shape)
            elif self.task == "regression":
                return ConvBranch3dRegression(input_shape)
        else:
            raise ValueError("Model variant not supported. Check the task and dim arguments.")
        
    def create_resnet18(self, in_channels: int):
        if self.dim == "2D":
            if self.task == "classification":
                return ResNet18Binary2d(in_channels)
            elif self.task == "regression":
                return ResNet18Regression2d(in_channels)
        elif self.dim == "3D":
            if self.task == "classification":
                return ResNet18Binary3d(in_channels)
            elif self.task == "regression":
                return ResNet18Regression3d(in_channels)
        else:
            raise ValueError("Model variant not supported. Check the task and dim arguments.")
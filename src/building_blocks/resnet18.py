from torch import nn
from torchvision.models import resnet18

class FlexibleResNet(nn.Module):
    def __init__(
        self, 
        input_channels=1, 
        output_dim=1,
        task_type='regression',
        output_activation=None
    ):
        super().__init__()
        
        # Load basic ResNet18 without pretrained weights
        self.resnet = resnet18(weights=None)
        
        # Modify first conv layer to accept different input channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Modify final fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, output_dim)
        
        # Set output activation based on task
        self.task_type = task_type
        self.output_activation = output_activation or self._get_default_activation()
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.output_activation(x)
        return x.squeeze()
    
    def _get_default_activation(self):
        if self.task_type == "regression":
            return nn.Identity()
        elif self.task_type == "classification":
            return nn.Sigmoid()
        else:  # multiclass
            ValueError(f"Task type {self.task_type} not supported.")

from torch import nn
from torchvision.models import resnet18

# class FlexibleResNet(nn.Module):
#     def __init__(
#         self, 
#         task_type="regression",
#         input_channels=1, 
#         output_dim=1,
#         output_activation=None
#     ):
#         super().__init__()
        
#         # Load basic ResNet18 without pretrained weights
#         self.resnet = resnet18(weights=None)
        
#         # Modify first conv layer to accept different input channels
#         self.resnet.conv1 = nn.Conv2d(
#             in_channels=input_channels,
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False
#         )
        
#         # Modify final fc layer
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, output_dim)
        
#         # Set output activation based on task
#         self.task_type = task_type
#         self.output_activation = output_activation or self._get_default_activation()
        
#     def forward(self, x):
#         x = self.resnet(x)
#         x = self.output_activation(x)
#         return x.squeeze()
    
#     def _get_default_activation(self):
#         if self.task_type == "regression":
#             return nn.Identity()
#         elif self.task_type == "classification":
#             return nn.Sigmoid()
#         else:  # multiclass
#             raise ValueError(f"Task type {self.task_type} not supported.")


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class BaseResNet18(nn.Module):
    def __init__(self, in_channels=3):
        super(BaseResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward_convolution(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class ResNet18Binary(BaseResNet18):
    def __init__(self, in_channels=3):
        super(ResNet18Binary, self).__init__(in_channels)
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.forward_convolution(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()

class ResNet18Regression(BaseResNet18):
    def __init__(self, in_channels=3):
        super(ResNet18Regression, self).__init__(in_channels)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.forward_convolution(x)
        out = self.fc(out)
        return out.squeeze()
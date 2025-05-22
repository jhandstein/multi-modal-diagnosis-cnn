from torch import nn
import torch


class BasicBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
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


class ResNet18Base2d(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet18Base2d, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock2d, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock2d, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock2d, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock2d, 512, 2, stride=2)

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


class ResNet18Binary2d(ResNet18Base2d):
    def __init__(self, in_channels=3):
        super(ResNet18Binary2d, self).__init__(in_channels)
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.forward_convolution(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.view(-1)


class ResNet18Regression2d(ResNet18Base2d):
    def __init__(self, in_channels=3):
        super(ResNet18Regression2d, self).__init__(in_channels)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.forward_convolution(x)
        out = self.fc(out)
        return out.view(-1)


class ResNet18DualModality2dBase(nn.Module):
    def __init__(self, anat_channels: int, func_channels: int):
        super(ResNet18DualModality2dBase, self).__init__()
        # Create base models without fc layers
        self.anat_branch = ResNet18Base2d(in_channels=anat_channels)
        self.func_branch = ResNet18Base2d(in_channels=func_channels)

    def forward_branches(self, x1, x2):
        out1 = self.anat_branch.forward_convolution(x1)
        out2 = self.func_branch.forward_convolution(x2)
        return torch.cat((out1, out2), dim=1)


class ResNet18Binary2dDualModality(ResNet18DualModality2dBase):
    def __init__(self, anat_channels: int, func_channels: int):
        super().__init__(anat_channels, func_channels)
        self.fc = nn.Linear(512 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        out = self.forward_branches(x1, x2)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.view(-1)


class ResNet18Regression2dDualModality(ResNet18DualModality2dBase):
    def __init__(self, anat_channels: int, func_channels: int):
        super().__init__(anat_channels, func_channels)
        self.fc = nn.Linear(512 * 2, 1)

    def forward(self, x1, x2):
        out = self.forward_branches(x1, x2)
        out = self.fc(out)
        return out.view(-1)


class BasicBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
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


class ResNet18Base3d(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet18Base3d, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock3d, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock3d, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock3d, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock3d, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

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


class ResNet18Binary3d(ResNet18Base3d):
    def __init__(self, in_channels=1):
        super(ResNet18Binary3d, self).__init__(in_channels)
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.forward_convolution(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.view(-1)


class ResNet18Regression3d(ResNet18Base3d):
    def __init__(self, in_channels=1):
        super(ResNet18Regression3d, self).__init__(in_channels)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.forward_convolution(x)
        out = self.fc(out)
        return out.view(-1)

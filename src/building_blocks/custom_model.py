import torch.nn as nn

from src.utils.calc_size_after_conv import ConvCalculator


class BaseConvBranch2d(nn.Module):
    """Base class for 2D convolutional branches that takes in a single feature map."""
    def __init__(self, input_shape: tuple):
        super().__init__()
        self.input_shape = input_shape

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
        )

        # Calculate output dimensions
        conv_calc = ConvCalculator(kernel_size=5, stride=1, padding=1)
        x, y = conv_calc.calculate_network_output_size(input_shape)
        self.feature_dim = 64 * x * y

        # Base layers
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 512)

    def forward_convolution(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x

class ConvBranch2dBinary(BaseConvBranch2d):
    def __init__(self, input_shape: tuple):
        super().__init__(input_shape)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.forward_convolution(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze()

class ConvBranch2dRegression(BaseConvBranch2d):
    def __init__(self, input_shape: tuple):
        super().__init__(input_shape)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.forward_convolution(x)
        x = self.fc2(x)
        return x.squeeze()
    

class BaseConvBranch3d(nn.Module):
    """Base class for 3D convolutional branches that takes in a single feature map."""
    def __init__(self, input_shape: tuple):
        super().__init__()
        self.input_shape = input_shape

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
        )

        # Calculate output dimensions
        conv_calc = ConvCalculator(kernel_size=5, stride=1, padding=1)
        x, y, z = conv_calc.calculate_network_output_size(input_shape)
        self.feature_dim = 64 * x * y * z

        # Base layers
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 512)

    def forward_convolution(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x
    

class ConvBranch3dBinary(BaseConvBranch3d):
    def __init__(self, input_shape: tuple):
        super().__init__(input_shape)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.forward_convolution(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze()
    

class ConvBranch3dRegression(BaseConvBranch3d):
    def __init__(self, input_shape: tuple):
        super().__init__(input_shape)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.forward_convolution(x)
        x = self.fc2(x)
        return x.squeeze()
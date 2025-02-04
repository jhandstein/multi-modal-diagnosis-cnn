from typing import Literal
import torch.nn as nn

from src.utils.calc_size_after_conv import ConvCalculator


class ConvBranch2d(nn.Module):
    def __init__(
        self, input_shape: tuple, task: Literal["classification", "regression"]
    ):
        super().__init__()
        self.input_shape = input_shape
        self.task = task

        # convolutional layer
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

        # dimensions of the image after 4 conv layers
        conv_calc = ConvCalculator(kernel_size=5, stride=1, padding=1)
        x, y = conv_calc.calculate_network_output_size(input_shape)

        # fully connected layers
        self.fc1 = nn.Linear(64 * x * y, 512)
        self.fc2 = nn.Linear(512, 1)

        # dropout layer for the input of the fully connected layers
        self.dropout = nn.Dropout(p=0.5)

        # sigmoid activation function to convert to class probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten tensor for fully connected layers (while keeping batch dimension)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.task == "classification":
            x = self.sigmoid(x)
        # Squeeze the output to match the shape of the labels
        return x.squeeze()


class ConvBranch3d(nn.Module):

    def __init__(
        self, input_shape: tuple, task: Literal["classification", "regression"]
    ):
        super().__init__()

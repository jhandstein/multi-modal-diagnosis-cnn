import torch.nn as nn

from src.utils.calc_size_after_conv import calculate_2d_conv_ouput_size

class ConvBranch2d(nn.Module):
    def __init__(self, input_shape: tuple):
        super().__init__()
        self.input_shape = input_shape
        # convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
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
        x, y = calculate_2d_conv_ouput_size(input_shape, kernel_size=5, stride=1, padding=1)
        # fully connected layers
        self.fc1 = nn.Linear(64 * x * y, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape, "before conv layers")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape, "after conv layers")
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        # Squeeze the output to match the shape of the labels
        return x.squeeze()
    


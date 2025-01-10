import torch.nn as nn

class ConvBranch2d(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=1),
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

        # fully connected layers
        x, y = 12, 12 # dimensions of the image after 4 conv layers
        self.fc1 = nn.Linear(64 * x * y, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    


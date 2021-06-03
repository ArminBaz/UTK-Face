'''
    Code for a Multi Label Neural Network

    The Initial (Shared) layers are high-level feature extractors using convolutions.
    The three lower level feature extractors consist of another convolutional layer followed by a few Dense layers.
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d

# High level feature extractor network (Adopted VGG type structure)
class highLevelNN(nn.Module):
    def __init__(self):
        super(highLevelNN, self).__init__()
        self.CNN = nn.Sequential(
            # first batch (32)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # second batch (64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Third Batch (128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.CNN(x)

        return out

# Low level feature extraction module
class lowLevelNN(nn.Module):
    def __init__(self, num_out, age=False):
        super(lowLevelNN, self).__init__()
        self.age = age
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=num_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=2, padding=1))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if self.age == True:
            return F.relu(x)
        else:
            return x


class TridentNN(nn.Module):
    def __init__(self, num_age, num_gen, num_eth):
        super(TridentNN, self).__init__()
        # Construct the high level neural network
        self.CNN = highLevelNN()
        # Construct the low level neural networks
        self.ageNN = lowLevelNN(num_out=1, age=True)
        self.genNN = lowLevelNN(num_out=num_gen, age=False)
        self.ethNN = lowLevelNN(num_out=num_eth, age=False)

    def forward(self, x):
        x = self.CNN(x)
        age = self.ageNN(x)
        gen = self.genNN(x)
        eth = self.ethNN(x)

        return age, gen, eth

if __name__ == '__main__':
    print('Testing out Multi-Label NN')
    mlNN = TridentNN(10, 10, 10)
    input = torch.randn(64, 1, 48, 48)
    y = mlNN(input)
    print(y[0].shape)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FullyConnectNet(nn.Module):
    def __init__(self, lr=1e-2):
        super(FullyConnectNet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 96, 500),
            nn.ReLU(),
            nn.Linear(500, 30)
        )
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet(nn.Module):
    '''
    TODO: structurally adapt from homework 3
    '''

    def __init__(self, lr=1e-2):
        super(ResNet, self).__init__()
        self.model = nn.Sequential()
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        x = self.model(x)
        return x

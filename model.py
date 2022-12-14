import torch.nn as nn
import torch.optim as optim


class FullyConnectNet:
    def __init__(self, lr=1e-2):
        self.model = nn.Sequential(
            nn.Linear(96 * 96, 500),
            nn.ReLU(),
            nn.Linear(500, 30)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

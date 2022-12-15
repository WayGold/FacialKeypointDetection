import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from resnet_utils import ResidualBlock, ResNetStem, ResNetStage, ResidualBottleneckBlock


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 96, 500),
            nn.ReLU(),
            nn.Linear(500, 30)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet(nn.Module):
    def __init__(self, stage_args, in_channels=1, block=ResidualBlock, num_classes=30):
        super().__init__()
        layers = [ResNetStem(in_channels, stage_args[0][0]), *[ResNetStage(*stage_arg, block) for stage_arg in stage_args]]
        self.cnn = nn.Sequential(*layers)
        self.fcc = nn.Linear(stage_args[-1][1], num_classes)

    def forward(self, x):
        x = self.cnn(x)
        H, W = x.shape[2], x.shape[3]
        x = F.avg_pool2d(x, kernel_size=(H, W))
        x = x.flatten(1, -1)
        x = self.fcc(x)
        return x


def resnet32():
    stage_args = [(8, 8, 5, False), (8, 16, 5, True), (16, 32, 5, True)]
    return ResNet(stage_args, block=ResidualBlock)


def resnet47():
    stage_args = [(32, 32, 5, False), (32, 64, 5, True), (64, 128, 5, True)]
    return ResNet(stage_args, block=ResidualBottleneckBlock)

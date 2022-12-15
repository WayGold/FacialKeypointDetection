import torch.nn as nn


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=[2 if downsample else 1], padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.block = PlainBlock(in_channels, out_channels, downsample)
        if not downsample:
            self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x = self.block(x) + self.shortcut(x)
        return x


class ResNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, downsample=True, block=ResidualBlock):
        super().__init__()
        blocks = [block(in_channels, out_channels, downsample)]
        for _ in range(num_blocks - 1):
            blocks.append(block(out_channels, out_channels))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=[2 if downsample else 1], padding=0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0)
        )
        if not downsample:
            self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x = self.block(x) + self.shortcut(x)
        return x

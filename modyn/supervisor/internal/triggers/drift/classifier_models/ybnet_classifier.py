import torch
from torch import nn

from modyn.models.coreset_methods_support import CoresetSupportingModule


class YearbookNetDriftDetector(CoresetSupportingModule):
    def __init__(self, num_input_channels: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            self.conv_block(num_input_channels, 32),
            self.conv_block(32, 32),
            self.conv_block(32, 32),
            self.conv_block(32, 32),
        )
        self.hid_dim = 32
        # Binary classifier for drift detection
        # see: https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/classifierdrift.html
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32, 2))

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.enc(data)
        data = torch.mean(data, dim=(2, 3))
        data = self.classifier(data)
        return data

    def get_last_layer(self) -> nn.Module:
        return self.classifier

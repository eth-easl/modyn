from typing import Any

import torch
from modyn.models.coreset_methods_support import CoresetSupportingModule
from torch import nn


class SmallYearbookNet:
    """
    Adapted from WildTime.
    Here you can find the original implementation:
    https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/networks/yearbook.py
    Can be used for experiments on RHO-LOSS as the IL model.
    """

    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = SmallYearbookNetModel(**model_configuration)
        self.model.to(device)


class SmallYearbookNetModel(CoresetSupportingModule):
    def __init__(self, num_input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            self.conv_block(num_input_channels, 16),
            self.conv_block(16, 16),
            self.conv_block(16, 16),
        )
        self.hid_dim = 16
        self.classifier = nn.Linear(16, num_classes)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.enc(data)
        data = torch.mean(data, dim=(2, 3))
        data = self.embedding_recorder(data)
        return self.classifier(data)

    def get_last_layer(self) -> nn.Module:
        return self.classifier

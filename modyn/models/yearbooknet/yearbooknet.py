from typing import Any, Optional

import torch
from modyn.models.coreset_methods_support import CoresetSupportingModule
from torch import nn


class YearbookNet:
    """
    Adapted from WildTime.
    Here you can find the original implementation:
    https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/networks/yearbook.py
    """

    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = YearbookNetModel(**model_configuration)
        self.model.to(device)


class YearbookNetModel(CoresetSupportingModule):
    def __init__(self, num_input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            self.conv_block(num_input_channels, 32),
            self.conv_block(32, 32),
            self.conv_block(32, 32),
            self.conv_block(32, 32),
        )
        self.hid_dim = 32
        self.classifier = nn.Linear(32, num_classes)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)
        )

    def forward(self, data: torch.Tensor, sample_ids: Optional[list[int]]) -> torch.Tensor:
        data = self.enc(data)
        data = torch.mean(data, dim=(2, 3))
        data = self.embedding_recorder(data)
        return self.classifier(data)

    def get_last_layer(self) -> nn.Module:
        return self.classifier

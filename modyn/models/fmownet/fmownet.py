from typing import Any, Optional

import torch
import torch.nn.functional as func
from modyn.models.coreset_methods_support import CoresetSupportingModule
from torch import nn
from torchvision.models import densenet121


class FmowNet:
    """
    Adapted from WildTime.
    Here you can find the original implementation:
    https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/networks/fmow.py
    """

    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = FmowNetModel(**model_configuration)
        self.model.to(device)


class FmowNetModel(CoresetSupportingModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.enc = densenet121(pretrained=True).features
        self.classifier = nn.Linear(1024, self.num_classes)

    def forward(self, data: torch.Tensor, sample_ids: Optional[list[int]] = None) -> torch.Tensor:
        features = self.enc(data)
        out = func.relu(features, inplace=True)
        out = func.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.embedding_recorder(out)
        return self.classifier(out)

    def get_last_layer(self) -> nn.Module:
        return self.classifier

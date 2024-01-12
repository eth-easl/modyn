from typing import Any

import torch
from modyn.models.coreset_methods_support import CoresetSupportingModule
from torch import Tensor, nn
from torchvision.models.resnet import Bottleneck, ResNet


class ResNet152:
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = ResNet152Modyn(model_configuration)
        self.model.to(device)


# the following class is adapted from
# torchvision https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


class ResNet152Modyn(ResNet, CoresetSupportingModule):
    def __init__(self, model_configuration: dict[str, Any]) -> None:
        super().__init__(Bottleneck, [3, 8, 36, 3], **model_configuration)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # the following line is the only difference compared to the original implementation
        x = self.embedding_recorder(x)
        x = self.fc(x)

        return x

    def get_last_layer(self) -> nn.Module:
        return self.fc

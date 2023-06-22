from typing import Any

import torch
import torch.nn.functional as F
from modyn.models.modyn_model import ModynModel
from torch import nn

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,
# https://github.com/PatrickZH/DeepCore/


class ResNet18:
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = ResNet18Model(**model_configuration)
        self.model.to(device)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(in_data)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(in_data)
        out = F.relu(out)
        return out


class ResNet18Model(ModynModel):
    def __init__(self, channel: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(channel, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(planes=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(planes=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(planes=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(planes=512, num_blocks=2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def get_last_layer(self) -> nn.Module:
        return self.linear

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for _stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, _stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(in_data)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.embedding_recorder(out)
        out = self.linear(out)
        return out

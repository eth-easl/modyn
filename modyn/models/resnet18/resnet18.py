from typing import Any, Optional

import torch
from modyn.models.coreset_methods_support import CoresetSupportingModule
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet, ResNet18_Weights


class ResNet18:
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = ResNet18Modyn(model_configuration)
        self.model.to(device)


# the following class is adapted from
# torchvision https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


class ResNet18Modyn(ResNet, CoresetSupportingModule):
    def __init__(self, model_configuration: dict[str, Any]) -> None:
        _num_classes = model_configuration.get("num_classes", None)
        weights = None
        if model_configuration.get("use_pretrained", False):
            weights = ResNet18_Weights.verify("ResNet18_Weights.DEFAULT")
            # We need to initialize the model with the number of classees
            # in the pretrained weights
            model_configuration["num_classes"] = len(weights.meta["categories"])
            del model_configuration["use_pretrained"]  # don't want to forward this to torchvision

        super().__init__(BasicBlock, [2, 2, 2, 2], **model_configuration)  # type: ignore

        if weights is not None:
            self.load_state_dict(weights.get_state_dict(progress=True))
            if _num_classes is not None:
                # we loaded pretrained weights - need to update linear layer
                self.fc: nn.Linear  # to satisfy mypy
                self.fc = nn.Linear(self.fc.in_features, _num_classes)

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

    def forward(self, data: torch.Tensor, sample_ids: Optional[list[int]] = None) -> torch.Tensor:
        return super().forward(data)

    def get_last_layer(self) -> nn.Module:
        return self.fc

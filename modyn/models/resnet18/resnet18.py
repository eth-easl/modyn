from typing import Any

from torchvision import models


class ResNet18:
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = models.__dict__["resnet18"](**model_configuration)
        self.model.to(device)

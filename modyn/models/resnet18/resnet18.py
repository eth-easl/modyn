from typing import Any

from torchvision import models


class ResNet18:
    def __init__(
        self,
        model_configuration: dict[str, Any],
    ) -> None:
        device = model_configuration.pop("device")
        model_configuration.pop("amp")
        self.model = models.__dict__["resnet18"](**model_configuration)
        self.model.to(device)

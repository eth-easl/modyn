from typing import Any

# torchvision resnet updated to add the embedding recorder.
from modyn.models.resnet18.resnet_torchvision import resnet18


class ResNet18:
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = resnet18(**model_configuration)
        self.model.to(device)

from typing import Any

from modyn.models.coreset_methods_support import CoresetSupportingModule
from torch import nn


class Dummy:
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = DummyModyn(model_configuration)
        self.model.to(device)


class DummyModyn(nn.Identity, CoresetSupportingModule):
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any]) -> None:
        super().__init__()

    def get_last_layer(self) -> nn.Module:
        return nn.Identity()

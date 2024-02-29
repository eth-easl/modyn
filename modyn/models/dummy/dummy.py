from typing import Any

from torch import Tensor, nn


class Dummy:
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = DummyModyn(model_configuration)
        self.model.to(device)


class DummyModyn(nn.Module):
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any]) -> None:
        super().__init__()
        self.output = nn.Linear(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(x)

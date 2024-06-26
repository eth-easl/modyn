import logging
from typing import Any, Optional

import torch
from modyn.utils import dynamic_module_import
from torch import nn

logger = logging.getLogger(__name__)


class RHOLOSSTwinModel:

    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = RHOLOSSTwinModelModyn(model_configuration, device, amp)
        self.model.to(device)


class RHOLOSSTwinModelModyn(nn.Module):

    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        super().__init__()
        self.device = device
        model_module = dynamic_module_import("modyn.models")
        rho_model_class = model_configuration["rho_real_model_class"]
        rho_model_config = model_configuration["rho_real_model_config"]
        model_handler = getattr(model_module, rho_model_class)
        # we only need the inner model, not the wrapper
        self._models = nn.ModuleList(
            [
                model_handler(rho_model_config, device, amp).model,
                model_handler(rho_model_config, device, amp).model,
            ]
        )
        self._models_seen_ids: list[set[int]] = [set(), set()]
        self._current_model = 0

    def get_extra_state(self) -> Any:
        return {
            "models_seen_ids": self._models_seen_ids,
        }

    def set_extra_state(self, state: dict) -> None:
        self._models_seen_ids = state["models_seen_ids"]
        # the second time we train on this model, we should switch to the other model
        self._current_model = 1

    def forward(self, data: torch.Tensor, sample_ids: Optional[list[int]] = None) -> torch.Tensor:
        assert sample_ids is not None
        # self.training is an internal attribute defined in nn.Module that is updated
        # whenever .eval() or .train() is called
        if self.training:
            output_tensor = self._training_forward(sample_ids, data)
        else:
            output_tensor = self._eval_forward(sample_ids, data)
        return output_tensor

    def _training_forward(self, sample_ids: list[int], data: torch.Tensor) -> torch.Tensor:
        self._models_seen_ids[self._current_model].update(sample_ids)
        return self._models[self._current_model](data)

    def _eval_forward(self, sample_ids: list[int], data: torch.Tensor) -> torch.Tensor:
        seen_by_model0 = torch.BoolTensor(
            [sample_id in self._models_seen_ids[0] for sample_id in sample_ids], device=self.device
        )
        seen_by_model1 = torch.BoolTensor(
            [sample_id in self._models_seen_ids[1] for sample_id in sample_ids], device=self.device
        )

        # if model 0 did not see any sample, we route all samples to model 0
        if not seen_by_model0.any():
            return self._models[0](data)
        # if model 1 did not see any sample, we route all samples to model 1
        if not seen_by_model1.any():
            return self._models[1](data)

        # when a sample is not seen by any model, we route it to model 0
        # unsqueeze to make seen_by_model1 broadcastable
        return torch.where(seen_by_model0.unsqueeze(1), self._models[1](data), self._models[0](data))

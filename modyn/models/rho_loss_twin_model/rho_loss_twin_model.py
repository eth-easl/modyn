from typing import Any, Optional

import torch
from modyn.utils import dynamic_module_import
from torch import nn


class RHOLOSSTwinModel:

    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = RHOLOSSTwinModelModyn(model_configuration, device, amp)
        self.model.to(device)


class RHOLOSSTwinModelModyn(nn.Module):

    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        super().__init__()
        model_module = dynamic_module_import("modyn.models")
        rho_model_class = model_configuration["rho_real_model_class"]
        rho_model_config = model_configuration["rho_real_model_config"]
        model_handler = getattr(model_module, rho_model_class)
        # we only need the inner model, not the wrapper
        self.model0 = model_handler(rho_model_config, device, amp).model
        self.model1 = model_handler(rho_model_config, device, amp).model
        self.model0_seen_ids: set[int] = set()
        self.model1_seen_ids: set[int] = set()

    def get_extra_state(self) -> Any:
        return {
            "model0_seen_ids": self.model0_seen_ids,
            "model1_seen_ids": self.model1_seen_ids,
        }

    def set_extra_state(self, state: dict) -> None:
        self.model0_seen_ids = state["model0_seen_ids"]
        self.model1_seen_ids = state["model1_seen_ids"]

    def forward(self, data: torch.Tensor, sample_ids: Optional[list[int]] = None) -> torch.Tensor:
        assert sample_ids is not None
        if self.training:
            output_tensor = self._training_forward(sample_ids, data)
        else:
            output_tensor = self._eval_forward(sample_ids, data)
        return output_tensor

    def _training_forward(self, sample_ids: list[int], data: torch.Tensor) -> torch.Tensor:
        is_seen_by_model0 = any(sample_id in self.model0_seen_ids for sample_id in sample_ids)
        is_seen_by_model1 = any(sample_id in self.model1_seen_ids for sample_id in sample_ids)
        if is_seen_by_model0 and is_seen_by_model1:
            raise ValueError("Sample ID is seen by both models; This shouldn't happen with IL model.")

        if is_seen_by_model0 or is_seen_by_model1:
            if is_seen_by_model0:
                return self.model0(data)
            return self.model1(data)

        train_by_model0 = len(self.model0_seen_ids) <= len(self.model1_seen_ids)
        if train_by_model0:
            self.model0_seen_ids.update(sample_ids)
            return self.model0(data)

        self.model1_seen_ids.update(sample_ids)
        return self.model1(data)

    def _eval_forward(self, sample_ids: list[int], data: torch.Tensor) -> torch.Tensor:
        is_seen_by_model1 = any(sample_id in self.model1_seen_ids for sample_id in sample_ids)
        if is_seen_by_model1:
            return self.model0(data)

        # when a sample is not seen by any model, we default to model0
        return self.model0(data)

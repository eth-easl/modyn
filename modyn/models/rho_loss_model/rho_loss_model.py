from typing import Any, Optional

from torch import nn
import torch

from modyn.utils import dynamic_module_import


class RHOLOSSModel:

    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = RHOLOSSModelModyn(model_configuration, device, amp)
        self.model.to(device)


class RHOLOSSModelModyn(nn.Module):

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

    def set_extra_state(self, state: dict):
        self.model0_seen_ids = state["model0_seen_ids"]
        self.model1_seen_ids = state["model1_seen_ids"]

    def forward(self, data: torch.Tensor, sample_ids: Optional[list[int]] = None):
        assert sample_ids is not None
        if self.training:
            return self._training_forward(sample_ids, data)
        else:
            return self._eval_forward(sample_ids, data)

    def _training_forward(self, sample_ids: list[int], data: torch.Tensor):
        is_seen_by_model0 = any(sample_id in self.model0_seen_ids for sample_id in sample_ids)
        is_seen_by_model1 = any(sample_id in self.model1_seen_ids for sample_id in sample_ids)
        if is_seen_by_model0 and is_seen_by_model1:
            raise ValueError("Sample ID is seen by both models; This shouldn't happen with IL model.")

        if is_seen_by_model0:
            return self.model0(data)
        elif is_seen_by_model1:
            return self.model1(data)
        else:
            train_by_model0 = len(self.model0_seen_ids) <= len(self.model1_seen_ids)
            if train_by_model0:
                self.model0_seen_ids.update(sample_ids)
                return self.model0(data)
            else:
                self.model1_seen_ids.update(sample_ids)
                return self.model1(data)

    def _eval_forward(self, sample_ids: list[int], data: torch.Tensor):
        is_seen_by_model0 = any(sample_id in self.model0_seen_ids for sample_id in sample_ids)
        is_seen_by_model1 = any(sample_id in self.model1_seen_ids for sample_id in sample_ids)

        if is_seen_by_model0:
            return self.model1(data)
        elif is_seen_by_model1:
            return self.model0(data)
        else:
            return self.model0(data)

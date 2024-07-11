import logging
from typing import Any, Optional

import torch
import copy
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
        model_module = dynamic_module_import("modyn.models")
        rho_model_class = model_configuration["rho_real_model_class"]
        rho_model_config = model_configuration["rho_real_model_config"]
        model_handler = getattr(model_module, rho_model_class)
        # we only need the inner model, not the wrapper
        self._models = nn.ModuleList(
            [
                # some models change the model config dict during initialization
                model_handler(copy.deepcopy(rho_model_config), device, amp).model,
                model_handler(copy.deepcopy(rho_model_config), device, amp).model,
            ]
        )
        self._models_seen_ids: list[set[int]] = [set(), set()]
        self._current_model = 0

    def get_extra_state(self) -> dict:
        return {
            "models_seen_ids": self._models_seen_ids,
            "current_model": self._current_model,
        }

    def set_extra_state(self, state: dict) -> None:
        self._models_seen_ids = state["models_seen_ids"]
        previous_model = state["current_model"]
        # the lifecycle of a twin model:
        # 1. initialized with random state or the previous twin model state depending
        # on whether `use_previous_model` is set. If the former, set_extra_state is not called so
        # self._current_model is 0. If the latter, set_extra_state is called with the state of the previous model
        # where "current_model" is 1. In the end self._current_model is also 1 - 1 = 0.
        #
        # 2. When the model is trained for the first time, self._current_model is 0. The first half is trained.
        # After the first training, it's state is stored with "current_model" = 0 (with a new model id).
        #
        # 3. When the model is trained for the second time, the state is loaded with "current_model" = 0.
        # So self._current_model = 1 - 0 = 1. The second half is trained.
        #
        # 4. The model is loaded again in the irreducible loss producer.
        # The value of self._current_model does not matter as we do not re-store it.
        self._current_model = 1 - previous_model

        if previous_model == 1 and self.training:
            logger.info("Finetune on a model that has been trained before. Resetting seen ids.")
            self._models_seen_ids = [set(), set()]

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
        seen_by_model0 = torch.tensor(
            [sample_id in self._models_seen_ids[0] for sample_id in sample_ids], dtype=torch.bool, device=data.device
        )
        seen_by_model1 = torch.tensor(
            [sample_id in self._models_seen_ids[1] for sample_id in sample_ids], dtype=torch.bool, device=data.device
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

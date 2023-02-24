# pylint: disable=too-many-instance-attributes
from typing import Any
import torch


class DLRMscheduler:
    def __init__(self, optimizers: list[torch.optim.Optimizer], scheduler_config: dict[str, Any]) -> None:
        self.current_step = 0
        self.optimizers = optimizers
        self.base_lrs = scheduler_config["base_lrs"]
        self.warmup_steps = scheduler_config["warmup_steps"]
        self.warmup_factor = scheduler_config["warmup_factor"]
        self.decay_steps = scheduler_config["decay_steps"]
        self.decay_start_step = scheduler_config["decay_start_step"]
        self.decay_power = scheduler_config["decay_power"]
        self.end_lr_factor = scheduler_config["end_lr_factor"]
        self.decay_end_step = self.decay_start_step + self.decay_steps

        if self.decay_start_step < self.warmup_steps:
            raise ValueError("Learning rate warmup must finish before decay starts")

    def _compute_lr_factor(self) -> int:
        lr_factor = 1

        if self.current_step <= self.warmup_steps:
            warmup_step = 1 / (self.warmup_steps * (2**self.warmup_factor))
            lr_factor = 1 - (self.warmup_steps - self.current_step) * warmup_step
        elif self.decay_start_step < self.current_step <= self.decay_end_step:
            lr_factor = ((self.decay_end_step - self.current_step) / self.decay_steps) ** self.decay_power
            lr_factor = max(lr_factor, self.end_lr_factor)
        elif self.current_step > self.decay_end_step:
            lr_factor = self.end_lr_factor

        return lr_factor

    def step(self) -> None:
        self.current_step += 1
        lr_factor = self._compute_lr_factor()

        for optim, base_lrs in zip(self.optimizers, self.base_lrs):
            for group_id, base_lr in enumerate(base_lrs):
                optim.param_groups[group_id]["lr"] = base_lr * lr_factor

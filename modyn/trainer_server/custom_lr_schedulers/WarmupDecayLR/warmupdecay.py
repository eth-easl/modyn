import math
from typing import Any

from torch.optim import Optimizer

WARMUP_LOG_RATE = "log"
WARMUP_LINEAR_RATE = "linear"


def get_torch_optimizer(optimizer: Any) -> Optimizer:
    if isinstance(optimizer, Optimizer):
        return optimizer

    if hasattr(optimizer, "optimizer") and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer

    raise TypeError(f"{type(optimizer).__name__} is not a subclass of torch.optim.Optimizer")


def update_lr(param_groups: list[dict[str, Any]], lrs: list[float]) -> list[float]:
    for param_group, lr in zip(param_groups, lrs):
        param_group["lr"] = lr
    return [group["lr"] for group in param_groups]


class WarmupLR:
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_min_lr: float | list[float] = 0.0,
        warmup_max_lr: float | list[float] = 0.001,
        warmup_num_steps: int = 1000,
        warmup_type: str = WARMUP_LOG_RATE,
        last_batch_iteration: int = -1,
    ) -> None:
        self.optimizer = get_torch_optimizer(optimizer)

        self.min_lrs = self._format_param(self.optimizer, warmup_min_lr, "min_lr")
        self.max_lrs = self._format_param(self.optimizer, warmup_max_lr, "max_lr")
        self.delta_lrs = [big - small for big, small in zip(self.max_lrs, self.min_lrs)]
        self.warmup_num_steps = max(2, warmup_num_steps)
        if warmup_type not in {WARMUP_LOG_RATE, WARMUP_LINEAR_RATE}:
            print("Using unknown warmup_type. The increasing function " "is set to default (log)")
            warmup_type = WARMUP_LOG_RATE
        self.warmup_type = warmup_type
        self.inverse_log_warm_up = 1.0 / math.log(self.warmup_num_steps)
        self.last_batch_iteration = last_batch_iteration
        if last_batch_iteration == -1:
            self._last_lr = update_lr(self.optimizer.param_groups, self.get_lr())

    def get_lr(self) -> list[float]:
        if self.last_batch_iteration < 0:
            print("Attempting to get learning rate from scheduler before it has started")
            return self.min_lrs
        gamma = self._get_gamma()
        return [min_lr + (delta_lr * gamma) for min_lr, delta_lr in zip(self.min_lrs, self.delta_lrs)]

    def get_last_lr(self) -> list[float]:
        assert getattr(self, "_last_lr", None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, last_batch_iteration: int | None = None) -> None:
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        self._last_lr = update_lr(self.optimizer.param_groups, self.get_lr())

    def state_dict(self) -> dict[str, int]:
        return {"last_batch_iteration": self.last_batch_iteration}

    def load_state_dict(self, sd: dict[str, int]) -> None:
        self.last_batch_iteration = sd["last_batch_iteration"]

    def _get_gamma(self) -> float:
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        return 1.0

    def _format_param(self, optimizer: Optimizer, param_value: float | list[float], param_name: str) -> list[float]:
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} value for {param_name}, got {FileNotFoundError(param_value)}"
                )
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)


class WarmupDecayLR(WarmupLR):
    def __init__(self, optimizers: list[Optimizer], scheduler_config: dict[str, Any]) -> None:
        if len(optimizers) != 1:
            raise ValueError("Only a single optimizer is supported.")

        self.optimizer = optimizers[0]
        self.current_step = 0

        self.total_num_steps = scheduler_config.get("total_num_steps", 10000)
        self.warmup_min_lr = scheduler_config.get("warmup_min_lr", 0.0)
        self.warmup_max_lr = scheduler_config.get("warmup_max_lr", 0.001)
        self.warmup_num_steps = scheduler_config.get("warmup_num_steps", 1000)
        self.warmup_type = scheduler_config.get("warmup_type", "log")
        self.last_batch_iteration = scheduler_config.get("last_batch_iteration", -1)

        if self.total_num_steps < self.warmup_num_steps:
            raise ValueError("total_num_steps must be greater than or equal to warmup_num_steps.")

        if self.warmup_min_lr >= self.warmup_max_lr:
            raise ValueError("warmup_min_lr must be less than warmup_max_lr.")

        super().__init__(
            self.optimizer,
            self.warmup_min_lr,
            self.warmup_max_lr,
            self.warmup_num_steps,
            self.warmup_type,
            self.last_batch_iteration,
        )

    def _get_gamma(self) -> float:
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        return max(
            0.0,
            float(self.total_num_steps - self.last_batch_iteration)
            / float(max(1.0, self.total_num_steps - self.warmup_num_steps)),
        )

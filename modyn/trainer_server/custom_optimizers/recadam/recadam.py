import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def anneal_function(function: str, step: int, k: float, t0: float, weight: float) -> float:
    """Computes the annealing factor for RecAdam optimization."""
    if function == "sigmoid":
        return float(1 / (1 + np.exp(-k * (step - t0)))) * weight
    elif function == "linear":
        return min(1, step / t0) * weight
    elif function == "constant":
        return weight
    else:
        raise ValueError(f"Invalid anneal function type: {function}")


class RecAdam(Optimizer):
    """Implementation of RecAdam optimizer, a variant of the Adam optimizer."""

    def __init__(
        self,
        params: list[Any] | list[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        anneal_fun: str = "sigmoid",
        anneal_k: float = 0.0,
        anneal_t0: float = 0.0,
        anneal_w: float = 1.0,
        pretrain_cof: float = 5000.0,
        pretrain_params: list[Tensor] | None = None,
    ) -> None:
        """Initializes the RecAdam optimizer."""
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            anneal_fun=anneal_fun,
            anneal_k=anneal_k,
            anneal_t0=anneal_t0,
            anneal_w=anneal_w,
            pretrain_cof=pretrain_cof,
            pretrain_params=pretrain_params,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A function that reevaluates the model
                and returns the loss. Defaults to None.

        Returns:
            Optional[float]: The loss value if a closure is provided, otherwise None.
        """
        loss: float | None = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            pretrain_params: list[Tensor] | None = group["pretrain_params"]
            if pretrain_params is None:
                continue

            for p, pp in zip(group["params"], pretrain_params):
                if p.grad is None:
                    continue
                grad: Tensor = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state: dict[str, Any] = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom: Tensor = exp_avg_sq.sqrt().add_(group["eps"])

                step_size: float = group["lr"]
                if group["correct_bias"]:
                    bias_correction1: float = 1.0 - beta1 ** state["step"]
                    bias_correction2: float = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Apply RecAdam adjustments
                if group["anneal_w"] > 0.0:
                    anneal_lambda: float = anneal_function(
                        group["anneal_fun"], state["step"], group["anneal_k"], group["anneal_t0"], group["anneal_w"]
                    )
                    assert anneal_lambda <= group["anneal_w"]
                    p.data.addcdiv_(-step_size * anneal_lambda, exp_avg, denom)
                    p.data.add_(
                        -group["lr"] * (group["anneal_w"] - anneal_lambda) * group["pretrain_cof"], p.data - pp.data
                    )
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                # Apply weight decay
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss

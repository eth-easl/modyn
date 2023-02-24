# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: ignore-errors

import torch
from torch.autograd import Function


class DotBasedInteract(Function):
    """Forward and Backward paths of cuda extension for dot-based feature interact."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, input, bottom_mlp_output):
        if torch.cuda.get_device_capability()[0] >= 8:
            from modyn.models.dlrm.cuda_ext import interaction_ampere as interaction
        else:
            from modyn.models.dlrm.cuda_ext import interaction_volta as interaction

        output = interaction.dotBasedInteractFwd(input, bottom_mlp_output)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        if torch.cuda.get_device_capability()[0] >= 8:
            from modyn.models.dlrm.cuda_ext import interaction_ampere as interaction
        else:
            from modyn.models.dlrm.cuda_ext import interaction_volta as interaction

        (input,) = ctx.saved_tensors
        grad, mlp_grad = interaction.dotBasedInteractBwd(input, grad_output)
        return grad, mlp_grad


dotBasedInteract = DotBasedInteract.apply

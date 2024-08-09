# pylint: disable=abstract-class-instantiated
from unittest.mock import patch

import torch

from modyn.config import ModynConfig
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_batch_then_sample_general(dummy_system_config: ModynConfig):
    downsampling_ratio = 50

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "ratio_max": 100}
    sampler = AbstractRemoteDownsamplingStrategy(
        154, 128, 64, params_from_selector, dummy_system_config.model_dump(by_alias=True), "cpu"
    )

    assert hasattr(sampler, "downsampling_ratio")
    assert sampler.downsampling_ratio == 50
    assert sampler.trigger_id == 128
    assert sampler.pipeline_id == 154
    assert sampler.batch_size == 64


@patch(
    "modyn.trainer_server.internal.trainer.remote_downsamplers"
    ".abstract_remote_downsampling_strategy.torch.autograd.grad",
    wraps=torch.autograd.grad,
)
def test__compute_last_layer_gradient_wrt_loss_sum(mock_torch_auto_grad):
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    forward_output = torch.randn((4, 2), requires_grad=True)
    # random target
    target = torch.randint(0, 2, (4,))
    last_layer_gradients = AbstractRemoteDownsamplingStrategy._compute_last_layer_gradient_wrt_loss_sum(
        per_sample_loss_fct, forward_output, target
    )
    # as we use CrossEntropyLoss, the gradient is computed in a closed form
    assert mock_torch_auto_grad.call_count == 0
    # verify that the gradients calculated via the closed form are equal to the ones calculated by autograd
    expected_grad = torch.autograd.grad(per_sample_loss_fct(forward_output, target).sum(), forward_output)[0]
    assert torch.allclose(last_layer_gradients, expected_grad)

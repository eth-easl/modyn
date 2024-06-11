from unittest.mock import patch

import pytest
import torch
from modyn.config import ModynConfig
from modyn.models import Dummy
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.irreducible_loss_producer import IrreducibleLossProducer
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_rho_loss_downsampling import (
    RemoteRHOLossDownsampling,
)


def dummy_model():
    return Dummy({}, "cpu", False)


@pytest.fixture
def dummy_init_params(dummy_system_config: ModynConfig):
    pipeline_id = 5
    trigger_id = 3
    batch_size = 32
    params_from_selector = {
        "rho_pipeline_id": 1,
        "il_model_id": 2,
        "downsampling_ratio": 50,
    }
    modyn_config = dummy_system_config.model_dump(by_alias=True)
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    device = "cpu"
    return {
        "pipeline_id": pipeline_id,
        "trigger_id": trigger_id,
        "batch_size": batch_size,
        "params_from_selector": params_from_selector,
        "modyn_config": modyn_config,
        "per_sample_loss": per_sample_loss_fct,
        "device": device,
    }


@patch.object(IrreducibleLossProducer, "_load_il_model", return_value=dummy_model())
@patch.object(AbstractRemoteDownsamplingStrategy, "__init__")
def test__init__(mock_abstract_sampler_init__, mock__load_il_model, dummy_init_params):
    sampler = RemoteRHOLossDownsampling(**dummy_init_params)
    assert sampler.per_sample_loss_fct == dummy_init_params["per_sample_loss"]
    assert not sampler.requires_grad
    mock_abstract_sampler_init__.assert_called_once_with(
        dummy_init_params["pipeline_id"],
        dummy_init_params["trigger_id"],
        dummy_init_params["batch_size"],
        dummy_init_params["params_from_selector"],
        dummy_init_params["modyn_config"],
        dummy_init_params["device"],
    )


@patch.object(IrreducibleLossProducer, "_load_il_model", return_value=dummy_model())
def test_inform_samples(mock__load_il_model, dummy_init_params):
    batch_size = 3

    def fake_per_sample_loss(forward_output, target):
        return torch.tensor(range(batch_size))

    dummy_init_params["per_sample_loss"] = fake_per_sample_loss
    dummy_init_params["batch_size"] = batch_size
    sampler = RemoteRHOLossDownsampling(**dummy_init_params)
    sample_ids = list(range(batch_size))
    forward_input = torch.randn(batch_size, 5)
    forward_output = torch.randn(batch_size, 5)
    target = torch.randint(5, (batch_size,))
    embedding = None
    fake_irreducible_loss = torch.tensor(range(batch_size, 0, -1))
    with patch.object(IrreducibleLossProducer, "get_irreducible_loss", return_value=fake_irreducible_loss):
        sampler.init_downsampler()
        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)
        assert sampler.index_sampleid_map == sample_ids
        assert sampler.rho_loss.shape == torch.Size([batch_size])
        assert sampler.number_of_points_seen == 3
        expected_rho_loss = torch.tensor([0, 1, 2]) - torch.tensor([3, 2, 1])
        assert torch.allclose(sampler.rho_loss, expected_rho_loss)


@patch.object(IrreducibleLossProducer, "_load_il_model", return_value=dummy_model())
def test_select_points(mock__load_il_model, dummy_init_params):
    dummy_init_params["batch_size"] = 5
    dummy_init_params["params_from_selector"]["downsampling_ratio"] = 60
    sampler = RemoteRHOLossDownsampling(**dummy_init_params)
    sampler.rho_loss = torch.tensor([32, 12, 5, 8, 3])
    sampler.number_of_points_seen = 5
    sampler.index_sampleid_map = [2, 4, 1, 3, 5]
    selected_ids, weights = sampler.select_points()
    assert selected_ids == [2, 4, 3]
    assert torch.allclose(weights, torch.tensor([1.0, 1.0, 1.0]))

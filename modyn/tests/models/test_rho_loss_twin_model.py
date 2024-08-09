import io
import tempfile
from unittest.mock import Mock, call, patch

import pytest
import torch

from modyn.models import RHOLOSSTwinModel
from modyn.models.dummy.dummy import Dummy, DummyModyn
from modyn.models.rho_loss_twin_model.rho_loss_twin_model import RHOLOSSTwinModelModyn


@pytest.fixture
def twin_model() -> RHOLOSSTwinModel:
    return RHOLOSSTwinModel(
        model_configuration={
            "rho_real_model_class": "Dummy",
            "rho_real_model_config": {"num_classes": 10},
        },
        device="cpu",
        amp=False,
    )


@patch("modyn.models.Dummy", wraps=Dummy)
def test_init(MockDummy):
    model = RHOLOSSTwinModel(
        model_configuration={
            "rho_real_model_class": "Dummy",
            "rho_real_model_config": {"num_classes": 10},
        },
        device="cpu",
        amp=False,
    )
    assert len(model.model._models) == 2
    assert isinstance(model.model._models[0], DummyModyn)
    assert isinstance(model.model._models[1], DummyModyn)
    assert model.model._models_seen_ids == [set(), set()]
    assert model.model._current_model == 0
    # assert called twice with the same arguments
    expected_call = call({"num_classes": 10}, "cpu", False)
    MockDummy.assert_has_calls([expected_call, expected_call])


def test_forward_missing_sample_ids(twin_model: RHOLOSSTwinModel):
    with pytest.raises(AssertionError):
        twin_model.model.train()
        twin_model.model(torch.randn(3, 2))
    with pytest.raises(AssertionError):
        twin_model.model.eval()
        twin_model.model(torch.randn(3, 2))


@pytest.mark.parametrize("current_model", [0, 1])
@patch.object(RHOLOSSTwinModelModyn, "_eval_forward")
def test_training_forward(mock__eval_forward, current_model, twin_model: RHOLOSSTwinModel):
    twin_model.model._models[1 - current_model].forward = Mock()
    twin_model.model._current_model = current_model
    assert twin_model.model._models_seen_ids == [set(), set()]
    twin_model.model.train()
    sample_ids = [2, 1, 3]
    forward_input = torch.randn(3, 2)
    twin_model.model(forward_input, sample_ids)
    assert twin_model.model._current_model == current_model
    assert twin_model.model._models_seen_ids[current_model] == set(sample_ids)
    assert twin_model.model._models_seen_ids[1 - current_model] == set()

    # another forward with the same sample_ids
    twin_model.model(forward_input, sample_ids)
    assert twin_model.model._current_model == current_model
    assert twin_model.model._models_seen_ids[current_model] == set(sample_ids)
    assert twin_model.model._models_seen_ids[1 - current_model] == set()

    sample_ids = [3, 4]
    twin_model.model(forward_input, sample_ids)
    assert twin_model.model._current_model == current_model
    assert twin_model.model._models_seen_ids[current_model] == {2, 1, 3, 4}
    assert twin_model.model._models_seen_ids[1 - current_model] == set()

    twin_model.model._models[1 - current_model].forward.assert_not_called()
    mock__eval_forward.assert_not_called()


@pytest.mark.parametrize("exclusive_model", [0, 1])
@patch.object(RHOLOSSTwinModelModyn, "_training_forward")
def test_eval_forward_exclusively_route_to_one_model(
    mock__training_forward, exclusive_model: int, twin_model: RHOLOSSTwinModel
):
    twin_model.model._models[1 - exclusive_model].forward = Mock(return_value=torch.zeros(3, 2))
    twin_model.model._models[exclusive_model].forward = Mock(return_value=torch.ones(3, 2))

    twin_model.model.eval()
    sample_ids = [1, 2, 3]
    forward_input = torch.randn(3, 2)

    twin_model.model._models_seen_ids[1 - exclusive_model] = set(sample_ids)
    twin_model.model._models_seen_ids[exclusive_model] = set()

    output = twin_model.model(forward_input, sample_ids)
    assert torch.allclose(output, torch.ones(3, 2))

    twin_model.model._models[1 - exclusive_model].forward.assert_not_called()
    twin_model.model._models[exclusive_model].forward.assert_called_once()
    # we never call _training_forward in eval mode
    assert mock__training_forward.call_count == 0


def test_eval_forward_mixed(twin_model: RHOLOSSTwinModel):
    def model0_mock_forward(data: torch.Tensor):
        return torch.zeros(data.shape[0], 10)

    def model1_mock_forward(data: torch.Tensor):
        return torch.ones(data.shape[0], 10)

    twin_model.model._models[0].forward = model0_mock_forward
    twin_model.model._models[1].forward = model1_mock_forward
    twin_model.model._models_seen_ids = [{1, 2}, {3, 4}]
    twin_model.model.eval()

    sample_ids = [1, 4, 3, 2, 5]
    forward_input = torch.randn(len(sample_ids), 2)
    output = twin_model.model(forward_input, sample_ids)
    assert torch.allclose(output, torch.tensor([[1.0] * 10, [0.0] * 10, [0.0] * 10, [1.0] * 10, [0.0] * 10]))


@pytest.mark.parametrize("current_model", [0, 1])
@pytest.mark.parametrize("training_mode", [True, False])
def test_backup_and_restore_state(current_model: int, training_mode: bool, twin_model: RHOLOSSTwinModel):
    model_seen_ids = [{1, 2, 3}, {4, 5}]
    twin_model.model._models_seen_ids = model_seen_ids
    twin_model.model._current_model = current_model
    twin_model.model._models[0].output.weight = torch.nn.Parameter(torch.zeros(2, 2))
    twin_model.model._models[0].output.bias = torch.nn.Parameter(torch.zeros(2))
    twin_model.model._models[1].output.weight = torch.nn.Parameter(torch.ones(2, 2))
    twin_model.model._models[1].output.bias = torch.nn.Parameter(torch.ones(2))

    with tempfile.NamedTemporaryFile() as model_file:
        torch.save({"model": twin_model.model.state_dict()}, model_file.name)
        new_twin_model = RHOLOSSTwinModel(
            model_configuration={
                "rho_real_model_class": "Dummy",
                "rho_real_model_config": {"num_classes": 10},
            },
            device="cpu",
            amp=False,
        )
        new_twin_model.model.train(training_mode)
        assert not torch.allclose(new_twin_model.model._models[0].output.weight, torch.zeros(2, 2))
        assert not torch.allclose(new_twin_model.model._models[0].output.bias, torch.zeros(2))
        assert not torch.allclose(new_twin_model.model._models[1].output.weight, torch.ones(2, 2))
        assert not torch.allclose(new_twin_model.model._models[1].output.bias, torch.ones(2))
        with open(model_file.name, "rb") as f:
            checkpoint = torch.load(io.BytesIO(f.read()), map_location="cpu")
            new_twin_model.model.load_state_dict(checkpoint["model"])

        if current_model == 1 and training_mode:
            assert new_twin_model.model._models_seen_ids == [set(), set()]
        else:
            assert new_twin_model.model._models_seen_ids == model_seen_ids
        assert new_twin_model.model._current_model == 1 - current_model
        assert torch.allclose(new_twin_model.model._models[0].output.weight, torch.zeros(2, 2))
        assert torch.allclose(new_twin_model.model._models[0].output.bias, torch.zeros(2))
        assert torch.allclose(new_twin_model.model._models[1].output.weight, torch.ones(2, 2))
        assert torch.allclose(new_twin_model.model._models[1].output.bias, torch.ones(2))
        assert new_twin_model.model.training == training_mode

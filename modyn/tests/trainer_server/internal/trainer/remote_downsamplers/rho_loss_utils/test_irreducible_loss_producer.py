import json
import os
import pathlib
import tempfile
from unittest.mock import ANY, Mock, patch

import pytest
import torch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.utils import ModelStorageStrategyConfig

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.models import Dummy
from modyn.models.dummy.dummy import DummyModyn
from modyn.trainer_server.internal.trainer.remote_downsamplers.rho_loss_utils.irreducible_loss_producer import (
    IrreducibleLossProducer,
)
from torch import Tensor

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"


@pytest.fixture
def minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "hostname": "",
            "port": "0",
            "database": f"{database_path}",
        },
        "model_storage": {
            "hostname": "localhost",
            "port": "50059",
        },
    }


def get_dummy_model():
    return Dummy({}, "cpu", False)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(minimal_modyn_config):
    with MetadataDatabaseConnection(minimal_modyn_config) as database:
        database.create_tables()
    yield


@patch("modyn.models.Dummy", wraps=Dummy)
def test__load_il_model_architecture(MockDummy, minimal_modyn_config):
    model_config = {"num_classes": 79}
    with MetadataDatabaseConnection(minimal_modyn_config) as database:
        pipeline_id = database.register_pipeline(
            2, "Dummy", json.dumps(model_config), True, "{}", "{}", ModelStorageStrategyConfig(name="PyTorchFullModel")
        )

    model = IrreducibleLossProducer._load_il_model_architecture(minimal_modyn_config, pipeline_id)
    assert isinstance(model, Dummy)
    MockDummy.assert_called_once_with(model_config, "cpu", True)


@patch.object(IrreducibleLossProducer, "_load_il_model_architecture", return_value=get_dummy_model())
def test__load_il_model(mock__load_il_model_architecture, minimal_modyn_config):
    expected_weight = torch.tensor([[-1231.0, 2.0], [3.0, 4.0]])
    expected_bias = torch.tensor([5.0, 667.0])
    with tempfile.NamedTemporaryFile() as model_weights_file:
        dummy_model = get_dummy_model()
        # hard code dummy_model's weights so that we can verify later if correct params are loaded
        dummy_model.model.output.weight = torch.nn.Parameter(expected_weight)
        dummy_model.model.output.bias = torch.nn.Parameter(expected_bias)
        # save weight and bias to temp_file
        torch.save({"model": dummy_model.model.state_dict()}, model_weights_file.name)

        mock_model_storage_stub = Mock(spec=["FetchModel"])
        mock_model_storage_stub.FetchModel.return_value = FetchModelResponse(
            success=True,
            model_path="dummy_model_path",
            checksum=bytes(12345),
        )
        # patch the connect_to_model_storage and download_trained_model so that we the downloaded model file is this
        # model_weights_file
        with (
            patch.object(IrreducibleLossProducer, "connect_to_model_storage", return_value=mock_model_storage_stub),
            patch(
                "modyn.trainer_server.internal.trainer.remote_downsamplers."
                "rho_loss_utils.irreducible_loss_producer.download_trained_model",
                return_value=model_weights_file.name,
            ) as mock_download_trained_model,
        ):
            rho_pipeline_id = 12
            il_model_id = 43
            model = IrreducibleLossProducer._load_il_model(minimal_modyn_config, rho_pipeline_id, il_model_id)
            assert isinstance(model, Dummy)
            assert torch.allclose(model.model.output.weight, expected_weight)
            assert torch.allclose(model.model.output.bias, expected_bias)

            mock_model_storage_stub.FetchModel.assert_called_once_with(
                FetchModelRequest(model_id=il_model_id, load_metadata=True)
            )

            mock_download_trained_model.assert_called_once_with(
                logger=ANY,
                model_storage_config=ANY,
                remote_path=pathlib.Path("dummy_model_path"),
                checksum=bytes(12345),
                identifier=il_model_id,
                base_directory=ANY,
            )


@patch.object(IrreducibleLossProducer, "_load_il_model", return_value=get_dummy_model())
def test_get_irreducible_loss_cached(minimal_modyn_config):
    def fake_per_sample_loss(forward_output, target):
        return 27 * torch.ones(forward_output.shape[0])

    mock_per_sample_loss = Mock(wraps=fake_per_sample_loss)

    il_loss_producer = IrreducibleLossProducer(mock_per_sample_loss, minimal_modyn_config, 43, 12, "cpu")
    sample_ids = [2, 1, 3]
    forward_input = torch.randn(3, 2)
    target = torch.randn(3, 5)
    il_loss_producer.loss_cache = {1: torch.tensor(-1.0), 2: torch.tensor(-2.0), 3: torch.tensor(-3.0)}
    il_loss = il_loss_producer.get_irreducible_loss(sample_ids, forward_input, target)
    # verify that we didn't calculate the loss on the fly
    mock_per_sample_loss.assert_not_called()
    assert torch.allclose(il_loss, torch.tensor([-2.0, -1.0, -3.0]))


@patch.object(IrreducibleLossProducer, "_load_il_model", return_value=get_dummy_model())
def test_get_irreducible_loss_uncached(minimal_modyn_config: dict):
    def fake_per_sample_loss(forward_output, target):
        return 27 * torch.ones(forward_output.shape[0])

    mock_per_sample_loss = Mock(wraps=fake_per_sample_loss)

    num_classes = 5

    def fake_forward(self, x: Tensor):
        return torch.zeros(x.shape[0], num_classes)

    with patch.object(DummyModyn, "forward", fake_forward):
        il_loss_producer = IrreducibleLossProducer(mock_per_sample_loss, minimal_modyn_config, 43, 12, "cpu")
        sample_ids = [2, 1, 3]
        forward_input = torch.randn(3, 2)
        target = torch.randn(3, num_classes)
        il_loss = il_loss_producer.get_irreducible_loss(sample_ids, forward_input, target)
        assert torch.allclose(il_loss, torch.tensor([27.0, 27.0, 27.0]))

        mock_per_sample_loss.assert_called_once()
        # verify that the per_sample_loss function was called with the correct arguments
        assert torch.allclose(mock_per_sample_loss.call_args[0][0], torch.zeros(3, num_classes))
        assert torch.allclose(mock_per_sample_loss.call_args[0][1], target)

        # verify that the newly computed loss was cached
        assert il_loss_producer.loss_cache.keys() == {1, 2, 3}
        for loss in il_loss_producer.loss_cache.values():
            assert torch.allclose(loss, torch.tensor(27.0))


@patch(
    "modyn.trainer_server.internal.trainer.remote_downsamplers.rho_loss_utils."
    "irreducible_loss_producer.grpc_connection_established",
    return_value=True,
)
def test_connect_to_model_storage(mock_grpc_connection_established):
    model_storage_stub = IrreducibleLossProducer.connect_to_model_storage("localhost:50059")
    assert model_storage_stub is not None
    mock_grpc_connection_established.assert_called_once()

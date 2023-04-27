import pathlib
import tempfile
from datetime import datetime
from unittest.mock import patch

from modyn.common.ftp.ftp_server import FTPServer
from modyn.metadata_database.models import TrainedModel

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.model_storage.internal.grpc.model_storage_grpc_servicer import ModelStorageGRPCServicer


def get_modyn_config():
    return {
        "model_storage": {"port": "50051", "ftp_port": "5223"},
        "trainer_server": {"hostname": "localhost", "ftp_port": "5222"},
    }


class MockSession:
    def get(self, entity, model_id) -> TrainedModel:  # pylint: disable=unused-argument
        assert model_id == 20
        return TrainedModel(
            model_id=model_id,
            pipeline_id=100,
            trigger_id=10,
            timestamp=datetime.fromtimestamp(1000),
            model_path="test_model.modyn",
        )


class MockDatabaseConnection:
    def __init__(self, modyn_config: dict):  # pylint: disable=super-init-not-called,unused-argument
        self.current_model_id = 50
        self.session = MockSession()

    # pylint: disable=unused-argument
    def add_trained_model(self, pipeline_id: int, trigger_id: int, model_path: str) -> int:
        model_id = self.current_model_id
        self.current_model_id += 1
        return model_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception):
        pass


@patch("modyn.model_storage.internal.grpc.model_storage_grpc_servicer.current_time_millis", return_value=100)
@patch(
    "modyn.model_storage.internal.grpc.model_storage_grpc_servicer.MetadataDatabaseConnection", MockDatabaseConnection
)
def test_register_model(current_time_millis):  # pylint: disable=unused-argument
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as storage_dir:
        storage_path = pathlib.Path(storage_dir)

        with open(storage_path / "test.txt", "wb") as file:
            file.write(b"Our test model")

        servicer = ModelStorageGRPCServicer(config, storage_path)
        assert servicer is not None

        with FTPServer(str(5222), storage_path):
            req = RegisterModelRequest(
                pipeline_id=10,
                trigger_id=42,
                hostname=config["trainer_server"]["hostname"],
                port=int(config["trainer_server"]["ftp_port"]),
                model_path="test.txt",
            )
            resp: RegisterModelResponse = servicer.RegisterModel(req, None)

            assert resp.success
            assert resp.model_id == 50

        # download file under path {current_time_millis}_{pipeline_id}_{trigger_id}.modyn
        with open(storage_path / "100_10_42.modyn", "rb") as file:
            assert file.read().decode("utf-8") == "Our test model"


@patch(
    "modyn.model_storage.internal.grpc.model_storage_grpc_servicer.MetadataDatabaseConnection", MockDatabaseConnection
)
def test_fetch_model():
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as storage_dir:
        storage_path = pathlib.Path(storage_dir)

        servicer = ModelStorageGRPCServicer(config, storage_path)
        assert servicer is not None

        req = FetchModelRequest(model_id=20)
        resp: FetchModelResponse = servicer.FetchModel(req, None)

        assert resp.success
        assert resp.model_path == "test_model.modyn"

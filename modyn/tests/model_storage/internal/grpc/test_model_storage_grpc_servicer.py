# pylint: disable=unused-argument
import pathlib
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import torch
from modyn.model_storage.internal import ModelStorageManager

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    DeleteModelRequest,
    DeleteModelResponse,
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.model_storage.internal.grpc.model_storage_grpc_servicer import ModelStorageGRPCServicer
from modyn.utils import calculate_checksum


def get_modyn_config():
    return {
        "model_storage": {"port": "50051", "ftp_port": "5223"},
        "trainer_server": {"hostname": "localhost", "ftp_port": "5222"},
    }


@patch("modyn.model_storage.internal.grpc.model_storage_grpc_servicer.download_file", return_value=True)
@patch("modyn.model_storage.internal.grpc.model_storage_grpc_servicer.current_time_millis", return_value=100)
@patch.object(ModelStorageManager, "__init__", return_value=None)
@patch.object(ModelStorageManager, "store_model", return_value=15)
@patch("os.remove")
def test_register_model(
    os_remove_mock: MagicMock,
    store_model_mock: MagicMock,
    init_manager_mock,
    current_time_millis,
    download_file_mock: MagicMock,
):
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as storage_dir:
        storage_path = pathlib.Path(storage_dir)

        with open(storage_path / "test.txt", "wb") as file:
            file.write(b"Our test model")

        servicer = ModelStorageGRPCServicer(config, storage_path, storage_path)
        assert servicer is not None

        req = RegisterModelRequest(
            pipeline_id=1,
            trigger_id=10,
            hostname=config["trainer_server"]["hostname"],
            port=int(config["trainer_server"]["ftp_port"]),
            model_path="test.txt",
            checksum=calculate_checksum(storage_path / "test.txt"),
        )

        resp: RegisterModelResponse = servicer.RegisterModel(req, None)

        download_file_mock.assert_called_once()
        kwargs = download_file_mock.call_args.kwargs

        remote_file_path = kwargs["remote_file_path"]
        local_file_path = kwargs["local_file_path"]

        shutil.copyfile(storage_path / remote_file_path, local_file_path)

        assert resp.success
        assert resp.model_id == 15

        # download file under path {current_time_millis}_{pipeline_id}_{trigger_id}.zip
        with open(storage_path / "100_1_10.modyn", "rb") as file:
            assert file.read().decode("utf-8") == "Our test model"

        assert calculate_checksum(storage_path / "100_1_10.modyn") == kwargs["checksum"]
        os_remove_mock.assert_called_with(storage_path / "100_1_10.modyn")


@patch("modyn.model_storage.internal.grpc.model_storage_grpc_servicer.download_file", return_value=False)
@patch("modyn.model_storage.internal.grpc.model_storage_grpc_servicer.current_time_millis", return_value=100)
@patch.object(ModelStorageManager, "__init__", return_value=None)
@patch.object(ModelStorageManager, "store_model")
def test_register_model_invalid(
    store_model_mock: MagicMock, init_manager_mock, current_time_millis, download_file_mock: MagicMock
):
    config = get_modyn_config()
    storage_path = pathlib.Path("storage_dir")
    servicer = ModelStorageGRPCServicer(config, storage_path, storage_path)

    assert servicer is not None
    req = RegisterModelRequest(
        pipeline_id=1,
        trigger_id=10,
        hostname=config["trainer_server"]["hostname"],
        port=int(config["trainer_server"]["ftp_port"]),
        model_path="test.txt",
        checksum=bytes([7, 1, 0]),
    )

    resp: RegisterModelResponse = servicer.RegisterModel(req, None)
    download_file_mock.assert_called_once()

    assert not resp.success
    store_model_mock.assert_not_called()


@patch("modyn.model_storage.internal.grpc.model_storage_grpc_servicer.current_time_millis", return_value=100)
@patch.object(ModelStorageManager, "__init__", return_value=None)
@patch.object(ModelStorageManager, "load_model", return_value={"model": {"conv_1": 1}, "metadata": True})
def test_fetch_model(load_model_mock: MagicMock, init_manager_mock, current_time_millis):
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as storage_dir:
        storage_path = pathlib.Path(storage_dir)

        servicer = ModelStorageGRPCServicer(config, storage_path, storage_path)
        assert servicer is not None

        req = FetchModelRequest(model_id=10, load_metadata=True)
        resp: FetchModelResponse = servicer.FetchModel(req, None)

        assert resp.success
        load_model_mock.assert_called_once_with(10, True)

        # store final model to {current_time_millis()}_{model_id}.zip
        assert resp.model_path == "100_10.modyn"

        assert torch.load(storage_path / resp.model_path) == {"model": {"conv_1": 1}, "metadata": True}


@patch.object(ModelStorageManager, "__init__", return_value=None)
@patch.object(ModelStorageManager, "load_model", return_value=None)
def test_fetch_model_invalid(load_model_mock: MagicMock, init_manager_mock):
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as storage_dir:
        storage_path = pathlib.Path(storage_dir)

        servicer = ModelStorageGRPCServicer(config, storage_path, storage_dir)
        assert servicer is not None

        req = FetchModelRequest(model_id=101, load_metadata=False)
        resp: FetchModelResponse = servicer.FetchModel(req, None)

        assert not resp.success

        req = FetchModelRequest(model_id=101, load_metadata=True)
        resp: FetchModelResponse = servicer.FetchModel(req, None)

        assert not resp.success


@patch.object(ModelStorageManager, "__init__", return_value=None)
@patch.object(ModelStorageManager, "delete_model", return_value=True)
def test_delete_model(delete_model_mock: MagicMock, init_manager_mock):
    config = get_modyn_config()
    servicer = ModelStorageGRPCServicer(config, pathlib.Path("storage_dir"), pathlib.Path("ftp_dir"))
    assert servicer is not None

    req = DeleteModelRequest(model_id=20)
    resp: DeleteModelResponse = servicer.DeleteModel(req, None)

    assert resp.success
    delete_model_mock.assert_called_once_with(20)


@patch.object(ModelStorageManager, "__init__", return_value=None)
@patch.object(ModelStorageManager, "delete_model", return_value=False)
def test_delete_model_invalid(delete_model_mock: MagicMock, init_manager_mock):
    config = get_modyn_config()
    servicer = ModelStorageGRPCServicer(config, pathlib.Path("storage_dir"), pathlib.Path("ftp_dir"))
    assert servicer is not None

    req = DeleteModelRequest(model_id=50)
    resp: DeleteModelResponse = servicer.DeleteModel(req, None)

    assert not resp.success
    delete_model_mock.assert_called_once_with(50)
